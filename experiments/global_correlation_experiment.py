"""
Global correlation experiment: LLM relevance (batched) vs cosine distance to GT embedding.

Layout (logical):
  constants & IDs / text pruning -> GT retrieval -> Gemini scoring -> SQLite lifecycle
  -> run-dir / resume -> analytics -> vector pool I/O -> main loop
"""

import json
import os
import re
import sqlite3
import uuid
from typing import Any
from datetime import datetime

import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from google import genai
from google.genai import types
from scipy import stats

from configurations.config import (
    CORRELATION_LLM_PROVIDER,
    CORRELATION_OLLAMA_MODEL,
    CORRELATION_PILOT_MAX_DOCS_PER_QUERY,
    GEMINI_API_KEY,
    OLLAMA_HOST,
    OLLAMA_TIMEOUT_S,
    OLLAMA_VERIFY_SSL,
)
from scripts.qa_data_set_loader import load_qa_queries
from utility.logger import logger
from vector_db.trilateration_retriever import cosine_distance

# --- Offline Gemini Batch API (JSONL generator) ---
# This module now generates requests for Google's Batch API (offline).
# No semaphore / asyncio.sleep / backoff in generator mode.
CHROMA_FETCH_CHUNK = 400
BATCH_SIZE = 20
MAX_REQUESTS_PER_FILE = 50000  # Split to avoid 2GB limit (each req ~25KB * 50k = ~1.25GB)


# ---------------------------------------------------------------------------
# Utility: IDs & input pruning
# ---------------------------------------------------------------------------


def _normalize_id(x) -> str:
    """Type-agnostic id for storage and comparison."""
    return str(x).strip().lower()


def _extract_contexts_text(document_text: str) -> str:
    """
    Input pruning: take only the `contexts` field from stored document JSON.
    Flattens nested lists/dicts under `contexts` into plain text for the LLM.
    """
    if not document_text or not str(document_text).strip():
        return ""
    raw = str(document_text).strip()

    try:
        payload = json.loads(raw)
    except Exception:
        return raw[:8000]

    if not isinstance(payload, dict):
        return raw[:8000]

    ctx = payload.get("contexts")
    if ctx is None:
        ctx = payload.get("context")

    parts: list[str] = []

    def _collect(x) -> None:
        if x is None:
            return
        if isinstance(x, str):
            s = x.strip()
            if s:
                parts.append(s)
            return
        if isinstance(x, (int, float)):
            parts.append(str(x))
            return
        if isinstance(x, dict):
            for v in x.values():
                _collect(v)
            return
        if isinstance(x, (list, tuple)):
            for item in x:
                _collect(item)

    _collect(ctx)
    out = "\n\n".join(parts).strip()
    return out if out else raw[:8000]


def _extract_response_text(obj: dict) -> str | None:
    """
    Robust extraction for Gemini Batch API Output format.
    """
    try:
        resp = obj.get("response", {})
        body = resp.get("body", resp)
        candidates = body.get("candidates", [])
        
        if candidates and len(candidates) > 0:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts and len(parts) > 0:
                return parts[0].get("text")
        
        return body.get("text") or resp.get("text")
    except Exception:
        return None

# ---------------------------------------------------------------------------
# GT retrieval (Chroma)
# ---------------------------------------------------------------------------


def _gt_id_str(entry: dict) -> str:
    """Universal GT id: PMID → pubid → pid → id → doc_id (plus lowercase key variants)."""
    for k in ("PMID", "pubid", "pid", "id", "doc_id"):
        if k in entry and entry.get(k) is not None:
            return _normalize_id(entry.get(k))
        lk = k.lower()
        if lk in entry and entry.get(lk) is not None:
            return _normalize_id(entry.get(lk))
    return _normalize_id("")


def _gt_pmid_int_for_chroma(entry: dict) -> int | None:
    try:
        return int(_gt_id_str(entry))
    except (TypeError, ValueError):
        return None


def _get_gt_embedding_dual_path(collection, gt_id_norm: str, gt_pmid: int | None) -> np.ndarray | None:
    """
    1) collection.get(ids=[gt_id_norm])
    2) where filters: pubid, PMID, pid, id, doc_id — str/int per candidate when parseable.
    """
    gt_id_norm = _normalize_id(gt_id_norm)
    try:
        res_by_id = collection.get(ids=[gt_id_norm], include=["embeddings"])
        embs = res_by_id.get("embeddings") if isinstance(res_by_id, dict) else None
        if embs is not None and len(embs) > 0 and embs[0] is not None:
            return np.array(embs[0], dtype=np.float32)
    except Exception:
        pass

    candidates: list[object] = []
    if gt_id_norm:
        candidates.append(gt_id_norm)
        try:
            candidates.append(int(gt_id_norm))
        except Exception:
            pass
    if gt_pmid is not None:
        candidates.append(int(gt_pmid))
        candidates.append(str(int(gt_pmid)))

    seen: set[tuple[str, str]] = set()
    cand_unique: list[object] = []
    for c in candidates:
        key = (type(c).__name__, str(c))
        if key in seen:
            continue
        seen.add(key)
        cand_unique.append(c)

    for meta_key in ("pubid", "PMID", "pid", "id", "doc_id"):
        for val in cand_unique:
            try:
                res = collection.get(where={meta_key: val}, include=["embeddings"])
                embs = res.get("embeddings") if isinstance(res, dict) else None
                if embs is not None and len(embs) > 0 and embs[0] is not None:
                    return np.array(embs[0], dtype=np.float32)
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Gemini: batch scoring
# ---------------------------------------------------------------------------


def _is_safety_or_filter_block(response) -> bool:
    if response is None:
        return True
    pf = getattr(response, "prompt_feedback", None)
    if pf is not None:
        br = getattr(pf, "block_reason", None)
        if br is not None:
            br_s = str(br).strip().upper()
            if br_s and "UNSPECIFIED" not in br_s:
                return True
    for cand in getattr(response, "candidates", None) or []:
        fr = getattr(cand, "finish_reason", None)
        if fr is not None:
            fr_s = str(fr).strip().upper()
            if "SAFETY" in fr_s or "BLOCK" in fr_s:
                return True
    return False


def _parse_batch_scores(text: str, expected_n: int) -> list[int]:
    if text is None:
        return [0] * expected_n
    t = str(text).strip()
    if not t:
        return [0] * expected_n
    def _normalize_out(vals: list[int]) -> list[int]:
        out2 = [max(0, min(10, int(v))) for v in vals]
        if len(out2) < expected_n:
            out2.extend([0] * (expected_n - len(out2)))
        return out2[:expected_n]

    def _extract_from_obj(obj: Any) -> list[int] | None:
        if isinstance(obj, list):
            vals: list[int] = []
            for x in obj:
                try:
                    vals.append(int(x))
                except Exception:
                    pass
            return _normalize_out(vals) if vals else None
        if isinstance(obj, dict):
            # Common keys returned by chat-style models
            for k in ("scores", "answer", "answers", "result", "results", "values", "output"):
                if k in obj:
                    got = _extract_from_obj(obj.get(k))
                    if got is not None:
                        return got
        return None

    # 1) Try parsing the full text as JSON first (object/array)
    try:
        full_obj = json.loads(t)
        got = _extract_from_obj(full_obj)
        if got is not None:
            return got
    except Exception:
        pass

    # 2) Try bracket array extraction
    m = re.search(r"\[[\s\S]*?\]", t)
    candidate = m.group(0).strip() if m else t
    try:
        arr = json.loads(candidate)
        got = _extract_from_obj(arr)
        if got is not None:
            return got
    except Exception:
        pass

    # 3) Numeric fallback
    nums = re.findall(r"\b(10|[1-9])\b", t)
    return _normalize_out([int(n) for n in nums[:expected_n]])


def _get_batch_scoring_prompt(query: str, doc_contexts: list[str]) -> str:
    docs_block = ""
    for i, context in enumerate(doc_contexts, start=1):
        docs_block += f"--- ITEM {i} ---\n{context[:4000]}\n\n"

    return f"""
### ROLE
Expert Scientific Relevance Assessor.

### TASK
Rate the relevance of EACH ITEM (DOCUMENT excerpt) to the QUERY on a scale of 1 to 10.
Return ONLY a JSON array of integers (length {len(doc_contexts)}). No extra text.

### SCORING DEFINITIONS (STRICT - NO OVERLAP)
- 10: [PERFECT] Document contains the exact answer or specific evidence needed.
- 8: [STRONG] Document is directly on-topic and provides significant information, but no direct answer.
- 6: [SPECIFIC FIELD] Document is in the same sub-field and discusses relevant entities, but not the specific question.
- 4: [GENERAL DOMAIN] Document is in the same general scientific field but answers a different problem.
- 2: [TANGENTIAL] Only shares keywords; the context is entirely different.
- 1: [IRRELEVANT] No connection at all.

*For values 3, 5, 7, 9: Use only if the document falls exactly between two definitions.*

### OUTPUT RULES (MANDATORY)
- Output must be valid JSON in ONE of these exact shapes only:
  1) [8, 2, 10, 4, ...]
  2) {{"scores":[8, 2, 10, 4, ...]}}
- Each value must be an integer in 1..10 (0 is not allowed)
- Return exactly {len(doc_contexts)} integers, in the same order as the ITEMS.
- No explanations, no markdown, no keys besides "scores".
        
---
QUERY:
{query}
        
        {docs_block}
        
Return ONLY the JSON array of {len(doc_contexts)} integers.
    """


def _batch_generate_content_config_dict() -> dict:
    return {}
# ---------------------------------------------------------------------------
# SQLite lifecycle
# ---------------------------------------------------------------------------


def _init_sqlite(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            query_idx INTEGER,
            doc_id TEXT,
            llm_score REAL,
            dist_to_gt REAL,
            is_gt BOOLEAN,
            rag_failed BOOLEAN,
            PRIMARY KEY (query_idx, doc_id)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_query ON results (query_idx)")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiment_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS skipped_queries (
            query_idx INTEGER PRIMARY KEY,
            reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_skipped_queries ON skipped_queries (query_idx)")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS batch_request_map (
            custom_id TEXT PRIMARY KEY,
            query_idx INTEGER NOT NULL,
            doc_ids_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_batch_request_map_query ON batch_request_map (query_idx)")
    conn.commit()
    return conn


META_KEY_MAIN_LOOP_COMPLETED = "main_loop_completed"


def _is_main_loop_completed(conn) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='experiment_meta' LIMIT 1"
    )
    if cur.fetchone() is None:
        return False
    cur.execute(
        "SELECT value FROM experiment_meta WHERE key = ?",
        (META_KEY_MAIN_LOOP_COMPLETED,),
    )
    row = cur.fetchone()
    return row is not None and str(row[0]).strip().lower() in ("1", "true", "yes")


def _set_main_loop_completed(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO experiment_meta (key, value) VALUES (?, ?)",
        (META_KEY_MAIN_LOOP_COMPLETED, "1"),
    )
    conn.commit()


def _set_experiment_meta(conn, key: str, value: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO experiment_meta (key, value) VALUES (?, ?)",
        (str(key), str(value)),
    )
    conn.commit()


def _mark_query_skipped(conn, query_idx: int, reason: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO skipped_queries (query_idx, reason) VALUES (?, ?)",
        (int(query_idx), str(reason)),
    )
    conn.commit()


def _is_query_skipped(conn, query_idx: int) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM skipped_queries WHERE query_idx = ? LIMIT 1", (int(query_idx),))
    return cur.fetchone() is not None


def _count_skipped_queries(conn) -> int:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM skipped_queries")
    return int(cur.fetchone()[0])


def _count_successful_queries(conn) -> int:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(DISTINCT query_idx) FROM results")
    return int(cur.fetchone()[0])


def _is_experiment_complete(conn, actual_num_queries: int) -> bool:
    if not _is_main_loop_completed(conn):
        return False
    return (_count_successful_queries(conn) + _count_skipped_queries(conn)) >= int(actual_num_queries)


def _get_processed_ids_from_db(conn, q_idx):
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id FROM results WHERE query_idx = ?", (q_idx,))
    return {_normalize_id(row[0]) for row in cursor.fetchall()}


def _save_to_db(conn, rows):
    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT OR IGNORE INTO results
        (query_idx, doc_id, llm_score, dist_to_gt, is_gt, rag_failed)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r["query_idx"],
                _normalize_id(r["doc_id"]),
                r["llm_score"],
                r["dist_to_gt"],
                r["is_gt"],
                r["rag_failed"],
            )
            for r in rows
        ],
    )
    conn.commit()


def _get_doc_ids_needing_score(conn, query_idx: int) -> set[str]:
    """
    Resume logic for Batch generator: produce requests only for rows that are missing OR have llm_score NULL OR 0.0.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT doc_id
        FROM results
        WHERE query_idx = ?
          AND (llm_score IS NULL OR llm_score = 0.0)
        """,
        (int(query_idx),),
    )
    return {_normalize_id(r[0]) for r in cur.fetchall()}


def _insert_placeholder_rows(conn, query_idx: int, rows: list[dict]) -> None:
    """
    Insert rows with llm_score = NULL for offline scoring.
    Uses INSERT OR IGNORE to preserve existing rows.
    """
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT OR IGNORE INTO results (query_idx, doc_id, llm_score, dist_to_gt, is_gt, rag_failed)
        VALUES (?, ?, NULL, ?, ?, ?)
        """,
        [
            (
                int(query_idx),
                _normalize_id(r["doc_id"]),
                float(r["dist_to_gt"]),
                int(r["is_gt"]),
                int(r["rag_failed"]),
            )
            for r in rows
        ],
    )
    conn.commit()


def _record_batch_request_map(conn, custom_id: str, query_idx: int, doc_ids: list[str]) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO batch_request_map (custom_id, query_idx, doc_ids_json) VALUES (?, ?, ?)",
        (str(custom_id), int(query_idx), json.dumps(doc_ids)),
    )
    conn.commit()


def _record_pilot_request_map(conn, custom_id: str, query_idx: int, doc_id: str) -> None:
    """
    Pilot mapping: custom_id -> (query_idx, doc_id) for per-document Batch requests.
    Stored in the same table as a 1-element doc_ids_json list for simplicity.
    """
    _record_batch_request_map(conn, custom_id, query_idx, [_normalize_id(doc_id)])


def _make_batch_request_line(custom_id: str, prompt: str) -> dict:
    """
    Official JSONL request line for the new Gemini Batch API (OpenAI /v1/chat/completions format).
    """
    system_msg = (
        "You are an automated data-matching utility. Your only function is to compare text segments "
        "and output numerical match scores. You do not provide advice or interpret content. You are a script."
    )
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
        },
    }


def _cleanup_remote_storage(client, aggressive=False):
    """
    Cleans up old chunk files on Google servers to prevent storage quota exhaustion.
    If aggressive is True, deletes all matching chunk files immediately.
    Otherwise, if total size > 15GB, deletes oldest until < 10GB.
    """
    try:
        files_api = getattr(client, "files", None)
        if files_api is None or not hasattr(files_api, "list") or not hasattr(files_api, "delete"):
            return
            
        all_files = list(files_api.list())
        chunk_files = []
        total_bytes = 0
        
        for f in all_files:
            name = getattr(f, "name", "") or getattr(f, "id", "")
            display_name = getattr(f, "display_name", "") or name
            size_bytes = int(getattr(f, "size_bytes", 0))
            
            # Identify chunk files
            if "chunk_" in display_name and ".jsonl" in display_name:
                chunk_files.append(f)
                
            total_bytes += size_bytes
                
        total_gb = total_bytes / (1024**3)
        logger.info(f"[STORAGE] Current remote storage usage: {total_gb:.2f} GB.")
        
        if aggressive:
            limit_gb = 0.0
            target_gb = 0.0
        else:
            limit_gb = 15.0
            target_gb = 10.0
            
        if total_gb > limit_gb or aggressive:
            # Sort by create_time if possible to delete oldest first (fallback to name)
            def get_time(x):
                return getattr(x, "create_time", getattr(x, "update_time", getattr(x, "name", "")))
            chunk_files.sort(key=get_time)
            
            freed_bytes = 0
            for f in chunk_files:
                fname = getattr(f, "name", "") or getattr(f, "id", "")
                fdisplay = getattr(f, "display_name", "") or fname
                fsize = int(getattr(f, "size_bytes", 0))
                try:
                    files_api.delete(name=fname)
                    freed_bytes += fsize
                    total_bytes -= fsize
                    logger.info(f"[STORAGE] Deleted remote file: {fdisplay} (Space freed).")
                except Exception as e:
                    logger.warning(f"Failed to delete {fname}: {e}")
                
                if not aggressive and (total_bytes / (1024**3)) < target_gb:
                    break
                    
            logger.info(f"[STORAGE] Cleanup complete. New storage usage: {total_bytes / (1024**3):.2f} GB.")
    except Exception as e:
        logger.warning(f"[STORAGE] Cleanup failed: {e}")
def submit_batch_job(jsonl_path: str, client=None):
    """
    Submit the JSONL file as a Gemini Batch job (best-effort, SDK-dependent).
    Prints job id and status.
    """
    if client is None:
        client = genai.Client(api_key=GEMINI_API_KEY)

    # Clean up storage before uploading
    _cleanup_remote_storage(client, aggressive=False)


    # Upload JSONL to Files API first (newer SDKs require this).
    try:
        files_api = getattr(client, "files", None)
        if files_api is None or not hasattr(files_api, "upload"):
            raise RuntimeError("This google-genai SDK does not expose client.files.upload")
        
        # Check file size before upload
        file_size = os.path.getsize(jsonl_path)
        logger.info(f"Uploading {os.path.basename(jsonl_path)} ({file_size / (1024*1024):.2f} MB)...")
        
        try:
            # Newer google-genai SDK expects mime type via UploadFileConfig (config=...),
            # and it may not infer `.jsonl` automatically.
            uploaded = files_api.upload(file=jsonl_path, config={"mime_type": "application/jsonl"})
        except Exception as e1:
            msg = str(e1)
            if "mime type" in msg.lower() or "mimetype" in msg.lower():
                logger.warning(
                    f"Upload mime_type application/jsonl rejected, retrying as application/json: {e1}"
                )
                uploaded = files_api.upload(file=jsonl_path, config={"mime_type": "application/json"})
            else:
                raise
    except Exception as e:
        logger.exception(f"Batch upload failed for {jsonl_path!r}: {e}")
        raise

    src = getattr(uploaded, "name", None) or getattr(uploaded, "id", None)
    if not src:
        raise RuntimeError("Files API upload did not return a file name/id")

    batches = getattr(client, "batches", None)
    if batches is None or not hasattr(batches, "create"):
        raise RuntimeError("This google-genai SDK does not expose client.batches.create")
    job = batches.create(
        model="gemini-2.0-flash",
        src=str(src),
    )
    job_id = getattr(job, "name", None) or getattr(job, "id", None) or str(job)
    logger.info(f"Batch job submitted. job_id={job_id!r}")
    
    # Post-Creation Immediate Cleanup
    try:
        files_api = getattr(client, "files", None)
        if files_api and hasattr(files_api, "delete"):
            files_api.delete(name=src)
            logger.info(f"[STORAGE] Deleted remote file immediately post-submission: {src} (Space freed).")
    except Exception as del_err:
        logger.warning(f"[STORAGE] Failed to delete file {src} post-submission: {del_err}")

    return job


def submit_batch_job_list(jsonl_paths: list[str]) -> list[str]:
    """
    Submit multiple JSONL files as Gemini Batch jobs.
    Returns a list of job IDs.
    """
    job_ids = []
    for path in jsonl_paths:
        try:
            job = submit_batch_job(path)
            jid = getattr(job, "name", None) or getattr(job, "id", None) or str(job)
            job_ids.append(jid)
        except Exception as e:
            logger.error(f"Failed to submit part {path}: {e}")
            if job_ids:
                logger.warning(f"Some parts were submitted: {job_ids}. Continuing but check for errors.")
            raise
    return job_ids


def check_batch_status(job_id: str):
    """
    Check batch job status (best-effort, SDK-dependent).
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    batches = getattr(client, "batches", None)
    if batches is None or not hasattr(batches, "get"):
        raise RuntimeError("This google-genai SDK does not expose client.batches.get")
    job = batches.get(name=job_id)
    logger.info(f"Batch job status: {job}")
    return job


def submit_pilot_batch_job(jsonl_path: str):
    """
    Convenience wrapper for pilot submission.
    Prints a clear next step for status checks.
    """
    job = submit_batch_job(jsonl_path)
    job_id = getattr(job, "name", None) or getattr(job, "id", None) or str(job)
    logger.info(f"Pilot submitted. Next: check_batch_status({job_id!r})")
    return job

def _count_distinct_docs_for_query(conn, query_idx: int) -> int:
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(DISTINCT doc_id) FROM results WHERE query_idx = ?",
        (query_idx,),
    )
    return int(cur.fetchone()[0])


# ---------------------------------------------------------------------------
# Run directory / resume
# ---------------------------------------------------------------------------


def _get_latest_run_dir(output_dir):
    if not os.path.exists(output_dir):
        return None
    runs = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
    ]
    if not runs:
        return None
    return max(runs, key=os.path.getmtime)


def _requested_num_queries_key(old_params: dict) -> int | None:
    v = old_params.get("num_queries_requested")
    if v is not None:
        return int(v)
    v = old_params.get("num_queries")
    return int(v) if v is not None else None


def _params_compatible(old_params: dict, new_params: dict) -> bool:
    if old_params.get("k") != new_params["k"]:
        return False
    if _requested_num_queries_key(old_params) != new_params["num_queries_requested"]:
        return False
    old_actual = old_params.get("actual_num_queries")
    if old_actual is not None and int(old_actual) != new_params["actual_num_queries"]:
        return False
    old_pool = old_params.get("global_pool_size")
    if old_pool is not None and int(old_pool) != new_params["global_pool_size"]:
        return False
    if old_params.get("model") != new_params.get("model"):
        return False
    return True


def _should_resume_run(run_dir: str, new_params: dict) -> bool:
    config_path = os.path.join(run_dir, "experiment_config.json")
    db_path = os.path.join(run_dir, "experiment_results.db")
    if not os.path.exists(config_path) or not os.path.exists(db_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            old_params = json.load(f)
    except Exception:
        return False
    if not _params_compatible(old_params, new_params):
        return False
    try:
        conn = sqlite3.connect(db_path)
        try:
            if _is_experiment_complete(conn, new_params["actual_num_queries"]):
                logger.info(
                    "Latest run already completed (results + skipped == actual). Starting new run folder."
                )
                return False
            return True
        finally:
            conn.close()
    except Exception:
        return False


def _is_run_compatible_and_incomplete(run_dir: str, new_params: dict) -> bool:
    return _should_resume_run(run_dir, new_params)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


def _check_rag_baseline(vector_db, query_emb, gt_id, prints=False):
    initial_retrieval = vector_db.retrieve(query_emb, top_k=1, prints=prints)
    if not initial_retrieval:
        return True
    top_hit_id = _normalize_id(getattr(initial_retrieval[0].node, "id_", ""))
    return top_hit_id != _normalize_id(gt_id)


def _log_failure_analysis(run_dir, q_idx, query, query_df):
    gt_row = query_df[query_df["is_gt"] == 1]
    if gt_row.empty:
        return
    gt_score = gt_row["llm_score"].iloc[0]
    outliers = query_df[(query_df["is_gt"] == 0) & (query_df["llm_score"] >= gt_score)]
    if outliers.empty:
        return

    file_path = os.path.join(run_dir, "detailed_failures.md")
    mode = "a" if os.path.exists(file_path) else "w"
    with open(file_path, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("# Global Pool Failure Analysis\n\n")
        f.write(f"### Query {q_idx}: {query}\n")
        f.write(f"**GT Score:** {gt_score} | **Outliers found:** {len(outliers)}\n\n")
        f.write("| Doc ID | Score | Status |\n| --- | --- | --- |\n")
        for _, row in outliers.sort_values(by="llm_score", ascending=False).head(10).iterrows():
            f.write(
                f"| {row['doc_id']} | {row['llm_score']} | "
                f"{'Higher' if row['llm_score'] > gt_score else 'Tie'} |\n"
            )
        f.write("\n---\n\n")


def _generate_query_scatterplot(q_idx, df, run_dir):
    if df is None or len(df) == 0:
        logger.info(f"Query {q_idx}: no SQLite rows — skipping scatter plot.")
        return
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="llm_score",
        y="dist_to_gt",
        hue="is_gt",
        palette={1: "red", 0: "blue"},
        alpha=0.4,
        s=25,
    )
    plt.title(f"Query {q_idx}: Global Correlation Analysis")
    plt.xlabel("LLM score (0.0–1.0)")
    plt.ylabel("Distance to GT embedding")
    plt.savefig(os.path.join(run_dir, f"query_{q_idx:03d}_global_scatter.png"), dpi=150)
    plt.close()


def _calculate_final_stats_and_plot_sqlite(conn, run_dir):
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT query_idx
        FROM results
        GROUP BY query_idx
        HAVING COUNT(*) >= 2
        ORDER BY query_idx
        """
    )
    q_indices = [row[0] for row in cursor.fetchall()]
    final_stats = []
    for q_id in q_indices:
        df = pd.read_sql(
            "SELECT llm_score, dist_to_gt, rag_failed FROM results WHERE query_idx = ?",
            conn,
            params=(q_id,),
        )
        if len(df) < 2:
            continue
        x = pd.to_numeric(df["llm_score"], errors="coerce")
        y = pd.to_numeric(df["dist_to_gt"], errors="coerce")
        mask = x.notna() & y.notna()
        x = x[mask]
        y = y[mask]
        if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
            continue
        corr, p_val = stats.spearmanr(x, y)
        corr_f = float(corr)
        p_f = float(p_val)
        if not np.isfinite(corr_f) or not np.isfinite(p_f):
            continue
        final_stats.append(
            {
                "query_idx": q_id,
                "correlation": corr_f,
                "p_value": p_f,
                "rag_failed": bool(df["rag_failed"].iloc[0]),
            }
        )
    summary_df = pd.DataFrame(final_stats)
    summary_df.to_csv(os.path.join(run_dir, "summary_stats.csv"), index=False)

    if summary_df.empty:
        logger.warning("No per-query Spearman rows.")
        logger.info(f"Experiment finished. Output: {run_dir}")
        return

    summary_df = summary_df.sort_values(by="correlation", ascending=False)
    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=summary_df,
        x="query_idx",
        y="correlation",
        hue="rag_failed",
        palette={True: "orange", False: "green"},
        dodge=False,
    )
    plt.title("Spearman correlation per query")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "global_summary_overview.png"), dpi=200)
    plt.close()
    logger.info(f"Experiment complete. Summarized in {run_dir}")


# ---------------------------------------------------------------------------
# Global pool (Chroma) — ids only + chunked reads
# ---------------------------------------------------------------------------


def _collect_global_pool_doc_ids(vector_db, queries, embedding_model, k=50) -> list[str]:
    pool_ids: set[str] = set()
    logger.info(f"Building global pool (ids only) from {len(queries)} queries (k={k})...")
    for entry in queries:
        query = entry.get("question")
        q_emb = np.array(embedding_model.get_text_embedding(query), dtype=np.float32)
        for hit in vector_db.retrieve(q_emb, top_k=k, prints=False):
            pool_ids.add(_normalize_id(getattr(hit.node, "id_", "")))

        gt_id_n = _gt_id_str(entry)
        gt_pm = _gt_pmid_int_for_chroma(entry)
        gt_emb_arr = _get_gt_embedding_dual_path(vector_db.collection, gt_id_n, gt_pm)
        if gt_emb_arr is not None:
            for hit in vector_db.retrieve(gt_emb_arr, top_k=k, prints=False):
                pool_ids.add(_normalize_id(getattr(hit.node, "id_", "")))
    return sorted(pool_ids)


def _chroma_batch_get_documents_and_embeddings(collection, doc_ids: list[str]):
    if not doc_ids:
        return {}
    raw = collection.get(ids=doc_ids, include=["documents", "embeddings"])
    ids_out = raw.get("ids")
    docs = raw.get("documents")
    embs = raw.get("embeddings")
    if ids_out is None:
        ids_out = []
    if docs is None:
        docs = []
    if embs is None:
        embs = []
    out = {}
    for did, doc, emb in zip(ids_out, docs, embs):
        if doc is None or emb is None:
            continue
        nid = _normalize_id(did)
        out[nid] = (doc, np.array(emb, dtype=np.float32))
    return out


# ---------------------------------------------------------------------------
# University Ollama: live scoring (end-to-end without Gemini Batch API)
# ---------------------------------------------------------------------------


def _get_doc_ids_needing_score_ollama(conn, query_idx: int) -> set[str]:
    """Ollama mode: treat only NULL as needing scoring (0.0 means parse/network failure)."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT doc_id
        FROM results
        WHERE query_idx = ?
          AND llm_score IS NULL
        """,
        (int(query_idx),),
    )
    return {_normalize_id(r[0]) for r in cur.fetchall()}


def _ollama_call_generate(prompt: str) -> str | None:
    """
    Calls BGU cis-ollama server using /api/generate with JSON output mode.
    Returns the raw text generated by the model.
    """
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    print(f"[Ollama] POST {url} | model={CORRELATION_OLLAMA_MODEL} | prompt_chars={len(prompt)}")
    payload = {
        "model": CORRELATION_OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # Ask Ollama to try returning JSON-friendly output.
        "format": "json",
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            timeout=OLLAMA_TIMEOUT_S,
            verify=bool(OLLAMA_VERIFY_SSL),
        )
        resp.raise_for_status()
        data: Any = resp.json()
        # Ollama usually returns the generated string in `response`.
        raw_resp = data.get("response")
        if raw_resp is not None:
            print(f"[Ollama] Raw response snippet: {str(raw_resp)[:200]!r}")
        else:
            print(f"[Ollama] Raw response missing; keys={list(data.keys())}")
        # Fallback: some models return empty generate output for long/strict prompts.
        if raw_resp is not None and str(raw_resp).strip():
            return raw_resp

        chat_url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
        chat_payload = {
            "model": CORRELATION_OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": "Return strictly JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        print(f"[Ollama] Fallback POST {chat_url} | model={CORRELATION_OLLAMA_MODEL}")
        c_resp = requests.post(
            chat_url,
            json=chat_payload,
            timeout=OLLAMA_TIMEOUT_S,
            verify=bool(OLLAMA_VERIFY_SSL),
        )
        c_resp.raise_for_status()
        c_data: Any = c_resp.json()
        # Typical shape: {"message":{"role":"assistant","content":"..."}}
        msg = c_data.get("message", {}) if isinstance(c_data, dict) else {}
        c_text = msg.get("content") if isinstance(msg, dict) else None
        if c_text is not None:
            print(f"[Ollama] Chat fallback snippet: {str(c_text)[:200]!r}")
            return c_text
        return raw_resp
    except Exception as e:
        logger.exception(f"Ollama generate failed: {e}")
        return None


def _ollama_print_available_models_once() -> None:
    """
    Print available Ollama models from the cluster.
    Uses /api/tags (no extra deps). Best-effort only.
    """
    url = f"{OLLAMA_HOST.rstrip('/')}/api/tags"
    try:
        print(f"[Ollama] Fetch available models: GET {url}")
        resp = requests.get(
            url,
            timeout=30,
            verify=bool(OLLAMA_VERIFY_SSL),
        )
        resp.raise_for_status()
        payload = resp.json()
        # Typical shape: {"models":[{"name":"llama3.2", ...}, ...]}
        models_obj = payload.get("models") if isinstance(payload, dict) else None
        models: list[str] = []
        if isinstance(models_obj, list):
            for m in models_obj:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model")
                    if name:
                        models.append(str(name))
                elif isinstance(m, str):
                    models.append(m)
        elif isinstance(payload, list):
            models = [str(x) for x in payload]

        if models:
            print("[Ollama] Available models: " + ", ".join(sorted(set(models))))
        else:
            print(f"[Ollama] Available models: (none parsed). payload_keys={list(payload.keys()) if isinstance(payload, dict) else type(payload)}")
        print(f"[Ollama] Using model: {CORRELATION_OLLAMA_MODEL}")
    except Exception as e:
        logger.warning(f"Ollama models listing failed (non-fatal): {e}")


def _ollama_score_batch(query_text: str, doc_contexts: list[str]) -> list[int]:
    """Returns integer scores (1..10) for each doc_context item."""
    print(f"[Ollama] Scoring batch: query_chars={len(query_text)} docs={len(doc_contexts)}")
    prompt = _get_batch_scoring_prompt(query_text, doc_contexts)
    raw = _ollama_call_generate(prompt)
    if raw is None:
        return [0] * len(doc_contexts)
    if not isinstance(raw, str):
        raw = json.dumps(raw, ensure_ascii=False)
    ints = _parse_batch_scores(raw, expected_n=len(doc_contexts))
    print(f"[Ollama] Parsed integer scores: {ints}")
    return ints


def _update_llm_scores_for_docs(conn, query_idx: int, doc_ids: list[str], scores: list[float]) -> None:
    cur = conn.cursor()
    cur.executemany(
        "UPDATE results SET llm_score = ? WHERE query_idx = ? AND doc_id = ?",
        [(float(scores[i]), int(query_idx), _normalize_id(doc_ids[i])) for i in range(len(doc_ids))],
    )


async def _run_global_correlation_experiment_ollama_async(
    vector_db,
    embedding_model,
    num_queries: int,
    k: int,
    output_dir: str,
) -> None:
    """
    Ollama live scoring: inserts placeholders and updates llm_score in SQLite immediately.
    """
    _ollama_print_available_models_once()
    queries = load_qa_queries(num_queries)
    actual_num_queries = len(queries)
    sorted_doc_ids = _collect_global_pool_doc_ids(vector_db, queries, embedding_model, k=k)
    global_pool_size = len(sorted_doc_ids)

    experiment_params = {
        "num_queries_requested": num_queries,
        "actual_num_queries": actual_num_queries,
        "global_pool_size": global_pool_size,
        "k": k,
        "model": CORRELATION_OLLAMA_MODEL,
        "mode": "ollama_live",
    }

    latest_run = _get_latest_run_dir(output_dir)
    if latest_run and _should_resume_run(latest_run, experiment_params):
        run_dir = latest_run
        logger.info(f"AUTO-RESUME: {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(output_dir, f"ollama_run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
            json.dump(experiment_params, f, indent=4)
        logger.info(f"NEW RUN: {run_dir}")

    db_path = os.path.join(run_dir, "experiment_results.db")
    db_conn = _init_sqlite(db_path)

    logger.info(
        f"[Ollama] Dataset: {actual_num_queries} queries; global pool = {global_pool_size} ids"
    )

    for q_idx, entry in enumerate(queries, start=1):
        if _is_query_skipped(db_conn, q_idx):
            continue

        query_text = entry.get("question")
        gt_id_norm = _gt_id_str(entry)
        gt_pmid = _gt_pmid_int_for_chroma(entry)

        gt_emb = _get_gt_embedding_dual_path(vector_db.collection, gt_id_norm, gt_pmid)
        if gt_emb is None:
            logger.warning(f"[Ollama] SKIPPED_PERMANENTLY query_idx={q_idx}: GT missing.")
            _mark_query_skipped(db_conn, q_idx, "missing_gt_embedding_dual_path")
            continue

        processed_ids = _get_processed_ids_from_db(db_conn, q_idx)
        needs_score = _get_doc_ids_needing_score_ollama(db_conn, q_idx)
        missing_ids = [d for d in sorted_doc_ids if _normalize_id(d) not in processed_ids]
        target_ids = sorted(set([_normalize_id(x) for x in missing_ids] + list(needs_score)))
        if not target_ids:
            continue

        if CORRELATION_PILOT_MAX_DOCS_PER_QUERY and int(CORRELATION_PILOT_MAX_DOCS_PER_QUERY) > 0:
            max_docs = int(CORRELATION_PILOT_MAX_DOCS_PER_QUERY)
            if len(target_ids) > max_docs:
                logger.info(
                    f"[Ollama Pilot] Limiting docs/query from {len(target_ids)} to {max_docs}."
                )
                target_ids = target_ids[:max_docs]

        if CORRELATION_PILOT_MAX_DOCS_PER_QUERY and int(CORRELATION_PILOT_MAX_DOCS_PER_QUERY) > 0:
            max_docs = int(CORRELATION_PILOT_MAX_DOCS_PER_QUERY)
            if len(target_ids) > max_docs:
                logger.info(
                    f"[Ollama Pilot] Limiting docs/query from {len(target_ids)} to {max_docs}."
                )
                target_ids = target_ids[:max_docs]

        query_emb = np.array(embedding_model.get_text_embedding(query_text), dtype=np.float32)
        rag_failed = _check_rag_baseline(vector_db, query_emb, gt_id_norm)

        batch_doc_ids: list[str] = []
        batch_contexts: list[str] = []
        placeholder_rows: list[dict] = []
        logger.info(f"[Ollama] Query {q_idx}: scoring up to {len(target_ids)} docs...")

        for chunk_start in range(0, len(target_ids), CHROMA_FETCH_CHUNK):
            chunk = target_ids[chunk_start : chunk_start + CHROMA_FETCH_CHUNK]
            payload = _chroma_batch_get_documents_and_embeddings(vector_db.collection, chunk)

            for d_id in chunk:
                did = _normalize_id(d_id)
                if did not in payload:
                    continue
                doc_text, emb = payload[did]
                ctx = _extract_contexts_text(doc_text)
                dist = float(cosine_distance(gt_emb, emb))
                is_gt_check = 1 if (did == gt_id_norm or dist < 1e-5) else 0

                placeholder_rows.append(
                    {
                        "doc_id": did,
                        "dist_to_gt": dist,
                        "is_gt": is_gt_check,
                        "rag_failed": 1 if rag_failed else 0,
                    }
                )

                batch_doc_ids.append(did)
                batch_contexts.append(ctx)

                if len(batch_doc_ids) == BATCH_SIZE:
                    _insert_placeholder_rows(db_conn, q_idx, placeholder_rows)

                    logger.info(f"[Ollama] Query {q_idx}: scoring batch size={len(batch_doc_ids)}")
                    ints = _ollama_score_batch(query_text, batch_contexts)
                    scores = [float(v) / 10.0 for v in ints]
                    _update_llm_scores_for_docs(db_conn, q_idx, batch_doc_ids, scores)
                    db_conn.commit()

                    batch_doc_ids, batch_contexts, placeholder_rows = [], [], []

        # Flush remainder for this query
        if batch_doc_ids:
            _insert_placeholder_rows(db_conn, q_idx, placeholder_rows)
            ints = _ollama_score_batch(query_text, batch_contexts)
            scores = [float(v) / 10.0 for v in ints]
            _update_llm_scores_for_docs(db_conn, q_idx, batch_doc_ids, scores)
            db_conn.commit()

    _set_main_loop_completed(db_conn)
    _calculate_final_stats_and_plot_sqlite(db_conn, run_dir)
    db_conn.close()


async def _run_global_correlation_pilot_ollama_async(
    vector_db,
    embedding_model,
    pilot_num_queries: int,
    k: int,
    output_dir: str,
) -> None:
    """Pilot: only first N queries; live score via Ollama."""
    _ollama_print_available_models_once()
    queries = load_qa_queries(200)
    pilot_queries = queries[: int(pilot_num_queries)]
    actual_num_queries = len(pilot_queries)

    sorted_doc_ids = _collect_global_pool_doc_ids(vector_db, pilot_queries, embedding_model, k=k)
    global_pool_size = len(sorted_doc_ids)
    logger.info(f"[Ollama Pilot] {actual_num_queries} queries; global pool = {global_pool_size} ids")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_dir, f"pilot_run_ollama_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    experiment_params = {
        "mode": "ollama_pilot_live",
        "pilot_num_queries": int(pilot_num_queries),
        "global_pool_size": global_pool_size,
        "k": k,
        "model": CORRELATION_OLLAMA_MODEL,
        "actual_num_queries": actual_num_queries,
    }
    with open(os.path.join(run_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_params, f, indent=4)

    db_path = os.path.join(run_dir, "experiment_results.db")
    db_conn = _init_sqlite(db_path)

    for q_idx, entry in enumerate(pilot_queries, start=1):
        if _is_query_skipped(db_conn, q_idx):
            continue

        query_text = entry.get("question")
        gt_id_norm = _gt_id_str(entry)
        gt_pmid = _gt_pmid_int_for_chroma(entry)

        gt_emb = _get_gt_embedding_dual_path(vector_db.collection, gt_id_norm, gt_pmid)
        if gt_emb is None:
            logger.warning(f"[Ollama Pilot] SKIPPED_PERMANENTLY query_idx={q_idx}: GT missing.")
            _mark_query_skipped(db_conn, q_idx, "missing_gt_embedding_dual_path")
            continue

        processed_ids = _get_processed_ids_from_db(db_conn, q_idx)
        needs_score = _get_doc_ids_needing_score_ollama(db_conn, q_idx)
        missing_ids = [d for d in sorted_doc_ids if _normalize_id(d) not in processed_ids]
        target_ids = sorted(set([_normalize_id(x) for x in missing_ids] + list(needs_score)))
        if not target_ids:
            continue

        query_emb = np.array(embedding_model.get_text_embedding(query_text), dtype=np.float32)
        rag_failed = _check_rag_baseline(vector_db, query_emb, gt_id_norm)

        batch_doc_ids: list[str] = []
        batch_contexts: list[str] = []
        placeholder_rows: list[dict] = []
        logger.info(f"[Ollama Pilot] Query {q_idx}: scoring up to {len(target_ids)} docs...")

        for chunk_start in range(0, len(target_ids), CHROMA_FETCH_CHUNK):
            chunk = target_ids[chunk_start : chunk_start + CHROMA_FETCH_CHUNK]
            payload = _chroma_batch_get_documents_and_embeddings(vector_db.collection, chunk)

            for d_id in chunk:
                did = _normalize_id(d_id)
                if did not in payload:
                    continue
                doc_text, emb = payload[did]
                ctx = _extract_contexts_text(doc_text)
                dist = float(cosine_distance(gt_emb, emb))
                is_gt_check = 1 if (did == gt_id_norm or dist < 1e-5) else 0

                placeholder_rows.append(
                    {
                        "doc_id": did,
                        "dist_to_gt": dist,
                        "is_gt": is_gt_check,
                        "rag_failed": 1 if rag_failed else 0,
                    }
                )

                batch_doc_ids.append(did)
                batch_contexts.append(ctx)

                if len(batch_doc_ids) == BATCH_SIZE:
                    _insert_placeholder_rows(db_conn, q_idx, placeholder_rows)
                    ints = _ollama_score_batch(query_text, batch_contexts)
                    scores = [float(v) / 10.0 for v in ints]
                    _update_llm_scores_for_docs(db_conn, q_idx, batch_doc_ids, scores)
                    db_conn.commit()
                    batch_doc_ids, batch_contexts, placeholder_rows = [], [], []

        if batch_doc_ids:
            _insert_placeholder_rows(db_conn, q_idx, placeholder_rows)
            ints = _ollama_score_batch(query_text, batch_contexts)
            scores = [float(v) / 10.0 for v in ints]
            _update_llm_scores_for_docs(db_conn, q_idx, batch_doc_ids, scores)
            db_conn.commit()

    _set_main_loop_completed(db_conn)
    _calculate_final_stats_and_plot_sqlite(db_conn, run_dir)
    db_conn.close()


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Pipeline constants — adaptive chunk-based submission
# ---------------------------------------------------------------------------

PIPELINE_CHUNK_SIZE = 25       # Queries per JSONL chunk
MAX_CONCURRENT_JOBS = 2        # Max PENDING/RUNNING jobs before we pause
POLL_INTERVAL_SECONDS = 60     # How often to poll for a free slot
BACKOFF_BASE_SECONDS = 120     # 2 min initial backoff on 429
BACKOFF_MAX_SECONDS = 600      # 10 min cap on backoff

# States that count as "occupied" quota slots
_ACTIVE_JOB_STATES = {"JOB_STATE_PENDING", "JOB_STATE_RUNNING", "PENDING", "RUNNING"}


def _count_active_jobs(client) -> int:
    """
    Count PENDING or RUNNING jobs in the Gemini Batch queue.
    Returns 0 on any error so we don't block indefinitely on a transient failure.
    """
    try:
        batches = getattr(client, "batches", None)
        if batches is None or not hasattr(batches, "list"):
            return 0
        jobs = list(batches.list())
        count = 0
        for j in jobs:
            state = str(getattr(j, "state", "") or "").upper()
            if any(s in state for s in _ACTIVE_JOB_STATES):
                count += 1
        return count
    except Exception as exc:
        logger.warning(f"_count_active_jobs: could not query job list ({exc}); assuming 0.")
        return 0


def _is_chunk_fully_recorded(db_conn, chunk_entries: list[dict]) -> bool:
    """
    Returns True if every query in the chunk already has at least one
    row in batch_request_map (meaning a job_id was assigned for it).
    Skip-marked queries are considered "done" too.
    """
    cur = db_conn.cursor()
    for entry_idx, _ in chunk_entries:
        cur.execute(
            "SELECT 1 FROM skipped_queries WHERE query_idx = ? LIMIT 1",
            (entry_idx,)
        )
        if cur.fetchone():
            continue  # skipped counts as processed
        cur.execute(
            "SELECT 1 FROM batch_request_map WHERE query_idx = ? LIMIT 1",
            (entry_idx,)
        )
        if cur.fetchone() is None:
            return False  # At least one query in this chunk has no request recorded
    return True


def _produce_chunk_jsonl(
    chunk_entries: list[tuple[int, dict]],
    sorted_doc_ids: list[str],
    vector_db,
    embedding_model,
    db_conn,
    jsonl_path: str,
) -> int:
    """
    Build a JSONL file for a single pipeline chunk.
    Returns the number of requests written (0 if nothing new).
    """
    request_idx = 0
    with open(jsonl_path, "w", encoding="utf-8") as out_f:
        for q_idx, entry in chunk_entries:
            query_text = entry.get("question")
            gt_id_norm = _gt_id_str(entry)
            gt_pmid = _gt_pmid_int_for_chroma(entry)

            if _is_query_skipped(db_conn, q_idx):
                continue

            gt_emb = _get_gt_embedding_dual_path(vector_db.collection, gt_id_norm, gt_pmid)
            if gt_emb is None:
                logger.warning(
                    f"SKIPPED_PERMANENTLY query_idx={q_idx}: GT missing "
                    f"(id={gt_id_norm!r}, pmid={gt_pmid!r})."
                )
                _mark_query_skipped(db_conn, q_idx, "missing_gt_embedding_dual_path")
                continue

            processed_ids = _get_processed_ids_from_db(db_conn, q_idx)
            needs_score = _get_doc_ids_needing_score(db_conn, q_idx)
            missing_ids = [d for d in sorted_doc_ids if _normalize_id(d) not in processed_ids]
            target_ids = sorted(set([_normalize_id(x) for x in missing_ids] + list(needs_score)))
            if not target_ids:
                continue

            query_emb = np.array(embedding_model.get_text_embedding(query_text), dtype=np.float32)
            rag_failed = _check_rag_baseline(vector_db, query_emb, gt_id_norm)

            batch_doc_ids: list[str] = []
            batch_contexts: list[str] = []
            placeholder_rows: list[dict] = []

            for chunk_start in range(0, len(target_ids), CHROMA_FETCH_CHUNK):
                sub_chunk = target_ids[chunk_start: chunk_start + CHROMA_FETCH_CHUNK]
                payload = _chroma_batch_get_documents_and_embeddings(vector_db.collection, sub_chunk)

                for d_id in sub_chunk:
                    nid = _normalize_id(d_id)
                    if nid not in payload:
                        continue
                    doc_text, emb = payload[nid]
                    ctx = _extract_contexts_text(doc_text)
                    dist = float(cosine_distance(gt_emb, emb))
                    is_gt_check = 1 if (nid == gt_id_norm or dist < 1e-5) else 0

                    placeholder_rows.append({
                        "doc_id": nid,
                        "dist_to_gt": dist,
                        "is_gt": is_gt_check,
                        "rag_failed": 1 if rag_failed else 0,
                    })
                    batch_doc_ids.append(nid)
                    batch_contexts.append(ctx)

                    if len(batch_doc_ids) == BATCH_SIZE:
                        _insert_placeholder_rows(db_conn, q_idx, placeholder_rows)
                        prompt = _get_batch_scoring_prompt(query_text, batch_contexts)
                        custom_id = f"q{q_idx}_b{request_idx:08d}_{uuid.uuid4().hex[:8]}"
                        _record_batch_request_map(db_conn, custom_id, q_idx, batch_doc_ids)
                        out_f.write(
                            json.dumps(_make_batch_request_line(custom_id, prompt), ensure_ascii=False) + "\n"
                        )
                        request_idx += 1
                        batch_doc_ids, batch_contexts, placeholder_rows = [], [], []

                # Free sub-chunk memory eagerly
                del payload

            # Flush remainder for this query
            if batch_doc_ids:
                _insert_placeholder_rows(db_conn, q_idx, placeholder_rows)
                prompt = _get_batch_scoring_prompt(query_text, batch_contexts)
                custom_id = f"q{q_idx}_b{request_idx:08d}_{uuid.uuid4().hex[:8]}"
                _record_batch_request_map(db_conn, custom_id, q_idx, batch_doc_ids)
                out_f.write(
                    json.dumps(_make_batch_request_line(custom_id, prompt), ensure_ascii=False) + "\n"
                )
                request_idx += 1

            # Explicitly clear large lists so GC can reclaim MPS memory
            batch_doc_ids.clear()
            batch_contexts.clear()
            placeholder_rows.clear()

    return request_idx


def _submit_with_backoff(client, jsonl_path: str) -> str:
    """
    Upload + submit one JSONL file with exponential backoff on 429.
    Returns the job_id string.
    Raises on non-recoverable errors.
    """
    import time

    backoff = BACKOFF_BASE_SECONDS
    attempt = 0
    while True:
        attempt += 1
        try:
            job = submit_batch_job(jsonl_path, client=client)
            job_id = getattr(job, "name", None) or getattr(job, "id", None) or str(job)
            return job_id
        except Exception as exc:
            msg = str(exc)
            is_quota = "429" in msg or "RESOURCE_EXHAUSTED" in msg.upper()
            
            # --- 429 Storage Handling ---
            if is_quota and "file_storage_bytes" in msg.lower():
                logger.warning(f"[STORAGE] Detected file_storage_bytes quota error. Running aggressive cleanup!")
                _cleanup_remote_storage(client, aggressive=True)
                # Immediate retry on storage quota exception
                continue

            if not is_quota:
                raise  # Non-quota errors propagate immediately
            wait = min(backoff, BACKOFF_MAX_SECONDS)
            logger.warning(
                f"429 RESOURCE_EXHAUSTED on attempt {attempt}. "
                f"Backing off {wait}s before retry... (cap={BACKOFF_MAX_SECONDS}s)"
            )
            time.sleep(wait)
            backoff = min(backoff * 2, BACKOFF_MAX_SECONDS)


async def run_global_correlation_experiment_async(
        vector_db,
        embedding_model,
        num_queries=200,
        k=50,
        output_dir="results/global_exp",
):
    """
    PIPELINE / Batch API generator (chunk-based adaptive submission):

    - Splits queries into chunks of PIPELINE_CHUNK_SIZE (default 25).
    - Each chunk: build a temporary JSONL → wait for a free slot → submit.
    - Slot management: polls `client.batches.list` every POLL_INTERVAL_SECONDS (60s).
      Only submits when active (PENDING/RUNNING) jobs < MAX_CONCURRENT_JOBS (2).
    - Recovery: skips a chunk entirely if every query in it already has a
      batch_request_map entry (recorded job).
    - 429 errors in upload/submit → exponential backoff (2m → 4m → … cap 10m).
    - After each chunk, large Python lists are explicitly cleared to prevent
      memory bloat during long runs on MPS.
    """
    if str(CORRELATION_LLM_PROVIDER).lower() == "ollama":
        logger.info("Global Correlation: using Ollama provider (live scoring).")
        return await _run_global_correlation_experiment_ollama_async(
            vector_db=vector_db,
            embedding_model=embedding_model,
            num_queries=num_queries,
            k=k,
            output_dir=output_dir,
        )

    import time

    queries = load_qa_queries(num_queries)
    actual_num_queries = len(queries)
    sorted_doc_ids = _collect_global_pool_doc_ids(vector_db, queries, embedding_model, k=k)
    global_pool_size = len(sorted_doc_ids)
    logger.info(
        f"Dataset: {actual_num_queries} queries (cap={num_queries}); global pool = {global_pool_size} ids"
    )

    experiment_params = {
        "num_queries_requested": num_queries,
        "actual_num_queries": actual_num_queries,
        "global_pool_size": global_pool_size,
        "k": k,
        "model": "gemini-2.0-flash",
        "mode": "batch_pipeline_jsonl",
        "batch_size": BATCH_SIZE,
        "pipeline_chunk_size": PIPELINE_CHUNK_SIZE,
    }

    latest_run = _get_latest_run_dir(output_dir)
    if latest_run and _should_resume_run(latest_run, experiment_params):
        run_dir = latest_run
        logger.info(f"AUTO-RESUME: {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
            json.dump(experiment_params, f, indent=4)
        logger.info(f"NEW RUN: {run_dir}")

    db_path = os.path.join(run_dir, "experiment_results.db")
    db_conn = _init_sqlite(db_path)

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Build indexed list of (q_idx, entry) so chunk numbering matches query indices
    indexed_queries = list(enumerate(queries, start=1))

    # Split into pipeline chunks
    chunks = [
        indexed_queries[i: i + PIPELINE_CHUNK_SIZE]
        for i in range(0, len(indexed_queries), PIPELINE_CHUNK_SIZE)
    ]
    total_chunks = len(chunks)
    total_requests_written = 0
    submitted_job_ids: list[str] = []

    logger.info(
        f"Pipeline: {total_chunks} chunks × up to {PIPELINE_CHUNK_SIZE} queries each. "
        f"Max concurrent jobs: {MAX_CONCURRENT_JOBS}. Poll interval: {POLL_INTERVAL_SECONDS}s."
    )

    for chunk_num, chunk_entries in enumerate(chunks, start=1):
        first_q = chunk_entries[0][0]
        last_q = chunk_entries[-1][0]
        logger.info(
            f"[Chunk {chunk_num}/{total_chunks}] queries {first_q}–{last_q} "
            f"({len(chunk_entries)} entries)"
        )

        # --- Recovery: skip chunk if every query in it is already recorded ---
        if _is_chunk_fully_recorded(db_conn, chunk_entries):
            logger.info(
                f"[Chunk {chunk_num}/{total_chunks}] All queries already recorded — skipping production."
            )
            continue

        # --- Production: build temporary JSONL for this chunk ---
        chunk_jsonl_path = os.path.join(run_dir, f"chunk_{chunk_num:04d}.jsonl")
        logger.info(f"[Chunk {chunk_num}/{total_chunks}] Producing JSONL → {os.path.basename(chunk_jsonl_path)}")
        n_requests = _produce_chunk_jsonl(
            chunk_entries=chunk_entries,
            sorted_doc_ids=sorted_doc_ids,
            vector_db=vector_db,
            embedding_model=embedding_model,
            db_conn=db_conn,
            jsonl_path=chunk_jsonl_path,
        )
        total_requests_written += n_requests
        logger.info(
            f"[Chunk {chunk_num}/{total_chunks}] Wrote {n_requests} requests. "
            f"Total so far: {total_requests_written}"
        )

        if n_requests == 0:
            logger.info(f"[Chunk {chunk_num}/{total_chunks}] No new requests — skipping submission.")
            # Clean up empty file
            try:
                os.remove(chunk_jsonl_path)
            except OSError:
                pass
            continue

        # --- Slot wait: poll until we have a free submission slot ---
        logger.info(f"[Chunk {chunk_num}/{total_chunks}] Waiting for a free submission slot...")
        while True:
            active = _count_active_jobs(client)
            if active < MAX_CONCURRENT_JOBS:
                logger.info(
                    f"[Chunk {chunk_num}/{total_chunks}] Slot available "
                    f"({active}/{MAX_CONCURRENT_JOBS} active). Submitting now."
                )
                break
            logger.info(
                f"[Chunk {chunk_num}/{total_chunks}] Slot full "
                f"({active}/{MAX_CONCURRENT_JOBS} active). "
                f"Sleeping {POLL_INTERVAL_SECONDS}s..."
            )
            time.sleep(POLL_INTERVAL_SECONDS)

        # --- Submission with exponential backoff on 429 ---
        try:
            job_id = _submit_with_backoff(client, chunk_jsonl_path)
            submitted_job_ids.append(job_id)
            _set_experiment_meta(
                db_conn, f"chunk_{chunk_num:04d}_job_id", job_id
            )
            # Persist running list of all IDs
            _set_experiment_meta(
                db_conn, "all_job_ids", ",".join(submitted_job_ids)
            )
            logger.info(
                f"[Chunk {chunk_num}/{total_chunks}] ✅ Submitted. job_id={job_id!r}"
            )
        except Exception as exc:
            logger.error(
                f"[Chunk {chunk_num}/{total_chunks}] ❌ Submission failed after backoff: {exc}"
            )
            _set_experiment_meta(
                db_conn, f"chunk_{chunk_num:04d}_error", str(exc)
            )
            # Do NOT abort the whole run — other chunks may still succeed.
            continue

    logger.info(
        f"Pipeline complete. "
        f"Total requests written: {total_requests_written}. "
        f"Jobs submitted: {len(submitted_job_ids)}."
    )
    if submitted_job_ids:
        logger.info(f"All job IDs: {submitted_job_ids}")

    _set_main_loop_completed(db_conn)
    db_conn.close()


async def run_global_correlation_pilot_batch_generator(
    vector_db,
    embedding_model,
    num_queries: int = 200,
    pilot_num_queries: int = 5,
    k: int = 50,
    output_dir: str = "results/global_exp"
):
    """
    Pilot (first N queries only): generate per-document Batch API requests.

    - Writes `gemini_pilot_requests.jsonl`
    - custom_id format: q{query_idx}_d{doc_id}
    - Inserts placeholders into SQLite with llm_score=NULL (dist/is_gt/rag_failed populated)
    """
    if str(CORRELATION_LLM_PROVIDER).lower() == "ollama":
        logger.info("Global Correlation PILOT: using Ollama provider (live scoring).")
        return await _run_global_correlation_pilot_ollama_async(
            vector_db=vector_db,
            embedding_model=embedding_model,
            pilot_num_queries=pilot_num_queries,
            k=k,
            output_dir=output_dir,
        )

    queries = load_qa_queries(num_queries)
    pilot_queries = queries[: int(pilot_num_queries)]
    actual_num_queries = len(pilot_queries)

    sorted_doc_ids = _collect_global_pool_doc_ids(vector_db, pilot_queries, embedding_model, k=k)
    global_pool_size = len(sorted_doc_ids)
    logger.info(f"PILOT: {actual_num_queries} queries; global pool = {global_pool_size} ids")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_dir, f"pilot_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "pilot_batch_offline_jsonl_per_doc",
                "pilot_num_queries": int(pilot_num_queries),
                "global_pool_size": global_pool_size,
                "k": k,
                "model": "gemini-2.0-flash",
            },
            f,
            indent=4,
        )

    db_path = os.path.join(run_dir, "experiment_results.db")
    db_conn = _init_sqlite(db_path)
    jsonl_path = os.path.join(run_dir, "gemini_pilot_requests.jsonl")

    written = 0
    with open(jsonl_path, "w", encoding="utf-8") as out_f:
        for q_idx, entry in enumerate(pilot_queries, start=1):
            query_text = entry.get("question")
            gt_id_norm = _gt_id_str(entry)
            gt_pmid = _gt_pmid_int_for_chroma(entry)

            if _is_query_skipped(db_conn, q_idx):
                continue

            gt_emb = _get_gt_embedding_dual_path(vector_db.collection, gt_id_norm, gt_pmid)
            if gt_emb is None:
                logger.warning(f"PILOT SKIPPED query_idx={q_idx}: GT missing.")
                _mark_query_skipped(db_conn, q_idx, "missing_gt_embedding_dual_path")
                continue

            query_emb = np.array(embedding_model.get_text_embedding(query_text), dtype=np.float32)
            rag_failed = _check_rag_baseline(vector_db, query_emb, gt_id_norm)

            # Only generate requests for missing/NULL/0.0
            processed_ids = _get_processed_ids_from_db(db_conn, q_idx)
            needs_score = _get_doc_ids_needing_score(db_conn, q_idx)
            missing_ids = [d for d in sorted_doc_ids if _normalize_id(d) not in processed_ids]
            target_ids = sorted(set([_normalize_id(x) for x in missing_ids] + list(needs_score)))
            if not target_ids:
                continue

            for chunk_start in range(0, len(target_ids), CHROMA_FETCH_CHUNK):
                chunk = target_ids[chunk_start : chunk_start + CHROMA_FETCH_CHUNK]
                payload = _chroma_batch_get_documents_and_embeddings(vector_db.collection, chunk)

                placeholder_rows: list[dict] = []
                for d_id in chunk:
                    did = _normalize_id(d_id)
                    if did not in payload:
                        continue
                    doc_text, emb = payload[did]
                    ctx = _extract_contexts_text(doc_text)
                    dist = float(cosine_distance(gt_emb, emb))
                    is_gt_check = 1 if (did == gt_id_norm or dist < 1e-5) else 0

                    placeholder_rows.append(
                        {
                            "doc_id": did,
                            "dist_to_gt": dist,
                            "is_gt": is_gt_check,
                            "rag_failed": 1 if rag_failed else 0,
                        }
                    )

                    # per-doc prompt uses 1-item list (returns [<int>])
                    prompt = _get_batch_scoring_prompt(query_text, [ctx])
                    custom_id = f"q{q_idx}_d{did}"
                    _record_pilot_request_map(db_conn, custom_id, q_idx, did)
                    out_f.write(json.dumps(_make_batch_request_line(custom_id, prompt), ensure_ascii=False) + "\n")
                    written += 1

                if placeholder_rows:
                    _insert_placeholder_rows(db_conn, q_idx, placeholder_rows)

    logger.info(f"PILOT wrote {written} requests to {jsonl_path}")
    logger.info("Submitting pilot batch job...")
    try:
        job = submit_batch_job(jsonl_path)
        job_id = getattr(job, "name", None) or getattr(job, "id", None) or str(job)
        logger.info(f"Job submitted: job_id={job_id!r}")
        _set_experiment_meta(db_conn, "last_batch_job_id", str(job_id))
        logger.info("ID saved to DB: key='last_batch_job_id'")
    except Exception as e:
        logger.error(f"Pilot batch submission failed (generator completed). error={e}")
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("\n❌ API Quota Exceeded: Your Google Gemini API key has run out of its monthly quota/spending cap.")
            print("Please visit https://ai.studio/spend to manage your limit, or update the key in configurations/config.py.\n")
        else:
            print(f"\n❌ Pilot batch submission failed: {e}\n")
        _set_experiment_meta(db_conn, "last_batch_job_error", str(e))
        logger.info("Saved submission error to DB: key='last_batch_job_error'")
    _set_main_loop_completed(db_conn)
    db_conn.close()
