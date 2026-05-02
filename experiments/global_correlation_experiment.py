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
import time
import uuid
import asyncio
from typing import Any, Callable
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ollama import Client
from google import genai
from google.genai import types
from scipy import stats

from configurations.config import (
    CORRELATION_LLM_PROVIDER,
    CORRELATION_OLLAMA_MODEL,
    OLLAMA_MAX_CONCURRENT_REQUESTS,
    OLLAMA_DOCS_PER_REQUEST,
    CORRELATION_PILOT_MAX_DOCS_PER_QUERY,
    GEMINI_API_KEY,
    OLLAMA_FAIL_FAST_ON_CONNECTION_ERROR,
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


def _setup_or_resume_run(
    output_dir: str,
    experiment_params: dict,
    run_prefix: str,
    *,
    support_resume: bool = True,
) -> str:
    """
    Returns the run directory to use.
    When support_resume=True and a compatible incomplete run exists, returns that
    directory.  Otherwise creates a new timestamped directory and writes
    experiment_config.json into it.
    """
    if support_resume:
        latest_run = _get_latest_run_dir(output_dir)
        if latest_run and _should_resume_run(latest_run, experiment_params):
            logger.info(f"AUTO-RESUME: {latest_run}")
            return latest_run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_dir, f"{run_prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_params, f, indent=4)
    logger.info(f"NEW RUN: {run_dir}")
    return run_dir


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


def _save_or_load_global_pool(run_dir: str, sorted_doc_ids: list[str]) -> list[str]:
    """
    On first run: save sorted_doc_ids to global_pool_ids.json in the run directory.
    On resume: load the exact original pool from disk, ignoring the freshly-recomputed one.

    This guarantees that the same doc set is scored regardless of when the run is
    resumed — even if the vector DB or k-NN tie-breaking changes in the interim.
    The file is written atomically to avoid partial-write corruption.
    """
    pool_path = os.path.join(run_dir, "global_pool_ids.json")
    if os.path.exists(pool_path):
        try:
            with open(pool_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if len(loaded) != len(sorted_doc_ids):
                logger.warning(
                    f"[POOL] Disk pool has {len(loaded)} ids, recomputed has {len(sorted_doc_ids)}. "
                    f"Using disk version to preserve consistency with the original run."
                )
            logger.info(f"[POOL] Loaded global pool from disk: {len(loaded)} ids")
            return loaded
        except Exception as exc:
            logger.warning(f"[POOL] Could not load global_pool_ids.json ({exc}). Using recomputed pool.")
            return sorted_doc_ids
    # First time: persist so every future resume uses the identical pool
    tmp_path = pool_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(sorted_doc_ids, f)
        os.replace(tmp_path, pool_path)
        logger.info(f"[POOL] Global pool saved to disk: {len(sorted_doc_ids)} ids")
    except Exception as exc:
        logger.warning(f"[POOL] Could not save global_pool_ids.json ({exc}).")
    return sorted_doc_ids


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


def _fetch_docs_in_batches(
    target_ids: list[str],
    collection,
    gt_emb: np.ndarray,
    gt_id_norm: str,
    rag_failed: bool,
    batch_size: int,
    on_batch_ready: Callable[[list[str], list[str], list[dict]], None],
) -> None:
    """
    Fetches target_ids from Chroma in CHROMA_FETCH_CHUNK sub-batches, computes
    distance and GT flags, then calls on_batch_ready(doc_ids, contexts,
    placeholder_rows) at every batch_size boundary and once for the final
    partial batch.  Frees each Chroma payload immediately after processing.
    """
    batch_doc_ids: list[str] = []
    batch_contexts: list[str] = []
    placeholder_rows: list[dict] = []

    for chunk_start in range(0, len(target_ids), CHROMA_FETCH_CHUNK):
        chunk = target_ids[chunk_start: chunk_start + CHROMA_FETCH_CHUNK]
        payload = _chroma_batch_get_documents_and_embeddings(collection, chunk)

        for d_id in chunk:
            did = _normalize_id(d_id)
            if did not in payload:
                continue
            doc_text, emb = payload[did]
            ctx = _extract_contexts_text(doc_text)
            dist = float(cosine_distance(gt_emb, emb))
            is_gt_check = 1 if (did == gt_id_norm or dist < 1e-5) else 0

            placeholder_rows.append({
                "doc_id": did,
                "dist_to_gt": dist,
                "is_gt": is_gt_check,
                "rag_failed": 1 if rag_failed else 0,
            })
            batch_doc_ids.append(did)
            batch_contexts.append(ctx)

            if len(batch_doc_ids) == batch_size:
                on_batch_ready(batch_doc_ids[:], batch_contexts[:], placeholder_rows[:])
                batch_doc_ids.clear()
                batch_contexts.clear()
                placeholder_rows.clear()

        del payload

    if batch_doc_ids:
        on_batch_ready(batch_doc_ids, batch_contexts, placeholder_rows)


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


def _resolve_query_state(
    conn,
    q_idx: int,
    entry: dict,
    sorted_doc_ids: list[str],
    collection,
    score_mode: str = "gemini",
) -> tuple[str, np.ndarray, list[str]] | None:
    """
    Returns (gt_id_norm, gt_emb, target_ids), or None when the GT embedding
    cannot be found (caller should skip/mark the query).

    score_mode "gemini": re-queues docs with llm_score NULL or 0.0.
    score_mode "ollama": re-queues only docs with llm_score NULL.
    """
    gt_id_norm = _gt_id_str(entry)
    gt_pmid = _gt_pmid_int_for_chroma(entry)
    gt_emb = _get_gt_embedding_dual_path(collection, gt_id_norm, gt_pmid)
    if gt_emb is None:
        return None
    processed_ids = _get_processed_ids_from_db(conn, q_idx)
    needs_score = (
        _get_doc_ids_needing_score_ollama(conn, q_idx)
        if score_mode == "ollama"
        else _get_doc_ids_needing_score(conn, q_idx)
    )
    missing_ids = [d for d in sorted_doc_ids if _normalize_id(d) not in processed_ids]
    target_ids = sorted(set(_normalize_id(x) for x in missing_ids) | needs_score)
    return gt_id_norm, gt_emb, target_ids


_LAST_OLLAMA_ERROR: str | None = None
_OLLAMA_CLIENT: Client | None = None
_OLLAMA_REQUEST_RETRIES = 3
_OLLAMA_RETRY_SLEEP_SECONDS = 2.0


def _get_ollama_client() -> Client:
    """Singleton Ollama client configured to work with BGU cluster."""
    global _OLLAMA_CLIENT
    if _OLLAMA_CLIENT is None:
        try:
            _OLLAMA_CLIENT = Client(
                host=OLLAMA_HOST,
                verify=bool(OLLAMA_VERIFY_SSL),
                timeout=float(OLLAMA_TIMEOUT_S),
            )
        except TypeError:
            # Some SDK versions may not support timeout in constructor.
            _OLLAMA_CLIENT = Client(
                host=OLLAMA_HOST,
                verify=bool(OLLAMA_VERIFY_SSL),
            )
    return _OLLAMA_CLIENT

def _cluster_llm_query(
    prompt: str,
    *,
    provider: str,
    model: str,
    system_prompt: str = "Return strictly JSON only.",
) -> str | None:
    """Generic provider-facing query function updated for BGU Ollama cluster."""
    p = str(provider).strip().lower()
    if p != "ollama":
        raise NotImplementedError(f"Unsupported LLM provider: {provider}")

    client = _get_ollama_client()
    last_err: Exception | None = None

    for attempt in range(1, _OLLAMA_REQUEST_RETRIES + 1):
        # Primary path: generate with explicit JSON format for stable score parsing.
        try:
            gen_resp = client.generate(
                model=model,
                prompt=prompt,
                stream=False,
                format="json",
            )
            gen_text = _extract_ollama_text_from_response(gen_resp)
            if gen_text:
                return gen_text
        except Exception as e:
            last_err = e
            logger.warning(
                f"Ollama generate attempt {attempt}/{_OLLAMA_REQUEST_RETRIES} failed: {e}"
            )

        # Fallback: chat endpoint, aligned with university examples.
        try:
            chat_resp = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                think="low",
            )
            chat_text = _extract_ollama_text_from_response(chat_resp)
            if chat_text:
                return chat_text
            last_err = RuntimeError("Ollama chat returned empty content")
        except TypeError:
            # Older SDKs may not support 'think'; retry without it.
            try:
                chat_resp = client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                chat_text = _extract_ollama_text_from_response(chat_resp)
                if chat_text:
                    return chat_text
                last_err = RuntimeError("Ollama chat returned empty content")
            except Exception as e:
                last_err = e
                logger.warning(
                    f"Ollama chat attempt {attempt}/{_OLLAMA_REQUEST_RETRIES} failed: {e}"
                )
        except Exception as e:
            last_err = e
            logger.warning(
                f"Ollama chat attempt {attempt}/{_OLLAMA_REQUEST_RETRIES} failed: {e}"
            )

        if attempt < _OLLAMA_REQUEST_RETRIES:
            time.sleep(_OLLAMA_RETRY_SLEEP_SECONDS)

    if last_err is not None:
        raise last_err
    raise RuntimeError("Ollama query failed: no response from generate/chat")

def _llm_response_to_dict(resp: Any) -> dict:
    """Convert provider SDK response object to dict safely."""
    if isinstance(resp, dict):
        return resp
    if resp is None:
        return {}
    if hasattr(resp, "model_dump"):
        try:
            out = resp.model_dump()
            return out if isinstance(out, dict) else {}
        except Exception:
            return {}
    return {}


def _extract_ollama_text_from_response(resp: Any) -> str | None:
    """
    Parse Ollama SDK response like the example:
      - generate: {"response":"..."}
      - chat: {"message":{"content":"..."}}
    Returns only the text needed for score parsing.
    """
    payload = _llm_response_to_dict(resp)
    direct = payload.get("response")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    msg = payload.get("message", {})
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, str) and content.strip():
        return content.strip()
    return None


def _validate_and_normalize_batch_scores(scores: list[int], expected_n: int) -> list[int]:
    """
    Post-parse validation:
    - exactly expected_n items
    - every item integer in 1..10
    Invalid/missing values are normalized and logged.
    """
    out: list[int] = []
    changes = 0
    for i in range(expected_n):
        raw = scores[i] if i < len(scores) else 1
        try:
            v = int(raw)
        except Exception:
            v = 1
            changes += 1
        if v < 1:
            v = 1
            changes += 1
        elif v > 10:
            v = 10
            changes += 1
        out.append(v)
    if changes > 0:
        logger.warning(
            f"[LLM] Score validation adjusted {changes} value(s); expected_n={expected_n}."
        )
    return out


def _ollama_call_generate(prompt: str) -> str | None:
    """
    Calls BGU cis-ollama server using /api/generate with JSON output mode.
    Returns the raw text generated by the model.
    """
    global _LAST_OLLAMA_ERROR
    _LAST_OLLAMA_ERROR = None
    try:
        out = _cluster_llm_query(
            prompt=prompt,
            provider="ollama",
            model=CORRELATION_OLLAMA_MODEL,
            system_prompt="Return strictly JSON only.",
        )
        if out is None or not str(out).strip():
            _LAST_OLLAMA_ERROR = "Empty response from Ollama cluster"
            return None
        return out
    except Exception as e:
        _LAST_OLLAMA_ERROR = str(e)
        logger.exception(f"Ollama generate failed: {e}")
        return None


def _ollama_print_available_models_once() -> None:
    """
    Print available Ollama models from the cluster.
    Uses /api/tags (no extra deps). Best-effort only.
    """
    try:
        client = _get_ollama_client()
        print(f"[Ollama] Fetch available models via client.list() | host={OLLAMA_HOST}")

        active = client.ps()
        print("[Ollama] Currently active models:")
        for m in getattr(active, "models", []) or []:
            size_vram = getattr(m, "size_vram", 0) or 0
            size_gb = float(size_vram) / (1024 ** 3)
            print(f"  - {getattr(m, 'model', 'unknown')} ({size_gb:.2f} GB)")

        models_payload = json.loads(client.list().model_dump_json())
        models_list = models_payload.get("models", []) if isinstance(models_payload, dict) else []
        models = sorted(
            {
                str(m.get("model") or m.get("name"))
                for m in models_list
                if isinstance(m, dict) and (m.get("model") or m.get("name"))
            }
        )

        if models:
            print("[Ollama] Available models: " + ", ".join(models))
        else:
            print("[Ollama] Available models: (none parsed)")
        print(f"[Ollama] Using model: {CORRELATION_OLLAMA_MODEL}")
    except Exception as e:
        logger.warning(f"Ollama models listing failed (non-fatal): {e}")


def _ollama_score_batch(query_text: str, doc_contexts: list[str]) -> list[int]:
    """Returns integer scores (1..10) for each doc_context item."""
    print(f"[Ollama] Scoring batch: query_chars={len(query_text)} docs={len(doc_contexts)}")
    prompt = _get_batch_scoring_prompt(query_text, doc_contexts)
    raw = _ollama_call_generate(prompt)
    if raw is None:
        if OLLAMA_FAIL_FAST_ON_CONNECTION_ERROR and _LAST_OLLAMA_ERROR:
            raise RuntimeError(f"Ollama connectivity failure: {_LAST_OLLAMA_ERROR}")
        return [0] * len(doc_contexts)
    if not isinstance(raw, str):
        raw = json.dumps(raw, ensure_ascii=False)
    ints = _parse_batch_scores(raw, expected_n=len(doc_contexts))
    ints = _validate_and_normalize_batch_scores(ints, expected_n=len(doc_contexts))
    print(f"[Ollama] Parsed integer scores: {ints}")
    return ints


def _update_llm_scores_for_docs(conn, query_idx: int, doc_ids: list[str], scores: list[float]) -> None:
    cur = conn.cursor()
    cur.executemany(
        "UPDATE results SET llm_score = ? WHERE query_idx = ? AND doc_id = ?",
        [(float(scores[i]), int(query_idx), _normalize_id(doc_ids[i])) for i in range(len(doc_ids))],
    )


def _score_batch_or_fail_fast(
    db_conn,
    q_idx: int,
    query_text: str,
    batch_contexts: list[str],
) -> list[int]:
    try:
        return _ollama_score_batch(query_text, batch_contexts)
    except Exception as e:
        _set_experiment_meta(db_conn, "fatal_llm_error", str(e))
        _set_experiment_meta(db_conn, "fatal_llm_error_query_idx", str(int(q_idx)))
        db_conn.commit()
        raise


def _record_fatal_llm_error(db_conn, q_idx: int, error: Exception) -> None:
    _set_experiment_meta(db_conn, "fatal_llm_error", str(error))
    _set_experiment_meta(db_conn, "fatal_llm_error_query_idx", str(int(q_idx)))
    db_conn.commit()


async def _score_batches_parallel_for_query(
    q_idx: int,
    query_text: str,
    batches_to_score: list[tuple[list[str], list[str]]],
    concurrency: int,
    on_batch_scored: Callable[[list[str], list[int]], None],
) -> None:
    """
    Score query batches in parallel with bounded concurrency.
    Calls on_batch_scored(doc_ids, int_scores) as each batch completes.
    """
    if not batches_to_score:
        return

    conc = max(1, int(concurrency))
    sem = asyncio.Semaphore(conc)

    async def _run_one(doc_ids: list[str], contexts: list[str]) -> tuple[list[str], list[int]]:
        async with sem:
            ints = await asyncio.to_thread(_ollama_score_batch, query_text, contexts)
            return doc_ids, ints

    tasks = [asyncio.create_task(_run_one(doc_ids, contexts)) for doc_ids, contexts in batches_to_score]
    try:
        for fut in asyncio.as_completed(tasks):
            doc_ids, ints = await fut
            on_batch_scored(doc_ids, ints)
        return
    except Exception:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _run_global_correlation_experiment_ollama_async(
    vector_db,
    embedding_model,
    num_queries: int,
    k: int,
    output_dir: str,
    *,
    is_pilot: bool = False,
) -> None:
    """
    Ollama live scoring: inserts placeholders and updates llm_score in SQLite immediately.

    is_pilot=True: always creates a new run (no resume), applies
    CORRELATION_PILOT_MAX_DOCS_PER_QUERY, and uses a 'pilot_run_ollama_' prefix.
    """
    _ollama_print_available_models_once()

    if is_pilot:
        all_queries = load_qa_queries(200)
        queries = all_queries[:int(num_queries)]
    else:
        queries = load_qa_queries(num_queries)

    actual_num_queries = len(queries)
    sorted_doc_ids = _collect_global_pool_doc_ids(vector_db, queries, embedding_model, k=k)
    global_pool_size = len(sorted_doc_ids)

    run_prefix = "pilot_run_ollama" if is_pilot else "ollama_run"
    experiment_params = {
        "num_queries_requested": num_queries,
        "actual_num_queries": actual_num_queries,
        "global_pool_size": global_pool_size,
        "k": k,
        "model": CORRELATION_OLLAMA_MODEL,
        "mode": "ollama_pilot_live" if is_pilot else "ollama_live",
    }

    run_dir = _setup_or_resume_run(
        output_dir, experiment_params, run_prefix, support_resume=not is_pilot
    )
    db_path = os.path.join(run_dir, "experiment_results.db")
    db_conn = _init_sqlite(db_path)

    sorted_doc_ids = _save_or_load_global_pool(run_dir, sorted_doc_ids)
    global_pool_size = len(sorted_doc_ids)

    tag = "[Ollama Pilot]" if is_pilot else "[Ollama]"
    logger.info(f"{tag} Dataset: {actual_num_queries} queries; global pool = {global_pool_size} ids")

    ollama_conc = max(1, int(OLLAMA_MAX_CONCURRENT_REQUESTS))
    ollama_batch_size = max(1, int(OLLAMA_DOCS_PER_REQUEST))
    logger.info(f"{tag} Concurrency per query: {ollama_conc}, docs per request: {ollama_batch_size}")

    for q_idx, entry in enumerate(queries, start=1):
        if _is_query_skipped(db_conn, q_idx):
            continue

        query_text = entry.get("question")
        state = _resolve_query_state(
            db_conn, q_idx, entry, sorted_doc_ids, vector_db.collection, score_mode="ollama"
        )
        if state is None:
            logger.warning(f"{tag} SKIPPED_PERMANENTLY query_idx={q_idx}: GT missing.")
            _mark_query_skipped(db_conn, q_idx, "missing_gt_embedding_dual_path")
            continue

        gt_id_norm, gt_emb, target_ids = state

        if is_pilot and CORRELATION_PILOT_MAX_DOCS_PER_QUERY and int(CORRELATION_PILOT_MAX_DOCS_PER_QUERY) > 0:
            max_docs = int(CORRELATION_PILOT_MAX_DOCS_PER_QUERY)
            if len(target_ids) > max_docs:
                logger.info(f"{tag} Limiting docs/query from {len(target_ids)} to {max_docs}.")
                target_ids = target_ids[:max_docs]

        if not target_ids:
            continue

        query_emb = np.array(embedding_model.get_text_embedding(query_text), dtype=np.float32)
        rag_failed = _check_rag_baseline(vector_db, query_emb, gt_id_norm)

        batches_to_score: list[tuple[list[str], list[str]]] = []

        def _on_batch(doc_ids, contexts, placeholder_rows):
            _insert_placeholder_rows(db_conn, q_idx, placeholder_rows)
            batches_to_score.append((doc_ids[:], contexts[:]))

        logger.info(f"{tag} Query {q_idx}: scoring up to {len(target_ids)} docs...")
        _fetch_docs_in_batches(
            target_ids, vector_db.collection, gt_emb, gt_id_norm, rag_failed, ollama_batch_size, _on_batch
        )

        if batches_to_score:
            logger.info(
                f"{tag} Query {q_idx}: scoring {len(batches_to_score)} batches "
                f"(batch_size={ollama_batch_size}, concurrency={ollama_conc})"
            )
            try:
                def _on_batch_scored(doc_ids: list[str], ints: list[int]) -> None:
                    scores = [float(v) / 10.0 for v in ints]
                    _update_llm_scores_for_docs(db_conn, q_idx, doc_ids, scores)
                    db_conn.commit()

                await _score_batches_parallel_for_query(
                    q_idx=q_idx,
                    query_text=query_text,
                    batches_to_score=batches_to_score,
                    concurrency=ollama_conc,
                    on_batch_scored=_on_batch_scored,
                )
            except Exception as e:
                _record_fatal_llm_error(db_conn, q_idx, e)
                raise

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
MAX_NETWORK_RETRIES = 5        # Max retries on transient network/connection errors

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


def _chunk_has_job_id(db_conn, chunk_num: int) -> bool:
    """Returns True if a batch job was successfully submitted and recorded for this chunk."""
    cur = db_conn.cursor()
    cur.execute(
        "SELECT 1 FROM experiment_meta WHERE key = ? LIMIT 1",
        (f"chunk_{chunk_num:04d}_job_id",),
    )
    return cur.fetchone() is not None


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
            if _is_query_skipped(db_conn, q_idx):
                continue

            query_text = entry.get("question")
            state = _resolve_query_state(db_conn, q_idx, entry, sorted_doc_ids, vector_db.collection)
            if state is None:
                logger.warning(
                    f"SKIPPED_PERMANENTLY query_idx={q_idx}: GT missing "
                    f"(id={_gt_id_str(entry)!r}, pmid={_gt_pmid_int_for_chroma(entry)!r})."
                )
                _mark_query_skipped(db_conn, q_idx, "missing_gt_embedding_dual_path")
                continue

            gt_id_norm, gt_emb, target_ids = state
            if not target_ids:
                continue

            query_emb = np.array(embedding_model.get_text_embedding(query_text), dtype=np.float32)
            rag_failed = _check_rag_baseline(vector_db, query_emb, gt_id_norm)

            def _on_batch(doc_ids, contexts, placeholder_rows):
                nonlocal request_idx
                _insert_placeholder_rows(db_conn, q_idx, placeholder_rows)
                prompt = _get_batch_scoring_prompt(query_text, contexts)
                custom_id = f"q{q_idx}_b{request_idx:08d}_{uuid.uuid4().hex[:8]}"
                _record_batch_request_map(db_conn, custom_id, q_idx, doc_ids)
                out_f.write(json.dumps(_make_batch_request_line(custom_id, prompt), ensure_ascii=False) + "\n")
                request_idx += 1

            _fetch_docs_in_batches(
                target_ids, vector_db.collection, gt_emb, gt_id_norm, rag_failed, BATCH_SIZE, _on_batch
            )

    return request_idx


def _submit_with_backoff(client, jsonl_path: str) -> str:
    """
    Upload + submit one JSONL file with exponential backoff on 429 and transient network errors.
    Returns the job_id string.
    Raises on non-recoverable errors or after MAX_NETWORK_RETRIES consecutive network failures.
    """
    import time

    backoff = BACKOFF_BASE_SECONDS
    attempt = 0
    network_attempts = 0
    while True:
        attempt += 1
        try:
            job = submit_batch_job(jsonl_path, client=client)
            job_id = getattr(job, "name", None) or getattr(job, "id", None) or str(job)
            return job_id
        except Exception as exc:
            msg = str(exc)
            msg_lower = msg.lower()
            is_quota = "429" in msg or "RESOURCE_EXHAUSTED" in msg.upper()
            is_network = (
                isinstance(exc, (ConnectionError, TimeoutError, OSError))
                or any(f in msg_lower for f in ("connection", "timeout", "503", "unavailable", "network"))
            )

            # --- 429 Storage quota: aggressive cleanup + immediate retry ---
            if is_quota and "file_storage_bytes" in msg_lower:
                logger.warning("[STORAGE] Detected file_storage_bytes quota error. Running aggressive cleanup!")
                _cleanup_remote_storage(client, aggressive=True)
                continue

            if not is_quota:
                if is_network:
                    network_attempts += 1
                    if network_attempts > MAX_NETWORK_RETRIES:
                        logger.error(
                            f"Network error exceeded {MAX_NETWORK_RETRIES} retries. Giving up: {exc}"
                        )
                        raise
                    wait = min(backoff, BACKOFF_MAX_SECONDS)
                    logger.warning(
                        f"Transient network error (attempt {network_attempts}/{MAX_NETWORK_RETRIES}). "
                        f"Backing off {wait}s: {exc}"
                    )
                    time.sleep(wait)
                    backoff = min(backoff * 2, BACKOFF_MAX_SECONDS)
                    continue
                raise  # Non-quota, non-network errors propagate immediately
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
    - Recovery: skips a chunk if every query has a batch_request_map entry AND
      a job_id was recorded. If the JSONL exists but no job_id (crash mid-submit),
      the existing JSONL is reused without regeneration.
    - 429 and transient network errors in upload/submit → exponential backoff with retry.
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

    run_dir = _setup_or_resume_run(output_dir, experiment_params, "run")

    db_path = os.path.join(run_dir, "experiment_results.db")
    db_conn = _init_sqlite(db_path)

    # Load the exact pool used in the original run (saves on new run; loads on all restarts)
    sorted_doc_ids = _save_or_load_global_pool(run_dir, sorted_doc_ids)
    global_pool_size = len(sorted_doc_ids)

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
    # Seed from DB so resumes accumulate job IDs rather than overwriting them
    try:
        _cur = db_conn.cursor()
        _cur.execute("SELECT value FROM experiment_meta WHERE key = 'all_job_ids'")
        _row = _cur.fetchone()
        submitted_job_ids: list[str] = (
            [j.strip() for j in str(_row[0]).split(",") if j.strip()]
            if (_row and _row[0]) else []
        )
    except Exception:
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

        # --- Recovery: skip only if queries recorded AND job was successfully submitted ---
        if _is_chunk_fully_recorded(db_conn, chunk_entries) and _chunk_has_job_id(db_conn, chunk_num):
            logger.info(
                f"[Chunk {chunk_num}/{total_chunks}] All queries recorded and job submitted — skipping."
            )
            continue

        # --- Production: reuse existing JSONL if present (previous submission failed) ---
        chunk_jsonl_path = os.path.join(run_dir, f"chunk_{chunk_num:04d}.jsonl")
        if os.path.exists(chunk_jsonl_path) and os.path.getsize(chunk_jsonl_path) > 0:
            n_requests = sum(1 for line in open(chunk_jsonl_path, encoding="utf-8") if line.strip())
            logger.info(
                f"[Chunk {chunk_num}/{total_chunks}] Reusing existing JSONL "
                f"({n_requests} requests) — previous submission failed."
            )
        else:
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
        return await _run_global_correlation_experiment_ollama_async(
            vector_db=vector_db,
            embedding_model=embedding_model,
            num_queries=pilot_num_queries,
            k=k,
            output_dir=output_dir,
            is_pilot=True,
        )

    queries = load_qa_queries(num_queries)
    pilot_queries = queries[: int(pilot_num_queries)]
    actual_num_queries = len(pilot_queries)

    sorted_doc_ids = _collect_global_pool_doc_ids(vector_db, pilot_queries, embedding_model, k=k)
    global_pool_size = len(sorted_doc_ids)
    logger.info(f"PILOT: {actual_num_queries} queries; global pool = {global_pool_size} ids")

    experiment_params = {
        "mode": "pilot_batch_offline_jsonl_per_doc",
        "pilot_num_queries": int(pilot_num_queries),
        "global_pool_size": global_pool_size,
        "k": k,
        "model": "gemini-2.0-flash",
    }
    run_dir = _setup_or_resume_run(output_dir, experiment_params, "pilot_run", support_resume=False)

    db_path = os.path.join(run_dir, "experiment_results.db")
    db_conn = _init_sqlite(db_path)
    jsonl_path = os.path.join(run_dir, "gemini_pilot_requests.jsonl")

    written = 0
    with open(jsonl_path, "w", encoding="utf-8") as out_f:
        for q_idx, entry in enumerate(pilot_queries, start=1):
            query_text = entry.get("question")

            if _is_query_skipped(db_conn, q_idx):
                continue

            state = _resolve_query_state(db_conn, q_idx, entry, sorted_doc_ids, vector_db.collection)
            if state is None:
                logger.warning(f"PILOT SKIPPED query_idx={q_idx}: GT missing.")
                _mark_query_skipped(db_conn, q_idx, "missing_gt_embedding_dual_path")
                continue

            gt_id_norm, gt_emb, target_ids = state
            if not target_ids:
                continue

            query_emb = np.array(embedding_model.get_text_embedding(query_text), dtype=np.float32)
            rag_failed = _check_rag_baseline(vector_db, query_emb, gt_id_norm)

            def _on_batch(doc_ids, contexts, placeholder_rows):
                nonlocal written
                _insert_placeholder_rows(db_conn, q_idx, placeholder_rows)
                prompt = _get_batch_scoring_prompt(query_text, contexts)
                custom_id = f"q{q_idx}_d{doc_ids[0]}"
                _record_pilot_request_map(db_conn, custom_id, q_idx, doc_ids[0])
                out_f.write(json.dumps(_make_batch_request_line(custom_id, prompt), ensure_ascii=False) + "\n")
                written += 1

            _fetch_docs_in_batches(target_ids, vector_db.collection, gt_emb, gt_id_norm, rag_failed, 1, _on_batch)

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
