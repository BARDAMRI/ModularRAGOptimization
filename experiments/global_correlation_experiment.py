"""
Global Correlation Experiment
=============================

Research question
-----------------
For a given query q and a set of documents D, is there a correlation between
dist(d, GT(q)) and LLM(d, q)?

  GT(q)       — the ground-truth document for query q.
  dist(d1,d2) — cosine distance between two document embeddings.
  LLM(d, q)   — relevance score (0.0–1.0) assigned by the LLM to document d for query q.

The experiment measures whether embedding distance to the GT predicts LLM-judged relevance,
across a large, diverse document pool.


Terms
-----
GT doc          The single ground-truth document for a query. Each of the 200 queries has
                exactly one GT doc (identified by PubMed ID).

dist_to_gt      Cosine distance from a candidate document's embedding to the GT doc's
                embedding, for the specific query being evaluated. This is the Y-axis of
                every per-query scatter plot.

llm_score       Relevance score (0.0–1.0) returned by the LLM for a (document, query) pair.
                This is the X-axis of every per-query scatter plot.

ranked_pool     A single global ordered list of unique doc IDs built once at run start.
                Construction:
                  1. For every query, find the top-max_ranked_per_gt nearest neighbours to
                     the GT embedding (gt_ranked) and the top-max_ranked_per_query nearest
                     neighbours to the query-text embedding (q_ranked).
                  2. Interleave gt_ranked and q_ranked alternately per query →
                     up to (max_ranked_per_gt + max_ranked_per_query) candidates per query.
                  3. Interleave all 200 per-query lists position-by-position, deduplicating
                     globally → one flat list of up to ~40 000 unique docs.

ranked_cursor   Integer offset into ranked_pool tracking how far the current run has
                progressed. Persisted in the manifest so runs can resume.

batch           stride × n_queries docs sliced from ranked_pool per stage.
                With stride=10 and n_queries=200, batch=2 000.

scoring_union   The set of new (delta) docs added to D in the current stage.

per_query_docs  For each query: scoring_union minus docs already scored for that query.
                Used to avoid rescoring (query, doc) pairs on resume.

per_query_seen  dict[query_idx → set[doc_id]] built from the SQLite results table.
                Guards against duplicate rows on resume.

manifest        JSON file (staged_manifest.json) stored in the run directory.
                Contains gt_ids, ranked_pool list, ranked_cursor, per-stage records,
                and run parameters. Re-loaded on resume to continue from where the
                run left off.

staging_spec_hash
                Short SHA-256 fingerprint of (n_queries, stride, max_ranked_per_gt,
                max_ranked_per_query). Used to detect parameter changes that would
                make an existing manifest incompatible with a new run.


Document set D
--------------
D is shared across all queries (global pool). It is built once and grown stage by stage:

  1. All GT docs                        — 200 unique docs (one per query)
  2. top-max_ranked_per_gt docs per GT  — up to 100 × 200 = 20 000 unique docs
  3. top-max_ranked_per_query docs per query — up to 100 × 200 = 20 000 unique docs

  Total unique docs in D: up to ~40 200 (exact count depends on overlap between queries).

Every query is scored against every doc in D (full cross-product):
  200 queries × ~40 200 docs ≈ 8 M (query, doc) pairs scored by the LLM.


Configuration (run_config.json → staged mode / config.py defaults)
------------------------------------------------------------------
  scoring_provider          LLM backend: "inference_api" (live) or "gemini" (batch).
  queries_to_load           Number of queries to use (default 200).
  staging_stride            Docs advanced per pool position per stage (default 10).
                            batch = stride × n_queries docs pulled per stage.
  staging_max_ranked_per_gt Max GT-embedding neighbours per query (default 100).
  staging_max_ranked_per_query
                            Max query-text-embedding neighbours per query (default 100).

  Keys read from config.py:
    STAGING_STRIDE, STAGING_MAX_RANKED_PER_GT, STAGING_MAX_RANKED_PER_QUERY
  API credentials loaded from .env:
    INFERENCE_API_KEY  (inference_api provider)
    GEMINI_API_KEY     (gemini provider)


Experiment steps
----------------
1. Load queries
   Load `queries_to_load` QA entries from the dataset. Each entry carries
   question text and a PMID that identifies its GT doc.

2. Build manifest (first run only)
   For every query:
     a. Retrieve top-max_ranked_per_gt neighbours of the GT embedding from Chroma.
     b. Embed the query text; retrieve top-max_ranked_per_query neighbours.
     c. Interleave the two lists alternately (GT-neighbour, query-neighbour, …).
   Interleave all 200 per-query lists position-by-position and deduplicate globally
   → ranked_pool. Save manifest with ranked_cursor=0.

3. Stage loop  (repeat until ranked_cursor exhausts ranked_pool)

   Stage 1
     new docs = all 200 GT docs
              + ranked_pool[0 : batch]
     ≈ 200 + 2 000 = ~2 200 unique docs
     pairs scored = 200 × ~2 200 = ~440 000

   Stage N (N > 1)
     new docs = ranked_pool[rc : rc + batch]   (rc = (N-1) × batch)
     ≈ 2 000 unique docs
     pairs scored = 200 × ~2 000 = ~400 000

   Per stage:
     i.   Build scoring_union (new docs for this stage).
     ii.  Advance ranked_cursor by batch.
     iii. For each query: send (query, doc) pairs to LLM; store (query_idx, doc_id,
          llm_score, dist_to_gt, is_gt) in SQLite.
     iv.  Mark stage "harvested" in manifest.

   Total stages ≈ ceil(len(ranked_pool) / batch)  ≈ ceil(40 000 / 2 000) = 20.
   Total pairs  ≈ 200 × 40 200 ≈ 8 M.

4. Analysis (after all stages complete)
   - Per-query Spearman correlation between llm_score and dist_to_gt.
   - 200 scatter plots: X = llm_score, Y = dist(d, GT(q)).
   - Overview bar chart: queries (X) sorted by Spearman ρ vs ρ value (Y).
   - Summary CSV with per-query ρ, p-value, and rag_failed flag.


Outputs (results/global_exp/<run_dir>/)
---------------------------------------
  experiment_results.db          SQLite database (results, experiment_meta tables).
  staged_manifest.json           Pool + cursor state for resume.
  query_<NNN>_global_scatter.png Per-query scatter: llm_score vs dist_to_gt.
  global_summary_overview.png    Overview: ρ per query sorted descending.
  summary_stats.csv              Per-query Spearman ρ and p-value.


Entry points (from main.py)
---------------------------
  run_global_correlation_staged_async       — staged mode (primary).
  run_global_correlation_experiment_async   — full one-shot run (live / Gemini batch).
  run_global_correlation_pilot_batch_generator — pilot (3 queries, quick check).
  retry_missing_batches                     — resubmit failed Gemini batch jobs.
  check_batch_status / submit_batch_job     — Gemini batch helpers.


Provider selection (config.CORRELATION_LLM_PROVIDER)
-----------------------------------------------------
  inference_api — live scoring via HTTPS (Vertex-style generateContent). Requires INFERENCE_API_KEY + INFERENCE_API_URL in .env.
  gemini        — offline batch scoring: write JSONL → submit Gemini Batch API job →
              harvest results → sync scores to SQLite. Requires GEMINI_API_KEY in .env.


File layout
-----------
  Provider helpers → IDs / prompts → SQLite → Batch API → resume / analytics
  → vector pool I/O → live scoring → pipeline harvest → staged mode
"""

import hashlib
import json
import math
import os
import sqlite3
import time
import uuid
import asyncio
from typing import Callable
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from configurations.config import (
    CORRELATION_OLLAMA_MODEL,
    OLLAMA_MAX_CONCURRENT_REQUESTS,
    OLLAMA_DOCS_PER_REQUEST,
    CORRELATION_PILOT_MAX_DOCS_PER_QUERY,
    CORRELATION_GEMINI_BATCH_MODEL,
    GEMINI_API_KEY,
    STAGING_STRIDE,
    STAGING_MAX_RANKED_PER_GT,
    OLLAMA_FAIL_FAST_ON_CONNECTION_ERROR,
    OLLAMA_HOST,
    OLLAMA_TIMEOUT_S,
    OLLAMA_VERIFY_SSL,
    correlation_live_model_name, STAGING_MAX_RANKED_PER_QUERY,
)
from scripts.qa_data_set_loader import load_qa_queries
from utility.logger import logger
from utility.llm_gateway import (
    LLMGatewayError,
    LLMRequestKind,
    await_relevance_scores_for_documents,
    build_relevance_scoring_prompt,
    get_gemini_client,
    get_llm_gateway,
    get_ollama_client,
    normalize_provider,
)
from vector_db.trilateration_retriever import cosine_distance

# --- Offline Gemini Batch API (JSONL generator) ---
# This module now generates requests for Google's Batch API (offline).
# No semaphore / asyncio.sleep / backoff in generator mode.
CHROMA_FETCH_CHUNK = 400
BATCH_SIZE = 20
MAX_REQUESTS_PER_FILE = 50000  # Split to avoid 2GB limit (each req ~25KB * 50k = ~1.25GB)

_LIVE_PROVIDER_ALIASES: dict[str, str] = {
    "ollama": "ollama",
    "inference_api": "inference_api",
    "inference_hub": "inference_api",
}


def _configured_correlation_provider() -> str:
    """Current ``CORRELATION_LLM_PROVIDER`` (reads config module so runtime overrides apply)."""
    from configurations import config as cfg

    return str(cfg.CORRELATION_LLM_PROVIDER).strip().lower()


def _normalize_live_provider(provider: str | None = None) -> str:
    """Return canonical live provider id: ``ollama`` or ``inference_api``."""
    raw = (provider or _configured_correlation_provider()).strip().lower()
    return _LIVE_PROVIDER_ALIASES.get(raw, raw)


def _is_live_correlation_provider(provider: str | None = None) -> bool:
    """Return True if scoring uses a live gateway provider (Ollama or inference API)."""
    return _normalize_live_provider(provider) in ("ollama", "inference_api")


def _is_gemini_batch_provider(provider: str | None = None) -> bool:
    """True when scoring uses Gemini Batch API (not live Ollama / Inference API)."""
    p = _configured_correlation_provider() if provider is None else str(provider).strip().lower()
    return p == "gemini"


def _live_provider_display_name(provider: str | None = None) -> str:
    """Human-readable label for logs (Ollama, Inference API, etc.)."""
    p = _normalize_live_provider(provider)
    if p == "inference_api":
        return "Inference_API"
    if p == "ollama":
        return "Ollama"
    return p.upper() or "LIVE_LLM"


def _live_run_prefix(*, is_pilot: bool, provider: str | None = None) -> str:
    """Directory name prefix for a live run folder (pilot vs full, per provider)."""
    p = _normalize_live_provider(provider)
    if p == "inference_api":
        return "pilot_run_inference_api" if is_pilot else "inference_api_run"
    return "pilot_run_ollama" if is_pilot else "ollama_run"


def _live_mode_name(*, is_pilot: bool, provider: str | None = None) -> str:
    """Value stored in experiment_config.json under ``mode`` for live runs."""
    p = _normalize_live_provider(provider)
    if p == "inference_api":
        return "inference_api_pilot_live" if is_pilot else "inference_api_live"
    return "ollama_pilot_live" if is_pilot else "ollama_live"


def _live_log_tag(*, is_pilot: bool = False, provider: str | None = None) -> str:
    """Bracketed log prefix including provider name and optional Pilot marker."""
    name = _live_provider_display_name(provider)
    return f"[{name} Pilot]" if is_pilot else f"[{name}]"


# ---------------------------------------------------------------------------
# Utility: IDs & input pruning (normalize ids, extract text for prompts)
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


# ---------------------------------------------------------------------------
# GT retrieval (Chroma) — locate ground-truth document embedding
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
    """Parse GT id as integer for Chroma metadata filters; None if not numeric."""
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
# SQLite lifecycle — results, meta, batch_request_map, skipped_queries
# ---------------------------------------------------------------------------

META_KEY_MAIN_LOOP_COMPLETED = "main_loop_completed"

BATCH_STATUS_SUBMITTED = "submitted"
BATCH_STATUS_SUCCEEDED = "succeeded"
BATCH_STATUS_MISSING = "missing"
BATCH_STATUS_FAILED = "failed"
BATCH_STATUS_PENDING = "pending"


def _init_sqlite(db_path):
    """Open (or create) experiment_results.db with results, meta, and batch map tables."""
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
    _ensure_batch_request_map_columns(conn)
    conn.commit()
    return conn


def _ensure_batch_request_map_columns(conn) -> None:
    """Add optional tracking columns on older DBs (idempotent)."""
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(batch_request_map)")
    cols = {str(row[1]) for row in cur.fetchall()}
    if "status" not in cols:
        cur.execute(
            "ALTER TABLE batch_request_map ADD COLUMN status TEXT "
            f"DEFAULT '{BATCH_STATUS_SUBMITTED}'"
        )
    if "updated_at" not in cols:
        cur.execute(
            "ALTER TABLE batch_request_map ADD COLUMN updated_at TEXT DEFAULT CURRENT_TIMESTAMP"
        )
    conn.commit()


def _snap_to_anchor(score: int) -> int:
    """Clamp parsed integer score to the 1..10 anchor scale."""
    try:
        v = int(score)
    except Exception:
        return 1
    return max(1, min(10, v))


def _is_main_loop_completed(conn) -> bool:
    """True when experiment_meta records that the main scoring loop finished."""
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
    """Mark the main scoring loop as finished in experiment_meta."""
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO experiment_meta (key, value) VALUES (?, ?)",
        (META_KEY_MAIN_LOOP_COMPLETED, "1"),
    )
    conn.commit()


def _set_experiment_meta(conn, key: str, value: str) -> None:
    """Upsert a key/value pair in the experiment_meta table."""
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO experiment_meta (key, value) VALUES (?, ?)",
        (str(key), str(value)),
    )
    conn.commit()


def _mark_query_skipped(conn, query_idx: int, reason: str) -> None:
    """Record a query as permanently skipped with a reason string."""
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO skipped_queries (query_idx, reason) VALUES (?, ?)",
        (int(query_idx), str(reason)),
    )
    conn.commit()


def _is_query_skipped(conn, query_idx: int) -> bool:
    """True if query_idx exists in skipped_queries."""
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM skipped_queries WHERE query_idx = ? LIMIT 1", (int(query_idx),))
    return cur.fetchone() is not None


def _count_skipped_queries(conn) -> int:
    """Number of rows in skipped_queries."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM skipped_queries")
    return int(cur.fetchone()[0])


def _count_successful_queries(conn) -> int:
    """Count of distinct query_idx values present in results."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(DISTINCT query_idx) FROM results")
    return int(cur.fetchone()[0])


def _is_experiment_complete(conn, actual_num_queries: int) -> bool:
    """True when main loop completed and processed+skipped queries cover the dataset."""
    if not _is_main_loop_completed(conn):
        return False
    return (_count_successful_queries(conn) + _count_skipped_queries(conn)) >= int(actual_num_queries)


def _get_processed_ids_from_db(conn, q_idx):
    """Set of doc_id values already stored for a query in results."""
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id FROM results WHERE query_idx = ?", (q_idx,))
    return {_normalize_id(row[0]) for row in cursor.fetchall()}


def _save_to_db(conn, rows):
    """Bulk INSERT OR IGNORE of fully populated result rows."""
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
    """Map a Gemini Batch custom_id to (query_idx, doc_ids) with status=submitted."""
    _ensure_batch_request_map_columns(conn)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO batch_request_map "
        "(custom_id, query_idx, doc_ids_json, status, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
        (str(custom_id), int(query_idx), json.dumps(doc_ids), BATCH_STATUS_SUBMITTED),
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
            "model": str(CORRELATION_GEMINI_BATCH_MODEL),
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
                """Sort key: prefer create_time, then update_time, then name."""
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

def submit_batch_job(jsonl_path: str, client=None, *, model: str | None = None):
    """
    Submit the JSONL file as a Gemini Batch job (best-effort, SDK-dependent).
    Prints job id and status.
    """
    batch_model = str(model or CORRELATION_GEMINI_BATCH_MODEL).strip()
    if client is None:
        client = get_gemini_client(GEMINI_API_KEY)
    get_llm_gateway().acquire("gemini", batch_model, LLMRequestKind.BATCH)

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
        model=batch_model,
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
    client = get_gemini_client(GEMINI_API_KEY)
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
    """How many distinct documents were recorded for one query."""
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(DISTINCT doc_id) FROM results WHERE query_idx = ?",
        (query_idx,),
    )
    return int(cur.fetchone()[0])


# ---------------------------------------------------------------------------
# Run directory / resume — run folders, experiment_config.json, resume logic
# ---------------------------------------------------------------------------


def _get_latest_run_dir(output_dir):
    """Most recently modified subdirectory under output_dir, or None."""
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
    """Read num_queries_requested (or legacy num_queries) from saved experiment params."""
    v = old_params.get("num_queries_requested")
    if v is not None:
        return int(v)
    v = old_params.get("num_queries")
    return int(v) if v is not None else None


def _params_compatible(old_params: dict, new_params: dict) -> bool:
    """True if an existing run's experiment_config matches new run parameters."""
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
    """True if latest run dir is compatible and not yet complete."""
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
# Analytics — correlation, scatter plots, failure reports
# ---------------------------------------------------------------------------


def _check_rag_baseline(vector_db, query_emb, gt_id, prints=False):
    """True when top-1 retrieval does not return the ground-truth document."""
    initial_retrieval = vector_db.retrieve(query_emb, top_k=1, prints=prints)
    if not initial_retrieval:
        return True
    top_hit_id = _normalize_id(getattr(initial_retrieval[0].node, "id_", ""))
    return top_hit_id != _normalize_id(gt_id)


def _log_failure_analysis(run_dir, q_idx, query, query_df):
    """Write per-query failure diagnostics to the run directory."""
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
    """Save llm_score vs dist_to_gt scatter PNG for one query."""
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
    """Aggregate SQLite results, log correlation stats, and write summary plots."""
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
    """Union of doc ids retrieved for all queries (global scoring pool)."""
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


def _save_or_load_queries(run_dir: str, queries: list[dict]) -> list[dict]:
    """
    On first run: persist the ordered query list to queries.json.
    On resume: load from disk, discarding the freshly-loaded list.

    This is the query-side counterpart of :func:`_save_or_load_global_pool`.
    Without it a resume that sees a changed/reordered source JSONL would silently
    misattribute every SQLite score — q_idx 3 would be analysed against a different
    question than was originally scored.
    """
    path = os.path.join(run_dir, "queries.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if len(loaded) != len(queries):
                logger.warning(
                    f"[QUERIES] Disk has {len(loaded)} queries, fresh load has {len(queries)}. "
                    "Using disk version to preserve q_idx consistency with the original run."
                )
            logger.info(f"[QUERIES] Loaded query list from disk: {len(loaded)} entries")
            return loaded
        except Exception as exc:
            logger.warning(f"[QUERIES] Could not load queries.json ({exc}). Using fresh load.")
            return queries
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(queries, f)
        os.replace(tmp_path, path)
        logger.info(f"[QUERIES] Query list persisted: {len(queries)} entries → {path}")
    except Exception as exc:
        logger.warning(f"[QUERIES] Could not save queries.json ({exc}).")
    return queries


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
    """Fetch document text and embeddings from Chroma in CHROMA_FETCH_CHUNK batches."""
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
# Live scoring — real-time scoring via llm_gateway (Ollama / Inference API)
# ---------------------------------------------------------------------------


def _get_doc_ids_needing_score_live(conn, query_idx: int) -> set[str]:
    """Live LLM mode (Ollama / Inference API): only NULL needs scoring (0.0 = parse/network failure)."""
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
    score_mode "live" (or legacy "ollama"): re-queues only docs with llm_score NULL.
    """
    gt_id_norm = _gt_id_str(entry)
    gt_pmid = _gt_pmid_int_for_chroma(entry)
    gt_emb = _get_gt_embedding_dual_path(collection, gt_id_norm, gt_pmid)
    if gt_emb is None:
        return None
    processed_ids = _get_processed_ids_from_db(conn, q_idx)
    needs_score = (
        _get_doc_ids_needing_score_live(conn, q_idx)
        if score_mode in ("live", "ollama")
        else _get_doc_ids_needing_score(conn, q_idx)
    )
    missing_ids = [d for d in sorted_doc_ids if _normalize_id(d) not in processed_ids]
    target_ids = sorted(set(_normalize_id(x) for x in missing_ids) | needs_score)
    return gt_id_norm, gt_emb, target_ids


def print_live_provider_info_once() -> None:
    """
    Print provider/model info for the configured live backend.
    Best-effort only (Ollama lists cluster models; Inference API prints model + key hint).
    """
    prov = _normalize_live_provider()
    if prov == "inference_api":
        from configurations.config import INFERENCE_API_KEY, INFERENCE_API_MODEL

        if not (INFERENCE_API_KEY or "").strip():
            print("[Inference_API] Warning: INFERENCE_API_KEY is not set.")
        print(f"[Inference_API] Model: {INFERENCE_API_MODEL}")
        return
    if prov != "ollama":
        print(f"[{_live_provider_display_name(prov)}] Live provider configured (no cluster listing).")
        return
    try:
        client = get_ollama_client(
            OLLAMA_HOST,
            verify_ssl=bool(OLLAMA_VERIFY_SSL),
            timeout_s=float(OLLAMA_TIMEOUT_S),
        )
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


# Backward-compatible name used by main.py.
ollama_print_available_models_once = print_live_provider_info_once


async def _live_score_batch(query_text: str, doc_contexts: list[str]) -> list[int]:
    """
    Score a batch of documents via the LLM gateway (provider + prompt in, integers 1..10 out).

    The experiment does not call providers or parse raw LLM text; that is handled in
    ``utility.llm_gateway.await_relevance_scores``.
    """
    label = _live_provider_display_name()
    n = len(doc_contexts)
    print(f"[{label}] Scoring batch: query_chars={len(query_text)} docs={n}")
    prov = _normalize_live_provider()
    if not _is_live_correlation_provider(prov):
        raise RuntimeError(
            f"Live scoring provider must be ollama or inference_api, got {prov!r}"
        )
    try:
        ints = await await_relevance_scores_for_documents(
            provider=prov,
            model=correlation_live_model_name(),
            query=query_text,
            doc_contexts=doc_contexts,
        )
    except LLMGatewayError as exc:
        logger.exception(f"{label} scoring failed: {exc}")
        if OLLAMA_FAIL_FAST_ON_CONNECTION_ERROR:
            raise RuntimeError(f"{label} request failure: {exc}") from exc
        return [0] * n
    print(f"[{label}] Scores: {ints}")
    return ints


def _update_llm_scores_for_docs(conn, query_idx: int, doc_ids: list[str], scores: list[float]) -> None:
    """Write normalized llm_score values (0.0–1.0) for doc_ids of one query."""
    cur = conn.cursor()
    cur.executemany(
        "UPDATE results SET llm_score = ? WHERE query_idx = ? AND doc_id = ?",
        [(float(scores[i]), int(query_idx), _normalize_id(doc_ids[i])) for i in range(len(doc_ids))],
    )


async def _score_batch_or_fail_fast(
    db_conn,
    q_idx: int,
    query_text: str,
    batch_contexts: list[str],
) -> list[int]:
    """Score one batch via live gateway; raise on connection error if configured."""
    try:
        return await _live_score_batch(query_text, batch_contexts)
    except Exception as e:
        _set_experiment_meta(db_conn, "fatal_llm_error", str(e))
        _set_experiment_meta(db_conn, "fatal_llm_error_query_idx", str(int(q_idx)))
        db_conn.commit()
        raise


def _record_fatal_llm_error(db_conn, q_idx: int, error: Exception) -> None:
    """Persist fatal LLM error text in experiment_meta for a query."""
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
            ints = await _live_score_batch(query_text, contexts)
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


async def _run_global_correlation_experiment_live_async(
    vector_db,
    embedding_model,
    num_queries: int,
    k: int,
    output_dir: str,
    *,
    is_pilot: bool = False,
) -> None:
    """
    Live scoring (Ollama or Inference API): inserts placeholders and updates llm_score in SQLite.

    Run steps
    ---------
    1. Load queries and build the global document pool.
    2. Create or resume run directory and initialize SQLite.
    3. Per query: GT, target docs, Chroma fetch, parallel scoring via gateway.
    4. Mark completion, compute stats and plots.

    ``is_pilot=True``: always a new run, cap docs per query, no resume.
    """
    live_prov = _normalize_live_provider()

    # --- Step 1: load queries and global document pool ---
    if is_pilot:
        all_queries = load_qa_queries(200)
        queries = all_queries[:int(num_queries)]
    else:
        queries = load_qa_queries(num_queries)

    actual_num_queries = len(queries)
    sorted_doc_ids = _collect_global_pool_doc_ids(vector_db, queries, embedding_model, k=k)
    global_pool_size = len(sorted_doc_ids)

    run_prefix = _live_run_prefix(is_pilot=is_pilot, provider=live_prov)
    experiment_params = {
        "num_queries_requested": num_queries,
        "actual_num_queries": actual_num_queries,
        "global_pool_size": global_pool_size,
        "k": k,
        "model": correlation_live_model_name(),
        "llm_backend": live_prov,
        "mode": _live_mode_name(is_pilot=is_pilot, provider=live_prov),
    }

    # --- Step 2: run dir, SQLite, persist pool/queries for resume ---
    run_dir = _setup_or_resume_run(
        output_dir, experiment_params, run_prefix, support_resume=not is_pilot
    )
    db_path = os.path.join(run_dir, "experiment_results.db")
    db_conn = _init_sqlite(db_path)

    sorted_doc_ids = _save_or_load_global_pool(run_dir, sorted_doc_ids)
    if not is_pilot:
        queries = _save_or_load_queries(run_dir, queries)
    global_pool_size = len(sorted_doc_ids)
    actual_num_queries = len(queries)

    tag = _live_log_tag(is_pilot=is_pilot, provider=live_prov)
    logger.info(f"{tag} Dataset: {actual_num_queries} queries; global pool = {global_pool_size} ids")

    live_conc = max(1, int(OLLAMA_MAX_CONCURRENT_REQUESTS))
    live_batch_size = max(1, int(OLLAMA_DOCS_PER_REQUEST))
    logger.info(f"{tag} Concurrency per query: {live_conc}, docs per request: {live_batch_size}")

    # --- Step 3: query loop — placeholders, LLM scoring, SQLite updates ---
    for q_idx, entry in enumerate(queries, start=1):
        if _is_query_skipped(db_conn, q_idx):
            continue

        query_text = entry.get("question")
        state = _resolve_query_state(
            db_conn, q_idx, entry, sorted_doc_ids, vector_db.collection, score_mode="live"
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
            target_ids, vector_db.collection, gt_emb, gt_id_norm, rag_failed, live_batch_size, _on_batch
        )

        if batches_to_score:
            logger.info(
                f"{tag} Query {q_idx}: scoring {len(batches_to_score)} batches "
                f"(batch_size={live_batch_size}, concurrency={live_conc})"
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
                    concurrency=live_conc,
                    on_batch_scored=_on_batch_scored,
                )
            except Exception as e:
                _record_fatal_llm_error(db_conn, q_idx, e)
                raise

    # --- Step 4: finalize — metadata, correlation, plots ---
    _set_main_loop_completed(db_conn)
    _calculate_final_stats_and_plot_sqlite(db_conn, run_dir)
    db_conn.close()


# Backward-compatible alias (main.py / docs may still reference the old name).
_run_global_correlation_experiment_ollama_async = _run_global_correlation_experiment_live_async


# ---------------------------------------------------------------------------
# Gemini Batch pipeline — chunk submission, harvest, retry
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
# Terminal states — job will not change further
_ENDED_JOB_STATES = {
    "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED", "ACTIVE", "FAILED",
}
# Subset of ended states that produced usable output
_SUCCEEDED_JOB_STATES = {"JOB_STATE_SUCCEEDED", "ACTIVE"}


def _download_job_output(client, job, run_dir: str) -> str | None:
    """
    Download the output JSONL for a completed Gemini Batch job from the Files API.
    Returns the local file path, or None on failure.
    """
    dest = getattr(job, "dest", None)
    if dest is None:
        logger.warning("[HARVEST] job.dest is None — cannot download output.")
        return None
    file_name = getattr(dest, "file_name", None)
    if not file_name:
        logger.warning("[HARVEST] job.dest.file_name is empty — cannot download output.")
        return None

    job_id = str(getattr(job, "name", None) or "unknown")
    safe_id = job_id.replace("/", "_").replace(":", "_").replace(".", "_")
    local_path = os.path.join(run_dir, f"batch_output_{safe_id}.jsonl")
    try:
        logger.info(f"[HARVEST] Downloading output file {file_name!r} → {os.path.basename(local_path)}")
        data: bytes = client.files.download(file=file_name)
        with open(local_path, "wb") as fh:
            fh.write(data)
        logger.info(f"[HARVEST] Saved {len(data):,} bytes → {local_path}")
        return local_path
    except Exception as exc:
        logger.error(f"[HARVEST] Download failed for {file_name!r}: {exc}")
        return None


def _poll_and_harvest_jobs(
    client,
    job_ids: list[str],
    run_dir: str,
    db_path: str,
    poll_interval: int = POLL_INTERVAL_SECONDS,
) -> list[str]:
    """
    Poll all submitted Gemini Batch jobs until each reaches a terminal state,
    download the output JSONL for each that succeeded, then sync scores into
    the SQLite DB via sync_batch_results.

    Returns the list of downloaded local JSONL paths.

    Harvest steps
    -------------
    1. Poll each job_id until a terminal state.
    2. Download output JSONL into the run directory.
    3. Call ``sync_batch_results`` to update llm_score in SQLite.
    """
    if not job_ids:
        logger.warning("[HARVEST] No job IDs to harvest.")
        return []

    # Import here to avoid a circular import at module load time.
    from sync_batch_results import sync_batch_results as _sync

    pending = list(job_ids)
    downloaded: list[str] = []

    # --- Step 1: wait for all jobs to finish ---
    logger.info(f"[HARVEST] Waiting for {len(pending)} job(s) to complete...")
    while pending:
        still_pending: list[str] = []
        for job_id in pending:
            try:
                job = client.batches.get(name=job_id)
                state = str(getattr(job, "state", "") or "").upper()
            except Exception as exc:
                logger.warning(f"[HARVEST] Could not poll {job_id!r}: {exc} — will retry.")
                still_pending.append(job_id)
                continue

            if not any(s in state for s in _ENDED_JOB_STATES):
                logger.info(f"[HARVEST] {job_id!r}: state={state} (running)")
                still_pending.append(job_id)
                continue

            if any(s in state for s in _SUCCEEDED_JOB_STATES):
                logger.info(f"[HARVEST] {job_id!r}: SUCCEEDED — downloading output")
                path = _download_job_output(client, job, run_dir)
                if path:
                    downloaded.append(path)
            else:
                logger.warning(f"[HARVEST] {job_id!r}: ended with state={state} — no output to download")

        pending = still_pending
        if pending:
            logger.info(f"[HARVEST] {len(pending)} job(s) still running. Sleeping {poll_interval}s...")
            time.sleep(poll_interval)

    if not downloaded:
        logger.warning("[HARVEST] No output files were downloaded; sync skipped.")
        return []

    # --- Steps 2–3: sync Batch responses into results table ---
    logger.info(f"[HARVEST] Syncing {len(downloaded)} output file(s) into {db_path}")
    _sync(db_path=db_path, output_paths_input=",".join(downloaded))
    return downloaded


def _index_jsonl_requests_by_custom_id(run_dir: str) -> dict[str, dict]:
    """Load request lines from all JSONL files in a run directory (key = custom_id)."""
    out: dict[str, dict] = {}
    if not os.path.isdir(run_dir):
        return out
    for name in sorted(os.listdir(run_dir)):
        if not name.endswith(".jsonl"):
            continue
        if name.startswith("batch_output_"):
            continue
        path = os.path.join(run_dir, name)
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    cid = obj.get("custom_id") or obj.get("key") or obj.get("id")
                    if cid:
                        out[str(cid)] = obj
        except OSError as exc:
            logger.warning(f"[RETRY] Could not read {path!r}: {exc}")
    return out


def _custom_ids_needing_batch_retry(conn) -> list[str]:
    """custom_ids whose mapped docs still lack a usable llm_score, or status is missing/failed."""
    _ensure_batch_request_map_columns(conn)
    cur = conn.cursor()
    cur.execute("SELECT custom_id, query_idx, doc_ids_json, status FROM batch_request_map")
    need: list[str] = []
    for custom_id, q_idx, doc_ids_json, status in cur.fetchall():
        st = str(status or BATCH_STATUS_SUBMITTED).strip().lower()
        if st in (BATCH_STATUS_MISSING, BATCH_STATUS_FAILED, BATCH_STATUS_PENDING):
            need.append(str(custom_id))
            continue
        try:
            doc_ids = json.loads(doc_ids_json)
        except Exception:
            doc_ids = []
        if not doc_ids:
            continue
        placeholders = ",".join(["?"] * len(doc_ids))
        cur.execute(
            f"SELECT COUNT(*) FROM results WHERE query_idx = ? AND doc_id IN ({placeholders}) "
            "AND (llm_score IS NULL OR llm_score = 0.0)",
            [int(q_idx), *[_normalize_id(d) for d in doc_ids]],
        )
        if int(cur.fetchone()[0]) > 0:
            need.append(str(custom_id))
    return sorted(set(need))


def retry_missing_batches(
    db_path: str,
    *,
    per_job_timeout_s: int | None = None,
) -> list[str]:
    """
    Re-submit Gemini Batch requests for rows still missing scores.

    Reuses existing request JSONL lines from the run directory when possible.

    Retry steps
    -----------
    1. Find custom_ids with missing scores or status missing/failed.
    2. Rebuild JSONL from original request files in the run directory.
    3. Submit and auto-harvest.
    """
    del per_job_timeout_s  # reserved for future per-job harvest timeout wiring
    if _is_live_correlation_provider():
        raise RuntimeError(
            "retry_missing_batches applies only to Gemini Batch mode "
            f"(current provider={_configured_correlation_provider()!r})."
        )
    if not os.path.isfile(db_path):
        raise FileNotFoundError(db_path)

    run_dir = os.path.dirname(os.path.abspath(db_path))
    conn = sqlite3.connect(db_path)
    _ensure_batch_request_map_columns(conn)
    # --- Step 1: identify requests needing retry ---
    custom_ids = _custom_ids_needing_batch_retry(conn)
    if not custom_ids:
        logger.info("[RETRY] No missing/failed batch requests found — nothing to resubmit.")
        conn.close()
        return []

    indexed = _index_jsonl_requests_by_custom_id(run_dir)
    missing_lines = [cid for cid in custom_ids if cid not in indexed]
    if missing_lines:
        logger.warning(
            f"[RETRY] {len(missing_lines)} custom_id(s) have no source JSONL line in {run_dir} "
            f"(first few: {missing_lines[:5]})."
        )

    # --- Step 2: rebuild JSONL from existing request files ---
    retry_path = os.path.join(run_dir, f"retry_missing_{int(time.time())}.jsonl")
    written = 0
    with open(retry_path, "w", encoding="utf-8") as out_f:
        for cid in custom_ids:
            obj = indexed.get(cid)
            if obj is None:
                continue
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1
    if written == 0:
        conn.close()
        raise RuntimeError(
            "[RETRY] Could not rebuild any request lines from run JSONL files."
        )

    logger.info(f"[RETRY] Wrote {written} request(s) → {retry_path}")
    client = get_gemini_client(GEMINI_API_KEY)
    job_id = _submit_with_backoff(client, retry_path)
    _set_experiment_meta(conn, "last_batch_job_id", job_id)
    prev = _get_experiment_meta(conn, "retry_job_ids") or ""
    merged = ",".join([j for j in (prev.split(",") if prev else []) + [job_id] if j])
    _set_experiment_meta(conn, "retry_job_ids", merged)
    conn.close()

    # --- Step 3: submit + harvest ---
    logger.info(f"[RETRY] Submitted job_id={job_id!r}; harvesting output...")
    _poll_and_harvest_jobs(
        client=client,
        job_ids=[job_id],
        run_dir=run_dir,
        db_path=db_path,
    )
    return [job_id]


def _get_experiment_meta(conn, key: str) -> str | None:
    """Read a single experiment_meta value by key, or None."""
    cur = conn.cursor()
    cur.execute("SELECT value FROM experiment_meta WHERE key = ?", (str(key),))
    row = cur.fetchone()
    return str(row[0]) if row and row[0] is not None else None


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
                prompt = build_relevance_scoring_prompt(query_text, contexts)
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

  Run steps (Gemini)
  ------------------
  1. If live provider — delegate to ``_run_global_correlation_experiment_live_async``.
  2. Build pool + run dir + SQLite.
  3. Split into chunks; per chunk: JSONL → wait for slot → submit with backoff.
  4. Harvest all job_ids and write scores into SQLite.
    """
    # --- Step 0: route to live path when not Gemini ---
    if _is_live_correlation_provider():
        label = _live_provider_display_name()
        logger.info(f"Global Correlation: using {label} provider (live scoring).")
        return await _run_global_correlation_experiment_live_async(
            vector_db=vector_db,
            embedding_model=embedding_model,
            num_queries=num_queries,
            k=k,
            output_dir=output_dir,
        )

    import time

    # --- Step 1: queries, global pool, run directory ---
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
        "model": CORRELATION_GEMINI_BATCH_MODEL,
        "llm_backend": "gemini",
        "mode": "batch_pipeline_jsonl",
        "batch_size": BATCH_SIZE,
        "pipeline_chunk_size": PIPELINE_CHUNK_SIZE,
    }

    run_dir = _setup_or_resume_run(output_dir, experiment_params, "run")

    db_path = os.path.join(run_dir, "experiment_results.db")
    db_conn = _init_sqlite(db_path)

    # Load the exact pool and query list used in the original run.
    sorted_doc_ids = _save_or_load_global_pool(run_dir, sorted_doc_ids)
    queries = _save_or_load_queries(run_dir, queries)
    global_pool_size = len(sorted_doc_ids)

    client = get_gemini_client(GEMINI_API_KEY)

    # --- Step 2: split into chunks (up to PIPELINE_CHUNK_SIZE queries each) ---
    indexed_queries = list(enumerate(queries, start=1))

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

    # --- Step 3: chunk loop — build/reuse JSONL, wait for slot, submit ---
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

    # --- Step 4: wait for jobs to complete and sync results into DB ---
    if submitted_job_ids:
        _poll_and_harvest_jobs(
            client=client,
            job_ids=submitted_job_ids,
            run_dir=run_dir,
            db_path=db_path,
        )


async def run_global_correlation_pilot_batch_generator(
    vector_db,
    embedding_model,
    num_queries: int = 200,
    pilot_num_queries: int = 5,
    k: int = 50,
    output_dir: str = "results/global_exp",
    pilot_batch_size: int = 20,
    per_job_timeout_s: float | None = None,
):
    """
    Pilot (first N queries only): generate per-document Batch API requests.

    - Writes `gemini_pilot_requests.jsonl`
    - custom_id format: q{query_idx}_d{doc_id}
    - Inserts placeholders into SQLite with llm_score=NULL (dist/is_gt/rag_failed populated)

  Run steps
  ---------
  1. Route to live pilot when provider is not Gemini.
  2. (Gemini) Build JSONL — one request per document, map in batch_request_map.
  3. Submit Batch job and store job_id in meta.
    """
    # --- Step 0: live pilot (Ollama / Inference API) ---
    if _is_live_correlation_provider():
        label = _live_provider_display_name()
        logger.info(f"Global Correlation PILOT: using {label} provider (live scoring).")
        return await _run_global_correlation_experiment_live_async(
            vector_db=vector_db,
            embedding_model=embedding_model,
            num_queries=pilot_num_queries,
            k=k,
            output_dir=output_dir,
            is_pilot=True,
        )

    # --- Step 1 (Gemini): pilot queries + pool + new run ---
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
        "model": CORRELATION_GEMINI_BATCH_MODEL,
        "llm_backend": "gemini",
    }
    run_dir = _setup_or_resume_run(output_dir, experiment_params, "pilot_run", support_resume=False)

    db_path = os.path.join(run_dir, "experiment_results.db")
    db_conn = _init_sqlite(db_path)
    jsonl_path = os.path.join(run_dir, "gemini_pilot_requests.jsonl")

    # --- Step 2: write JSONL (one doc per line) + SQLite placeholders ---
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
                prompt = build_relevance_scoring_prompt(query_text, contexts)
                custom_id = f"q{q_idx}_d{doc_ids[0]}"
                _record_pilot_request_map(db_conn, custom_id, q_idx, doc_ids[0])
                out_f.write(json.dumps(_make_batch_request_line(custom_id, prompt), ensure_ascii=False) + "\n")
                written += 1

            _fetch_docs_in_batches(target_ids, vector_db.collection, gt_emb, gt_id_norm, rag_failed, 1, _on_batch)

    logger.info(f"PILOT wrote {written} requests to {jsonl_path}")
    # --- Step 3: submit job to Gemini Batch API ---
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


# ---------------------------------------------------------------------------
# Staged additive-pool mode — grow document pool in stages (ranked + random)
# ---------------------------------------------------------------------------

MANIFEST_FILENAME = "correlation_pool_manifest.json"


def _manifest_path(run_dir: str) -> str:
    """Absolute path to correlation_pool_manifest.json inside a run directory."""
    return os.path.join(run_dir, MANIFEST_FILENAME)


def _save_manifest(manifest: dict, run_dir: str) -> None:
    """Atomically write the staged-run manifest JSON to disk."""
    path = _manifest_path(run_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)


def _load_manifest(run_dir: str) -> dict | None:
    """Load staged-run manifest from disk, or None if missing."""
    path = _manifest_path(run_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _staging_spec_hash(
    n_queries: int, stride: int, max_ranked_gt: int, max_ranked_q: int
) -> str:
    """Short hash fingerprint of staged mode parameters for resume compatibility."""
    payload = f"{n_queries}|{stride}|{max_ranked_gt}|{max_ranked_q}|global_pool_v2"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _staged_params_compatible(old_params: dict, new_params: dict) -> bool:
    """True if an existing staged run matches new staging parameters."""
    if old_params.get("stage_mode") != new_params.get("stage_mode"):
        return False
    if old_params.get("staging_spec_hash") != new_params.get("staging_spec_hash"):
        return False
    if _requested_num_queries_key(old_params) != new_params.get("num_queries_requested"):
        return False
    old_actual = old_params.get("actual_num_queries")
    if old_actual is not None and int(old_actual) != int(new_params["actual_num_queries"]):
        return False
    if old_params.get("llm_backend") != new_params.get("llm_backend"):
        return False
    if old_params.get("model") != new_params.get("model"):
        return False
    return True


def _setup_or_resume_staged_run(output_dir: str, experiment_params: dict) -> str:
    """Return staged run directory (resume compatible run or create new)."""
    latest_run = _get_latest_run_dir(output_dir)
    if latest_run and os.path.exists(os.path.join(latest_run, "experiment_config.json")):
        try:
            with open(os.path.join(latest_run, "experiment_config.json"), "r", encoding="utf-8") as f:
                old_params = json.load(f)
            if _staged_params_compatible(old_params, experiment_params):
                logger.info(f"AUTO-RESUME staged run: {latest_run}")
                return latest_run
        except Exception as exc:
            logger.warning(f"Could not resume staged run from {latest_run}: {exc}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_dir, f"staged_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_params, f, indent=4)
    logger.info(f"NEW STAGED RUN: {run_dir}")
    return run_dir


def _get_per_query_seen_for_docs(conn, doc_ids: set[str]) -> dict[str, set[str]]:
    """Map of query_idx -> set of already-scored doc_ids, restricted to doc_ids.

    Loads only the rows relevant to the given doc set instead of the full results
    table, keeping memory proportional to the stage size (~2 000 docs) not the
    cumulative run size (~8 M rows by stage 20).
    """
    if not doc_ids:
        return {}
    per_query: dict[str, set[str]] = {}
    doc_list = list(doc_ids)
    chunk_size = 900  # stay under SQLite's 999-parameter limit
    cur = conn.cursor()
    for i in range(0, len(doc_list), chunk_size):
        chunk = doc_list[i: i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        cur.execute(
            f"SELECT query_idx, doc_id FROM results WHERE doc_id IN ({placeholders})",
            chunk,
        )
        for q_idx, doc_id in cur.fetchall():
            if doc_id:
                per_query.setdefault(str(q_idx), set()).add(_normalize_id(doc_id))
    return per_query


def _build_ranked_neighbor_ids(
    vector_db,
    *,
    anchor_emb: np.ndarray,
    gt_id_norm: str,
    max_ranked: int,
) -> list[str]:
    """Top similar doc ids to anchor_emb, excluding GT itself."""
    seen: set[str] = set()
    out: list[str] = []
    top_k = int(max_ranked) + 10
    for hit in vector_db.retrieve(anchor_emb, top_k=top_k, prints=False):
        nid = _normalize_id(getattr(hit.node, "id_", ""))
        if not nid or nid == gt_id_norm or nid in seen:
            continue
        seen.add(nid)
        out.append(nid)
        if len(out) >= int(max_ranked):
            break
    return out


def _interleave_ranked_ids(gt_ranked: list[str], q_ranked: list[str]) -> list[str]:
    """Interleave GT-based and query-based neighbor lists, deduplicating."""
    from itertools import zip_longest
    seen: set[str] = set()
    out: list[str] = []
    for gt_doc, q_doc in zip_longest(gt_ranked, q_ranked, fillvalue=None):
        if gt_doc and gt_doc not in seen:
            seen.add(gt_doc)
            out.append(gt_doc)
        if q_doc and q_doc not in seen:
            seen.add(q_doc)
            out.append(q_doc)
    return out


def _interleave_across_queries(per_query_lists: dict[str, list[str]]) -> list[str]:
    """Build global pool by interleaving per-query lists position-by-position, deduplicating."""
    max_len = max((len(v) for v in per_query_lists.values()), default=0)
    seen: set[str] = set()
    result: list[str] = []
    for pos in range(max_len):
        for q_idx_str in sorted(per_query_lists.keys(), key=int):
            lst = per_query_lists[q_idx_str]
            if pos < len(lst):
                doc = lst[pos]
                if doc and doc not in seen:
                    seen.add(doc)
                    result.append(doc)
    return result


def _load_or_build_manifest(
    run_dir: str,
    queries: list[dict],
    vector_db,
    embedding_model,
    *,
    stride: int,
    max_ranked_gt: int,
    max_ranked_query: int,
) -> dict:
    """Load staged manifest from disk or build the global ranked pool.

    D = 200 GT docs + top-max_ranked_gt neighbours per GT embedding
                    + top-max_ranked_query neighbours per query embedding.
    ranked_pool is a single global deduplicated list interleaved across all queries.
    Each stage adds stride*n_queries docs from ranked_pool and scores every query
    against every new doc (full cross-product).
    """
    existing = _load_manifest(run_dir)
    if existing is not None:
        return existing

    per_query_ranked: dict[str, list[str]] = {}
    gt_ids: dict[str, str] = {}

    for q_idx, entry in enumerate(queries, start=1):
        q_idx_str = str(q_idx)
        gt_id_norm = _gt_id_str(entry)
        gt_pmid = _gt_pmid_int_for_chroma(entry)
        gt_emb = _get_gt_embedding_dual_path(vector_db.collection, gt_id_norm, gt_pmid)

        gt_ranked: list[str] = []
        q_ranked: list[str] = []
        if gt_emb is not None:
            gt_ranked = _build_ranked_neighbor_ids(
                vector_db, anchor_emb=gt_emb, gt_id_norm=gt_id_norm, max_ranked=max_ranked_gt,
            )
        if max_ranked_query > 0:
            query_text = entry.get("question", "")
            q_emb = np.array(embedding_model.get_text_embedding(query_text), dtype=np.float32)
            q_ranked = _build_ranked_neighbor_ids(
                vector_db, anchor_emb=q_emb, gt_id_norm=gt_id_norm, max_ranked=max_ranked_query,
            )

        gt_ids[q_idx_str] = gt_id_norm
        per_query_ranked[q_idx_str] = _interleave_ranked_ids(gt_ranked, q_ranked)

    ranked_pool = _interleave_across_queries(per_query_ranked)

    manifest: dict = {
        "staging_stride": int(stride),
        "max_ranked_per_gt": int(max_ranked_gt),
        "max_ranked_per_query": int(max_ranked_query),
        "gt_ids": gt_ids,
        "ranked_pool": ranked_pool,
        "ranked_cursor": 0,
        "stages": {},
        "staging_last_completed_stage": 0,
    }

    _save_manifest(manifest, run_dir)
    logger.info(
        f"[STAGED] Built global pool manifest: "
        f"ranked_pool={len(ranked_pool)}, stride={stride}, n_queries={len(queries)}."
    )
    return manifest


def _is_staged_run_complete(manifest: dict) -> bool:
    """True when the ranked_pool cursor is exhausted."""
    rc = int(manifest.get("ranked_cursor", 0))
    return rc >= len(manifest.get("ranked_pool", []))


def _build_stage_scoring_set(
    manifest: dict,
    stage: int,
    per_query_seen: dict[str, set[str]],
    stride: int,
) -> tuple[set[str], dict[str, list[str]], dict[str, int]]:
    """
    Build the global delta of new docs for this stage, then produce per-query
    doc lists for cross-product scoring.

    Each stage pulls stride*n_queries docs from ranked_pool. Every query is
    scored against every new doc (full cross-product), filtered per-query to
    skip already-scored (query, doc) pairs.

    Returns:
        scoring_union  — set of new doc IDs
        per_query_docs — {q_idx_str: [doc_ids to score for this query]}
        new_cursors    — {"ranked": new_rc}
    """
    gt_ids = manifest.get("gt_ids", {})
    ranked_pool = manifest.get("ranked_pool", [])
    rc = int(manifest.get("ranked_cursor", 0))
    n_queries = len(gt_ids)
    batch = stride * n_queries  # docs to pull from ranked_pool per stage

    new_docs: list[str] = []
    stage_seen: set[str] = set()  # dedup within new docs of this stage

    # Stage 1: include GT doc for every query
    if stage == 1:
        for gt_id in gt_ids.values():
            nid = _normalize_id(gt_id)
            if nid and nid not in stage_seen:
                stage_seen.add(nid)
                new_docs.append(nid)

    for doc_id in ranked_pool[rc : rc + batch]:
        nid = _normalize_id(doc_id)
        if nid and nid not in stage_seen:
            stage_seen.add(nid)
            new_docs.append(nid)

    scoring_union: set[str] = set(new_docs)

    # Cross-product: every query scores every new doc, filtered per-query
    per_query_docs: dict[str, list[str]] = {}
    for q_idx_str in gt_ids:
        query_seen = per_query_seen.get(q_idx_str, set())
        q_docs = [d for d in new_docs if d not in query_seen]
        if q_docs:
            per_query_docs[q_idx_str] = q_docs

    new_cursors = {"ranked": min(rc + batch, len(ranked_pool))}
    return scoring_union, per_query_docs, new_cursors


def _produce_stage_gemini_jsonl(
    *,
    stage: int,
    scoring_union: set[str],
    queries: list[dict],
    vector_db,
    embedding_model,
    db_conn,
    jsonl_path: str,
) -> int:
    """Write one stage JSONL; returns number of batch requests written."""
    if not scoring_union:
        return 0

    scoring_list = sorted(scoring_union)
    request_idx = 0
    with open(jsonl_path, "w", encoding="utf-8") as out_f:
        for q_idx, entry in enumerate(queries, start=1):
            if _is_query_skipped(db_conn, q_idx):
                continue

            query_text = entry.get("question", "")
            state = _resolve_query_state(
                db_conn, q_idx, entry, scoring_list, vector_db.collection, score_mode="gemini"
            )
            if state is None:
                logger.warning(f"[STAGE {stage}] SKIPPED query_idx={q_idx}: GT missing.")
                _mark_query_skipped(db_conn, q_idx, "missing_gt_embedding_dual_path")
                continue

            gt_id_norm, gt_emb, target_ids = state
            if not target_ids:
                continue

            query_emb = np.array(embedding_model.get_text_embedding(query_text), dtype=np.float32)
            rag_failed = _check_rag_baseline(vector_db, query_emb, gt_id_norm)

            def _on_batch(doc_ids, contexts, placeholder_rows, _q=q_idx):
                nonlocal request_idx
                _insert_placeholder_rows(db_conn, _q, placeholder_rows)
                prompt = build_relevance_scoring_prompt(query_text, contexts)
                custom_id = f"q{_q}_s{stage:04d}_b{request_idx:08d}_{uuid.uuid4().hex[:8]}"
                _record_batch_request_map(db_conn, custom_id, _q, doc_ids)
                out_f.write(
                    json.dumps(_make_batch_request_line(custom_id, prompt), ensure_ascii=False) + "\n"
                )
                request_idx += 1

            _fetch_docs_in_batches(
                target_ids,
                vector_db.collection,
                gt_emb,
                gt_id_norm,
                rag_failed,
                BATCH_SIZE,
                _on_batch,
            )

    return request_idx


async def _run_staged_live(
    *,
    stage: int,
    total_stages: int,
    scoring_union: set[str],
    per_query_docs: dict[str, list[str]],
    queries: list[dict],
    vector_db,
    embedding_model,
    db_conn,
    run_dir: str,
) -> None:
    """
    Live scoring for one staged step (Ollama or Inference API).

    Each query scores only its OWN new docs for this stage (its ranked slice +
    random slice + GT on stage 1), not the full cross-query union.  The global
    ``scoring_union`` is kept for logging only.
    """
    del run_dir, scoring_union
    if not per_query_docs:
        return

    live_conc = max(1, int(OLLAMA_MAX_CONCURRENT_REQUESTS))
    live_batch_size = max(1, int(OLLAMA_DOCS_PER_REQUEST))
    tag = _live_log_tag()

    for q_idx, entry in enumerate(queries, start=1):
        if _is_query_skipped(db_conn, q_idx):
            continue
        query_text = entry.get("question", "")
        query_doc_list = per_query_docs.get(str(q_idx)) or []
        if not query_doc_list:
            continue
        state = _resolve_query_state(
            db_conn, q_idx, entry, query_doc_list, vector_db.collection, score_mode="live"
        )
        if state is None:
            _mark_query_skipped(db_conn, q_idx, "missing_gt_embedding_dual_path")
            continue
        gt_id_norm, gt_emb, target_ids = state
        if not target_ids:
            continue

        query_emb = np.array(embedding_model.get_text_embedding(query_text), dtype=np.float32)
        rag_failed = _check_rag_baseline(vector_db, query_emb, gt_id_norm)

        def _on_batch_scored(doc_ids: list[str], ints: list[int], _q=q_idx) -> None:
            scores = [float(v) / 10.0 for v in ints]
            _update_llm_scores_for_docs(db_conn, _q, doc_ids, scores)
            db_conn.commit()

        # Stream: fetch one Chroma chunk → score it → release → next chunk.
        # Peak memory = CHROMA_FETCH_CHUNK doc texts at a time, not all target_ids.
        for chunk_start in range(0, len(target_ids), CHROMA_FETCH_CHUNK):
            chunk_ids = target_ids[chunk_start: chunk_start + CHROMA_FETCH_CHUNK]
            chunk_batches: list[tuple[list[str], list[str]]] = []

            def _on_batch(doc_ids, contexts, placeholder_rows, _q=q_idx):
                _insert_placeholder_rows(db_conn, _q, placeholder_rows)
                chunk_batches.append((doc_ids[:], contexts[:]))

            _fetch_docs_in_batches(
                chunk_ids,
                vector_db.collection,
                gt_emb,
                gt_id_norm,
                rag_failed,
                live_batch_size,
                _on_batch,
            )

            if chunk_batches:
                try:
                    await _score_batches_parallel_for_query(
                        q_idx=q_idx,
                        query_text=query_text,
                        batches_to_score=chunk_batches,
                        concurrency=live_conc,
                        on_batch_scored=_on_batch_scored,
                    )
                except Exception as exc:
                    _record_fatal_llm_error(db_conn, q_idx, exc)
                    raise
                finally:
                    chunk_batches.clear()

    logger.info(f"[STAGE {stage}/{total_stages}] {tag} scoring complete")


async def _run_staged_gemini(
    *,
    stage: int,
    total_stages: int,
    scoring_union: set[str],
    queries: list[dict],
    vector_db,
    embedding_model,
    db_conn,
    db_path: str,
    run_dir: str,
) -> None:
    """
    Gemini Batch scoring for one staged step.

    Steps: build ``stage_NNNN.jsonl`` → wait for slot → submit → harvest into SQLite.
    """
    if not scoring_union:
        return

    # --- Step 1: build stage JSONL ---
    jsonl_path = os.path.join(run_dir, f"stage_{stage:04d}.jsonl")
    n_requests = _produce_stage_gemini_jsonl(
        stage=stage,
        scoring_union=scoring_union,
        queries=queries,
        vector_db=vector_db,
        embedding_model=embedding_model,
        db_conn=db_conn,
        jsonl_path=jsonl_path,
    )
    db_conn.commit()

    if n_requests == 0:
        logger.info(f"[STAGE {stage}/{total_stages}] No new Gemini requests — skipping submit.")
        try:
            if os.path.exists(jsonl_path):
                os.remove(jsonl_path)
        except OSError:
            pass
        return

    # --- Step 2: wait for slot and submit Batch job ---
    client = get_gemini_client(GEMINI_API_KEY)
    logger.info(f"[STAGE {stage}/{total_stages}] Submitting {n_requests} requests...")
    while True:
        active = _count_active_jobs(client)
        if active < MAX_CONCURRENT_JOBS:
            break
        logger.info(
            f"[STAGE {stage}/{total_stages}] Waiting for batch slot "
            f"({active}/{MAX_CONCURRENT_JOBS} active)..."
        )
        time.sleep(POLL_INTERVAL_SECONDS)

    job_id = _submit_with_backoff(client, jsonl_path)
    _set_experiment_meta(db_conn, f"stage_{stage:04d}_job_id", job_id)
    db_conn.commit()
    logger.info(f"[STAGE {stage}/{total_stages}] Submitted job_id={job_id!r}")

    # --- Step 3: download output and sync scores ---
    _poll_and_harvest_jobs(
        client=client,
        job_ids=[job_id],
        run_dir=run_dir,
        db_path=db_path,
    )


async def run_global_correlation_staged_async(
    vector_db,
    embedding_model,
    num_queries: int = 200,
    stride: int | None = None,
    max_ranked: int | None = None,
    max_ranked_query: int | None = None,
    output_dir: str = "results/global_exp",
    per_job_timeout_s: int | None = None,
) -> None:
    """
    Staged global correlation experiment.

    D = all GT docs + top-max_ranked GT-embedding neighbours per query
                    + top-max_ranked_query query-embedding neighbours per query.
    Each stage pulls stride*n_queries docs from the global ranked_pool and scores
    every query against every new doc (full cross-product).

    Run steps
    ---------
    1. Create/load manifest (global ranked_pool interleaved across all queries).
    2. Per stage: build scoring_union, advance cursor, score (live or Gemini batch).
    3. Mark stage harvested; at end — stats and plots.
    """
    del per_job_timeout_s  # reserved for future per-job harvest timeout wiring

    # --- Step 0: resolve parameters ---
    stride = int(stride if stride is not None else STAGING_STRIDE)
    max_ranked = int(max_ranked if max_ranked is not None else STAGING_MAX_RANKED_PER_GT)
    max_ranked_query = int(max_ranked_query if max_ranked_query is not None else STAGING_MAX_RANKED_PER_QUERY)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    queries = load_qa_queries(num_queries)
    n_queries = len(queries)
    if n_queries == 0:
        logger.error("[STAGED] No queries loaded — aborting.")
        return

    provider = _configured_correlation_provider()

    # --- Step 1: set up run + manifest ---
    max_ranked_combined = max_ranked + max_ranked_query
    total_stages = max(math.ceil(max_ranked_combined / stride) if max_ranked_combined > 0 else 0, 1)

    experiment_params: dict = {
        "num_queries_requested": num_queries,
        "actual_num_queries": n_queries,
        "stage_mode": "additive_pool",
        "staging_stride": stride,
        "max_ranked_per_gt": max_ranked,
        "max_ranked_per_query": max_ranked_query,
        "staging_spec_hash": _staging_spec_hash(
            n_queries, stride, max_ranked, max_ranked_query
        ),
        "staging_total_stages_expected": total_stages,
        "model": (
            CORRELATION_GEMINI_BATCH_MODEL
            if _is_gemini_batch_provider(provider)
            else correlation_live_model_name()
        ),
        "llm_backend": _normalize_live_provider(provider)
        if _is_live_correlation_provider(provider)
        else provider,
    }

    run_dir = _setup_or_resume_staged_run(output_dir, experiment_params)
    db_path = os.path.join(run_dir, "experiment_results.db")
    db_conn = _init_sqlite(db_path)

    queries = _save_or_load_queries(run_dir, queries)
    n_queries = len(queries)

    manifest = _load_or_build_manifest(
        run_dir,
        queries,
        vector_db,
        embedding_model,
        stride=stride,
        max_ranked_gt=max_ranked,
        max_ranked_query=max_ranked_query,
    )

    # Recompute total_stages from actual pool size (dedup reduces theoretical max)
    batch = stride * n_queries
    ranked_pool_size = len(manifest.get("ranked_pool", []))
    total_stages = max(math.ceil(ranked_pool_size / batch) if ranked_pool_size > 0 else 0, 1)
    manifest["staging_total_stages_expected"] = total_stages
    _save_manifest(manifest, run_dir)

    last_completed = int(manifest.get("staging_last_completed_stage", 0))
    logger.info(
        f"[STAGED] N={n_queries}, stride={stride}, "
        f"ranked_pool={ranked_pool_size}, batch_per_stage={batch}, "
        f"total_stages={total_stages}, resume_from={last_completed + 1}, provider={provider}"
    )

    # --- Step 2: stage loop ---
    for stage in range(last_completed + 1, total_stages + 1):
        if _is_staged_run_complete(manifest):
            logger.info("[STAGED] All cursors exhausted — run is complete.")
            break

        stage_entry = manifest.get("stages", {}).get(str(stage), {})

        # 2a. resume pending stage, or build new scoring set
        if stage_entry.get("status") == "pending":
            scoring_union = set(_normalize_id(x) for x in (stage_entry.get("scoring_union") or []))
            # Rebuild per_query_docs scoped to only this stage's docs — avoids
            # loading the full results table (grows to ~8 M rows by stage 20).
            per_query_seen = _get_per_query_seen_for_docs(db_conn, scoring_union)
            new_docs = sorted(scoring_union)
            per_query_docs = {
                q_idx_str: [d for d in new_docs if d not in per_query_seen.get(q_idx_str, set())]
                for q_idx_str in manifest.get("gt_ids", {})
            }
            per_query_docs = {k: v for k, v in per_query_docs.items() if v}
            logger.info(
                f"[STAGE {stage}/{total_stages}] Resuming pending stage "
                f"({len(scoring_union)} docs × {len(per_query_docs)} queries)."
            )
            if not scoring_union:
                manifest["stages"][str(stage)]["status"] = "harvested"
                manifest["staging_last_completed_stage"] = stage
                _save_manifest(manifest, run_dir)
                _set_experiment_meta(db_conn, "staging_last_completed_stage", str(stage))
                db_conn.commit()
                continue
        else:
            # Fresh stage: new docs from the pool are guaranteed unseen.
            # Pass empty seen-set; _resolve_query_state handles per-query
            # crash recovery via its own targeted DB lookup.
            scoring_union, per_query_docs, new_cursors = _build_stage_scoring_set(
                manifest, stage, {}, stride
            )

            manifest["ranked_cursor"] = new_cursors["ranked"]

            if not scoring_union:
                logger.info(
                    f"[STAGE {stage}/{total_stages}] Empty scoring set — cursors advanced."
                )
                manifest["stages"][str(stage)] = {
                    "scoring_union": [],
                    "status": "harvested",
                }
                manifest["staging_last_completed_stage"] = stage
                _save_manifest(manifest, run_dir)
                _set_experiment_meta(db_conn, "staging_last_completed_stage", str(stage))
                db_conn.commit()
                continue

            manifest["stages"][str(stage)] = {
                "scoring_union": sorted(scoring_union),
                "status": "pending",
            }
            _save_manifest(manifest, run_dir)
            logger.info(
                f"[STAGE {stage}/{total_stages}] New docs: {len(scoring_union)} "
                f"→ ~{len(scoring_union) * len(manifest.get('gt_ids', {}))} pairs."
            )

        # 2b. score this stage (provider-specific)
        if _is_gemini_batch_provider(provider):
            await _run_staged_gemini(
                stage=stage,
                total_stages=total_stages,
                scoring_union=scoring_union,
                queries=queries,
                vector_db=vector_db,
                embedding_model=embedding_model,
                db_conn=db_conn,
                db_path=db_path,
                run_dir=run_dir,
            )
        elif _is_live_correlation_provider(provider):
            await _run_staged_live(
                stage=stage,
                total_stages=total_stages,
                scoring_union=scoring_union,
                per_query_docs=per_query_docs,
                queries=queries,
                vector_db=vector_db,
                embedding_model=embedding_model,
                db_conn=db_conn,
                run_dir=run_dir,
            )
        else:
            raise RuntimeError(
                f"Unknown CORRELATION_LLM_PROVIDER={provider!r} "
                f"(expected gemini, ollama, or inference_api)"
            )

        stage_entry = manifest["stages"][str(stage)]
        stage_entry["status"] = "harvested"
        stage_entry.pop("scoring_union", None)  # no longer needed; frees manifest RAM
        manifest["staging_last_completed_stage"] = stage
        _save_manifest(manifest, run_dir)
        _set_experiment_meta(db_conn, "staging_last_completed_stage", str(stage))
        db_conn.commit()

    # --- Step 3: finalize staged experiment ---
    # Release the ranked_pool list (no longer needed after all stages complete)
    manifest.pop("ranked_pool", None)
    _save_manifest(manifest, run_dir)

    _set_experiment_meta(db_conn, "stages_complete", "1")
    _set_main_loop_completed(db_conn)
    _calculate_final_stats_and_plot_sqlite(db_conn, run_dir)
    db_conn.close()
    logger.info(f"[STAGED] Experiment complete. Output: {run_dir}")
