"""
Global correlation experiment: LLM relevance (batched) vs cosine distance to GT embedding.

Layout (logical):
  constants & IDs / text pruning -> GT retrieval -> Gemini scoring -> SQLite lifecycle
  -> run-dir / resume -> analytics -> vector pool I/O -> main loop
"""

import asyncio
import json
import os
import re
import sqlite3
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from google import genai
from google.genai import types
from scipy import stats

from configurations.config import GEMINI_API_KEY
from scripts.qa_data_set_loader import load_qa_queries
from utility.logger import logger
from vector_db.trilateration_retriever import cosine_distance

# --- API-friendly pacing: fewer concurrent calls + longer backoff + delay between batches (TPM) ---
MAX_CONCURRENT_REQUESTS = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
CHROMA_FETCH_CHUNK = 400
TASK_SUBMIT_INTERVAL_S = 0.5
RATE_LIMIT_BACKOFF_BASE_S = 20.0


# ---------------------------------------------------------------------------
# Utility: IDs & input pruning
# ---------------------------------------------------------------------------


def _normalize_id(x) -> str:
    """Type-agnostic id for storage and comparison."""
    return str(x).strip().lower()


def _extract_contexts_text(document_text: str) -> str:
    """
    Strict pruning: extract ONLY text from JSON `contexts` array.
    On failure, return raw string.
    """
    try:
        obj = json.loads(document_text)
    except Exception:
        return document_text

    if not isinstance(obj, dict):
        return document_text

    contexts = obj.get("contexts")
    if contexts is None:
        return document_text

    parts: list[str] = []
    if isinstance(contexts, list):
        for c in contexts:
            if c is None:
                continue
            if isinstance(c, str):
                parts.append(c.strip())
            elif isinstance(c, dict):
                if "text" in c and isinstance(c["text"], str):
                    parts.append(c["text"].strip())
                elif "content" in c and isinstance(c["content"], str):
                    parts.append(c["content"].strip())
                else:
                    for v in c.values():
                        if isinstance(v, str) and v.strip():
                            parts.append(v.strip())
            else:
                s = str(c).strip()
                if s:
                    parts.append(s)
    elif isinstance(contexts, str):
        parts.append(contexts.strip())

    pruned = "\n".join([p for p in parts if p])
    return pruned if pruned else document_text


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
    m = re.search(r"\[[\s\S]*?\]", t)
    candidate = m.group(0).strip() if m else t
    try:
        arr = json.loads(candidate)
        if isinstance(arr, list):
            out: list[int] = []
            for x in arr:
                try:
                    v = int(x)
                except Exception:
                    v = 0
                out.append(max(0, min(10, v)))
            if len(out) < expected_n:
                out.extend([0] * (expected_n - len(out)))
            return out[:expected_n]
    except Exception:
        pass
    nums = re.findall(r"\b(10|[1-9])\b", t)
    out = [int(n) for n in nums[:expected_n]]
    if len(out) < expected_n:
        out.extend([0] * (expected_n - len(out)))
    return out


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
- Output must be a valid JSON array: [8, 2, 10, 4, ...]
- Each value must be an integer in 1..10 (0 is not allowed)
- Return exactly {len(doc_contexts)} integers, in the same order as the ITEMS.
        
---
QUERY:
{query}
        
        {docs_block}
        
Return ONLY the JSON array of {len(doc_contexts)} integers.
    """


async def _async_gemini_score_batch(
        client,
        query: str,
        doc_contexts: list[str],
        max_retries: int = 8,
) -> list[float]:
    """
    Single implementation: system instruction + JSON batch + poisoned-batch per-doc fallback.
    """
    system_msg = (
        "You are an automated data-matching utility. Your only function is to compare text segments "
        "and output numerical match scores. You do not provide advice or interpret content. You are a script."
    )
    config = types.GenerateContentConfig(
        system_instruction=system_msg,
        temperature=0.0,
        response_mime_type="application/json",
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ],
    )

    async with semaphore:
        prompt = _get_batch_scoring_prompt(query, doc_contexts)
        for attempt in range(max_retries):
            try:
                response = await client.aio.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config=config,
                )
                if _is_safety_or_filter_block(response):
                    logger.warning(
                        f"Safety/Filter Block: batch blocked → 0.0 × {len(doc_contexts)} docs"
                    )
                    return [0.0] * len(doc_contexts)
                if response is None or not getattr(response, "text", None):
                    logger.warning(
                        f"Safety/Filter Block: empty response → 0.0 × {len(doc_contexts)} docs"
                    )
                    return [0.0] * len(doc_contexts)

                ints = _parse_batch_scores(response.text, len(doc_contexts))

                if all(v == 0 for v in ints) and len(doc_contexts) > 1:
                    logger.warning(
                        f"Batch poisoned (all zeros). Per-doc fallback for {len(doc_contexts)} docs."
                    )
                    individual_tasks = [
                        _async_gemini_score_batch(client, query, [doc], max_retries=2)
                        for doc in doc_contexts
                    ]
                    indiv_results = await asyncio.gather(*individual_tasks)
                    return [res[0] for res in indiv_results]

                return [float(v) / 10.0 for v in ints]

            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "resource_exhausted" in err_msg:
                    wait_time = min(RATE_LIMIT_BACKOFF_BASE_S * (2 ** attempt), 300.0)
                    logger.warning(
                        f"Gemini 429; backoff {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                elif "none" in err_msg and "strip" in err_msg:
                    logger.warning(f"Safety/Filter Block: empty text error → batch 0.0 ({e})")
                    return [0.0] * len(doc_contexts)
                elif attempt == max_retries - 1:
                    logger.warning(f"Safety/Filter Block: giving up → batch 0.0 ({e})")
                    return [0.0] * len(doc_contexts)
                else:
                    await asyncio.sleep(3)
        return [0.0] * len(doc_contexts)


async def _gather_scores_batched_smooth_paced(
        client,
        query_text: str,
        doc_contexts: list[str],
        batch_size: int = 20,
) -> list[float]:
    """
    Run batch API calls sequentially with TASK_SUBMIT_INTERVAL_S between starts
    (lowers burst TPM vs parallel create_task + gather).
    """
    batches = [doc_contexts[i: i + batch_size] for i in range(0, len(doc_contexts), batch_size)]
    flat: list[float] = []
    for i, batch in enumerate(batches):
        if i > 0:
            await asyncio.sleep(TASK_SUBMIT_INTERVAL_S)
        scores = await _async_gemini_score_batch(client, query_text, batch)
        flat.extend(list(scores))
    if len(flat) < len(doc_contexts):
        flat.extend([0.0] * (len(doc_contexts) - len(flat)))
    return flat[: len(doc_contexts)]


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
# Main experiment loop
# ---------------------------------------------------------------------------


async def run_global_correlation_experiment_async(
        vector_db,
        embedding_model,
        num_queries=200,
        k=50,
        output_dir="results/global_exp",
):
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

    for q_idx, entry in enumerate(queries, start=1):
        query_text = entry.get("question")
        gt_id_norm = _gt_id_str(entry)
        gt_pmid = _gt_pmid_int_for_chroma(entry)

        if _is_query_skipped(db_conn, q_idx):
            logger.info(f"Query {q_idx}/{actual_num_queries}: skipped (SQLite).")
            continue

        stored_docs = _count_distinct_docs_for_query(db_conn, q_idx)
        if stored_docs >= global_pool_size:
            logger.info(
                f"Query {q_idx}/{actual_num_queries}: complete ({stored_docs}/{global_pool_size})."
            )
            query_df = pd.read_sql(
                "SELECT * FROM results WHERE query_idx = ?",
                db_conn,
                params=(q_idx,),
            )
            _log_failure_analysis(run_dir, q_idx, query_text, query_df)
            _generate_query_scatterplot(q_idx, query_df, run_dir)
            continue

        processed_ids = _get_processed_ids_from_db(db_conn, q_idx)
        remaining_ids = [d_id for d_id in sorted_doc_ids if _normalize_id(d_id) not in processed_ids]

        if not remaining_ids:
            if stored_docs < global_pool_size:
                logger.warning(
                    f"Query {q_idx}: no remaining pool ids ({stored_docs}/{global_pool_size})."
                )
            query_df = pd.read_sql(
                "SELECT * FROM results WHERE query_idx = ?",
                db_conn,
                params=(q_idx,),
            )
            _log_failure_analysis(run_dir, q_idx, query_text, query_df)
            _generate_query_scatterplot(q_idx, query_df, run_dir)
            continue

        gt_emb = _get_gt_embedding_dual_path(vector_db.collection, gt_id_norm, gt_pmid)
        if gt_emb is None:
            logger.warning(
                f"SKIPPED_PERMANENTLY query_idx={q_idx}: GT missing (id={gt_id_norm!r}, pmid={gt_pmid!r})."
            )
            _mark_query_skipped(db_conn, q_idx, "missing_gt_embedding_dual_path")
            continue

        logger.info(
            f"Query {q_idx}/{actual_num_queries}: {len(remaining_ids)} docs left "
            f"(pool {global_pool_size}, chunk {CHROMA_FETCH_CHUNK})"
        )
        query_emb = np.array(embedding_model.get_text_embedding(query_text), dtype=np.float32)
        rag_failed = _check_rag_baseline(vector_db, query_emb, gt_id_norm)

        for chunk_start in range(0, len(remaining_ids), CHROMA_FETCH_CHUNK):
            chunk = remaining_ids[chunk_start: chunk_start + CHROMA_FETCH_CHUNK]
            payload = _chroma_batch_get_documents_and_embeddings(vector_db.collection, chunk)
            documents: list[str] = []
            embs: list[np.ndarray] = []
            chunk_keys: list[str] = []
            for d_id in chunk:
                nid = _normalize_id(d_id)
                if nid not in payload:
                    logger.warning(f"Chroma missing id={nid!r}")
                    continue
                text, emb = payload[nid]
                text = _extract_contexts_text(text)
                chunk_keys.append(nid)
                documents.append(text)
                embs.append(emb)

            if not documents:
                continue

            scores = await _gather_scores_batched_smooth_paced(
                client, query_text, documents, batch_size=20
            )

            query_rows = []
            for nid, score, emb in zip(chunk_keys, scores, embs):
                dist = float(cosine_distance(gt_emb, emb))
                is_gt_check = 1 if (nid == gt_id_norm or dist < 1e-5) else 0
                query_rows.append(
                    {
                        "query_idx": q_idx,
                        "doc_id": nid,
                        "llm_score": score,
                        "dist_to_gt": dist,
                        "is_gt": is_gt_check,
                        "rag_failed": 1 if rag_failed else 0,
                    }
                )
            _save_to_db(db_conn, query_rows)

        query_df = pd.read_sql(
            "SELECT * FROM results WHERE query_idx = ?",
            db_conn,
            params=(q_idx,),
        )
        _log_failure_analysis(run_dir, q_idx, query_text, query_df)
        _generate_query_scatterplot(q_idx, query_df, run_dir)

    _set_main_loop_completed(db_conn)
    logger.info("Main loop done. Writing global Spearman summary.")
    _calculate_final_stats_and_plot_sqlite(db_conn, run_dir)
    db_conn.close()
