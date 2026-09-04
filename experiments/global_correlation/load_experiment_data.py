"""
Shared data loader for group geometry experiments.

Guarantees the same 196 queries and pool as the original group analysis:
  - Query list from master_summary.csv (196 queries, original experiment)
  - Pool docs from expanded_scoring_queue.parquet (5,026 docs)
  - Missing GT texts fetched from scoring_queue.parquet
  - Embeddings via sentence-transformers/all-mpnet-base-v2
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "utility"))
from similarity_calculator import cosine_distance  # noqa: F401 — re-exported for experiment scripts

RUN_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "results/cloud_runs/staged_run_2026-08-03_08-32-45"
)
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def extract_doc_text(s: str) -> str:
    try:
        d = json.loads(s)
        raw = d.get("contexts") or d.get("CONTEXTS") or ""
        return " ".join(str(c) for c in raw if c) if isinstance(raw, list) else str(raw)
    except Exception:
        return str(s)


def load_data(verbose: bool = True) -> dict:
    """
    Returns a dict with:
      queries_meta  — DataFrame(query_idx, query_text, gt_id, rag_failed), 196 rows
      pool_docs     — DataFrame(doc_id, doc_text), 5 026+ docs (incl. missing GTs)
      scores_df     — DataFrame(query_idx, doc_id, llm_score) from experiment_results.db
      query_vecs    — np.ndarray (196, 768)
      doc_vecs      — np.ndarray (n_pool, 768)
      query_idx_to_vec — dict {query_idx: vec}
      doc_id_to_idx    — dict {doc_id: int}
      all_doc_ids      — list[str], ordered, len = n_pool
      model            — SentenceTransformer instance
    """
    log = print if verbose else (lambda *a, **k: None)

    # ------------------------------------------------------------------
    # 1. 196 canonical query_idxs from the original group analysis
    # ------------------------------------------------------------------
    master = pd.read_csv(RUN_DIR / "analysis/csv/group_analysis/master_summary.csv")
    canonical_q_idxs = set(master["query_idx"].tolist())
    log(f"Canonical queries (original analysis): {len(canonical_q_idxs)}")

    # ------------------------------------------------------------------
    # 2. Pool docs from expanded_scoring_queue (5,026 unique docs)
    # ------------------------------------------------------------------
    log("Loading expanded_scoring_queue.parquet …")
    esq = pd.read_parquet(
        RUN_DIR / "expanded_scoring_queue.parquet",
        columns=["query_idx", "query_text", "gt_id", "doc_id", "doc_text", "is_gt", "rag_failed"],
    )
    esq["doc_id"] = esq["doc_id"].astype(str)
    esq["gt_id"]  = esq["gt_id"].astype(str)
    log(f"  {len(esq):,} rows, {esq['doc_id'].nunique():,} unique docs")

    # Filter to canonical queries only (exclude any extra queries not in original)
    esq = esq[esq["query_idx"].isin(canonical_q_idxs)].copy()

    queries_meta = (
        esq[["query_idx", "query_text", "gt_id", "rag_failed"]]
        .drop_duplicates("query_idx")
        .sort_values("query_idx")
        .reset_index(drop=True)
    )
    log(f"Queries after canonical filter: {len(queries_meta)}")

    pool_docs = esq[["doc_id", "doc_text"]].drop_duplicates("doc_id").reset_index(drop=True)
    pool_doc_ids = set(pool_docs["doc_id"].tolist())

    # ------------------------------------------------------------------
    # 3. Add missing GTs from scoring_queue.parquet
    # ------------------------------------------------------------------
    gt_ids_all = set(queries_meta["gt_id"].tolist())
    missing_gts = gt_ids_all - pool_doc_ids

    if missing_gts:
        log(f"GT docs missing from pool ({len(missing_gts)}): {sorted(missing_gts)}")
        log("  Fetching from scoring_queue.parquet …")
        sq = pd.read_parquet(
            RUN_DIR / "scoring_queue.parquet",
            columns=["doc_id", "doc_text"],
        )
        sq["doc_id"] = sq["doc_id"].astype(str)
        gt_rows = sq[sq["doc_id"].isin(missing_gts)].drop_duplicates("doc_id")
        pool_docs = pd.concat([pool_docs, gt_rows], ignore_index=True).drop_duplicates("doc_id")
        log(f"  Pool docs after adding missing GTs: {len(pool_docs)}")
    else:
        log("All GTs present in pool.")

    log(f"Final pool size: {len(pool_docs)}")

    # ------------------------------------------------------------------
    # 4. LLM scores
    # ------------------------------------------------------------------
    log("Loading experiment_results.db …")
    conn = sqlite3.connect(RUN_DIR / "experiment_results.db")
    scores_df = pd.read_sql("SELECT query_idx, doc_id, llm_score FROM results", conn)
    conn.close()
    scores_df["doc_id"] = scores_df["doc_id"].astype(str)
    scores_df = scores_df[scores_df["query_idx"].isin(canonical_q_idxs)].copy()
    log(f"  Scored pairs (canonical queries): {len(scores_df):,}")

    # ------------------------------------------------------------------
    # 5. Embed
    # ------------------------------------------------------------------
    log(f"\nLoading model: {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME)

    log("Embedding queries …")
    query_vecs = model.encode(
        queries_meta["query_text"].tolist(),
        batch_size=64, normalize_embeddings=True, show_progress_bar=verbose,
    )

    log("Embedding pool documents …")
    doc_texts = [extract_doc_text(t) for t in pool_docs["doc_text"].tolist()]
    doc_vecs = model.encode(
        doc_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=verbose,
    )

    query_idx_to_vec = dict(zip(queries_meta["query_idx"].tolist(), query_vecs))
    all_doc_ids = pool_docs["doc_id"].tolist()
    doc_id_to_idx = {d: i for i, d in enumerate(all_doc_ids)}

    log(f"\nEmbedded {len(query_vecs)} queries, {len(doc_vecs)} pool docs.")

    return dict(
        queries_meta     = queries_meta,
        pool_docs        = pool_docs,
        scores_df        = scores_df,
        query_vecs       = query_vecs,
        doc_vecs         = doc_vecs,
        query_idx_to_vec = query_idx_to_vec,
        doc_id_to_idx    = doc_id_to_idx,
        all_doc_ids      = all_doc_ids,
        model            = model,
    )
