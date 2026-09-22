"""
Shared data loader for group geometry experiments.

Guarantees the same 196 queries and pool as the original group analysis:
  - Query list from master_summary.csv (196 queries, original experiment)
  - Pool docs from expanded_scoring_queue.parquet (5,026 docs)
  - Missing GT texts fetched from scoring_queue.parquet
  - Embeddings via embedding_factory (cached in <run_dir>/embeddings/)
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "utility"))
from similarity_calculator import cosine_distance  # noqa: F401 — re-exported for callers

from embedding_factory import (  # noqa: F401 — MODEL_NAME, extract_doc_text re-exported for callers
    MODEL_NAME, extract_doc_text,
    load_model, encode_texts,
    save_embeddings, load_embeddings,
)

RUN_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "results/cloud_runs/staged_run_2026-08-03_08-32-45"
)


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
    # 3. Add missing GTs — three fallback sources in order
    # ------------------------------------------------------------------
    gt_ids_all  = set(queries_meta["gt_id"].tolist())
    missing_gts = gt_ids_all - pool_doc_ids

    if missing_gts:
        log(f"GT docs missing from expanded_scoring_queue ({len(missing_gts)}): {sorted(missing_gts)}")

        # Fallback 1: scoring_queue.parquet (original pre-expansion queue)
        log("  [Fallback 1] Searching scoring_queue.parquet …")
        sq_path = RUN_DIR / "scoring_queue.parquet"
        if sq_path.exists():
            sq = pd.read_parquet(sq_path, columns=["doc_id", "doc_text"])
            sq["doc_id"] = sq["doc_id"].astype(str)
            found1 = sq[sq["doc_id"].isin(missing_gts)].drop_duplicates("doc_id")
            if not found1.empty:
                pool_docs   = pd.concat([pool_docs, found1], ignore_index=True).drop_duplicates("doc_id")
                missing_gts -= set(found1["doc_id"].tolist())
                log(f"    Recovered {len(found1)} GT(s). Still missing: {len(missing_gts)}")
            else:
                log("    Not found in scoring_queue.parquet.")
        else:
            log("    scoring_queue.parquet not found — skipping.")

        # Fallback 2: experiment_results.db — any table with (doc_id, doc_text) columns
        if missing_gts:
            log("  [Fallback 2] Searching experiment_results.db …")
            conn2  = sqlite3.connect(RUN_DIR / "experiment_results.db")
            tables = [r[0] for r in conn2.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            log(f"    DB tables: {tables}")
            found2 = pd.DataFrame()
            for tbl in tables:
                cols = [r[1] for r in conn2.execute(f"PRAGMA table_info({tbl})").fetchall()]
                if "doc_id" in cols and "doc_text" in cols:
                    placeholders = ",".join("?" * len(missing_gts))
                    rows = pd.read_sql(
                        f"SELECT doc_id, doc_text FROM {tbl} WHERE doc_id IN ({placeholders})",
                        conn2, params=list(missing_gts),
                    )
                    rows["doc_id"] = rows["doc_id"].astype(str)
                    found2 = pd.concat([found2, rows], ignore_index=True).drop_duplicates("doc_id")
            conn2.close()
            if not found2.empty:
                pool_docs   = pd.concat([pool_docs, found2], ignore_index=True).drop_duplicates("doc_id")
                missing_gts -= set(found2["doc_id"].tolist())
                log(f"    Recovered {len(found2)} GT(s) from DB. Still missing: {len(missing_gts)}")
            else:
                log("    No table with (doc_id, doc_text) columns found in DB.")

        # Final guard — abort rather than silently skip queries
        if missing_gts:
            raise ValueError(
                f"FATAL: {len(missing_gts)} GT doc(s) could not be found in any source "
                f"(expanded_scoring_queue, scoring_queue, experiment_results.db).\n"
                f"Missing GT ids: {sorted(missing_gts)}\n"
                f"Cannot proceed — ranks for queries with these GTs would be undefined."
            )

        log(f"  All GTs recovered. Pool size: {len(pool_docs)}")
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
    # 5. Embed — load from cache or compute and save
    # ------------------------------------------------------------------
    query_ids   = queries_meta["query_idx"].tolist()
    all_doc_ids = pool_docs["doc_id"].tolist()

    cached = load_embeddings(RUN_DIR, query_ids, all_doc_ids)
    if cached is not None:
        query_vecs, doc_vecs = cached
        log(f"Embeddings loaded from cache ({RUN_DIR / 'embeddings'}).")
        model = None
    else:
        log(f"\nNo embedding cache found — computing with {MODEL_NAME} …")
        model      = load_model()
        query_vecs = encode_texts(queries_meta["query_text"].tolist(), model,
                                  show_progress_bar=verbose)
        doc_texts  = [extract_doc_text(t) for t in pool_docs["doc_text"].tolist()]
        doc_vecs   = encode_texts(doc_texts, model, show_progress_bar=verbose)
        save_embeddings(RUN_DIR, query_vecs, doc_vecs, query_ids, all_doc_ids)
        log(f"Embeddings cached → {RUN_DIR / 'embeddings'}")

    query_idx_to_vec = dict(zip(query_ids, query_vecs))
    doc_id_to_idx    = {d: i for i, d in enumerate(all_doc_ids)}
    log(f"Vectors ready: {len(query_vecs)} queries, {len(doc_vecs)} pool docs.")

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
