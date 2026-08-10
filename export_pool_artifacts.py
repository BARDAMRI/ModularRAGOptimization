#!/usr/bin/env python3
"""
export_pool_artifacts.py — Safety-net artifact exporter
========================================================
Reads the Chroma DB and scoring_queue.parquet produced by Phase 1, then
exports a self-contained bundle with everything needed to re-run, debug,
or extend the experiment WITHOUT access to Chroma again.

Artifacts saved to <run_dir>/:
    scoring_queue.parquet     — main Phase-2 input (already there)
    pool_embeddings.npy       — float32 [n_unique_docs × dim]
    pool_doc_ids.json         — ordered list of doc IDs (matches .npy rows)
    pool_metadata.parquet     — PMID, title, text_snippet per pool doc
    gt_embeddings.npy         — float32 [n_queries × dim]
    gt_doc_ids.json           — GT doc ID per query (index = query_idx - 1)
    run_metadata.json         — full provenance record

Usage:
    python export_pool_artifacts.py --run-dir results/phase1_queue/staged_run_XXX
    python export_pool_artifacts.py --run-dir ... --chroma-source pubmed_full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configurations.config import HF_MODEL_NAME
from utility.logger import logger


def export_pool_artifacts(
    run_dir: Path,
    chroma_source: str = "pubmed_selected_articles",
) -> None:
    run_dir = run_dir.resolve()
    queue_path = run_dir / "scoring_queue.parquet"

    if not queue_path.exists():
        sys.exit(f"ERROR: scoring_queue.parquet not found in {run_dir}")

    logger.info(f"[EXPORT] Loading queue from {queue_path}")
    df = pd.read_parquet(queue_path)
    unique_doc_ids = df["doc_id"].unique().tolist()
    n_unique       = len(unique_doc_ids)
    n_queries      = df["query_idx"].nunique()

    logger.info(
        f"[EXPORT] Queue: {len(df):,} rows | "
        f"{n_queries} queries | "
        f"{n_unique:,} unique docs"
    )

    # --- Open Chroma ---
    logger.info("[EXPORT] Opening ChromaVectorDB...")
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from vector_db.chroma_vector_db import ChromaVectorDB
    from utility.distance_metrics import DistanceMetric

    emb_model = HuggingFaceEmbedding(model_name=HF_MODEL_NAME)
    vdb = ChromaVectorDB(
        source_path=chroma_source,
        embedding_model=emb_model,
        distance_metric=DistanceMetric.COSINE,
    )
    collection   = vdb.collection
    chroma_total = collection.count()
    logger.info(f"[EXPORT] Chroma collection: {chroma_total:,} docs")

    # --- Fetch pool doc embeddings + metadata ---
    logger.info("[EXPORT] Fetching pool doc embeddings and metadata...")

    CHUNK   = 400
    all_ids:    list[str]        = []
    all_embs:   list[np.ndarray] = []
    meta_rows:  list[dict]       = []
    t0 = time.time()

    for chunk_start in range(0, n_unique, CHUNK):
        chunk_ids = unique_doc_ids[chunk_start: chunk_start + CHUNK]
        raw = collection.get(
            ids=chunk_ids,
            include=["embeddings", "documents", "metadatas"],
        )
        ids_out  = raw.get("ids") or []
        embs_raw = raw.get("embeddings")
        docs_out = raw.get("documents") or []
        metas    = raw.get("metadatas") or []
        embs_out = embs_raw if embs_raw is not None else []

        for did, emb, doc, meta in zip(ids_out, embs_out, docs_out, metas):
            if emb is None:
                continue
            all_ids.append(did)
            all_embs.append(np.array(emb, dtype=np.float32))

            pmid  = meta.get("PMID") or meta.get("pubid") if meta else None
            title = (meta.get("Title") or meta.get("title") or "") if meta else ""
            meta_rows.append({
                "doc_id":       did,
                "pmid":         pmid,
                "title":        str(title),
                "text_snippet": str(doc)[:500] if doc else "",
            })

        done = chunk_start + len(chunk_ids)
        if done % 5000 < CHUNK or done >= n_unique:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            logger.info(
                f"[EXPORT]   {done:,}/{n_unique:,} docs "
                f"({100*done/n_unique:.1f}%) | {rate:.0f} docs/s"
            )

    # --- Build embedding matrix ---
    if all_embs:
        emb_matrix = np.stack(all_embs).astype(np.float32)
    else:
        emb_matrix = np.empty((0, 768), dtype=np.float32)
    dim = emb_matrix.shape[1] if emb_matrix.ndim > 1 and emb_matrix.shape[0] > 0 else None

    logger.info(
        f"[EXPORT] Pool embedding matrix: {emb_matrix.shape} | "
        f"{emb_matrix.nbytes/1e6:.1f} MB | "
        f"missing: {n_unique - len(all_ids)}"
    )

    # --- Fetch GT embeddings per query ---
    logger.info("[EXPORT] Fetching GT embeddings per query...")
    from experiments.global_correlation_experiment import (
        _gt_id_str,
        _gt_pmid_int_for_chroma,
        _get_gt_embedding_dual_path,
    )
    from scripts.qa_data_set_loader import load_qa_queries

    queries_path = run_dir / "queries.json"
    if queries_path.exists():
        queries = json.loads(queries_path.read_text())
        logger.info(f"[EXPORT] Loaded {len(queries)} queries from {queries_path}")
    else:
        queries = load_qa_queries(df["query_idx"].max())
        logger.info(f"[EXPORT] Loaded {len(queries)} queries from dataset")

    gt_embs:    list[np.ndarray] = []
    gt_doc_ids: list[str]        = []
    gt_found = 0

    for entry in queries:
        gt_id   = _gt_id_str(entry)
        gt_pmid = _gt_pmid_int_for_chroma(entry)
        gt_emb  = _get_gt_embedding_dual_path(collection, gt_id, gt_pmid)
        if gt_emb is not None:
            gt_embs.append(gt_emb.astype(np.float32))
            gt_found += 1
        else:
            zero = np.zeros(dim if dim else 768, dtype=np.float32)
            gt_embs.append(zero)
        gt_doc_ids.append(gt_id)

    if gt_embs:
        gt_matrix = np.stack(gt_embs).astype(np.float32)
    else:
        gt_matrix = np.empty((0, dim or 768), dtype=np.float32)

    logger.info(
        f"[EXPORT] GT embedding matrix: {gt_matrix.shape} | "
        f"found {gt_found}/{len(queries)} embeddings"
    )

    # --- Save all artifacts ---
    logger.info("[EXPORT] Saving artifacts...")

    np.save(run_dir / "pool_embeddings.npy", emb_matrix)
    logger.info(f"[EXPORT] pool_embeddings.npy  {emb_matrix.shape}  {emb_matrix.nbytes/1e6:.0f} MB")

    (run_dir / "pool_doc_ids.json").write_text(json.dumps(all_ids))
    logger.info(f"[EXPORT] pool_doc_ids.json  {len(all_ids):,} IDs")

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(run_dir / "pool_metadata.parquet", index=False, compression="snappy")
    logger.info(f"[EXPORT] pool_metadata.parquet  {len(meta_df):,} rows")

    np.save(run_dir / "gt_embeddings.npy", gt_matrix)
    logger.info(f"[EXPORT] gt_embeddings.npy  {gt_matrix.shape}")

    (run_dir / "gt_doc_ids.json").write_text(json.dumps(gt_doc_ids))
    logger.info(f"[EXPORT] gt_doc_ids.json  {len(gt_doc_ids)} entries")

    # --- Provenance metadata ---
    run_meta = {
        "exported_at":                  datetime.now(timezone.utc).isoformat(),
        "chroma_source":                chroma_source,
        "chroma_total_docs":            chroma_total,
        "embedding_model":              HF_MODEL_NAME,
        "embedding_dim":                dim,
        "pool_requested_docs":          n_unique,
        "pool_docs_with_embeddings":    len(all_ids),
        "pool_docs_missing_embeddings": n_unique - len(all_ids),
        "n_queries":                    len(queries),
        "gt_embeddings_found":          gt_found,
        "queue_rows":                   int(len(df)),
        "queue_queries":                int(n_queries),
        "queue_is_gt_count":            int(df["is_gt"].sum()),
        "queue_rag_failed_count":       int(df["rag_failed"].sum()),
        "queue_dist_to_gt_mean":        float(df["dist_to_gt"].mean()),
        "queue_dist_to_gt_std":         float(df["dist_to_gt"].std()),
        "queue_dist_to_gt_min":         float(df["dist_to_gt"].min()),
        "queue_dist_to_gt_max":         float(df["dist_to_gt"].max()),
        "artifacts": {
            "scoring_queue.parquet":    "Main Phase-2 input — (query_idx, doc_id, doc_text, dist_to_gt, is_gt, rag_failed)",
            "pool_embeddings.npy":      f"float32 [{len(all_ids)} × {dim}] — pool doc embeddings in pool_doc_ids.json order",
            "pool_doc_ids.json":        "Ordered list of doc IDs; row i matches pool_embeddings.npy[i]",
            "pool_metadata.parquet":    "PMID, title, text_snippet per pool doc",
            "gt_embeddings.npy":        f"float32 [{len(queries)} × {dim}] — GT embedding per query; row = query_idx - 1",
            "gt_doc_ids.json":          "GT doc ID per query; index = query_idx - 1",
            "run_metadata.json":        "This file — full provenance",
            "queries.json":             "Full query list used in experiment",
            "correlation_pool_manifest.json": "Ranked pool manifest from Phase 1",
        },
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
    logger.info("[EXPORT] run_metadata.json saved")

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"Export complete  ({elapsed/60:.1f} min)")
    print(f"  Run dir       : {run_dir}")
    print(f"  Pool docs     : {len(all_ids):,}  (requested {n_unique:,})")
    print(f"  Emb matrix    : {emb_matrix.shape}  {emb_matrix.nbytes/1e6:.0f} MB")
    print(f"  GT matrix     : {gt_matrix.shape}")
    print(f"  Queue rows    : {len(df):,}")
    print(f"\nPhase 2 command:")
    print(f"  python score_from_queue.py --queue {queue_path}")
    print(f"{'='*55}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export rich pool artifacts from Chroma")
    ap.add_argument("--run-dir",        type=Path, required=True, help="Phase-1 run directory")
    ap.add_argument("--chroma-source",  default="pubmed_selected_articles",
                    help="Chroma source path (default: pubmed_selected_articles; use 'pubmed_full' for 36M)")
    args = ap.parse_args()
    export_pool_artifacts(args.run_dir, chroma_source=args.chroma_source)


if __name__ == "__main__":
    main()
