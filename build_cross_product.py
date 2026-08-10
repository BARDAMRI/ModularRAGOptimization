#!/usr/bin/env python3
"""
build_cross_product.py  — Phase 1.5
=====================================
Takes scoring_queue.parquet (199 × 200 rows) from embed_and_rank.py,
builds a global document pool D (union of all unique docs, deduplicated),
re-embeds everything locally with all-mpnet-base-v2, then produces an
expanded_scoring_queue.parquet of ~1M (query, doc) pairs ready for
score_from_queue.py.

Steps
-----
1. Load scoring_queue.parquet
2. Build D = union of all unique doc_ids (~39,507 docs after dedup)
3. Re-embed D + all query texts + GT texts with all-mpnet-base-v2
4. K = min(|D|, ceil(1_000_000 / n_queries))  →  per-query top-K from D
5. For each query: pick top-K docs from D by cosine similarity to that query
6. Compute dist_to_gt = 1 - cosine(doc_emb, gt_emb) for every pair
7. Write expanded_scoring_queue.parquet  (~1M rows, same schema as input)

Output columns (compatible with score_from_queue.py):
    query_idx, query_text, gt_id, doc_id, doc_text, dist_to_gt, is_gt, rag_failed

Usage
-----
    python build_cross_product.py --queue results/cloud_runs/.../scoring_queue.parquet
    python build_cross_product.py --queue <path> --target-pairs 1000000
    python build_cross_product.py --queue <path> --output expanded.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))

from utility.logger import logger

MODEL_NAME   = "sentence-transformers/all-mpnet-base-v2"
TARGET_PAIRS = 1_000_000
EMBED_BATCH  = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_plain_text(doc_text_json: str) -> str:
    """Pull plain text out of {"contexts": "..."} JSON stored in parquet."""
    try:
        obj = json.loads(doc_text_json)
        if isinstance(obj, dict):
            return str(obj.get("contexts") or obj.get("context") or "")[:8000]
    except Exception:
        pass
    return str(doc_text_json)[:8000]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_cross_product(
    queue_path: Path,
    output_path: Path | None = None,
    target_pairs: int = TARGET_PAIRS,
) -> Path:

    # ------------------------------------------------------------------
    # 1. Load scoring_queue.parquet
    # ------------------------------------------------------------------
    logger.info(f"[XP] Loading {queue_path}")
    df = pd.read_parquet(queue_path)
    df["doc_id"]   = df["doc_id"].astype(str)
    df["gt_id"]    = df["gt_id"].astype(str)
    df["query_idx"] = df["query_idx"].astype(int)

    n_queries  = df["query_idx"].nunique()
    query_idxs = sorted(df["query_idx"].unique())
    logger.info(
        f"[XP] Queue: {len(df):,} rows | {n_queries} queries | "
        f"{df['doc_id'].nunique():,} unique docs"
    )

    # ------------------------------------------------------------------
    # 2. Build global pool D
    # ------------------------------------------------------------------
    pool_df = (
        df.drop_duplicates("doc_id")[["doc_id", "doc_text"]]
        .reset_index(drop=True)
        .copy()
    )
    doc_ids        = pool_df["doc_id"].tolist()
    doc_texts_json = pool_df["doc_text"].tolist()
    doc_texts_plain = [_extract_plain_text(t) for t in doc_texts_json]
    D = len(doc_ids)
    logger.info(f"[XP] Global pool |D| = {D:,} unique docs")

    # ------------------------------------------------------------------
    # 3. Per-query metadata
    # ------------------------------------------------------------------
    per_query: dict[int, dict] = {}
    for q_idx in query_idxs:
        qdf        = df[df["query_idx"] == q_idx]
        query_text = str(qdf["query_text"].iloc[0])
        gt_id      = str(qdf["gt_id"].iloc[0])
        rag_failed = int(qdf["rag_failed"].iloc[0])

        gt_rows = qdf[qdf["is_gt"] == 1]
        if not gt_rows.empty:
            gt_text = _extract_plain_text(str(gt_rows["doc_text"].iloc[0]))
        else:
            # GT wasn't found in the 23.9M corpus — fall back to query text
            gt_text = query_text

        per_query[q_idx] = {
            "query_text": query_text,
            "gt_id":      gt_id,
            "gt_text":    gt_text,
            "rag_failed": rag_failed,
        }

    # ------------------------------------------------------------------
    # 4. Embed everything
    # ------------------------------------------------------------------
    import torch
    from sentence_transformers import SentenceTransformer

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"[XP] Embedding device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    logger.info(f"[XP] Embedding {D:,} docs in pool D …")
    doc_embs = model.encode(
        doc_texts_plain,
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)   # (D, 768)

    query_texts = [per_query[q]["query_text"] for q in query_idxs]
    logger.info(f"[XP] Embedding {n_queries} query texts …")
    query_embs = model.encode(
        query_texts,
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)   # (n_q, 768)

    gt_texts = [per_query[q]["gt_text"] for q in query_idxs]
    logger.info(f"[XP] Embedding {n_queries} GT texts …")
    gt_embs = model.encode(
        gt_texts,
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)   # (n_q, 768)

    del model  # free memory before building the big matrix

    # ------------------------------------------------------------------
    # 5. Build global ranked pool S by interleaving per-query rankings
    #    (mirrors global_correlation_experiment.py ranked_pool construction)
    #    Each query's docs are already ordered by cosine similarity (pos 0 = best).
    # ------------------------------------------------------------------
    logger.info("[XP] Building global ranked pool S via interleaving …")

    # per_query_doc_ids[q_idx] = ordered list of doc_ids for that query
    per_query_doc_ids: dict[int, list[str]] = {}
    for q_idx in query_idxs:
        qdf = df[df["query_idx"] == q_idx].copy()
        # rows are in heap extraction order (best first); preserve that order
        per_query_doc_ids[q_idx] = qdf["doc_id"].astype(str).tolist()

    max_pos = max(len(v) for v in per_query_doc_ids.values())
    ranked_pool: list[str] = []   # globally-ordered unique doc_ids
    seen_in_pool: set[str] = set()

    for pos in range(max_pos):
        for q_idx in query_idxs:
            docs = per_query_doc_ids[q_idx]
            if pos < len(docs):
                did = docs[pos]
                if did not in seen_in_pool:
                    seen_in_pool.add(did)
                    ranked_pool.append(did)

    logger.info(f"[XP] ranked_pool size: {len(ranked_pool):,} unique docs")

    # ------------------------------------------------------------------
    # 6. Select global fixed subset S of size K (same for ALL queries)
    # ------------------------------------------------------------------
    K = min(len(ranked_pool), math.ceil(target_pairs / n_queries))
    S_ids = ranked_pool[:K]   # first K docs from globally-interleaved list
    S_set = set(S_ids)

    logger.info(f"[XP] K = min({len(ranked_pool):,}, ceil({target_pairs:,}/{n_queries})) = {K:,}")
    logger.info(f"[XP] Global pool S: {K:,} docs  →  {n_queries} × {K:,} = {n_queries * K:,} pairs")

    # Map S doc_ids → pool indices for fast embedding lookup
    doc_id_to_pool_idx = {did: i for i, did in enumerate(doc_ids)}
    S_pool_idx = np.array([doc_id_to_pool_idx[did] for did in S_ids], dtype=np.int32)
    S_embs     = doc_embs[S_pool_idx]          # (K, 768)
    S_texts    = [doc_texts_json[i] for i in S_pool_idx]

    # ------------------------------------------------------------------
    # 7. Build cross-product: all queries × same S — write with pyarrow
    # ------------------------------------------------------------------
    if output_path is None:
        output_path = queue_path.parent / "expanded_scoring_queue.parquet"

    schema = pa.schema([
        pa.field("query_idx",  pa.int32()),
        pa.field("query_text", pa.string()),
        pa.field("gt_id",      pa.string()),
        pa.field("doc_id",     pa.string()),
        pa.field("doc_text",   pa.string()),
        pa.field("dist_to_gt", pa.float32()),
        pa.field("is_gt",      pa.int8()),
        pa.field("rag_failed", pa.int8()),
    ])

    writer = pq.ParquetWriter(str(output_path), schema, compression="snappy")
    total_rows = 0

    for i, q_idx in enumerate(query_idxs):
        meta   = per_query[q_idx]
        gt_id  = meta["gt_id"]
        gt_emb = gt_embs[i]   # (768,) normalized

        # dist_to_gt for all K docs in S — vectorized
        cos_sims_gt = S_embs @ gt_emb               # (K,) — both L2-normalised
        dists       = (1.0 - np.clip(cos_sims_gt, -1.0, 1.0)).astype(np.float32)
        is_gt_flags = np.array([1 if did == gt_id else 0 for did in S_ids], dtype=np.int8)

        batch = pa.record_batch({
            "query_idx":  pa.array([q_idx] * K, type=pa.int32()),
            "query_text": pa.array([meta["query_text"]] * K, type=pa.string()),
            "gt_id":      pa.array([gt_id] * K, type=pa.string()),
            "doc_id":     pa.array(S_ids, type=pa.string()),
            "doc_text":   pa.array(S_texts, type=pa.string()),
            "dist_to_gt": pa.array(dists, type=pa.float32()),
            "is_gt":      pa.array(is_gt_flags, type=pa.int8()),
            "rag_failed": pa.array([meta["rag_failed"]] * K, type=pa.int8()),
        }, schema=schema)

        writer.write_batch(batch)
        total_rows += K

        if (i + 1) % 20 == 0 or (i + 1) == n_queries:
            logger.info(f"[XP] {i+1}/{n_queries} queries | {total_rows:,} rows written")

    writer.close()

    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"[XP] Saved → {output_path}  ({size_mb:.0f} MB, {total_rows:,} rows)")

    # Quick summary
    check = pd.read_parquet(output_path, columns=["query_idx", "is_gt", "rag_failed"])
    n_gt  = int(check.groupby("query_idx")["is_gt"].max().sum())
    logger.info(f"[XP] GT found in cross-product: {n_gt}/{n_queries} queries")
    logger.info(f"[XP] RAG failed: {int((check.groupby('query_idx')['rag_failed'].first()==1).sum())}/{n_queries} queries")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.5: build ~1M cross-product scoring queue from scoring_queue.parquet"
    )
    parser.add_argument(
        "--queue", type=Path, required=True,
        help="Path to scoring_queue.parquet produced by embed_and_rank.py"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for expanded_scoring_queue.parquet (default: same dir as --queue)"
    )
    parser.add_argument(
        "--target-pairs", type=int, default=TARGET_PAIRS,
        help=f"Target number of (query, doc) pairs (default: {TARGET_PAIRS:,})"
    )
    args = parser.parse_args()

    queue_path = args.queue.resolve()
    if not queue_path.exists():
        print(f"ERROR: {queue_path} not found", file=sys.stderr)
        sys.exit(1)

    out = build_cross_product(queue_path, args.output, args.target_pairs)

    summary = pd.read_parquet(out, columns=["query_idx", "doc_id"])
    print(f"\nExpanded queue ready: {out}")
    print(f"  Rows:        {len(summary):,}")
    print(f"  Queries:     {summary['query_idx'].nunique()}")
    print(f"  Unique docs: {summary['doc_id'].nunique():,}")


if __name__ == "__main__":
    main()
