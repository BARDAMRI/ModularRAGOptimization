#!/usr/bin/env python3
"""
build_scoring_queue.py — Phase 1 (Cloud / GPU)
================================================
Builds the global ranked_pool from Chroma and pre-computes all (query, doc)
distances, exporting a self-contained Parquet file that Phase 2 can score
with LLM calls without any Chroma / GPU access.

Output
------
<run_dir>/scoring_queue.parquet
  Columns: query_idx, query_text, gt_id, doc_id, doc_text, dist_to_gt,
           is_gt, rag_failed

Usage
-----
    python build_scoring_queue.py                          # default corpus (272K)
    python build_scoring_queue.py --chroma-source pubmed_full  # 36M corpus
    python build_scoring_queue.py --num-queries 199
    python build_scoring_queue.py --run-dir results/global_exp/staged_run_XXXX
    python build_scoring_queue.py --max-ranked-gt 100 --max-ranked-query 100 --stride 10
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from configurations.config import (
    GLOBAL_EXPERIMENT_OUTPUT_DIR,
    STAGING_STRIDE,
    STAGING_MAX_RANKED_PER_GT,
    STAGING_MAX_RANKED_PER_QUERY,
    INDEX_SOURCE_URL,
    HF_MODEL_NAME,
)
from scripts.qa_data_set_loader import load_qa_queries
from utility.logger import logger
from vector_db.trilateration_retriever import cosine_distance

# Import reusable internals from the experiment module
from experiments.global_correlation_experiment import (
    _load_or_build_manifest,
    _gt_id_str,
    _gt_pmid_int_for_chroma,
    _get_gt_embedding_dual_path,
    _normalize_id,
    _extract_contexts_text,
    _check_rag_baseline,
    _interleave_across_queries,
    MANIFEST_FILENAME,
    CHROMA_FETCH_CHUNK,
)

QUEUE_FILENAME  = "scoring_queue.parquet"
RUN_STATE_FILE  = "run_state.json"


# ---------------------------------------------------------------------------
# Per-run state helpers (shared format with score_from_queue.py)
# ---------------------------------------------------------------------------

def _load_run_state(run_dir: Path) -> dict:
    p = run_dir / RUN_STATE_FILE
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_run_state(run_dir: Path, updates: dict) -> None:
    p = run_dir / RUN_STATE_FILE
    state = _load_run_state(run_dir)
    state.update(updates)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(p)


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / MANIFEST_FILENAME


def _save_manifest(manifest: dict, run_dir: Path) -> None:
    path = _manifest_path(run_dir)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(path)


def _load_manifest(run_dir: Path) -> dict | None:
    path = _manifest_path(run_dir)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _setup_run_dir(output_dir: str, params: dict) -> Path:
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_dir) / f"staged_run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "experiment_params.json", "w") as f:
        json.dump(params, f, indent=2)
    return run_dir


def _find_or_create_run_dir(
    output_dir: str,
    resume_run_dir: str | None,
    params: dict,
    fresh: bool = False,
) -> Path:
    if resume_run_dir:
        rd = Path(resume_run_dir)
        if not rd.exists():
            raise FileNotFoundError(f"Run dir not found: {rd}")
        logger.info(f"[PHASE1] Resuming explicit run: {rd}")
        return rd
    if not fresh:
        base = Path(output_dir)
        if base.exists():
            runs = sorted(base.glob("staged_run_*"), reverse=True)
            if runs:
                state = _load_run_state(runs[0])
                if state.get("phase1_complete"):
                    logger.info(
                        f"[PHASE1] Latest run is already complete: {runs[0]}\n"
                        "         Pass --fresh to start a new run."
                    )
                else:
                    logger.info(f"[PHASE1] Resuming incomplete run: {runs[0]}")
                return runs[0]
    if fresh:
        logger.info("[PHASE1] --fresh: starting a new run directory")
    return _setup_run_dir(output_dir, params)


def _open_vector_db(chroma_source: str):
    """
    Open a ChromaVectorDB for the given source path.
    Works for both the default 272K corpus and the full pubmed_full corpus.
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from vector_db.chroma_vector_db import ChromaVectorDB
    from utility.distance_metrics import DistanceMetric

    embedding_model = HuggingFaceEmbedding(model_name=HF_MODEL_NAME)
    vector_db = ChromaVectorDB(
        source_path=chroma_source,
        embedding_model=embedding_model,
        distance_metric=DistanceMetric.COSINE,
    )
    count = vector_db.collection.count() if hasattr(vector_db, "collection") else -1
    if count == 0:
        raise RuntimeError(
            f"Chroma collection for '{chroma_source}' is empty (0 docs). "
            "Phase 0 (embed_full_pubmed.py) must complete successfully before Phase 1."
        )
    return vector_db, embedding_model


def build_scoring_queue(
    *,
    num_queries: int = 200,
    stride: int | None = None,
    max_ranked_gt: int | None = None,
    max_ranked_query: int | None = None,
    output_dir: str | None = None,
    resume_run_dir: str | None = None,
    chroma_source: str | None = None,
    fresh: bool = False,
) -> Path:
    """
    Build (or resume) the scoring queue.

    Returns path to the saved scoring_queue.parquet.
    """
    stride = int(stride or STAGING_STRIDE)
    max_ranked_gt = int(max_ranked_gt or STAGING_MAX_RANKED_PER_GT)
    max_ranked_query = int(max_ranked_query or STAGING_MAX_RANKED_PER_QUERY)
    output_dir = output_dir or GLOBAL_EXPERIMENT_OUTPUT_DIR
    chroma_source = chroma_source or INDEX_SOURCE_URL

    queries = load_qa_queries(num_queries)
    n_queries = len(queries)
    logger.info(f"[PHASE1] Loaded {n_queries} queries")
    logger.info(f"[PHASE1] Chroma source: {chroma_source}")

    params = {
        "num_queries": n_queries,
        "stride": stride,
        "max_ranked_per_gt": max_ranked_gt,
        "max_ranked_per_query": max_ranked_query,
        "phase": "phase1_build_queue",
        "chroma_source": chroma_source,
    }
    run_dir = _find_or_create_run_dir(output_dir, resume_run_dir, params, fresh=fresh)
    queue_path = run_dir / QUEUE_FILENAME

    # Check if queue already exists and is complete
    if queue_path.exists():
        existing = pd.read_parquet(queue_path)
        logger.info(f"[PHASE1] Found existing queue: {len(existing):,} rows — resuming from checkpoint")
        existing_pairs = set(zip(existing["query_idx"], existing["doc_id"]))
    else:
        existing = pd.DataFrame()
        existing_pairs = set()

    # Load vector DB (Chroma)
    logger.info("[PHASE1] Loading vector DB...")
    vector_db, embedding_model_inst = _open_vector_db(chroma_source)
    collection = vector_db.collection

    # Build / load manifest (ranked_pool)
    manifest = _load_or_build_manifest(
        str(run_dir),
        queries,
        vector_db,
        embedding_model_inst,
        stride=stride,
        max_ranked_gt=max_ranked_gt,
        max_ranked_query=max_ranked_query,
    )

    ranked_pool: list[str] = manifest.get("ranked_pool", [])
    gt_ids: dict[str, str] = manifest.get("gt_ids", {})
    logger.info(f"[PHASE1] ranked_pool size: {len(ranked_pool):,} docs")

    # Save queries.json for Phase 2 (needed for run resume)
    queries_path = run_dir / "queries.json"
    if not queries_path.exists():
        with open(queries_path, "w") as f:
            json.dump(queries, f, indent=2)

    # Build all (query_idx, doc_id) pairs to process
    all_pairs: list[tuple[int, str]] = []
    for q_idx, _ in enumerate(queries, start=1):
        for doc_id in ranked_pool:
            if (q_idx, _normalize_id(doc_id)) not in existing_pairs:
                all_pairs.append((q_idx, doc_id))

    logger.info(
        f"[PHASE1] Total pairs in pool: {len(ranked_pool) * n_queries:,} | "
        f"Already done: {len(existing_pairs):,} | "
        f"Remaining: {len(all_pairs):,}"
    )

    if not all_pairs:
        logger.info("[PHASE1] Queue already complete.")
        return queue_path

    # Pre-compute GT embeddings per query
    logger.info("[PHASE1] Fetching GT embeddings...")
    gt_embeddings: dict[int, np.ndarray] = {}
    for q_idx, entry in enumerate(queries, start=1):
        gt_id_norm = _gt_id_str(entry)
        gt_pmid = _gt_pmid_int_for_chroma(entry)
        gt_emb = _get_gt_embedding_dual_path(collection, gt_id_norm, gt_pmid)
        if gt_emb is not None:
            gt_embeddings[q_idx] = gt_emb

    logger.info(f"[PHASE1] GT embeddings found: {len(gt_embeddings)}/{n_queries}")

    # Build rag_failed per query (top-1 retrieval check)
    logger.info("[PHASE1] Computing rag_failed per query...")
    rag_failed_map: dict[int, int] = {}
    for q_idx, entry in enumerate(queries, start=1):
        query_text = entry.get("question", "")
        gt_id_norm = gt_ids.get(str(q_idx), _gt_id_str(entry))
        q_emb = np.array(
            embedding_model_inst.get_text_embedding(query_text), dtype=np.float32
        )
        failed = _check_rag_baseline(vector_db, q_emb, gt_id_norm)
        rag_failed_map[q_idx] = 1 if failed else 0

    # Process docs in chunks — fetch text+embedding from Chroma, compute dist_to_gt
    logger.info("[PHASE1] Fetching doc texts and computing distances...")

    # Group pairs by doc_id to batch Chroma fetches efficiently
    doc_to_queries: dict[str, list[int]] = {}
    for q_idx, doc_id in all_pairs:
        nid = _normalize_id(doc_id)
        doc_to_queries.setdefault(nid, []).append(q_idx)

    unique_doc_ids = list(doc_to_queries.keys())
    rows_buffer: list[dict] = []
    FLUSH_EVERY = 50_000
    total_docs = len(unique_doc_ids)

    for chunk_start in range(0, total_docs, CHROMA_FETCH_CHUNK):
        chunk_ids = unique_doc_ids[chunk_start: chunk_start + CHROMA_FETCH_CHUNK]

        # Fetch from Chroma
        raw = collection.get(ids=chunk_ids, include=["documents", "embeddings"])
        ids_out = raw.get("ids") or []
        docs_out = raw.get("documents") or []
        embs_raw = raw.get("embeddings")
        embs_out = embs_raw if embs_raw is not None else []

        for did, doc_text, emb_raw in zip(ids_out, docs_out, embs_out):
            if doc_text is None or emb_raw is None:
                continue
            nid = _normalize_id(did)
            doc_emb = np.array(emb_raw, dtype=np.float32)
            ctx = _extract_contexts_text(doc_text)

            for q_idx in doc_to_queries.get(nid, []):
                entry = queries[q_idx - 1]
                query_text = entry.get("question", "")
                gt_id_norm = gt_ids.get(str(q_idx), _gt_id_str(entry))
                gt_emb = gt_embeddings.get(q_idx)
                if gt_emb is None:
                    continue

                dist = float(cosine_distance(gt_emb, doc_emb))
                is_gt = 1 if (nid == _normalize_id(gt_id_norm) or dist < 1e-5) else 0

                rows_buffer.append({
                    "query_idx": q_idx,
                    "query_text": query_text,
                    "gt_id": gt_id_norm,
                    "doc_id": nid,
                    "doc_text": ctx,
                    "dist_to_gt": dist,
                    "is_gt": is_gt,
                    "rag_failed": rag_failed_map.get(q_idx, 0),
                })

        # Flush buffer to parquet periodically
        if len(rows_buffer) >= FLUSH_EVERY:
            chunk_df = pd.DataFrame(rows_buffer)
            combined = pd.concat([existing, chunk_df], ignore_index=True) if not existing.empty else chunk_df
            combined.to_parquet(queue_path, index=False, compression="snappy")
            existing = combined
            rows_buffer = []
            done = chunk_start + len(chunk_ids)
            logger.info(
                f"[PHASE1] Progress: {done:,}/{total_docs:,} docs "
                f"({100*done/total_docs:.1f}%) | queue rows: {len(existing):,}"
            )

    # Final flush
    if rows_buffer:
        chunk_df = pd.DataFrame(rows_buffer)
        combined = pd.concat([existing, chunk_df], ignore_index=True) if not existing.empty else chunk_df
        combined.to_parquet(queue_path, index=False, compression="snappy")
        logger.info(f"[PHASE1] Done. Queue saved: {len(combined):,} rows → {queue_path}")

    # Persist completion marker (loaded by score_from_queue.py and run_cloud_phase1.py)
    final_df = pd.read_parquet(queue_path) if queue_path.exists() else existing
    _save_run_state(run_dir, {
        "phase1_complete":       True,
        "phase1_completed_at":   datetime.now(timezone.utc).isoformat(),
        "phase1_queue_rows":     int(len(final_df)),
        "phase1_unique_docs":    int(final_df["doc_id"].nunique()) if not final_df.empty else 0,
        "phase1_n_queries":      n_queries,
        "phase1_chroma_source":  chroma_source,
        "phase1_queue_path":     str(queue_path),
    })
    logger.info(f"[PHASE1] run_state.json updated → phase1_complete=True")

    return queue_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: build scoring queue from Chroma")
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-ranked-gt", type=int, default=None)
    parser.add_argument("--max-ranked-query", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None, help="Resume existing run")
    parser.add_argument(
        "--chroma-source", type=str, default=None,
        help=(
            "Chroma corpus source path (default: INDEX_SOURCE_URL from config). "
            "Use 'pubmed_full' for the 36M corpus built by embed_full_pubmed.py."
        ),
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help=(
            "Ignore any existing run and start a brand-new staged_run_* directory. "
            "Without this flag the script auto-resumes the most recent incomplete run, "
            "or re-uses a complete run as-is."
        ),
    )
    args = parser.parse_args()

    queue_path = build_scoring_queue(
        num_queries=args.num_queries,
        stride=args.stride,
        max_ranked_gt=args.max_ranked_gt,
        max_ranked_query=args.max_ranked_query,
        output_dir=args.output_dir,
        resume_run_dir=args.run_dir,
        chroma_source=args.chroma_source,
        fresh=args.fresh,
    )
    print(f"\nScoring queue ready: {queue_path}")
    df = pd.read_parquet(queue_path)
    print(f"  Rows:        {len(df):,}")
    print(f"  Queries:     {df['query_idx'].nunique()}")
    print(f"  Unique docs: {df['doc_id'].nunique():,}")
    print(f"  File size:   {os.path.getsize(queue_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
