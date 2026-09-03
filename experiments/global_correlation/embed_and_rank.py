#!/usr/bin/env python3
"""
embed_and_rank.py — Phase 0+1 (Cloud / GPU) — NO CHROMA / NO HNSW
===================================================================
Streams through all 23M PubMed documents, embeds on GPU, and directly builds
scoring_queue.parquet WITHOUT building a Chroma/HNSW index.

Why: hnswlib HNSW construction scales O(N log N) on CPU, making a 23M-doc
index take 6–20+ days on 4 cores. This script bypasses the index entirely:
  - Embed 199 queries once (seconds)
  - Stream all 23M docs, embed on GPU in large batches (~700 docs/s)
  - Matmul: cosine_sim(199 queries, batch) in microseconds on GPU
  - Maintain min-heap of top-K per query (~140 MB RAM)
  - Write scoring_queue.parquet at the end

Estimated wall-clock: ~10 hours on g4dn.xlarge (T4) vs 6–20+ days for HNSW.

Output — same schema as build_scoring_queue.py (Phase 1):
  <run_dir>/scoring_queue.parquet
  Columns: query_idx, query_text, gt_id, doc_id, doc_text,
           dist_to_gt, is_gt, rag_failed

Usage
-----
    python embed_and_rank.py
    python embed_and_rank.py --dataset data/pubmed_parquet_slim --top-k 200
    python embed_and_rank.py --run-dir results/global_exp/staged_run_XXXX --resume
    # MedRAG/pubmed Parquet (default fields: PMID / title / content)
    python embed_and_rank.py --pmid-field PMID --title-field title --abstract-field content
"""

from __future__ import annotations

import argparse
import glob
import heapq
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from utility.logger import logger
from configurations.config import GLOBAL_EXPERIMENT_OUTPUT_DIR, HF_MODEL_NAME
from scripts.qa_data_set_loader import load_qa_queries
from experiments.global_correlation_experiment import (
    _gt_id_str,
    _gt_pmid_int_for_chroma,
    _normalize_id,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUEUE_FILENAME = "scoring_queue.parquet"
RUN_STATE_FILE = "run_state.json"
CKPT_FILE      = "embed_rank_ckpt.pkl"

# MedRAG/pubmed slim Parquet columns (created by embed_full_pubmed.py JSONL→Parquet)
DEFAULT_DATASET       = "data/pubmed_parquet_slim"
DEFAULT_MODEL         = HF_MODEL_NAME
DEFAULT_TOP_K         = 200
DEFAULT_BATCH         = 2048          # docs to buffer before flush (GPU encode uses GPU_ENCODE_BATCH)
GPU_ENCODE_BATCH      = 256           # actual GPU forward-pass size (fits T4 16GB)
DEFAULT_PMID_FIELD    = "PMID"
DEFAULT_TITLE_FIELD   = "title"       # MedRAG/pubmed slim Parquet column
DEFAULT_ABSTRACT_FIELD = "content"   # MedRAG/pubmed slim Parquet column

LOG_EVERY  = 100_000     # log every N docs
CKPT_EVERY = 50          # checkpoint every N Parquet files (~1M docs)


# ---------------------------------------------------------------------------
# Direct PyArrow iterator (no HF datasets overhead)
# ---------------------------------------------------------------------------

def _iter_parquet(files: list[str], cols: list[str]) -> Iterator[dict]:
    import pyarrow.parquet as pq
    for f in files:
        try:
            tbl = pq.read_table(f, columns=cols)
        except Exception as e:
            logger.warning(f"[RANK] Skipping {f}: {e}")
            continue
        for batch in tbl.to_batches(max_chunksize=8192):
            for i in range(batch.num_rows):
                yield {c: batch.column(c)[i].as_py() for c in cols}


# ---------------------------------------------------------------------------
# Per-query top-K heap (min by similarity — worst item at top of heap)
# ---------------------------------------------------------------------------

class TopKHeap:
    """
    Min-heap keeping the top-K (highest-similarity) items.
    Uses a tie-breaking counter so numpy arrays are never compared directly.
    """

    __slots__ = ("k", "_heap", "_ctr")

    def __init__(self, k: int) -> None:
        self.k    = k
        self._heap: list[tuple[float, int, str, str, bytes]] = []
        self._ctr  = 0

    def offer(self, sim: float, doc_id: str, text: str, emb: np.ndarray) -> None:
        if len(self._heap) >= self.k and sim <= self._heap[0][0]:
            return
        self._ctr += 1
        emb_b = emb.astype(np.float32).tobytes()
        heapq.heappush(self._heap, (sim, self._ctr, doc_id, text, emb_b))
        if len(self._heap) > self.k:
            heapq.heappop(self._heap)

    def min_sim(self) -> float:
        return self._heap[0][0] if self._heap else -2.0

    def top_k(self) -> list[tuple[float, str, str, np.ndarray]]:
        """Return (sim, doc_id, text, emb) sorted best-first."""
        return sorted(
            [
                (s, d, t, np.frombuffer(e, dtype=np.float32))
                for s, _, d, t, e in self._heap
            ],
            key=lambda x: -x[0],
        )

    def state(self) -> list[tuple[float, str, str, bytes]]:
        """Serialisable heap state."""
        return [(s, d, t, e) for s, _, d, t, e in self._heap]

    @classmethod
    def from_state(cls, k: int, items: list[tuple]) -> "TopKHeap":
        h = cls(k)
        for s, d, t, e in items:
            h.offer(s, d, t, np.frombuffer(e, dtype=np.float32))
        return h


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(ckpt_path: Path, next_file_idx: int, total_seen: int,
                     heaps: list[TopKHeap]) -> None:
    data = {
        "next_file_idx": next_file_idx,
        "total_seen":    total_seen,
        "heaps":         [h.state() for h in heaps],
    }
    tmp = ckpt_path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=4)
    tmp.replace(ckpt_path)
    logger.info(f"[RANK] Checkpoint saved: file #{next_file_idx}, {total_seen:,} docs")


def _load_checkpoint(ckpt_path: Path, top_k: int) -> tuple[int, int, list[TopKHeap]] | None:
    if not ckpt_path.exists():
        return None
    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)
    heaps = [TopKHeap.from_state(top_k, items) for items in data["heaps"]]
    return data["next_file_idx"], data["total_seen"], heaps


# ---------------------------------------------------------------------------
# run_state.json helpers (shared format with score_from_queue.py)
# ---------------------------------------------------------------------------

def _load_run_state(run_dir: Path) -> dict:
    p = run_dir / RUN_STATE_FILE
    return json.loads(p.read_text()) if p.exists() else {}


def _save_run_state(run_dir: Path, updates: dict) -> None:
    p    = run_dir / RUN_STATE_FILE
    state = _load_run_state(run_dir)
    state.update(updates)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(p)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def embed_and_rank(
    *,
    dataset: str             = DEFAULT_DATASET,
    model_name: str          = DEFAULT_MODEL,
    embed_batch: int         = DEFAULT_BATCH,
    top_k: int               = DEFAULT_TOP_K,
    num_queries: int         = 199,
    pmid_field: str          = DEFAULT_PMID_FIELD,
    title_field: str         = DEFAULT_TITLE_FIELD,
    abstract_field: str      = DEFAULT_ABSTRACT_FIELD,
    output_dir: str | None   = None,
    run_dir_override: str | None = None,
    resume: bool             = False,
) -> Path:
    import torch
    from sentence_transformers import SentenceTransformer

    output_dir = output_dir or GLOBAL_EXPERIMENT_OUTPUT_DIR

    # --- run dir ---
    if run_dir_override:
        run_dir = Path(run_dir_override)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path(output_dir) / f"staged_run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

    queue_path = run_dir / QUEUE_FILENAME
    ckpt_path  = run_dir / CKPT_FILE
    logger.info(f"[RANK] Run dir: {run_dir}")

    # --- device ---
    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    logger.info(f"[RANK] Device: {device}")

    # --- model ---
    logger.info(f"[RANK] Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    dim = model.get_sentence_embedding_dimension()
    logger.info(f"[RANK] Embedding dim: {dim}")

    # --- queries ---
    queries     = load_qa_queries(num_queries)
    n_q         = len(queries)
    query_texts = [q.get("question", "") for q in queries]
    gt_ids      = [_normalize_id(_gt_id_str(q)) for q in queries]
    gt_pmids    = [_gt_pmid_int_for_chroma(q) for q in queries]
    logger.info(f"[RANK] Queries: {n_q}")

    # Pre-embed queries and GT documents (from CONTEXTS field)
    def _contexts_text(entry: dict) -> str:
        raw = entry.get("CONTEXTS") or entry.get("contexts") or []
        if isinstance(raw, list):
            return " ".join(str(c) for c in raw if c)
        return str(raw)

    logger.info("[RANK] Embedding queries and GT contexts...")
    gt_texts   = [_contexts_text(q) for q in queries]
    query_vecs = model.encode(
        query_texts, batch_size=min(64, n_q), normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    gt_vecs = model.encode(
        gt_texts, batch_size=min(64, n_q), normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    logger.info(f"[RANK] Queries/GT embedded. Shape: {query_vecs.shape}")

    # Switch model to FP16 for 3-4× faster inference on tensor cores (T4/V100)
    if device == "cuda":
        model = model.half()
        logger.info("[RANK] Model converted to FP16 for faster GPU inference")

    # GPU query matrix in FP16
    Q = torch.from_numpy(query_vecs).half().to(device) if device == "cuda" else torch.from_numpy(query_vecs).to(device)   # (n_q, dim)

    # --- Parquet files ---
    if not os.path.isdir(dataset):
        raise FileNotFoundError(f"Dataset dir not found: {dataset}")
    parquet_files = sorted(glob.glob(os.path.join(dataset, "**", "*.parquet"), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found under {dataset}")
    logger.info(f"[RANK] {len(parquet_files)} Parquet files in {dataset}")

    # --- Resume from checkpoint ---
    start_file_idx = 0
    total_seen     = 0
    heaps: list[TopKHeap] = [TopKHeap(top_k) for _ in range(n_q)]
    gt_vecs_updated = [False] * n_q

    if resume:
        ckpt = _load_checkpoint(ckpt_path, top_k)
        if ckpt is not None:
            start_file_idx, total_seen, heaps = ckpt
            logger.info(f"[RANK] Resumed: file #{start_file_idx}, {total_seen:,} docs seen")
        else:
            logger.info("[RANK] No checkpoint found — starting fresh")

    # GT PMID → query indices (for updating GT vecs when we see the doc in stream)
    gt_pmid_to_qidx: dict[int, list[int]] = {}
    for q_i, pm in enumerate(gt_pmids):
        if pm is not None:
            gt_pmid_to_qidx.setdefault(pm, []).append(q_i)

    # --- Streaming embed + rank ---
    t0   = time.time()
    cols = [c for c in [pmid_field, title_field, abstract_field] if c]

    buf_pmids: list[int | None] = []
    buf_ids:   list[str]        = []
    buf_texts: list[str]        = []

    def _flush() -> None:
        nonlocal total_seen
        if not buf_texts:
            return

        # GPU encode — use GPU_ENCODE_BATCH to prevent OOM (separate from buffer size)
        doc_vecs_t = model.encode(
            buf_texts,
            batch_size=GPU_ENCODE_BATCH,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=device,
        )  # (B, dim)

        # Cosine similarity: (n_q, B) — both already L2-normalised
        # Cast to float32 for matmul if Q is fp16 (tensor cores) then back to float32 numpy
        sims_np = torch.mm(Q, doc_vecs_t.T).float().cpu().numpy()  # (n_q, B)
        doc_vecs_np = doc_vecs_t.float().cpu().numpy().astype(np.float32)

        # Update GT vecs if we encounter the GT doc in the corpus
        for b_i, pm in enumerate(buf_pmids):
            if pm is not None and pm in gt_pmid_to_qidx:
                for q_i in gt_pmid_to_qidx[pm]:
                    if not gt_vecs_updated[q_i]:
                        gt_vecs[q_i] = doc_vecs_np[b_i]
                        gt_vecs_updated[q_i] = True

        # Update heaps — vectorized: numpy mask to find candidates, Python only for actual inserts
        for q_i in range(n_q):
            heap = heaps[q_i]
            min_s = heap.min_sim()
            if len(heap._heap) < top_k:
                # Heap not full yet — accept all (warm-up phase)
                for b_i in range(len(buf_ids)):
                    heap.offer(float(sims_np[q_i, b_i]), buf_ids[b_i], buf_texts[b_i], doc_vecs_np[b_i])
            else:
                # Heap full — use numpy to find only candidates that beat threshold
                good = np.where(sims_np[q_i] > min_s)[0]
                for b_i in good:
                    heap.offer(float(sims_np[q_i, b_i]), buf_ids[b_i], buf_texts[b_i], doc_vecs_np[b_i])
                    min_s = heap.min_sim()

        total_seen += len(buf_ids)
        buf_pmids.clear()
        buf_ids.clear()
        buf_texts.clear()

    for file_idx, pf in enumerate(parquet_files):
        if file_idx < start_file_idx:
            continue

        for rec in _iter_parquet([pf], cols):
            pmid_raw = rec.get(pmid_field)
            title    = str(rec.get(title_field, "") or "").strip()
            abstract = str(rec.get(abstract_field, "") or "").strip()

            if not abstract and not title:
                continue

            if pmid_raw is None:
                doc_id   = f"doc_{total_seen + len(buf_ids)}"
                pmid_int = None
            else:
                doc_id = _normalize_id(pmid_raw)
                try:
                    pmid_int = int(str(pmid_raw).split("_")[0])
                except ValueError:
                    pmid_int = None

            text = f"{title}. {abstract}".strip(" .")
            buf_pmids.append(pmid_int)
            buf_ids.append(doc_id)
            buf_texts.append(text)

            if len(buf_ids) >= embed_batch:
                _flush()

                if total_seen % LOG_EVERY < embed_batch:
                    elapsed = time.time() - t0
                    rate = total_seen / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"[RANK] {total_seen:,} docs | {rate:.0f} docs/s | "
                        f"file {file_idx+1}/{len(parquet_files)} | "
                        f"eta {(23_000_000 - total_seen) / max(rate,1) / 3600:.1f}h"
                    )

        # Checkpoint every CKPT_EVERY files
        if (file_idx + 1) % CKPT_EVERY == 0:
            _flush()
            _save_checkpoint(ckpt_path, file_idx + 1, total_seen, heaps)

    _flush()  # final partial batch

    elapsed = time.time() - t0
    logger.info(
        f"[RANK] Scan complete. {total_seen:,} docs | {elapsed/3600:.2f}h | "
        f"{total_seen/elapsed:.0f} docs/s"
    )

    # --- Build scoring_queue.parquet ---
    logger.info("[RANK] Building scoring_queue.parquet...")

    rag_failed: dict[int, int] = {}
    for q_i in range(n_q):
        top = heaps[q_i].top_k()
        top1_id = top[0][1] if top else ""
        rag_failed[q_i] = 0 if _normalize_id(top1_id) == gt_ids[q_i] else 1

    rows: list[dict] = []
    for q_i, query in enumerate(queries):
        q_idx      = q_i + 1
        query_text = query_texts[q_i]
        gt_id      = gt_ids[q_i]
        gt_emb     = gt_vecs[q_i]
        gt_norm    = np.linalg.norm(gt_emb)

        for sim, doc_id, doc_text, doc_emb in heaps[q_i].top_k():
            doc_text_json = json.dumps({"contexts": doc_text})

            doc_norm = np.linalg.norm(doc_emb)
            denom    = (gt_norm * doc_norm) + 1e-10
            cos_sim  = float(np.dot(gt_emb, doc_emb) / denom)
            dist     = float(1.0 - np.clip(cos_sim, -1.0, 1.0))

            rows.append({
                "query_idx":  q_idx,
                "query_text": query_text,
                "gt_id":      gt_id,
                "doc_id":     _normalize_id(doc_id),
                "doc_text":   doc_text_json,
                "dist_to_gt": dist,
                "is_gt":      1 if _normalize_id(doc_id) == gt_id else 0,
                "rag_failed": rag_failed[q_i],
            })

    df = pd.DataFrame(rows)
    df.to_parquet(queue_path, index=False, compression="snappy")
    logger.info(f"[RANK] Saved {len(df):,} rows → {queue_path}")

    # Mark phase1 complete (detected by run_cloud_phase1.py + score_from_queue.py)
    _save_run_state(run_dir, {
        "phase1_complete":      True,
        "phase1_completed_at":  datetime.now(timezone.utc).isoformat(),
        "phase1_queue_rows":    int(len(df)),
        "phase1_unique_docs":   int(df["doc_id"].nunique()),
        "phase1_n_queries":     n_q,
        "phase1_queue_path":    str(queue_path),
        "phase0_method":        "streaming_gpu_rank",
    })

    queries_path = run_dir / "queries.json"
    if not queries_path.exists():
        with open(queries_path, "w") as f:
            json.dump(queries, f, indent=2)

    logger.info(
        f"[RANK] Done. "
        f"GT found in corpus: {sum(gt_vecs_updated)}/{n_q} | "
        f"RAG failed: {sum(rag_failed.values())}/{n_q}"
    )
    return queue_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 0+1: stream PubMed, embed on GPU, build scoring queue (no HNSW)"
    )
    ap.add_argument("--dataset",        default=DEFAULT_DATASET)
    ap.add_argument("--model",          default=DEFAULT_MODEL)
    ap.add_argument("--embed-batch",    type=int, default=DEFAULT_BATCH)
    ap.add_argument("--top-k",          type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--num-queries",    type=int, default=199)
    ap.add_argument("--pmid-field",     default=DEFAULT_PMID_FIELD)
    ap.add_argument("--title-field",    default=DEFAULT_TITLE_FIELD)
    ap.add_argument("--abstract-field", default=DEFAULT_ABSTRACT_FIELD)
    ap.add_argument("--output-dir",     default=None)
    ap.add_argument("--run-dir",        default=None, help="Specific run dir")
    ap.add_argument("--resume",         action="store_true", help="Resume from checkpoint")
    args = ap.parse_args()

    queue_path = embed_and_rank(
        dataset        = args.dataset,
        model_name     = args.model,
        embed_batch    = args.embed_batch,
        top_k          = args.top_k,
        num_queries    = args.num_queries,
        pmid_field     = args.pmid_field,
        title_field    = args.title_field,
        abstract_field = args.abstract_field,
        output_dir     = args.output_dir,
        run_dir_override = args.run_dir,
        resume         = args.resume,
    )

    df = pd.read_parquet(queue_path)
    print(f"\nScoring queue ready: {queue_path}")
    print(f"  Rows:        {len(df):,}")
    print(f"  Queries:     {df['query_idx'].nunique()}")
    print(f"  Unique docs: {df['doc_id'].nunique():,}")
    if len(df):
        print(f"  GT found:    {int(df.groupby('query_idx')['is_gt'].max().sum())}/{df['query_idx'].nunique()} queries")
        print(f"  RAG failed:  {int((df.groupby('query_idx')['rag_failed'].first()==1).sum())}/{df['query_idx'].nunique()} queries")


if __name__ == "__main__":
    main()
