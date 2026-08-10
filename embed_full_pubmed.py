#!/usr/bin/env python3
"""
embed_full_pubmed.py — Phase 0 (Cloud / GPU)
=============================================
Downloads full PubMed from HuggingFace in streaming mode and embeds all
documents into a Chroma collection at:
    storage/chroma/pubmed_full/cosine/

Uses sentence_transformers directly (not llama_index) for 10× faster
GPU-batched embedding.  Fully resumable via a checkpoint file.

Resulting collection is compatible with build_scoring_queue.py via:
    python build_scoring_queue.py --chroma-source pubmed_full

Usage
-----
    python embed_full_pubmed.py
    python embed_full_pubmed.py --dataset ncbi/pubmed --embed-batch 256
    python embed_full_pubmed.py --resume          # continue interrupted run
    python embed_full_pubmed.py --dry-run 1000    # embed first 1 000 docs only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from utility.logger import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_STORAGE = PROJECT_ROOT / "storage" / "chroma" / "pubmed_full" / "cosine"
COLLECTION_NAME = "collection_pubmed_full"
CHECKPOINT_FILE = CHROMA_STORAGE / "embed_progress.json"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATASET = "ncbi/pubmed"
DEFAULT_SPLIT = "train"
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_EMBED_BATCH = 2048     # docs sent to GPU at once (tune to VRAM)
CHROMA_UPSERT_BATCH = 5_000   # docs per collection.upsert() call
CHECKPOINT_EVERY = 50_000     # save checkpoint every N docs


# ---------------------------------------------------------------------------
# Schema extractors
# ---------------------------------------------------------------------------

def _extract_ncbi_pubmed(record: dict) -> tuple[str | None, str, str]:
    """
    Extract (pmid, title, abstract) from a raw ncbi/pubmed HF record.
    The dataset stores MEDLINE XML as nested dicts.
    """
    try:
        citation = record.get("MedlineCitation", {})
        pmid_field = citation.get("PMID", {})
        if isinstance(pmid_field, dict):
            pmid = pmid_field.get("#text") or pmid_field.get("$", "")
        else:
            pmid = str(pmid_field)

        article = citation.get("Article", {})
        title = article.get("ArticleTitle", "") or ""
        if isinstance(title, dict):
            title = title.get("#text") or title.get("$", "") or ""

        abstract_field = article.get("Abstract", {})
        if isinstance(abstract_field, dict):
            abstract_text = abstract_field.get("AbstractText", "")
        else:
            abstract_text = abstract_field or ""

        if isinstance(abstract_text, list):
            parts = []
            for part in abstract_text:
                if isinstance(part, dict):
                    parts.append(part.get("#text") or part.get("$", "") or "")
                else:
                    parts.append(str(part))
            abstract = " ".join(p for p in parts if p)
        else:
            abstract = str(abstract_text) if abstract_text else ""

        return pmid or None, str(title).strip(), abstract.strip()
    except Exception:
        return None, "", ""


def _extract_flat(record: dict, pmid_field: str, title_field: str, abstract_field: str) -> tuple[str | None, str, str]:
    """Extract from a flat-field HF record (e.g. uiyunkim-hub/pubmed-abstract)."""
    pmid = record.get(pmid_field)
    pmid = str(pmid) if pmid is not None else None
    title = str(record.get(title_field, "") or "").strip()
    abstract = str(record.get(abstract_field, "") or "").strip()
    return pmid, title, abstract


def _build_doc_text(title: str, abstract: str) -> str:
    """Wrap text in the JSON format expected by _extract_contexts_text."""
    combined = f"{title}. {abstract}".strip(" .")
    return json.dumps({"contexts": combined})


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _load_checkpoint() -> int:
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text())
            return int(data.get("indexed", 0))
        except Exception:
            pass
    return 0


def _save_checkpoint(indexed: int) -> None:
    CHECKPOINT_FILE.write_text(json.dumps({"indexed": indexed, "ts": time.time()}))


# ---------------------------------------------------------------------------
# Core embedding loop
# ---------------------------------------------------------------------------

def embed_full_pubmed(
    *,
    dataset: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    model_name: str = DEFAULT_MODEL,
    embed_batch: int = DEFAULT_EMBED_BATCH,
    resume: bool = True,
    dry_run: int | None = None,
    pmid_field: str = "PMID",
    title_field: str = "ArticleTitle",
    abstract_field: str = "AbstractText",
    schema: str = "auto",
) -> None:
    # --- imports deferred so the module loads fast ---
    import torch
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings

    # --- GPU / device ---
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"[EMBED] Device: {device}")

    # --- Load embedding model ---
    logger.info(f"[EMBED] Loading model: {model_name}")
    embed_model = SentenceTransformer(model_name, device=device)
    embedding_dim = embed_model.get_sentence_embedding_dimension()
    logger.info(f"[EMBED] Embedding dim: {embedding_dim}")

    # --- Open Chroma collection ---
    CHROMA_STORAGE.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_STORAGE))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:M": 16,
            "hnsw:construction_ef": 100,
            "hnsw:sync_threshold": 1_000_000,  # sync every 1M docs instead of every upsert
            "hnsw:batch_size": 1_000_000,
        },
    )
    existing_count = collection.count()
    logger.info(f"[EMBED] Collection '{COLLECTION_NAME}' — existing docs: {existing_count:,}")

    # --- Resume support ---
    skip_docs = 0
    if resume:
        skip_docs = _load_checkpoint()
        if skip_docs:
            logger.info(f"[EMBED] Resuming from checkpoint: {skip_docs:,} docs already indexed")

    # --- Load dataset (local dir = fast; HF Hub = streaming fallback) ---
    import os, glob as _glob
    if os.path.isdir(dataset):
        logger.info(f"[EMBED] Loading from local directory: {dataset}")
        parquet_files = sorted(_glob.glob(os.path.join(dataset, "**", "*.parquet"), recursive=True))
        jsonl_files   = sorted(_glob.glob(os.path.join(dataset, "**", "*.jsonl"),   recursive=True))
        from datasets import load_dataset
        if parquet_files:
            logger.info(f"[EMBED] Found {len(parquet_files)} parquet files")
            ds = load_dataset("parquet", data_files=parquet_files, split="train", streaming=True)
        elif jsonl_files:
            logger.info(f"[EMBED] Found {len(jsonl_files)} jsonl files — converting to slim Parquet for fast GPU feed")
            import pyarrow.json as _paj
            import pyarrow.parquet as _pq
            _slim_dir = os.path.join(os.path.dirname(dataset.rstrip("/")), "pubmed_parquet_slim")
            os.makedirs(_slim_dir, exist_ok=True)
            _need_cols = [f for f in [pmid_field, title_field, abstract_field] if f]
            _t = time.time()
            _done = 0
            for _jf in jsonl_files:
                _pf = os.path.join(_slim_dir, Path(_jf).stem + ".parquet")
                if os.path.exists(_pf):
                    _done += 1
                    continue
                if os.path.getsize(_jf) == 0:
                    _done += 1
                    continue
                try:
                    _tbl = _paj.read_json(_jf)
                except Exception as _e:
                    logger.warning(f"[EMBED] Skipping {_jf}: {_e}")
                    _done += 1
                    continue
                _cols = [c for c in _need_cols if c in _tbl.schema.names]
                _pq.write_table(_tbl.select(_cols), _pf, compression="snappy")
                _done += 1
                if _done % 100 == 0 or _done <= 3:
                    _elapsed = time.time() - _t
                    _rate = _done / _elapsed if _elapsed > 0 else 1
                    _eta = (len(jsonl_files) - _done) / _rate / 60
                    logger.info(f"[EMBED] JSONL→Parquet: {_done}/{len(jsonl_files)} | ETA: {_eta:.1f} min")
            logger.info(f"[EMBED] Conversion done in {(time.time()-_t)/60:.1f} min. Switching to Parquet.")
            parquet_files = sorted(_glob.glob(os.path.join(_slim_dir, "*.parquet")))
            ds = load_dataset("parquet", data_files=parquet_files, split="train", streaming=True)
        else:
            raise FileNotFoundError(f"No .parquet or .jsonl files found under {dataset}")
    else:
        logger.info(f"[EMBED] Streaming dataset from Hub: {dataset} / {split}")
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset, split=split, streaming=True, trust_remote_code=True)
        except Exception as exc:
            logger.error(f"[EMBED] Failed to load dataset: {exc}")
            raise

    # --- Schema detection ---
    if schema == "auto":
        if "ncbi/pubmed" in dataset.lower() or "pubmed" == dataset.lower():
            schema = "ncbi_pubmed"
        else:
            schema = "flat"
    logger.info(f"[EMBED] Schema: {schema}")

    def _extract(record: dict) -> tuple[str | None, str, str]:
        if schema == "ncbi_pubmed":
            return _extract_ncbi_pubmed(record)
        return _extract_flat(record, pmid_field, title_field, abstract_field)

    _first_record_logged = False

    # --- Embedding loop ---
    buffer_ids: list[str] = []
    buffer_texts: list[str] = []
    buffer_pmids: list[int | None] = []

    total_indexed = 0
    total_skipped = 0
    total_empty = 0
    t0 = time.time()
    seen = 0

    def _flush_buffer() -> None:
        nonlocal total_indexed
        if not buffer_texts:
            return

        embeddings = embed_model.encode(
            buffer_texts,
            batch_size=embed_batch,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        # Build Chroma-compatible doc strings and metadatas
        documents = [_build_doc_text("", text) for text in buffer_texts]
        # Chroma rejects empty {} metadata; always supply at least PMID key
        metadatas = [{"PMID": pmid} if pmid is not None else {"PMID": -1} for pmid in buffer_pmids]

        # Upsert in sub-batches to avoid memory spikes
        for i in range(0, len(buffer_ids), CHROMA_UPSERT_BATCH):
            collection.upsert(
                ids=buffer_ids[i: i + CHROMA_UPSERT_BATCH],
                embeddings=embeddings[i: i + CHROMA_UPSERT_BATCH],
                documents=documents[i: i + CHROMA_UPSERT_BATCH],
                metadatas=metadatas[i: i + CHROMA_UPSERT_BATCH],
            )

        total_indexed += len(buffer_ids)
        buffer_ids.clear()
        buffer_texts.clear()
        buffer_pmids.clear()

    for record in ds:
        seen += 1

        if dry_run is not None and seen > dry_run:
            break

        # Skip already-indexed docs when resuming
        if seen <= skip_docs:
            continue

        pmid, title, abstract = _extract(record)

        if not _first_record_logged:
            _first_record_logged = True
            logger.info(f"[EMBED] First record keys: {list(record.keys())}")
            logger.info(f"[EMBED] First record pmid={pmid!r} title={str(title)[:60]!r}")

        if not abstract and not title:
            total_empty += 1
            continue

        if pmid is None:
            # Fallback ID: sequential
            doc_id = f"doc_{seen}"
            pmid_int = None
        else:
            doc_id = str(pmid).strip()
            try:
                pmid_int = int(pmid)
            except ValueError:
                # Handle chunked IDs like "21645374_0" (MedRAG format: PMID_chunkindex)
                base = pmid.split("_")[0]
                try:
                    pmid_int = int(base)
                except ValueError:
                    pmid_int = None

        text = f"{title}. {abstract}".strip(" .")
        buffer_ids.append(doc_id)
        buffer_texts.append(text)
        buffer_pmids.append(pmid_int)

        if len(buffer_ids) >= embed_batch:
            _flush_buffer()

            if total_indexed % CHECKPOINT_EVERY < embed_batch:
                _save_checkpoint(skip_docs + total_indexed)
                elapsed = time.time() - t0
                rate = total_indexed / elapsed if elapsed > 0 else 0
                logger.info(
                    f"[EMBED] {total_indexed:,} indexed | "
                    f"{total_skipped:,} skipped | "
                    f"{rate:.0f} docs/s | "
                    f"collection size: {collection.count():,}"
                )

    _flush_buffer()

    _save_checkpoint(skip_docs + total_indexed)
    elapsed = time.time() - t0

    logger.info(
        f"[EMBED] Done. Indexed: {total_indexed:,} | "
        f"Empty: {total_empty:,} | "
        f"Time: {elapsed/60:.1f} min | "
        f"Collection total: {collection.count():,}"
    )
    print(f"\nEmbedding complete.")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Path       : {CHROMA_STORAGE}")
    print(f"  Total docs : {collection.count():,}")
    print(f"  Time       : {elapsed/60:.1f} min")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0: embed full PubMed into Chroma (GPU-accelerated)"
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help=f"HuggingFace dataset name (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--split", default=DEFAULT_SPLIT,
        help=f"Dataset split (default: {DEFAULT_SPLIT})"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Embedding model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--embed-batch", type=int, default=DEFAULT_EMBED_BATCH,
        help=f"GPU embedding batch size (default: {DEFAULT_EMBED_BATCH})"
    )
    parser.add_argument(
        "--schema", default="auto", choices=["auto", "ncbi_pubmed", "flat"],
        help="Dataset field schema (default: auto-detect)"
    )
    parser.add_argument(
        "--pmid-field", default="PMID",
        help="Field name for PMID in flat schema (default: PMID)"
    )
    parser.add_argument(
        "--title-field", default="ArticleTitle",
        help="Field name for title in flat schema (default: ArticleTitle)"
    )
    parser.add_argument(
        "--abstract-field", default="AbstractText",
        help="Field name for abstract in flat schema (default: AbstractText)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start from scratch even if checkpoint exists"
    )
    parser.add_argument(
        "--dry-run", type=int, default=None, metavar="N",
        help="Embed only first N documents (for testing)"
    )
    args = parser.parse_args()

    embed_full_pubmed(
        dataset=args.dataset,
        split=args.split,
        model_name=args.model,
        embed_batch=args.embed_batch,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        pmid_field=args.pmid_field,
        title_field=args.title_field,
        abstract_field=args.abstract_field,
        schema=args.schema,
    )


if __name__ == "__main__":
    main()
