"""
embedding_factory.py — single source of truth for all embedding operations.

All experiment scripts must import from here instead of defining their own
model name, extract_doc_text, or encode calls. This guarantees that every
script uses identical model, parameters, and text pre-processing.

Usage:
    from embedding_factory import MODEL_NAME, load_model, encode_texts, extract_doc_text
    from embedding_factory import save_embeddings, load_embeddings

    model     = load_model()                    # or load_model(device="cuda")
    doc_texts = [extract_doc_text(t) for t in raw_texts]
    vecs      = encode_texts(doc_texts, model)  # (N, 768) float32, L2-normalised

Caching:
    # First run: compute and cache
    emb = load_embeddings(run_dir, query_ids, doc_ids)
    if emb is None:
        model = load_model()
        query_vecs = encode_texts(query_texts, model)
        doc_vecs   = encode_texts(doc_texts,   model)
        save_embeddings(run_dir, query_vecs, doc_vecs, query_ids, doc_ids)
    else:
        query_vecs, doc_vecs = emb
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_NAME    = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE    = 64
VECTOR_DIM    = 768
_CACHE_SUBDIR = "embeddings"


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_doc_text(s: str) -> str:
    """
    Unwrap the {"contexts": ...} JSON stored in parquet → plain text string.
    Falls back to str(s) if the input is not valid JSON or has no contexts key.
    """
    try:
        d   = json.loads(s)
        raw = d.get("contexts") or d.get("CONTEXTS") or ""
        return " ".join(str(c) for c in raw if c) if isinstance(raw, list) else str(raw)
    except Exception:
        return str(s)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(device: str | None = None):
    """
    Load and return the SentenceTransformer model.

    Args:
        device: "cpu", "cuda", "mps", or None (auto-detect).
    Returns:
        SentenceTransformer instance ready for encode().
    """
    from sentence_transformers import SentenceTransformer  # lazy import — heavy dependency

    kwargs = {"device": device} if device is not None else {}
    return SentenceTransformer(MODEL_NAME, **kwargs)


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_texts(
    texts: list[str],
    model,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """
    Encode a list of strings with the shared parameters.

    Always uses:
        batch_size           = BATCH_SIZE (64)
        normalize_embeddings = True  → L2-unit vectors, cosine = dot product
        convert_to_numpy     = True

    Returns:
        np.ndarray of shape (len(texts), VECTOR_DIM), dtype float32.
    """
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )


# ── Embedding cache (.npy) ────────────────────────────────────────────────────

def _cache_dir(run_dir: Path) -> Path:
    return Path(run_dir) / _CACHE_SUBDIR


def save_embeddings(
    run_dir: Path,
    query_vecs: np.ndarray,
    doc_vecs: np.ndarray,
    query_ids: list,
    doc_ids: list[str],
) -> None:
    """
    Persist embeddings to <run_dir>/embeddings/.

    Files written:
        query_vecs.npy   (n_queries, 768) float32
        doc_vecs.npy     (n_docs,    768) float32
        query_ids.json   ordered list of query_idx ints
        doc_ids.json     ordered list of doc_id strings
        meta.json        model name + shapes (for cache validation)
    """
    d = _cache_dir(run_dir)
    d.mkdir(parents=True, exist_ok=True)

    np.save(d / "query_vecs.npy", query_vecs.astype(np.float32))
    np.save(d / "doc_vecs.npy",   doc_vecs.astype(np.float32))
    (d / "query_ids.json").write_text(json.dumps(list(query_ids)))
    (d / "doc_ids.json").write_text(json.dumps(list(doc_ids)))
    (d / "meta.json").write_text(json.dumps({
        "model":      MODEL_NAME,
        "n_queries":  int(query_vecs.shape[0]),
        "n_docs":     int(doc_vecs.shape[0]),
        "vector_dim": int(query_vecs.shape[1]),
    }, indent=2))


def load_embeddings(
    run_dir: Path,
    query_ids: list,
    doc_ids: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load cached embeddings if they exist and are valid.

    Validates:
      - all cache files present
      - model name matches MODEL_NAME
      - query_ids and doc_ids order matches exactly

    Returns:
        (query_vecs, doc_vecs) on cache hit, or None on any mismatch/miss.
    """
    d = _cache_dir(run_dir)
    required = ["query_vecs.npy", "doc_vecs.npy", "query_ids.json", "doc_ids.json", "meta.json"]
    if not all((d / f).exists() for f in required):
        return None

    meta = json.loads((d / "meta.json").read_text())
    if meta.get("model") != MODEL_NAME:
        return None

    cached_query_ids = json.loads((d / "query_ids.json").read_text())
    cached_doc_ids   = json.loads((d / "doc_ids.json").read_text())
    if cached_query_ids != list(query_ids) or cached_doc_ids != list(doc_ids):
        return None

    query_vecs = np.load(d / "query_vecs.npy")
    doc_vecs   = np.load(d / "doc_vecs.npy")
    return query_vecs, doc_vecs
