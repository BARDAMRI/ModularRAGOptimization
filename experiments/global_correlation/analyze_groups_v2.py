"""
Group Analysis v2 — refactored from analyze_groups.py.

Identical computation, restructured into functions:
  cosine_distance() — imported from utility/similarity_calculator
  docs_in_circle()  — shared membership test for G1 and G2
  compute_g3()      — score-based G3 membership
  compute_centroid()— L2-normalised centroid of G3

Output goes to analysis/csv/group_analysis_v2/ for comparison with original.
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "utility"))
from similarity_calculator import cosine_distance

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RUN_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "results/cloud_runs/staged_run_2026-08-03_08-32-45"
)
OUT_DIR = RUN_DIR / "analysis/csv/group_analysis_v2"
PER_QUERY_DIR = OUT_DIR / "per_query"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_QUERY_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_doc_text(doc_text_json: str) -> str:
    """Parse {"contexts": "..."} JSON stored in parquet → plain text."""
    try:
        d = json.loads(doc_text_json)
        raw = d.get("contexts") or d.get("CONTEXTS") or ""
        if isinstance(raw, list):
            return " ".join(str(c) for c in raw if c)
        return str(raw)
    except Exception:
        return str(doc_text_json)


# cosine_distance imported from utility/similarity_calculator


def docs_in_circle(
    vecs_q: np.ndarray,
    doc_ids: list,
    center_vec: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, set]:
    """
    Shared membership function for G1 and G2.

    Returns (flags, id_set) for every pool doc whose cosine distance
    from center_vec is ≤ radius.

    vecs_q     : (n_pool, dim) — L2-normalised pool embeddings for this query.
    center_vec : (dim,)        — L2-normalised centre (query vec for G1,
                                 centroid for G2).
    radius     : float         — dist(center, GT).
    """
    distances = 1.0 - vecs_q @ center_vec          # (n_pool,) cosine distances
    flags = (distances <= radius).astype(int)
    return flags, set(np.array(doc_ids)[flags.astype(bool)])


def compute_g3(
    q_scores: pd.DataFrame,
    doc_ids_q: list,
) -> tuple[np.ndarray, set]:
    """G3: all docs with llm_score == 1.0 (score-based, no geometry)."""
    flags = (q_scores["llm_score"].values == 1.0).astype(int)
    return flags, set(np.array(doc_ids_q)[flags.astype(bool)])


def compute_centroid(vecs_q: np.ndarray, g3_flags: np.ndarray) -> np.ndarray | None:
    """L2-normalised mean of G3 embeddings. Returns None when G3 is empty."""
    indices = np.where(g3_flags)[0]
    if len(indices) == 0:
        return None
    centroid = vecs_q[indices].mean(axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm > 0 else None


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading scoring_queue.parquet …")
sq = pd.read_parquet(RUN_DIR / "scoring_queue.parquet",
                     columns=["query_idx", "query_text", "gt_id", "doc_id", "doc_text", "is_gt"])

print("Loading experiment_results.db …")
conn = sqlite3.connect(RUN_DIR / "experiment_results.db")
scores_df = pd.read_sql(
    "SELECT query_idx, doc_id, llm_score, dist_to_gt, is_gt, rag_failed FROM results",
    conn
)
conn.close()
print(f"  Loaded {len(scores_df):,} scored pairs")

# ---------------------------------------------------------------------------
# Unique queries
# ---------------------------------------------------------------------------
queries_meta = (
    sq[["query_idx", "query_text", "gt_id"]]
    .drop_duplicates("query_idx")
    .sort_values("query_idx")
    .reset_index(drop=True)
)
print(f"Unique queries: {len(queries_meta)}")

# ---------------------------------------------------------------------------
# Unique pool docs
# ---------------------------------------------------------------------------
pool_docs = sq[["doc_id", "doc_text"]].drop_duplicates("doc_id").reset_index(drop=True)
print(f"Unique pool docs: {len(pool_docs)}")

gt_ids_all = set(queries_meta["gt_id"].tolist())
pool_doc_ids = set(pool_docs["doc_id"].tolist())
missing_gts = gt_ids_all - pool_doc_ids

if missing_gts:
    print(f"GT docs missing from pool ({len(missing_gts)}), fetching from expanded_scoring_queue …")
    esq = pd.read_parquet(
        RUN_DIR / "expanded_scoring_queue.parquet",
        columns=["doc_id", "doc_text", "is_gt"]
    )
    gt_rows = esq[esq["is_gt"] == 1][["doc_id", "doc_text"]].drop_duplicates("doc_id")
    gt_rows = gt_rows[gt_rows["doc_id"].isin(missing_gts)]
    pool_docs = pd.concat([pool_docs, gt_rows], ignore_index=True).drop_duplicates("doc_id")
    del esq
    print(f"  Pool docs after adding missing GTs: {len(pool_docs)}")

# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------
print(f"\nLoading model: {MODEL_NAME} …")
model = SentenceTransformer(MODEL_NAME)

print("Embedding queries …")
query_vecs = model.encode(
    queries_meta["query_text"].tolist(),
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True,
)

print("Embedding pool documents …")
doc_texts_raw = [extract_doc_text(t) for t in pool_docs["doc_text"].tolist()]
doc_vecs = model.encode(
    doc_texts_raw,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True,
)

query_idx_to_vec: dict[int, np.ndarray] = dict(
    zip(queries_meta["query_idx"].tolist(), query_vecs)
)
doc_id_to_vec: dict[str, np.ndarray] = dict(
    zip(pool_docs["doc_id"].tolist(), doc_vecs)
)

# ---------------------------------------------------------------------------
# Per-query group analysis
# ---------------------------------------------------------------------------
print("\nRunning per-query group analysis …")

summary_rows = []

for _, qrow in queries_meta.iterrows():
    q_idx = int(qrow["query_idx"])
    gt_id = str(qrow["gt_id"])
    q_vec = query_idx_to_vec[q_idx]
    gt_vec = doc_id_to_vec.get(gt_id)

    if gt_vec is None:
        print(f"  [SKIP] query {q_idx}: GT embedding not available")
        continue

    q_scores = scores_df[scores_df["query_idx"] == q_idx].copy()
    if q_scores.empty:
        continue

    rag_failed = int(q_scores["rag_failed"].iloc[0])

    valid_mask = q_scores["doc_id"].isin(doc_id_to_vec)
    q_scores = q_scores[valid_mask].copy()
    doc_ids_q = q_scores["doc_id"].tolist()
    n_pool = len(doc_ids_q)

    if n_pool == 0:
        continue

    vecs_q = np.stack([doc_id_to_vec[d] for d in doc_ids_q])  # (n_pool, dim)

    dist_q_gt = cosine_distance(q_vec, gt_vec)
    dist_doc_to_gt = 1.0 - vecs_q @ gt_vec   # kept for per-query CSV

    # --- G1 and G3 ---
    g1_flags, g1_doc_ids = docs_in_circle(vecs_q, doc_ids_q, q_vec, dist_q_gt)
    g3_flags, g3_doc_ids = compute_g3(q_scores, doc_ids_q)

    # --- G2 ---
    centroid_norm = compute_centroid(vecs_q, g3_flags)
    if centroid_norm is not None:
        dist_centroid_gt = cosine_distance(centroid_norm, gt_vec)
        g2_flags, g2_doc_ids = docs_in_circle(vecs_q, doc_ids_q, centroid_norm, dist_centroid_gt)
    else:
        dist_centroid_gt = None
        g2_flags = np.zeros(n_pool, dtype=int)
        g2_doc_ids = set()

    # --- Overlaps ---
    g1_g2 = g1_doc_ids & g2_doc_ids
    g1_g3 = g1_doc_ids & g3_doc_ids
    g2_g3 = g2_doc_ids & g3_doc_ids
    g1_g2_g3 = g1_doc_ids & g2_doc_ids & g3_doc_ids

    gt_in_g1 = int(gt_id in g1_doc_ids)
    gt_in_g2 = int(gt_id in g2_doc_ids)
    gt_in_g3 = int(gt_id in g3_doc_ids)

    # --- Per-query detailed CSV ---
    dist_to_centroid = (1.0 - vecs_q @ centroid_norm) if centroid_norm is not None else np.nan
    per_query_df = pd.DataFrame({
        "doc_id":            doc_ids_q,
        "llm_score":         q_scores["llm_score"].values,
        "dist_to_gt_db":     q_scores["dist_to_gt"].values,
        "dist_to_gt_reembed": dist_doc_to_gt,
        "dist_to_query":     1.0 - vecs_q @ q_vec,
        "dist_to_centroid":  dist_to_centroid,
        "is_gt":             q_scores["is_gt"].values,
        "in_g1":             g1_flags,
        "in_g2":             g2_flags,
        "in_g3":             g3_flags,
    })
    per_query_df["in_g1_and_g2"] = (per_query_df["in_g1"] & per_query_df["in_g2"]).astype(int)
    per_query_df["in_g1_and_g3"] = (per_query_df["in_g1"] & per_query_df["in_g3"]).astype(int)
    per_query_df["in_g2_and_g3"] = (per_query_df["in_g2"] & per_query_df["in_g3"]).astype(int)
    per_query_df["in_all"] = (per_query_df["in_g1"] & per_query_df["in_g2"] & per_query_df["in_g3"]).astype(int)
    per_query_df.sort_values("llm_score", ascending=False, inplace=True)
    per_query_df.to_csv(PER_QUERY_DIR / f"query_{q_idx:03d}.csv", index=False)

    summary_rows.append({
        "query_idx":        q_idx,
        "gt_id":            gt_id,
        "rag_failed":       rag_failed,
        "n_pool":           n_pool,
        "dist_query_to_gt":    dist_q_gt,
        "dist_centroid_to_gt": dist_centroid_gt,
        "n_g1":  len(g1_doc_ids),
        "n_g2":  len(g2_doc_ids),
        "n_g3":  len(g3_doc_ids),
        "n_g1_g2":    len(g1_g2),
        "n_g1_g3":    len(g1_g3),
        "n_g2_g3":    len(g2_g3),
        "n_g1_g2_g3": len(g1_g2_g3),
        "jaccard_g1_g2":    jaccard(g1_doc_ids, g2_doc_ids),
        "jaccard_g1_g3":    jaccard(g1_doc_ids, g3_doc_ids),
        "jaccard_g2_g3":    jaccard(g2_doc_ids, g3_doc_ids),
        "jaccard_g1_g2_g3": len(g1_g2_g3) / len(g1_doc_ids | g2_doc_ids | g3_doc_ids)
                            if (g1_doc_ids | g2_doc_ids | g3_doc_ids) else 0.0,
        "gt_in_g1":  gt_in_g1,
        "gt_in_g2":  gt_in_g2,
        "gt_in_g3":  gt_in_g3,
        "gt_in_all": int(gt_in_g1 and gt_in_g2 and gt_in_g3),
        "g1_doc_ids": "|".join(sorted(str(d) for d in g1_doc_ids)),
        "g2_doc_ids": "|".join(sorted(str(d) for d in g2_doc_ids)),
        "g3_doc_ids": "|".join(sorted(str(d) for d in g3_doc_ids)),
    })

    if q_idx % 20 == 0:
        print(f"  query {q_idx}: G1={len(g1_doc_ids)}, G2={len(g2_doc_ids)}, G3={len(g3_doc_ids)}"
              f"  GT∈G1={gt_in_g1} G2={gt_in_g2} G3={gt_in_g3}")

# ---------------------------------------------------------------------------
# Master summary CSV
# ---------------------------------------------------------------------------
master_df = pd.DataFrame(summary_rows)
master_df.to_csv(OUT_DIR / "master_summary.csv", index=False)
print(f"\nSaved master_summary.csv ({len(master_df)} rows)")

numeric_cols = [
    "n_pool", "n_g1", "n_g2", "n_g3",
    "n_g1_g2", "n_g1_g3", "n_g2_g3", "n_g1_g2_g3",
    "jaccard_g1_g2", "jaccard_g1_g3", "jaccard_g2_g3", "jaccard_g1_g2_g3",
    "gt_in_g1", "gt_in_g2", "gt_in_g3", "gt_in_all",
    "dist_query_to_gt", "dist_centroid_to_gt",
]
agg = master_df[numeric_cols].agg(["mean", "median", "std", "min", "max"])
agg.index.name = "stat"
agg.reset_index().to_csv(OUT_DIR / "global_aggregates.csv", index=False)
print("Saved global_aggregates.csv")

print("\n=== GLOBAL SUMMARY ===")
print(f"Queries analysed        : {len(master_df)}")
for col in ["n_g1", "n_g2", "n_g3"]:
    print(f"  {col} mean/median      : {master_df[col].mean():.1f} / {master_df[col].median():.1f}")
for col in ["jaccard_g1_g2", "jaccard_g1_g3", "jaccard_g2_g3"]:
    print(f"  {col} mean            : {master_df[col].mean():.3f}")
print(f"  GT in G1 (%)          : {master_df['gt_in_g1'].mean()*100:.1f}%")
print(f"  GT in G2 (%)          : {master_df['gt_in_g2'].mean()*100:.1f}%")
print(f"  GT in G3 (%)          : {master_df['gt_in_g3'].mean()*100:.1f}%")
print(f"  GT in all 3 (%)       : {master_df['gt_in_all'].mean()*100:.1f}%")
print(f"\nPer-query CSVs saved to : {PER_QUERY_DIR}")
print(f"Master summary          : {OUT_DIR / 'master_summary.csv'}")
