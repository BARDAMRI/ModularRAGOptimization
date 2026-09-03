"""
Group Analysis — 3 geometric groups per query, pairwise overlap.

Groups (in embedding space):
  G1: docs within circle centered at QUERY, radius = dist(query, GT)
  G2: docs within circle centered at CENTROID of G3, radius = dist(centroid, GT)
  G3: all docs with llm_score == 1.0

Output:
  analysis/csv/group_analysis/per_query/query_<idx>.csv  — per-doc membership
  analysis/csv/group_analysis/master_summary.csv          — per-query stats + global aggregates
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RUN_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "results/cloud_runs/staged_run_2026-08-03_08-32-45"
)
OUT_DIR = RUN_DIR / "analysis/csv/group_analysis"
PER_QUERY_DIR = OUT_DIR / "per_query"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_QUERY_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_doc_text(doc_text_json: str) -> str:
    """Parse {"contexts": "..."} JSON stored in parquet → plain text for encoding."""
    try:
        d = json.loads(doc_text_json)
        raw = d.get("contexts") or d.get("CONTEXTS") or ""
        if isinstance(raw, list):
            return " ".join(str(c) for c in raw if c)
        return str(raw)
    except Exception:
        return str(doc_text_json)


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two L2-normalised vectors (= 1 - dot)."""
    return float(1.0 - np.dot(a, b))


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
# Unique pool docs  (union of all top-200 retrieval results)
# ---------------------------------------------------------------------------
pool_docs = sq[["doc_id", "doc_text"]].drop_duplicates("doc_id").reset_index(drop=True)
print(f"Unique pool docs: {len(pool_docs)}")

# Some GT docs might not be in the pool (rag_failed queries where GT wasn't retrieved).
# Find any GT not in pool and collect its text from the scored pairs (expanded).
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
query_texts_raw = queries_meta["query_text"].tolist()
query_vecs = model.encode(
    query_texts_raw,
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

# Build lookup maps
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

    # All scored docs for this query
    q_scores = scores_df[scores_df["query_idx"] == q_idx].copy()
    if q_scores.empty:
        continue

    rag_failed = int(q_scores["rag_failed"].iloc[0])

    # Filter to docs we have embeddings for
    valid_mask = q_scores["doc_id"].isin(doc_id_to_vec)
    q_scores = q_scores[valid_mask].copy()
    doc_ids_q = q_scores["doc_id"].tolist()
    n_pool = len(doc_ids_q)

    if n_pool == 0:
        continue

    # Stack embeddings for this query's pool docs
    vecs_q = np.stack([doc_id_to_vec[d] for d in doc_ids_q])  # (n_pool, dim)

    # dist(query, GT) = cosine distance (both L2-normalised → 1 - dot)
    dist_q_gt = cosine_dist(q_vec, gt_vec)

    # dist(doc_i, query) for each pool doc
    dist_doc_to_query = 1.0 - vecs_q @ q_vec  # (n_pool,)

    # dist(doc_i, GT) for each pool doc  (use pre-computed from DB where available)
    dist_doc_to_gt = 1.0 - vecs_q @ gt_vec  # (n_pool,)

    # --- G1: dist(doc, query) ≤ dist(query, GT) ---
    g1_flags = (dist_doc_to_query <= dist_q_gt).astype(int)
    g1_doc_ids = set(np.array(doc_ids_q)[g1_flags.astype(bool)])

    # --- G3: llm_score == 1.0 ---
    g3_flags = (q_scores["llm_score"].values == 1.0).astype(int)
    g3_doc_ids = set(np.array(doc_ids_q)[g3_flags.astype(bool)])

    # --- G2: centroid of G3, radius = dist(centroid, GT) ---
    g3_indices = np.where(g3_flags)[0]
    if len(g3_indices) > 0:
        g3_vecs = vecs_q[g3_indices]
        centroid = g3_vecs.mean(axis=0)
        centroid_norm = centroid / np.linalg.norm(centroid)
        dist_centroid_gt = cosine_dist(centroid_norm, gt_vec)
        dist_doc_to_centroid = 1.0 - vecs_q @ centroid_norm
        g2_flags = (dist_doc_to_centroid <= dist_centroid_gt).astype(int)
        g2_doc_ids = set(np.array(doc_ids_q)[g2_flags.astype(bool)])
    else:
        centroid_norm = None
        dist_centroid_gt = None
        g2_flags = np.zeros(n_pool, dtype=int)
        g2_doc_ids = set()

    # --- Overlaps ---
    def jaccard(a: set, b: set) -> float:
        union = a | b
        return len(a & b) / len(union) if union else 0.0

    g1_g2 = g1_doc_ids & g2_doc_ids
    g1_g3 = g1_doc_ids & g3_doc_ids
    g2_g3 = g2_doc_ids & g3_doc_ids
    g1_g2_g3 = g1_doc_ids & g2_doc_ids & g3_doc_ids

    gt_in_g1 = int(gt_id in g1_doc_ids)
    gt_in_g2 = int(gt_id in g2_doc_ids)
    gt_in_g3 = int(gt_id in g3_doc_ids)

    # --- Per-query detailed CSV ---
    per_query_df = pd.DataFrame({
        "doc_id": doc_ids_q,
        "llm_score": q_scores["llm_score"].values,
        "dist_to_gt_db": q_scores["dist_to_gt"].values,
        "dist_to_gt_reembed": dist_doc_to_gt,
        "dist_to_query": dist_doc_to_query,
        "dist_to_centroid": (1.0 - vecs_q @ centroid_norm) if centroid_norm is not None else np.nan,
        "is_gt": q_scores["is_gt"].values,
        "in_g1": g1_flags,
        "in_g2": g2_flags,
        "in_g3": g3_flags,
    })
    per_query_df["in_g1_and_g2"] = (per_query_df["in_g1"] & per_query_df["in_g2"]).astype(int)
    per_query_df["in_g1_and_g3"] = (per_query_df["in_g1"] & per_query_df["in_g3"]).astype(int)
    per_query_df["in_g2_and_g3"] = (per_query_df["in_g2"] & per_query_df["in_g3"]).astype(int)
    per_query_df["in_all"] = (per_query_df["in_g1"] & per_query_df["in_g2"] & per_query_df["in_g3"]).astype(int)
    per_query_df.sort_values("llm_score", ascending=False, inplace=True)
    per_query_df.to_csv(PER_QUERY_DIR / f"query_{q_idx:03d}.csv", index=False)

    # --- Summary row ---
    summary_rows.append({
        "query_idx": q_idx,
        "gt_id": gt_id,
        "rag_failed": rag_failed,
        "n_pool": n_pool,
        "dist_query_to_gt": dist_q_gt,
        "dist_centroid_to_gt": dist_centroid_gt,
        # group sizes
        "n_g1": len(g1_doc_ids),
        "n_g2": len(g2_doc_ids),
        "n_g3": len(g3_doc_ids),
        # pairwise intersections
        "n_g1_g2": len(g1_g2),
        "n_g1_g3": len(g1_g3),
        "n_g2_g3": len(g2_g3),
        "n_g1_g2_g3": len(g1_g2_g3),
        # Jaccard
        "jaccard_g1_g2": jaccard(g1_doc_ids, g2_doc_ids),
        "jaccard_g1_g3": jaccard(g1_doc_ids, g3_doc_ids),
        "jaccard_g2_g3": jaccard(g2_doc_ids, g3_doc_ids),
        "jaccard_g1_g2_g3": len(g1_g2_g3) / len(g1_doc_ids | g2_doc_ids | g3_doc_ids) if (g1_doc_ids | g2_doc_ids | g3_doc_ids) else 0.0,
        # GT membership
        "gt_in_g1": gt_in_g1,
        "gt_in_g2": gt_in_g2,
        "gt_in_g3": gt_in_g3,
        "gt_in_all": int(gt_in_g1 and gt_in_g2 and gt_in_g3),
        # doc_id lists (pipe-separated for inspection)
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

# ---------------------------------------------------------------------------
# Global aggregate stats (appended as extra rows)
# ---------------------------------------------------------------------------
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

# Print summary to console
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
print(f"Global aggregates       : {OUT_DIR / 'global_aggregates.csv'}")
