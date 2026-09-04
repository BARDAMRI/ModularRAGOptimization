"""
Pool-Doc Centroid Experiment

Simulates a "random centroid" using actual pool documents instead of synthetic
random points (as in analyze_halfpoint_groups.py).

For each query:
  1. Find all pool docs where dist(doc, GT) < r1  (closer to GT than the query is)
  2. Randomly sample up to N_SAMPLES of them as "centroid candidates"
  3. For each candidate doc d:
       - Use d as the circle centre (like the G3 centroid in G2)
       - Set radius = dist(d, GT)   (same convention as G2)
       - Count how many pool docs fall inside this circle
  4. Compare against G1 (centre=query, radius=r1) and G2 (true centroid)

Key question from the supervisor meeting (3.9.26):
  "If you choose a random document closer to GT than Q as the centroid,
   does it give N2 >> N1, or is the true G3 centroid special?"

Output: analysis/csv/pooldoc_centroid/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_experiment_data import RUN_DIR, load_data, cosine_distance

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUT_DIR = RUN_DIR / "analysis/csv/pooldoc_centroid"
PER_QUERY_DIR = OUT_DIR / "per_query"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_QUERY_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 10
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# cosine_distance imported from utility/similarity_calculator via load_experiment_data


def docs_in_circle(vecs: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    return (1.0 - vecs @ center) <= radius


def jaccard_masks(a: np.ndarray, b: np.ndarray) -> float:
    intersection = int((a & b).sum())
    union = int((a | b).sum())
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Load data — shared loader guarantees same 196 queries as original analysis
# ---------------------------------------------------------------------------
data = load_data()
queries_meta     = data["queries_meta"]
scores_df        = data["scores_df"]
query_idx_to_vec = data["query_idx_to_vec"]
doc_id_to_idx    = data["doc_id_to_idx"]
all_doc_ids      = data["all_doc_ids"]
all_vecs         = data["doc_vecs"]
n_pool_global    = len(all_doc_ids)

print("Building LLM score lookup …")
scores_by_query = {}
for q_idx, grp in scores_df.groupby("query_idx"):
    scores_by_query[int(q_idx)] = dict(zip(grp["doc_id"].tolist(), grp["llm_score"].tolist()))

rng = np.random.default_rng(RANDOM_SEED)

# Pre-compute all pairwise distances to GT per query is too expensive.
# Instead compute per-query on the fly.

# ---------------------------------------------------------------------------
# Per-query analysis
# ---------------------------------------------------------------------------
print("\nRunning pool-doc centroid experiment …")

summary_rows = []
candidate_rows = []

for _, qrow in queries_meta.iterrows():
    q_idx = int(qrow["query_idx"])
    gt_id = str(qrow["gt_id"])
    rag_failed = int(qrow["rag_failed"])
    q_vec = query_idx_to_vec[q_idx]

    gt_idx = doc_id_to_idx.get(gt_id)
    if gt_idx is None:
        print(f"  [SKIP] query {q_idx}: GT not in pool")
        continue
    gt_vec = all_vecs[gt_idx]

    # -----------------------------------------------------------------------
    # G1
    # -----------------------------------------------------------------------
    r1 = cosine_distance(q_vec, gt_vec)
    g1_mask = docs_in_circle(all_vecs, q_vec, r1)
    n_g1 = int(g1_mask.sum())

    # -----------------------------------------------------------------------
    # G3 and G2 (true centroid — reference)
    # -----------------------------------------------------------------------
    q_score_lut = scores_by_query.get(q_idx, {})
    g3_mask = np.array([q_score_lut.get(d, 0.0) == 1.0 for d in all_doc_ids])
    g3_indices = np.where(g3_mask)[0]

    if len(g3_indices) > 0:
        centroid = all_vecs[g3_indices].mean(axis=0)
        centroid_norm = centroid / np.linalg.norm(centroid)
        dist_centroid_gt = cosine_distance(centroid_norm, gt_vec)
        g2_mask = docs_in_circle(all_vecs, centroid_norm, dist_centroid_gt)
        n_g2 = int(g2_mask.sum())
        n_g3 = int(g3_mask.sum())
    else:
        dist_centroid_gt = float("nan")
        n_g2 = 0
        n_g3 = 0

    # -----------------------------------------------------------------------
    # Find pool docs closer to GT than the query is (dist < r1), excluding GT
    # -----------------------------------------------------------------------
    dist_all_to_gt = 1.0 - all_vecs @ gt_vec   # (5026,)
    closer_mask = (dist_all_to_gt < r1) & (np.array(all_doc_ids) != gt_id)
    closer_indices = np.where(closer_mask)[0]
    n_closer = len(closer_indices)

    if n_closer == 0:
        # No pool doc is closer to GT than the query — record and skip candidates
        per_q_rows = []
    else:
        # Sample up to N_SAMPLES candidate docs
        sample_size = min(N_SAMPLES, n_closer)
        sampled_indices = rng.choice(closer_indices, size=sample_size, replace=False)

        per_q_rows = []
        for cand_i, pool_idx in enumerate(sampled_indices):
            cand_vec = all_vecs[pool_idx]
            cand_doc_id = all_doc_ids[pool_idx]
            cand_dist_gt = float(dist_all_to_gt[pool_idx])   # = dist(cand, GT)

            # G2-like circle: centre = cand_vec, radius = dist(cand, GT)
            cand_mask = docs_in_circle(all_vecs, cand_vec, cand_dist_gt)
            n_cand = int(cand_mask.sum())
            gt_in_cand = int(cand_mask[gt_idx])
            j_vs_g1 = jaccard_masks(g1_mask, cand_mask)
            j_vs_g2 = jaccard_masks(g2_mask, cand_mask) if n_g2 > 0 else 0.0

            row = {
                "query_idx": q_idx,
                "candidate_idx": cand_i,
                "cand_doc_id": cand_doc_id,
                "r1": r1,
                "cand_dist_gt": cand_dist_gt,      # radius for the candidate circle
                "dist_centroid_gt": dist_centroid_gt,
                "n_g1": n_g1,
                "n_g2_true": n_g2,
                "n_cand_circle": n_cand,
                "gt_in_cand": gt_in_cand,
                "jaccard_vs_g1": j_vs_g1,
                "jaccard_vs_g2": j_vs_g2,
                "density_g1": n_g1 / r1 if r1 > 0 else 0.0,
                "density_cand": n_cand / cand_dist_gt if cand_dist_gt > 0 else 0.0,
                "n_closer_docs": n_closer,
                "rag_failed": rag_failed,
            }
            per_q_rows.append(row)
            candidate_rows.append(row)

    # Per-query CSV
    if per_q_rows:
        pd.DataFrame(per_q_rows).to_csv(PER_QUERY_DIR / f"query_{q_idx:03d}.csv", index=False)

    # Per-query summary
    n_cand_vals = [r["n_cand_circle"] for r in per_q_rows]
    density_cand_vals = [r["density_cand"] for r in per_q_rows]

    summary_rows.append({
        "query_idx": q_idx,
        "rag_failed": rag_failed,
        "r1": r1,
        "n_g1": n_g1,
        "n_g2_true": n_g2,
        "n_g3": n_g3,
        "dist_centroid_gt": dist_centroid_gt,
        "n_closer_docs": n_closer,
        "n_candidates_sampled": len(per_q_rows),
        "mean_n_cand": float(np.mean(n_cand_vals)) if n_cand_vals else float("nan"),
        "median_n_cand": float(np.median(n_cand_vals)) if n_cand_vals else float("nan"),
        "min_n_cand": float(np.min(n_cand_vals)) if n_cand_vals else float("nan"),
        "max_n_cand": float(np.max(n_cand_vals)) if n_cand_vals else float("nan"),
        "std_n_cand": float(np.std(n_cand_vals)) if n_cand_vals else float("nan"),
        "density_g1": n_g1 / r1 if r1 > 0 else 0.0,
        "mean_density_cand": float(np.mean(density_cand_vals)) if density_cand_vals else float("nan"),
        "cand_gt_g1": int(np.mean(n_cand_vals) > n_g1) if n_cand_vals else 0,
        "n_pool": n_pool_global,
    })

    if q_idx % 20 == 0 or q_idx <= 2:
        mean_cand = np.mean(n_cand_vals) if n_cand_vals else float("nan")
        print(
            f"  query {q_idx:3d}: r1={r1:.4f}  G1={n_g1:4d}  "
            f"cand mean={mean_cand:6.1f}  G2_true={n_g2:4d}  "
            f"closer_docs={n_closer}"
        )

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
print("\nSaving results …")

all_cands_df = pd.DataFrame(candidate_rows)
all_cands_df.to_csv(OUT_DIR / "all_candidates.csv", index=False)
print(f"  all_candidates.csv      — {len(all_cands_df)} rows")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "per_query_summary.csv", index=False)
print(f"  per_query_summary.csv   — {len(summary_df)} rows")

# ---------------------------------------------------------------------------
# Global statistics
# ---------------------------------------------------------------------------
valid = summary_df.dropna(subset=["mean_n_cand"])

print("\n" + "=" * 65)
print("POOL-DOC CENTROID EXPERIMENT — GLOBAL SUMMARY")
print("=" * 65)
print(f"\nQueries analysed : {len(summary_df)}")
print(f"  with ≥1 closer doc to GT  : {len(valid)} ({len(valid)/len(summary_df)*100:.1f}%)")
print(f"  with 0 closer docs        : {len(summary_df)-len(valid)}")
print(f"\nAvg closer docs per query  : {summary_df['n_closer_docs'].mean():.1f} "
      f"(median {summary_df['n_closer_docs'].median():.1f})")

print(f"\nGroup sizes (mean / median):")
print(f"  G1  (centre=query, radius=r1)          : "
      f"{summary_df['n_g1'].mean():.1f} / {summary_df['n_g1'].median():.1f}")
print(f"  Cand (centre=pool doc, radius=dist(d,GT)): "
      f"{valid['mean_n_cand'].mean():.1f} / {valid['median_n_cand'].median():.1f}")
print(f"  G2  (true centroid, reference)          : "
      f"{summary_df['n_g2_true'].mean():.1f} / {summary_df['n_g2_true'].median():.1f}")

print(f"\nDensity  docs/radius  (mean):")
print(f"  G1   : {summary_df['density_g1'].mean():.1f}")
print(f"  Cand : {valid['mean_density_cand'].mean():.1f}")

pct_gt_g1 = valid["cand_gt_g1"].mean() * 100
print(f"\nQueries where cand mean > G1 : {pct_gt_g1:.1f}%")
print(f"  (cf. G2_true > G1 in all queries where n_g3 > 0)")

agg = {
    "n_queries": len(summary_df),
    "n_queries_with_closer_docs": len(valid),
    "g1_mean": summary_df["n_g1"].mean(),
    "g1_median": summary_df["n_g1"].median(),
    "cand_mean": valid["mean_n_cand"].mean(),
    "cand_median": valid["median_n_cand"].median(),
    "g2_true_mean": summary_df["n_g2_true"].mean(),
    "g2_true_median": summary_df["n_g2_true"].median(),
    "density_g1_mean": summary_df["density_g1"].mean(),
    "density_cand_mean": valid["mean_density_cand"].mean(),
    "pct_cand_gt_g1": pct_gt_g1,
}
pd.DataFrame([agg]).to_csv(OUT_DIR / "global_aggregates.csv", index=False)
print(f"\nOutputs in: {OUT_DIR}")
print("=" * 65)
