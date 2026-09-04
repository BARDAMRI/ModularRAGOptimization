"""
Halfpoint Groups Experiment

For each query, generates N_ANCHORS random anchor points at cosine distance r1/2
from the GT, where r1 = dist(query, GT). For each anchor, computes a "halfpoint
circle" containing all pool docs within r1/2 of that anchor.

Rationale: G2 uses the G3 centroid (at distance dist_centroid_gt < r1) as its
center and captures more docs than G1 despite a smaller radius. Is this specific
to the centroid's position, or is it a general property of the GT neighbourhood?

This experiment answers that by placing 10 random points at exactly r1/2 from GT
and asking: how many pool docs do they capture compared to G1?

Output: analysis/csv/halfpoint_groups/
  all_anchors.csv        — one row per (query, anchor) = 196 × 10 rows
  per_query_summary.csv  — one row per query, aggregated over 10 anchors
  global_aggregates.csv  — overall statistics
  per_query/query_NNN.csv — per-query anchor detail
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_experiment_data import RUN_DIR, load_data, cosine_distance

OUT_DIR = RUN_DIR / "analysis/csv/halfpoint_groups"
PER_QUERY_DIR = OUT_DIR / "per_query"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_QUERY_DIR.mkdir(parents=True, exist_ok=True)

N_ANCHORS = 10
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_doc_text(s: str) -> str:
    try:
        d = json.loads(s)
        raw = d.get("contexts") or d.get("CONTEXTS") or ""
        return " ".join(str(c) for c in raw if c) if isinstance(raw, list) else str(raw)
    except Exception:
        return str(s)


# cosine_distance imported from utility/similarity_calculator via load_experiment_data


def docs_in_circle(vecs: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Boolean mask: pool docs whose cosine distance from center ≤ radius."""
    return (1.0 - vecs @ center) <= radius


def random_anchor_at_dist(
    gt_vec: np.ndarray, target_dist: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Random unit vector at cosine distance target_dist from gt_vec.

    Constructs the point as:
        P = cos(θ) * GT  +  sin(θ) * u_perp
    where cos(θ) = 1 − target_dist and u_perp is a random unit vector ⊥ GT.

    Verification: dot(P, GT) = cos(θ) = 1 − target_dist
                  cosine_distance(P, GT) = 1 − (1 − target_dist) = target_dist  ✓
    """
    v = rng.standard_normal(gt_vec.shape[0])
    v -= np.dot(v, gt_vec) * gt_vec     # remove GT component (Gram-Schmidt)
    v /= np.linalg.norm(v)              # unit vector perpendicular to GT

    cos_theta = 1.0 - target_dist
    sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta ** 2)))

    anchor = cos_theta * gt_vec + sin_theta * v
    return anchor / np.linalg.norm(anchor)   # re-normalise for numerical safety


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
query_vecs       = data["query_vecs"]   # kept for compatibility

# Pre-build LLM score lookup
print("Building LLM score lookup …")
scores_by_query: dict = {}
for q_idx, grp in scores_df.groupby("query_idx"):
    scores_by_query[int(q_idx)] = dict(zip(grp["doc_id"].tolist(), grp["llm_score"].tolist()))

# ---------------------------------------------------------------------------
# Per-query analysis
# ---------------------------------------------------------------------------
print("Running halfpoint experiment …")
rng = np.random.default_rng(RANDOM_SEED)

summary_rows = []
anchor_rows = []

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

    # ------------------------------------------------------------------
    # G1  —  circle centred on query, radius = dist(query, GT)
    # ------------------------------------------------------------------
    r1 = cosine_distance(q_vec, gt_vec)
    g1_mask = docs_in_circle(all_vecs, q_vec, r1)   # (5026,) bool
    n_g1 = int(g1_mask.sum())

    # ------------------------------------------------------------------
    # G3 and G2 (centroid-based, kept as reference)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Halfpoint anchors — 10 random points at cosine distance r1/2 from GT
    # Each anchor's circle has radius = dist(anchor, GT) = r1/2
    # ------------------------------------------------------------------
    target_dist = r1 / 2.0    # = dist(anchor, GT) by construction

    anchor_ns = []
    gt_pos = next((i for i, d in enumerate(all_doc_ids) if d == gt_id), None)

    per_q_rows = []
    for anchor_i in range(N_ANCHORS):
        anchor = random_anchor_at_dist(gt_vec, target_dist, rng)

        # Verify: cosine_distance(anchor, gt_vec) should equal target_dist
        actual_anchor_dist = cosine_distance(anchor, gt_vec)

        hp_mask = docs_in_circle(all_vecs, anchor, target_dist)
        n_hp = int(hp_mask.sum())
        gt_in_hp = int(hp_mask[gt_pos]) if gt_pos is not None else 0
        j_vs_g1 = jaccard_masks(g1_mask, hp_mask)

        anchor_ns.append(n_hp)

        row = {
            "query_idx": q_idx,
            "anchor_idx": anchor_i,
            "r1": r1,
            "target_dist": target_dist,
            "anchor_dist_to_gt": actual_anchor_dist,   # sanity check ≈ target_dist
            "n_g1": n_g1,
            "n_halfpoint": n_hp,
            "gt_in_halfpoint": gt_in_hp,
            "jaccard_halfpoint_vs_g1": j_vs_g1,
            "density_g1": n_g1 / r1 if r1 > 0 else 0.0,
            "density_halfpoint": n_hp / target_dist if target_dist > 0 else 0.0,
            "rag_failed": rag_failed,
        }
        per_q_rows.append(row)
        anchor_rows.append(row)

    # Per-query CSV (N_ANCHORS rows)
    pd.DataFrame(per_q_rows).to_csv(PER_QUERY_DIR / f"query_{q_idx:03d}.csv", index=False)

    # Per-query aggregates
    mean_n_hp = float(np.mean(anchor_ns))
    median_n_hp = float(np.median(anchor_ns))
    density_g1 = n_g1 / r1 if r1 > 0 else 0.0
    mean_density_hp = float(np.mean([n / target_dist for n in anchor_ns])) if target_dist > 0 else 0.0

    summary_rows.append({
        "query_idx": q_idx,
        "rag_failed": rag_failed,
        "r1": r1,
        "target_dist": target_dist,
        "n_g1": n_g1,
        "n_g2_centroid": n_g2,
        "n_g3": n_g3,
        "dist_centroid_gt": dist_centroid_gt,
        "mean_n_halfpoint": mean_n_hp,
        "median_n_halfpoint": median_n_hp,
        "min_n_halfpoint": float(np.min(anchor_ns)),
        "max_n_halfpoint": float(np.max(anchor_ns)),
        "std_n_halfpoint": float(np.std(anchor_ns)),
        "density_g1": density_g1,
        "mean_density_halfpoint": mean_density_hp,
        "halfpoint_gt_g1": int(mean_n_hp > n_g1),
        "n_pool": n_pool_global,
    })

    if q_idx % 20 == 0 or q_idx <= 2:
        print(
            f"  query {q_idx:3d}: r1={r1:.4f}  G1={n_g1:4d}  "
            f"halfpoint mean={mean_n_hp:6.1f}  G2_centroid={n_g2:4d}"
        )

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
print("\nSaving results …")

all_anchors_df = pd.DataFrame(anchor_rows)
all_anchors_df.to_csv(OUT_DIR / "all_anchors.csv", index=False)
print(f"  all_anchors.csv          — {len(all_anchors_df)} rows")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "per_query_summary.csv", index=False)
print(f"  per_query_summary.csv    — {len(summary_df)} rows")

# ---------------------------------------------------------------------------
# Global statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("HALFPOINT EXPERIMENT — GLOBAL SUMMARY")
print("=" * 60)
print(f"\nQueries analysed : {len(summary_df)}")
print(f"\nGroup sizes (mean / median) — full pool = {n_pool_global} docs:")
print(f"  G1  (center=query,  radius=r1)   : "
      f"{summary_df['n_g1'].mean():6.1f} / {summary_df['n_g1'].median():6.1f}")
print(f"  HP  (center=anchor, radius=r1/2) : "
      f"{summary_df['mean_n_halfpoint'].mean():6.1f} / {summary_df['median_n_halfpoint'].median():6.1f}")
print(f"  G2  (centroid-based, reference)  : "
      f"{summary_df['n_g2_centroid'].mean():6.1f} / {summary_df['n_g2_centroid'].median():6.1f}")

print(f"\nDensity  docs / radius  (mean / median):")
print(f"  G1 : {summary_df['density_g1'].mean():8.1f} / {summary_df['density_g1'].median():8.1f}")
print(f"  HP : {summary_df['mean_density_halfpoint'].mean():8.1f} / "
      f"{summary_df['mean_density_halfpoint'].median():.1f}")

pct_hp_gt_g1 = summary_df["halfpoint_gt_g1"].mean() * 100
print(f"\nQueries where halfpoint mean > G1 : {pct_hp_gt_g1:.1f}%")

agg = {
    "n_queries": len(summary_df),
    "n_pool": n_pool_global,
    "g1_mean": summary_df["n_g1"].mean(),
    "g1_median": summary_df["n_g1"].median(),
    "hp_mean": summary_df["mean_n_halfpoint"].mean(),
    "hp_median": summary_df["median_n_halfpoint"].median(),
    "g2_centroid_mean": summary_df["n_g2_centroid"].mean(),
    "g2_centroid_median": summary_df["n_g2_centroid"].median(),
    "density_g1_mean": summary_df["density_g1"].mean(),
    "density_hp_mean": summary_df["mean_density_halfpoint"].mean(),
    "pct_queries_hp_gt_g1": pct_hp_gt_g1,
}
pd.DataFrame([agg]).to_csv(OUT_DIR / "global_aggregates.csv", index=False)

print(f"\nOutputs in: {OUT_DIR}")
print("=" * 60)
