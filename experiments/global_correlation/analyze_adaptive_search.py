"""
analyze_adaptive_search.py  —  Adaptive Sequential Search

Problem (from meeting notes):
  Input:  pool d_1..d_n, subset d_1..d_m already LLM-scored, LLM(d_i) per scored doc
  Output: which document to check with the LLM NEXT

This is a sequential decision problem — each LLM call is expensive, and we want
to find GT with the fewest calls possible. The metric is not "rank of GT in a
pre-sorted list" but "how many LLM calls until GT is found."

Baseline (greedy dist Q):
  Always pick the unscored doc closest to Q.
  Calls to find GT = rank_b1 (mean 3.3 in our data).

Adaptive policy:
  Same as baseline until the first G3 doc is found (LLM score >= 1.0).
  After that: shift the search toward the centroid of all G3 docs found so far.
  Score(d) = alpha * sim(d, Q) + (1-alpha) * sim(d, G3_centroid)
  Pick the highest-scoring unscored doc next.

Key difference from PRF:
  PRF computed centroid from k docs, then re-ranked all 5035.
  Here we update the selection score AFTER EVERY SINGLE LLM CALL,
  using only the G3 docs discovered so far — truly adaptive.

All simulated offline using existing LLM scores.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_experiment_data import RUN_DIR, load_data

OUT_DIR = RUN_DIR / "analysis/csv/adaptive_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]   # 1.0 = pure dist(Q) = baseline

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
data             = load_data()
queries_meta     = data["queries_meta"]
scores_df        = data["scores_df"]
query_idx_to_vec = data["query_idx_to_vec"]
doc_id_to_idx    = data["doc_id_to_idx"]
all_doc_ids      = data["all_doc_ids"]
all_vecs         = data["doc_vecs"]
n_pool           = len(all_doc_ids)
print(f"Pool: {n_pool} docs | Queries: {len(queries_meta)}")

scores_by_query: dict = {}
for q_idx, grp in scores_df.groupby("query_idx"):
    scores_by_query[int(q_idx)] = dict(zip(
        grp["doc_id"].tolist(), grp["llm_score"].tolist()))


def simulate(sim_q: np.ndarray, llm_scores: np.ndarray,
             gt_idx: int, alpha: float) -> dict:
    """
    Simulate adaptive sequential search. Returns:
      calls_to_find  : number of LLM calls until GT is scored
      g3_found_at    : call number when first G3 doc was found (or -1)
      g3_count       : total G3 docs found before GT
    """
    # sim_q[i] = cosine similarity of doc i with Q  (higher = more promising)
    scored    = np.zeros(n_pool, dtype=bool)
    g3_vecs   = []          # accumulate found G3 doc vectors
    centroid  = None

    for call in range(1, n_pool + 1):
        # --- Select next doc ---
        if centroid is None or alpha == 1.0:
            # No G3 found yet (or pure baseline): use sim(Q) only
            sel_scores = np.where(~scored, sim_q, -np.inf)
        else:
            sim_c      = all_vecs @ centroid          # sim to current G3 centroid
            sel_scores = np.where(~scored,
                                  alpha * sim_q + (1 - alpha) * sim_c,
                                  -np.inf)

        chosen = int(np.argmax(sel_scores))
        scored[chosen] = True

        if chosen == gt_idx:
            return {"calls": call,
                    "g3_before_gt": len(g3_vecs),
                    "centroid_active": centroid is not None}

        # --- Update centroid if new G3 doc found ---
        if llm_scores[chosen] >= 1.0:
            g3_vecs.append(all_vecs[chosen])
            raw      = np.array(g3_vecs).mean(axis=0)
            norm     = np.linalg.norm(raw)
            centroid = raw / norm if norm > 1e-10 else centroid

    return {"calls": n_pool, "g3_before_gt": len(g3_vecs),
            "centroid_active": centroid is not None}


# ── Per-query simulation ──────────────────────────────────────────────────────
print("\nSimulating adaptive search ...")
rows = []

for i, (_, qrow) in enumerate(queries_meta.iterrows()):
    q_idx  = int(qrow["query_idx"])
    gt_id  = str(qrow["gt_id"])
    q_vec  = query_idx_to_vec[q_idx]

    gt_idx = doc_id_to_idx.get(gt_id)
    if gt_idx is None:
        print(f"  [SKIP] query {q_idx}")
        continue

    sim_q      = all_vecs @ q_vec   # cosine similarity (L2-norm → = 1 - dist)
    q_lut      = scores_by_query.get(q_idx, {})
    llm_scores = np.array([q_lut.get(d, 0.0) for d in all_doc_ids])

    row = {"query_idx": q_idx, "rag_failed": int(qrow["rag_failed"])}

    for alpha in ALPHAS:
        result = simulate(sim_q, llm_scores, gt_idx, alpha)
        row[f"calls_a{int(alpha*10):02d}"] = result["calls"]
        if alpha == 0.5:
            row["g3_before_gt"] = result["g3_before_gt"]

    rows.append(row)
    if (i + 1) % 50 == 0 or i == 0:
        print(f"  {i+1}/{len(queries_meta)}")

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "adaptive_search_results.csv", index=False)

# ── Print results ─────────────────────────────────────────────────────────────
alpha_col = {1.0: "calls_a10", 0.9: "calls_a09", 0.7: "calls_a07",
             0.5: "calls_a05", 0.3: "calls_a03", 0.1: "calls_a01"}

print("\n" + "=" * 85)
print("ADAPTIVE SEQUENTIAL SEARCH — calls to find GT (lower = better)")
print("=" * 85)
print(f"{'Strategy':<35} {'Mean':>6} {'Med':>5} {'<=1':>7} {'<=3':>7} {'<=5':>7} {'<=10':>7}")
print("-" * 85)

for alpha in ALPHAS:
    col   = alpha_col[alpha]
    c     = df[col]
    label = f"alpha={alpha:.1f}  {'← baseline' if alpha==1.0 else ''}"
    print(f"{label:<35} {c.mean():>6.2f} {c.median():>5.0f}"
          f" {(c<=1).mean()*100:>6.1f}%"
          f" {(c<=3).mean()*100:>6.1f}%"
          f" {(c<=5).mean()*100:>6.1f}%"
          f" {(c<=10).mean()*100:>6.1f}%")

# Best alpha
best_alpha = min(ALPHAS, key=lambda a: df[alpha_col[a]].mean())
best_col   = alpha_col[best_alpha]
base_col   = alpha_col[1.0]
improvement = df[base_col].mean() - df[best_col].mean()
print(f"\nBest alpha: {best_alpha}  (mean calls: {df[best_col].mean():.2f} "
      f"vs baseline {df[base_col].mean():.2f} — delta: {improvement:+.2f})")

# Breakdown: how much does adaptive help on hard cases?
hard = df[df[base_col] > 1]   # queries where baseline needs >1 call
easy = df[df[base_col] == 1]
print(f"\nBreakdown on HARD queries (GT not rank 1 under dist(Q)): {len(hard)} queries")
print(f"{'Strategy':<35} {'Mean':>6} {'<=1':>7} {'<=3':>7} {'<=5':>7}")
print("-" * 60)
for alpha in [1.0, 0.9, 0.7, 0.5, 0.3]:
    col = alpha_col[alpha]
    c   = hard[col]
    label = f"alpha={alpha:.1f}"
    print(f"{label:<35} {c.mean():>6.2f}"
          f" {(c<=1).mean()*100:>6.1f}%"
          f" {(c<=3).mean()*100:>6.1f}%"
          f" {(c<=5).mean()*100:>6.1f}%")

print(f"\nOutput -> {OUT_DIR}/adaptive_search_results.csv")
