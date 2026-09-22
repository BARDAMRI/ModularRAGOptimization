"""
analyze_iterative_prf.py  —  Iterative Pseudo-Relevance Feedback (PRF)

Motivation (from previous experiments):
  - Baseline 1 dist(Q):  Top-1=70.4%  — good starting point
  - G3 centroid is 44% closer to GT than Q  — centroid is a better anchor
  - But centroid is an oracle (requires LLM scores on all 5035 docs)
  - LLM re-ranking collapses Top-1 to 32-40% due to score quantization
  - Goal: use a small LLM budget (score only top-k docs) to discover
    a good centroid, then re-retrieve from the full pool with it.

The "Semantle-style" loop:
  Round 0 : rank all docs by dist(Q)
  Round r  : take top-k by current ranking
             → simulate LLM scoring those k docs (use existing scores)
             → build centroid of high-scored docs (score >= threshold)
             → if centroid found: re-rank all 5035 docs by dist(centroid)
             → else: stay with previous ranking
  Repeat for R rounds (default 3).

Key difference from re-ranking:
  Re-ranking: re-orders k docs among themselves (score quantization kills it)
  PRF:        uses k docs to *update the anchor vector*, then searches all 5035
              → escapes the k-doc window entirely

All data is offline — no new LLM calls.

k values tested: 5, 10, 20, 50
Rounds: up to 3
Score threshold for centroid: >= 0.5 (G2-style), == 1.0 (G3-style)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_experiment_data import RUN_DIR, load_data

OUT_DIR = RUN_DIR / "analysis/csv/iterative_prf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES   = [5, 10, 20, 50]
N_ROUNDS   = 3
THRESHOLDS = {"g3": 1.0, "g2": 0.5}   # LLM score threshold to include in centroid

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

# Per-query LLM score lookup (already in DB for all 5035 docs)
scores_by_query: dict = {}
for q_idx, grp in scores_df.groupby("query_idx"):
    scores_by_query[int(q_idx)] = dict(zip(
        grp["doc_id"].tolist(), grp["llm_score"].tolist()))


def rank_of(dist: np.ndarray, target_idx: int) -> int:
    return int(np.where(np.argsort(dist) == target_idx)[0][0]) + 1


def build_centroid(candidate_idx: np.ndarray, vecs: np.ndarray) -> np.ndarray | None:
    """L2-normalised mean of candidate vectors. Returns None if empty."""
    if len(candidate_idx) == 0:
        return None
    raw = vecs[candidate_idx].mean(axis=0)
    norm = np.linalg.norm(raw)
    if norm < 1e-10:
        return None
    return raw / norm


# ── Per-query loop ────────────────────────────────────────────────────────────
print("\nRunning iterative PRF experiment ...")
rows = []

for i, (_, qrow) in enumerate(queries_meta.iterrows()):
    q_idx = int(qrow["query_idx"])
    gt_id = str(qrow["gt_id"])
    q_vec = query_idx_to_vec[q_idx]

    gt_idx = doc_id_to_idx.get(gt_id)
    if gt_idx is None:
        print(f"  [SKIP] query {q_idx}")
        continue

    # Full LLM score array aligned with pool
    q_lut      = scores_by_query.get(q_idx, {})
    llm_scores = np.array([q_lut.get(d, 0.0) for d in all_doc_ids])

    row = {
        "query_idx":      q_idx,
        "rag_failed":     int(qrow["rag_failed"]),
        "rank_baseline1": rank_of(1.0 - all_vecs @ q_vec, gt_idx),
    }

    for k in K_VALUES:
        for thr_name, threshold in THRESHOLDS.items():
            col_prefix = f"k{k:03d}_{thr_name}"

            # Start from dist(Q)
            current_dist = 1.0 - all_vecs @ q_vec
            anchor       = q_vec.copy()

            for r in range(1, N_ROUNDS + 1):
                # Step 1: sample top-k docs by current distance (simulates LLM call budget)
                top_k_idx = np.argpartition(current_dist, min(k, n_pool - 1))[:k]

                # Step 2: find high-scored docs among the top-k
                high_scored = top_k_idx[llm_scores[top_k_idx] >= threshold]

                # Step 3: build centroid of high-scored docs
                centroid = build_centroid(high_scored, all_vecs)

                if centroid is not None:
                    # Step 4: re-rank ALL 5035 docs by dist(centroid)
                    current_dist = 1.0 - all_vecs @ centroid
                    anchor       = centroid
                # else: no update — stay with current ranking

                rank_r = rank_of(current_dist, gt_idx)
                row[f"{col_prefix}_r{r}"] = rank_r
                row[f"{col_prefix}_r{r}_n_high"] = len(high_scored)

    rows.append(row)
    if (i + 1) % 50 == 0 or i == 0:
        print(f"  {i+1}/{len(queries_meta)}")

# ── Save ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "prf_results.csv", index=False)

# ── Print results ─────────────────────────────────────────────────────────────
b1 = df["rank_baseline1"]

print("\n" + "=" * 80)
print("ITERATIVE PRF RESULTS")
print("=" * 80)
print(f"{'Strategy':<36} {'Mean':>6} {'Med':>5} {'Top-1':>7} {'Top-10':>8} {'Top-50':>8}")
print("-" * 80)
print(f"{'Baseline 1  dist(Q)':<36} {b1.mean():>6.1f} {b1.median():>5.0f}"
      f" {(b1<=1).mean()*100:>6.1f}% {(b1<=10).mean()*100:>7.1f}%"
      f" {(b1<=50).mean()*100:>7.1f}%")

for k in K_VALUES:
    for thr_name in THRESHOLDS:
        col_prefix = f"k{k:03d}_{thr_name}"
        for r in range(1, N_ROUNDS + 1):
            col = f"{col_prefix}_r{r}"
            if col not in df.columns:
                continue
            c        = df[col]
            n_hi_col = f"{col_prefix}_r{r}_n_high"
            n_hi_avg = df[n_hi_col].mean() if n_hi_col in df.columns else float("nan")
            label    = f"PRF k={k:>3} thr={thr_name} round={r}"
            print(f"{label:<36} {c.mean():>6.1f} {c.median():>5.0f}"
                  f" {(c<=1).mean()*100:>6.1f}% {(c<=10).mean()*100:>7.1f}%"
                  f" {(c<=50).mean()*100:>7.1f}%  (avg high-scored: {n_hi_avg:.1f})")

print(f"\nOutput -> {OUT_DIR}/prf_results.csv")
