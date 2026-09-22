"""
analyze_llm_rerank.py  —  LLM Re-ranking (offline simulation)

For each query:
  1. Retrieve top-k pool docs by dist(Q)            [standard retrieval]
  2. Re-rank those k docs by existing LLM score     [re-ranking step]
  3. Report rank of GT in the re-ranked list

No new LLM calls — all scores already exist in experiment_results.db.

k values: 5, 10, 20, 50, 100
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_experiment_data import RUN_DIR, load_data

OUT_DIR = RUN_DIR / "analysis/csv/llm_rerank"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [5, 10, 20, 50, 100]

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

# Per-query LLM score lookup
scores_by_query: dict = {}
for q_idx, grp in scores_df.groupby("query_idx"):
    scores_by_query[int(q_idx)] = dict(zip(
        grp["doc_id"].tolist(), grp["llm_score"].tolist()))

# ── Per-query loop ────────────────────────────────────────────────────────────
print("\nRunning re-ranking experiment ...")
rows = []

for i, (_, qrow) in enumerate(queries_meta.iterrows()):
    q_idx  = int(qrow["query_idx"])
    gt_id  = str(qrow["gt_id"])
    q_vec  = query_idx_to_vec[q_idx]

    gt_idx = doc_id_to_idx.get(gt_id)
    if gt_idx is None:
        print(f"  [SKIP] query {q_idx}: GT not in pool")
        continue

    dist_q = 1.0 - all_vecs @ q_vec

    # Baseline 1: rank all docs by dist(Q)
    rank_b1 = int(np.where(np.argsort(dist_q) == gt_idx)[0][0]) + 1

    # LLM score array aligned with pool
    q_lut      = scores_by_query.get(q_idx, {})
    llm_scores = np.array([q_lut.get(d, 0.0) for d in all_doc_ids])

    row = {
        "query_idx":      q_idx,
        "rag_failed":     int(qrow["rag_failed"]),
        "rank_baseline1": rank_b1,
    }

    for k in K_VALUES:
        # Step 1: keep top-k by dist(Q)
        top_k_idx = np.argpartition(dist_q, min(k, n_pool - 1))[:k]

        if gt_idx not in top_k_idx:
            row[f"gt_in_top{k}"]    = 0
            row[f"rank_rerank_k{k}"] = k + 1   # penalty: GT was filtered out
        else:
            row[f"gt_in_top{k}"] = 1
            # Step 2: re-rank those k docs by LLM score (higher = better)
            reranked = top_k_idx[np.argsort(-llm_scores[top_k_idx])]
            rank     = int(np.where(reranked == gt_idx)[0][0]) + 1
            row[f"rank_rerank_k{k}"] = rank

    rows.append(row)
    if (i + 1) % 50 == 0 or i == 0:
        print(f"  {i+1}/{len(queries_meta)}")

# ── Save ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "llm_rerank_results.csv", index=False)

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("LLM RE-RANKING RESULTS")
print("=" * 72)
print(f"{'Strategy':<28} {'Mean':>6} {'Med':>5} {'Top-1':>7} {'Top-10':>8}  GT in top-k")
print("-" * 72)

b1 = df["rank_baseline1"]
print(f"{'Baseline 1  dist(Q)':<28} {b1.mean():>6.1f} {b1.median():>5.0f}"
      f" {(b1<=1).mean()*100:>6.1f}% {(b1<=10).mean()*100:>7.1f}%  (no filter)")

for k in K_VALUES:
    col   = f"rank_rerank_k{k}"
    c     = df[col]
    in_k  = df[f"gt_in_top{k}"].mean() * 100
    label = f"Re-rank k={k}"
    print(f"{label:<28} {c.mean():>6.1f} {c.median():>5.0f}"
          f" {(c<=1).mean()*100:>6.1f}% {(c<=10).mean()*100:>7.1f}%  {in_k:.1f}%")

print(f"\nOutput -> {OUT_DIR}/llm_rerank_results.csv")
