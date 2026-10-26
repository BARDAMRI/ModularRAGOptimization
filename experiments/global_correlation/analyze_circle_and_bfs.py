"""
Two experiments on the group-geometry dataset.

=============================================================================
Experiment 1 — Q_CIRCLE vs GT_CIRCLE  (radius = r1/2)
=============================================================================
  Hypothesis: query embeddings are further from document embeddings than
  documents are from each other → query circle is smaller than GT circle.

  For each query, place two circles of radius r1/2:
    Q_CIRCLE  = { docs : dist(doc, Q)  ≤ r1/2 }
    GT_CIRCLE = { docs : dist(doc, GT) ≤ r1/2 } \ {GT itself}
  Compare sizes across all 196 queries.

=============================================================================
Experiment 2 — BFS rank comparison
=============================================================================
  Hypothesis: best to search mostly by dist(Q), but also a bit by dist(Centroid).

  "BFS with f" = rank all 5035 pool docs by f; report rank of GT.
  Lower rank = fewer iterations = better.

  Strategies (exactly as specified):
    Baseline 1  : f = dist(Q)
    Baseline 2  : f = dist(Centroid)   [oracle — G3 centroid]
    Hybrid 1    : f = alpha*dist(Q) + beta*dist(Centroid),  alpha > beta
    Hybrid 2    : f = (dist(Q), dist(Centroid)) lexicographic
    Hybrid 3    : select k closest to Q → rerank by dist(Centroid)

  Main metric: rank of GT (= number of BFS iterations until GT is found).
  Also reported: % found in top-1, top-10, top-50.

Output: results/cloud_runs/.../analysis/csv/circle_bfs/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_experiment_data import RUN_DIR, load_data, cosine_distance

OUT_DIR = RUN_DIR / "analysis/csv/circle_bfs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hybrid 1: alpha values (all satisfy alpha > beta = 1-alpha, i.e. alpha > 0.5)
ALPHAS_H1 = [0.99, 0.97, 0.95, 0.93, 0.91, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.51]

# ── Hybrid 3: k values
K_VALUES = [10, 20, 50, 100, 200, 500]

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data …")
data             = load_data()
queries_meta     = data["queries_meta"]
scores_df        = data["scores_df"]
query_idx_to_vec = data["query_idx_to_vec"]
doc_id_to_idx    = data["doc_id_to_idx"]
all_doc_ids      = data["all_doc_ids"]
all_vecs         = data["doc_vecs"]          # (5035, 768) L2-normalised
n_pool           = len(all_doc_ids)
print(f"Pool: {n_pool} docs | Queries: {len(queries_meta)}")

# LLM score lookup per query
scores_by_query: dict = {}
for q_idx, grp in scores_df.groupby("query_idx"):
    scores_by_query[int(q_idx)] = dict(zip(
        grp["doc_id"].tolist(), grp["llm_score"].tolist()))

# ── Helpers ──────────────────────────────────────────────────────────────────

def rank_of(f_vals: np.ndarray, target_idx: int) -> int:
    """
    Sort all pool docs by f_vals (ascending) and return the position of
    target_idx in that sorted order. Position 1 = closest / best.

    Example: if target_idx is the 3rd smallest value in f_vals, returns 3.
    """
    return int(np.where(np.argsort(f_vals) == target_idx)[0][0]) + 1


def rank_lex(primary: np.ndarray, secondary: np.ndarray, target_idx: int) -> int:
    """
    Same as rank_of, but sorts by two distance arrays instead of one:
    sort by primary first; break ties using secondary.

    In practice ties almost never occur with continuous float distances,
    so the result is nearly always identical to rank_of(primary, target_idx).
    """
    order = np.lexsort((secondary, primary))   # last key = primary
    return int(np.where(order == target_idx)[0][0]) + 1


def rank_hybrid3(dist_q: np.ndarray, dist_c: np.ndarray,
                 target_idx: int, k: int) -> int:
    """
    Two-stage retrieval: first keep only the k docs closest to Q,
    then re-rank those k docs by distance to the centroid.
    Returns the position of target_idx in the re-ranked list.

    If target_idx is not among the k closest to Q, returns k+1 (penalty).
    """
    top_k = np.argpartition(dist_q, min(k, n_pool - 1))[:k]
    if target_idx not in top_k:
        return k + 1
    reranked = top_k[np.argsort(dist_c[top_k])]
    return int(np.where(reranked == target_idx)[0][0]) + 1


# ── Per-query loop ───────────────────────────────────────────────────────────
print("\nRunning experiments …")

exp1_rows, exp2_rows = [], []
# Loop over queries and compute metrics for both experiments
for i, (_, qrow) in enumerate(queries_meta.iterrows()):
    q_idx = int(qrow["query_idx"])
    gt_id = str(qrow["gt_id"])
    rag_f = int(qrow["rag_failed"])
    q_vec = query_idx_to_vec[q_idx]

    # Get GT index in pool 
    gt_idx = doc_id_to_idx.get(gt_id)
    # Skip queries whose GT is not in the pool (e.g. RAG failed)
    if gt_idx is None:
        print(f"  [SKIP] query {q_idx}: GT not in pool")
        continue

    gt_vec = all_vecs[gt_idx]
    r1     = cosine_distance(q_vec, gt_vec)

    # Distance from every pool doc to Q and GT
    dist_q  = 1.0 - all_vecs @ q_vec
    dist_gt = 1.0 - all_vecs @ gt_vec

    # ── G3 centroid (oracle) ─────────────────────────────────────────────
    # get scores table for this query, as a dict {doc_id: score}
    q_lut   = scores_by_query.get(q_idx, {})
    # Get all perfect scores for this query, and their indices in the pool
    g3_mask = np.array([q_lut.get(d, 0.0) == 1.0 for d in all_doc_ids])
    # Get indices of G3 docs in the pool
    g3_idx  = np.where(g3_mask)[0]

    if len(g3_idx) > 0:
        c_raw         = all_vecs[g3_idx].mean(axis=0)
        c_norm        = c_raw / np.linalg.norm(c_raw)
        dist_c        = 1.0 - all_vecs @ c_norm
        have_centroid = True
        n_g3          = len(g3_idx)
    else:
        dist_c        = None
        have_centroid = False
        n_g3          = 0

    # ════════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: Q_CIRCLE vs GT_CIRCLE  (radius = r1/2)
    # ════════════════════════════════════════════════════════════════════
    radius = r1 / 2.0

    q_circle_mask       = dist_q  <= radius # docs within radius of query. in the form of a boolean mask
    gt_circle_mask      = dist_gt <= radius # docs within radius of GT. in the form of a boolean mask
    gt_circle_mask[gt_idx] = False   # exclude GT itself

    n_q  = int(q_circle_mask.sum())
    n_gt = int(gt_circle_mask.sum())

    exp1_rows.append({
        "query_idx":   q_idx,
        "rag_failed":  rag_f,
        "r1":          r1,
        "radius":      radius,
        "n_q_circle":  n_q,
        "n_gt_circle": n_gt,
        "gt_minus_q":  n_gt - n_q,
        "ratio_gt_q":  (n_gt / n_q) if n_q > 0 else np.nan,
        "n_g3":        n_g3,
    })

    # ════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: BFS rank comparison
    # ════════════════════════════════════════════════════════════════════
    row2 = {
        "query_idx":     q_idx,
        "rag_failed":    rag_f,
        "r1":            r1,
        "n_g3":          n_g3,
        "have_centroid": int(have_centroid),
    }

    # Baseline 1: f = dist(Q)
    row2["rank_query"] = rank_of(dist_q, gt_idx)

    # Baseline 2: f = dist(Centroid)
    row2["rank_centroid"] = rank_of(dist_c, gt_idx) if have_centroid else None

    # Hybrid 1: f = alpha * dist(Q) + beta * dist(Centroid),  alpha > beta
    for alpha in ALPHAS_H1:
        beta = 1.0 - alpha
        # in case centroid wasnt found we need to skip this hybrid and in the results we would like to see that this hybrid was not computed for this query
        if not have_centroid:
            row2[f"rank_weighted_a{int(round(alpha*100)):03d}"] = None
            print(f"  [SKIP] query {q_idx}: no centroid for hybrid 1 with alpha={alpha:.2f}")
            continue
        f = alpha * dist_q + beta * dist_c if have_centroid else None
        col = f"rank_weighted_a{int(round(alpha*100)):03d}"
        row2[col] = rank_of(f, gt_idx)

    # Hybrid 2: lexicographic (dist_Q primary, dist_C secondary)
    row2["rank_lex"] = (
        rank_lex(dist_q, dist_c, gt_idx) if have_centroid else row2["rank_query"]
    )

    # Hybrid 3: top-k by dist(Q), rerank by dist(Centroid)
    for k in K_VALUES:
        col = f"rank_topk_k{k:04d}"
        row2[col] = (
            rank_hybrid3(dist_q, dist_c, gt_idx, k) if have_centroid else row2["rank_query"]
        )

    exp2_rows.append(row2)

    if (i + 1) % 50 == 0 or i == 0:
        print(f"  {i+1}/{len(queries_meta)}")

# ── Save ─────────────────────────────────────────────────────────────────────
exp1_df = pd.DataFrame(exp1_rows)
exp2_df = pd.DataFrame(exp2_rows)
exp1_df.to_csv(OUT_DIR / "exp1_circle_comparison.csv", index=False)
exp2_df.to_csv(OUT_DIR / "exp2_bfs_ranks.csv", index=False)

valid = exp2_df.dropna(subset=["rank_centroid"])

# ── Print Experiment 1 ────────────────────────────────────────────────────────
print("\n" + "="*65)
print("EXPERIMENT 1  —  Q_CIRCLE vs GT_CIRCLE  (radius = r1/2)")
print("="*65)
print(f"Queries analysed       : {len(exp1_df)}")
print(f"Q_CIRCLE  always empty : {(exp1_df['n_q_circle']==0).sum()} / {len(exp1_df)}")
print(f"GT_CIRCLE always empty : {(exp1_df['n_gt_circle']==0).sum()} / {len(exp1_df)}")
print(f"GT_CIRCLE > Q_CIRCLE   : {(exp1_df['n_gt_circle']>exp1_df['n_q_circle']).sum()} queries  "
      f"({(exp1_df['n_gt_circle']>exp1_df['n_q_circle']).mean()*100:.1f}%)")
print(f"Q_CIRCLE  > GT_CIRCLE  : {(exp1_df['n_q_circle']>exp1_df['n_gt_circle']).sum()} queries")
print(f"\nMean  Q_CIRCLE         : {exp1_df['n_q_circle'].mean():.3f}")
print(f"Mean  GT_CIRCLE        : {exp1_df['n_gt_circle'].mean():.3f}")
print(f"Max   GT_CIRCLE        : {exp1_df['n_gt_circle'].max()}")
nz = exp1_df[exp1_df['n_gt_circle']>0]
print(f"GT_CIRCLE mean (non-0) : {nz['n_gt_circle'].mean():.2f}  ({len(nz)} queries)")

# ── Print Experiment 2 ────────────────────────────────────────────────────────
def stats(col):
    c = valid[col].dropna()
    return (c.mean(), c.median(),
            (c<=1).mean()*100, (c<=10).mean()*100, (c<=50).mean()*100)

print("\n" + "="*75)
print("EXPERIMENT 2  —  BFS RANK COMPARISON  (oracle centroid)")
print("="*75)
print(f"{'Strategy':<32} {'Mean':>6} {'Med':>5} {'Top-1':>7} {'Top-10':>8} {'Top-50':>8}")
print("-"*75)

def pr(name, col):
    m,med,t1,t10,t50 = stats(col)
    print(f"{name:<32} {m:>6.1f} {med:>5.0f} {t1:>6.1f}% {t10:>7.1f}% {t50:>7.1f}%")

pr("Baseline 1  f=dist(Q)",         "rank_query")
pr("Baseline 2  f=dist(Centroid)",   "rank_centroid")
print()
for alpha in ALPHAS_H1:
    col = f"rank_weighted_a{int(round(alpha*100)):03d}"
    pr(f"Hybrid 1   α={alpha:.2f} β={1-alpha:.2f}", col)
print()
pr("Hybrid 2  lex(dist_Q, dist_C)", "rank_lex")
print()
for k in K_VALUES:
    pr(f"Hybrid 3  top-{k:<4} → Centroid", f"rank_topk_k{k:04d}")

print(f"\nOutputs → {OUT_DIR}")
