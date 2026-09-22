"""
analyze_hyde.py  —  HyDE: Hypothetical Document Embeddings

For each query, asks an LLM to write a short hypothetical answer document,
embeds it with the same model used for pool docs, then ranks pool docs by
distance to the hypothetical embedding instead of distance to the query.

Motivation: the query is a geometric outlier in document space (Q_CIRCLE
always empty). A hypothetical *answer document* lives in document space and
should sit closer to the relevant cluster.

Pipeline:
  query text  ->  LLM  ->  hypothetical doc text  ->  embed  ->  rank pool

Comparison:
  Baseline 1 : rank by dist(Q)
  HyDE       : rank by dist(hypothetical_doc)

Hypothetical docs are cached in <OUT_DIR>/hyde_docs.json so re-runs are free.

Usage:
  python analyze_hyde.py
  (requires INFERENCE_API_KEY + INFERENCE_API_URL env vars)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_experiment_data import RUN_DIR, load_data
from embedding_factory import load_model, encode_texts

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility.llm_gateway import complete_prompt
from configurations.config import INFERENCE_API_MODEL

OUT_DIR    = RUN_DIR / "analysis/csv/hyde"
HYDE_CACHE = OUT_DIR / "hyde_docs.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a biomedical expert. Write factual, information-dense answers "
    "in the style of a scientific paper."
)
HYDE_PROMPT = (
    "Write a short biomedical passage (3-5 sentences) that directly and "
    "specifically answers the following question. Write as if it is an "
    "excerpt from a published paper — no introductory phrasing, just the "
    "factual content.\n\nQuestion: {query}"
)

# ── Load experiment data ──────────────────────────────────────────────────────
print("Loading data ...")
data             = load_data()
queries_meta     = data["queries_meta"]
query_idx_to_vec = data["query_idx_to_vec"]
doc_id_to_idx    = data["doc_id_to_idx"]
all_doc_ids      = data["all_doc_ids"]
all_vecs         = data["doc_vecs"]
n_pool           = len(all_doc_ids)
print(f"Pool: {n_pool} docs | Queries: {len(queries_meta)}")

# ── Generate or load hypothetical docs ───────────────────────────────────────
hyde_docs: dict[str, str] = {}
if HYDE_CACHE.exists():
    hyde_docs = json.loads(HYDE_CACHE.read_text())
    print(f"Loaded {len(hyde_docs)} cached HyDE docs from {HYDE_CACHE}")

to_generate = [
    (int(row["query_idx"]), str(row["query_text"]))
    for _, row in queries_meta.iterrows()
    if str(int(row["query_idx"])) not in hyde_docs
]

if to_generate:
    print(f"\nGenerating {len(to_generate)} hypothetical docs "
          f"via {INFERENCE_API_MODEL} ...")
    for j, (q_idx, q_text) in enumerate(to_generate):
        prompt = HYDE_PROMPT.format(query=q_text)
        try:
            doc = complete_prompt(
                provider="inference_api",
                model=INFERENCE_API_MODEL,
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
            )
        except Exception as exc:
            print(f"  [ERROR] query {q_idx}: {exc} — using query text as fallback")
            doc = q_text
        hyde_docs[str(q_idx)] = doc
        # Save incrementally so a crash loses at most one doc
        HYDE_CACHE.write_text(json.dumps(hyde_docs, indent=2, ensure_ascii=False))
        if (j + 1) % 20 == 0 or j == 0:
            print(f"  {j+1}/{len(to_generate)}")
    print("Generation done.")
else:
    print("All HyDE docs already cached — skipping generation.")

# ── Embed hypothetical docs ───────────────────────────────────────────────────
q_idxs   = [int(row["query_idx"]) for _, row in queries_meta.iterrows()]
texts    = [hyde_docs.get(str(qi), "") for qi in q_idxs]

print("\nEmbedding hypothetical docs ...")
model     = load_model()
hyde_vecs = encode_texts(texts, model)   # (n_queries, 768) L2-normalised

# ── Per-query rank comparison ─────────────────────────────────────────────────
print("\nComputing ranks ...")
rows = []

for i, (_, qrow) in enumerate(queries_meta.iterrows()):
    q_idx = int(qrow["query_idx"])
    gt_id = str(qrow["gt_id"])

    gt_idx = doc_id_to_idx.get(gt_id)
    if gt_idx is None:
        print(f"  [SKIP] query {q_idx}: GT not in pool")
        continue

    q_vec    = query_idx_to_vec[q_idx]
    hyde_vec = hyde_vecs[i]

    dist_q    = 1.0 - all_vecs @ q_vec
    dist_hyde = 1.0 - all_vecs @ hyde_vec

    rank_b1   = int(np.where(np.argsort(dist_q)    == gt_idx)[0][0]) + 1
    rank_hyde = int(np.where(np.argsort(dist_hyde)  == gt_idx)[0][0]) + 1

    rows.append({
        "query_idx":      q_idx,
        "rag_failed":     int(qrow["rag_failed"]),
        "rank_baseline1": rank_b1,
        "rank_hyde":      rank_hyde,
        "hyde_better":    int(rank_hyde < rank_b1),
        "b1_better":      int(rank_b1   < rank_hyde),
        "tie":            int(rank_b1  == rank_hyde),
        "delta":          rank_b1 - rank_hyde,   # positive = HyDE improved
    })

# ── Save & print ──────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "hyde_results.csv", index=False)

print("\n" + "=" * 65)
print("HyDE EXPERIMENT  —  RESULTS")
print("=" * 65)
print(f"{'Strategy':<22} {'Mean':>6} {'Med':>5} {'Top-1':>7} {'Top-10':>8} {'Top-50':>8}")
print("-" * 65)
for name, col in [("Baseline 1  dist(Q)", "rank_baseline1"),
                  ("HyDE  dist(hyp_doc)", "rank_hyde")]:
    c = df[col]
    print(f"{name:<22} {c.mean():>6.1f} {c.median():>5.0f}"
          f" {(c<=1).mean()*100:>6.1f}% {(c<=10).mean()*100:>7.1f}%"
          f" {(c<=50).mean()*100:>7.1f}%")

print(f"\nHyDE better : {df['hyde_better'].sum()} / {len(df)} queries "
      f"({df['hyde_better'].mean()*100:.1f}%)")
print(f"B1 better   : {df['b1_better'].sum()} / {len(df)} queries "
      f"({df['b1_better'].mean()*100:.1f}%)")
print(f"Tie         : {df['tie'].sum()} / {len(df)} queries")
print(f"\nMean rank improvement (B1 - HyDE): {df['delta'].mean():.2f} ranks")
print(f"\nOutput -> {OUT_DIR}/")
