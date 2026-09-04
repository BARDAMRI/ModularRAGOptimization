# Why Centroid-Based Retrieval Groups Consistently Outperform Query-Centered Groups

Project: ModularRAGOptimization  
Run: staged_run_2026-08-03_08-32-45  
Date: 2026-08-25  
Queries: 196 · Pool: 5,035 documents  
Embedding model: sentence-transformers/all-mpnet-base-v2

---

## Overview

This document reports three linked geometric-group experiments that explain why the
centroid-based retrieval group (G2) consistently captures more relevant documents than
the query-centered group (G1).

### Group definitions

| Group | Center | Radius | Membership criterion |
|-------|--------|--------|---------------------|
| G1 | Query vector Q | r₁ = cosine_dist(Q, GT) | cosine_dist(doc, Q) ≤ r₁ |
| G2 | Normalized centroid of G3 | r₂ = cosine_dist(centroid, GT) | cosine_dist(doc, centroid) ≤ r₂ |
| G3 | - | - | LLM score = 1.0 |

All distances are cosine distances over L2-normalized embeddings: `dist(a, b) = 1 − dot(a, b)`.

---

## Experiment 1 - G1 vs G2 vs G3 Group Analysis

Script: `experiments/global_correlation/analyze_groups.py`  
Output: `analysis/csv/group_analysis/`

### Results

| Metric | G1 | G2 | G3 |
|--------|----|----|-----|
| Mean size (docs) | 3.13 | 9.72 | 17.05 |
| Median size (docs) | 1.0 | 6.0 | 18.0 |
| GT included (%) | 84.7% | 85.2% | 94.9% |
| Mean density (docs/radius) | 15.4 | 92.5 | - |
| Mean center–GT distance | 0.173 | 0.098 | - |

Pairwise overlaps (mean Jaccard):

| Pair | Jaccard |
|------|---------|
| G1 ∩ G2 | 0.326 |
| G1 ∩ G3 | 0.125 |
| G2 ∩ G3 | 0.336 |

G2 > G1 (more docs): 152 / 196 queries = 77.6%

### Key observation

G2 is approximately 3× larger than G1 on average and has 6× higher density (92.5 vs 15.4 docs/radius). The centroid is also much closer to GT (0.098 vs 0.173 mean cosine distance). This raises two questions:

1. Is the GT neighborhood inherently denser (so any point near GT gives a large group)?
2. Is the pool structurally concentrated around GT?

---

## Experiment 2 - Halfpoint Experiment (Random Space Points)

Script: `experiments/global_correlation/analyze_halfpoint_groups.py`  
Output: `analysis/csv/halfpoint_groups/`

Design: For each of 196 queries, 10 random unit vectors are generated at cosine
distance r₁/2 from GT (using Gram-Schmidt orthogonalization). Each anchor's circle
has radius r₁/2 (same as its distance to GT - matching the G2 convention).

Construction:

```
u_perp = random_normal − (dot(random_normal, GT)) * GT   # orthogonal component
u_perp /= ||u_perp||                                      # normalize
anchor = cos(θ) * GT + sin(θ) * u_perp                   # where cos(θ) = 1 − target_dist
```

### Results

| Group | Mean size | Median size | Mean density |
|-------|-----------|-------------|--------------|
| G1 (query, r₁) | 3.18 | 1.0 | 15.6 |
| HP (random anchor, r₁/2) | 0.82 | 1.0 | 11.0 |
| G2 (centroid, reference) | 9.78 | 6.0 | - |

Queries where HP mean > G1: 7.1% (14/196)

### Conclusion

Random points at the same distance from GT as the centroid give fewer docs (0.82) than
G1 (3.18), not more. The GT neighborhood is not inherently denser - arbitrary
directions near GT yield sparse circles. The centroid's advantage is not a general
property of proximity to GT.

---

## Experiment 3 - Pool-Document Centroid Experiment

Script: `experiments/global_correlation/analyze_pooldoc_centroid.py`  
Output: `analysis/csv/pooldoc_centroid/`

Design: For each query, find all pool documents where `dist(doc, GT) < r₁`
(closer to GT than the query). Sample up to 10 of them. Use each as a circle
center with radius = `dist(doc, GT)` - identical convention to G2.

This tests whether the pool itself is dense around GT, independently of the G3 centroid.

### Results

| Group | Mean size | Median size | Mean density |
|-------|-----------|-------------|--------------|
| G1 (query, r₁) | 3.18 | 1.0 | 15.6 |
| Pool-doc candidate | 10.12 | 6.5 | 59.6 |
| G2 true centroid | 9.78 | 6.0 | - |

Queries where 159/196 (81.1%) have ≥1 pool doc closer to GT than Q.  
Queries where candidate mean > G1: 91.8% (146/159 valid queries)

### Conclusion

Pool documents closer to GT than Q give 10.12 docs on average - almost exactly
matching G2's 9.78. The pool is structurally concentrated near GT: any real pool
document close to GT as an anchor yields a dense circle, not just the centroid.

---

## Combined Interpretation

The three experiments together resolve why G2 >> G1:

### Cause 1: Query-Document Gap

Queries and documents occupy different regions of embedding space. The retriever
ranks GT as the nearest document to Q (by cosine distance), so almost nothing in the
pool is closer to Q than GT - G1 collapses to 1–3 docs on average.

Meanwhile, GT is surrounded by other semantically similar documents in document
space. The pool naturally clusters near GT.

### Cause 2: The Pool is Structured Around GT

The document pool is not uniformly distributed in embedding space. Documents
with high LLM relevance scores (G3) form a coherent cluster near GT. Pool
documents within this cluster capture many neighbors when used as circle centers
(10.12 docs on average), explaining why G2 >> G1.

### Cause 3: The Centroid Amplifies the Effect

The G3 centroid is the center of mass of the relevant document cluster. It
minimizes total distance to all G3 members, landing at cosine distance 0.098 from
GT (vs 0.173 for the query). This gives the centroid:

- A smaller radius r₂ (0.098 vs 0.173) - but this smaller circle still captures
  more documents because it's positioned inside the dense document cluster
- The highest density (92.5 vs 59.6 for random pool docs vs 15.4 for G1)

The centroid is not unique in capturing many documents - any pool doc near GT does
similarly well - but it is the optimal center for the cluster by construction.

### Density Summary

| Anchor | Mean docs | Mean radius | Docs/radius |
|--------|-----------|-------------|-------------|
| Query Q (G1) | 3.18 | 0.173 | 15.6 |
| Random point at r₁/2 from GT | 0.82 | 0.087 | 11.0 |
| Pool doc closer to GT than Q | 10.12 | ~0.060–0.100 | 59.6 |
| G3 centroid (G2) | 9.78 | 0.098 | 92.5 |

---

## Conclusions

1. G2 systematically outperforms G1 (77.6% of queries, ~3× more docs on average)
   because the centroid anchors the circle inside the dense document cluster, not in
   query space.

2. The GT neighborhood is not uniformly dense. Random points at equal distance
   from GT capture only 0.82 docs on average - fewer than G1.

3. The document pool is concentrated near GT. Real pool docs closer to GT
   than Q capture 10.12 docs, matching G2's performance.

4. The query–document gap drives the G1/G2 disparity. Queries sit in a
   different embedding region than documents; using Q as the circle center
   misses the document cluster.

5. The centroid is special in density (92.5 docs/radius) but not in count.
   Any real pool doc near GT gives similar raw count (10.12 ≈ 9.78). The centroid
   has the highest density because it minimizes total cluster distance, yielding
   the smallest radius for the same coverage.

---

## Output Files

| File | Description |
|------|-------------|
| `csv/group_analysis/master_summary.csv` | Per-query G1/G2/G3 sizes, overlaps, GT membership |
| `csv/group_analysis/global_aggregates.csv` | Global mean/median/std for all metrics |
| `csv/group_analysis/per_query/query_NNN.csv` | Per-doc membership flags per query |
| `csv/halfpoint_groups/all_anchors.csv` | 1960 rows (196 × 10 anchors) with HP group sizes |
| `csv/halfpoint_groups/per_query_summary.csv` | Per-query HP aggregates vs G1 |
| `csv/halfpoint_groups/global_aggregates.csv` | Global halfpoint statistics |
| `csv/pooldoc_centroid/all_candidates.csv` | 1052 candidate-doc rows |
| `csv/pooldoc_centroid/per_query_summary.csv` | Per-query pool-doc centroid aggregates |
| `csv/pooldoc_centroid/global_aggregates.csv` | Global pool-doc centroid statistics |

---

*Generated from: staged_run_2026-08-03_08-32-45 · ModularRAGOptimization*
