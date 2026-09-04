# Analysis Index - staged_run_2026-08-03_08-32-45

Run completed: 2026-08-08 00:17 UTC  
Scored pairs: 1,000,174  
Scoring model: gcp/google/gemini-2.5-flash-lite (NVIDIA IH)  
Pool: 5,035 docs (5,026 from expanded_scoring_queue + 9 missing GTs)  
Canonical dataset: 196 queries (filtered from 199 - 3 missing GT in pool)

---

## Reports (start here)

| File | Format | Topic | Date |
|------|--------|-------|------|
| `report_full.html` / `.pdf` | HTML + PDF | Complete report - correlation analysis + all group geometry experiments | 2026-08-25 |
| `report_summary.html` / `.pdf` | HTML + PDF | Summary of main correlation experiment | 2026-08-10 |
| `group_analysis_extended.html` / `.pdf` | HTML + PDF | Extended G1/G2/G3 group analysis with charts | 2026-08-10 |
| `experiment_centroid_geometry.html` / `.pdf` / `.md` | HTML + PDF + MD | Standalone centroid geometry experiment report | 2026-08-25 |

> Quick start: Open `report_full.pdf` for everything in one document.

---

## Topic 1 - LLM Score vs Embedding Distance Correlation

*Do LLM relevance scores reflect cosine-distance proximity to GT in embedding space?*  
*Answer: Yes - strong negative correlation (Spearman −0.290, ROC-AUC 0.9974), GT recovery 99.5%.*

### Global metrics (`csv/global/`)

| File | Description |
|------|-------------|
| `correlation_summary.csv` | Median/mean Spearman, Pearson, Kendall-τ across queries |
| `llm_recovery_at_k.csv` | GT in top-K by LLM score (all queries) |
| `mrr_and_precision_at_k.csv` | Optimistic MRR / P@K |
| `roc_pr_summary.csv` | Global ROC-AUC / average precision |
| `score_distribution_summary.csv` | GT vs non-GT score statistics |
| `distance_quantile_analysis.csv` | LLM score by distance decile (non-GT docs) |
| `topk_jaccard_summary.csv` | LLM vs distance top-K agreement (aggregated) |

### Per-query metrics (`csv/per_query/`) - one row per query_idx

| File | Key columns |
|------|-------------|
| `per_query_overview.csv` | Master merge - start here for per-query review |
| `per_query_correlation.csv` | spearman_r, pearson_r, kendall_tau |
| `gt_llm_rank.csv` | gt_llm_rank, gt_llm_score, gt_rank_pct |
| `score_gap.csv` | GT score − best non-GT score |
| `topk_jaccard.csv` | Jaccard vs distance ranking per query × K |

---

## Topic 2 - Geometric Group Analysis (G1 / G2 / G3)

*Why does the centroid-based group (G2) consistently capture more docs than the query-centered group (G1)?*  
*Three linked experiments; all use the canonical 196-query × 5,035-doc dataset.*

### Group definitions

| Group | Center | Radius | Rule |
|-------|--------|--------|------|
| G1 | Query vector Q | dist(Q, GT) | docs within query circle |
| G2 | Normalised centroid of G3 | dist(centroid, GT) | docs within centroid circle |
| G3 | - | - | LLM score = 1.0 |

### Experiment A - G1 / G2 / G3 sizes and overlaps (`csv/group_analysis/`)

Run: 2026-08-XX · Script: `analyze_groups.py`

Key results: G2 mean 9.72 vs G1 mean 3.13 (77.6% of queries). Density: G2 92.5 vs G1 15.4 docs/radius.

| File | Description |
|------|-------------|
| `master_summary.csv` | Per-query G1/G2/G3 sizes, overlaps, Jaccard, GT membership (196 rows) |
| `global_aggregates.csv` | Global mean / median / std for all metrics |
| `per_query/query_NNN.csv` | Per-doc membership flags, distances, LLM scores per query |

### Experiment B - Halfpoint (random space points) (`csv/halfpoint_groups/`)

Run: 2026-08-XX · Script: `analyze_halfpoint_groups.py`

Key result: Random points at dist r₁/2 from GT yield 0.82 docs (< G1's 3.18). GT neighbourhood is not inherently dense.

| File | Description |
|------|-------------|
| `all_anchors.csv` | 1,960 rows (196 queries × 10 anchors) - n_halfpoint, density, Jaccard vs G1 |
| `per_query_summary.csv` | Per-query aggregate over 10 anchors (196 rows) |
| `global_aggregates.csv` | Global halfpoint statistics |
| `per_query/query_NNN.csv` | Per-anchor detail per query |

### Experiment C - Pool-document centroid (`csv/pooldoc_centroid/`)

Run: 2026-08-XX · Script: `analyze_pooldoc_centroid.py`

Key result: Real pool docs closer to GT than Q yield 10.12 docs (91.8% beat G1). Pool is structurally dense around GT.

| File | Description |
|------|-------------|
| `all_candidates.csv` | 1,052 candidate rows - n_cand_circle, density, Jaccard vs G1/G2 |
| `per_query_summary.csv` | Per-query aggregate over sampled candidates (196 rows) |
| `global_aggregates.csv` | Global pool-doc centroid statistics |
| `per_query/query_NNN.csv` | Per-candidate detail per query |

---

## Summaries (`summaries/`)

| File | Description |
|------|-------------|
| `analysis_summary.txt` | Base correlation run summary + run health |
| `group_analysis_summary.md` | Narrative summary of G1/G2/G3 group analysis findings |

## Charts (`charts/`)

| Folder | Source script | Contents |
|--------|---------------|----------|
| `charts/base/` | `analyze_experiment.py` | Correlation + score distribution |
| `charts/extended/` | `analyze_experiment_extended.py` | Extended analysis |
| `charts/vector_space/radial/` | `visualize_vector_space.py` | 196 radial density PNGs |
| `charts/vector_space/geometric/` | `visualize_vector_space.py` | 196 trilateration scatter PNGs |
| `charts/vector_space/umap/` | `visualize_vector_space.py` | 196 UMAP scatter PNGs |

---

## Interactive Visualizations (`charts/vector_space/`)

Plotly HTML files — open in any browser, supports zoom / pan / hover.

### Trilateration maps — exact cosine distances (recommended)

```
charts/vector_space/geo_interactive/
  index.html          ← start here: 196 query tiles with gap values
  query_NNN.html      ← per-query interactive plot
```

**How to open:**
```bash
open results/cloud_runs/staged_run_2026-08-03_08-32-45/analysis/charts/vector_space/geo_interactive/index.html
# or a specific query:
open .../geo_interactive/query_118.html
```

**Axes:** X = cosine projection onto Q→GT (Q at 0, GT at r1). Y = perpendicular distance (folded ≥ 0).
All 5,035 docs shown. G1 circle is exact. Initial view zoomed to area of interest; scroll to zoom out.

### UMAP maps — neighbourhood-preserving (not distance-accurate)

```
charts/vector_space/umap_interactive/
  index.html
  query_NNN.html
```

---

## Extreme Examples (`csv/extreme_examples/`)

Top 10 queries by G2−G1 gap — generated by `analyze_extreme_examples.py`.

| File | Description |
|------|-------------|
| `top10_extreme_examples.csv` | Full stats for top-10 queries |
| `g2_only_query_NNN.csv` | Docs in G2 but not G1 for each query |
| `analysis.md` | Detailed per-query breakdown + conclusions |

**Mechanism classification:** A = centroid displaced (r2 ≥ r1, lands in dense zone); B = tight G3 cluster (r2 < r1, coherent centroid).

---

## Session Brief

Full narrative brief of all analysis steps since group experiment results:
**https://claude.ai/code/artifact/44397523-a8bc-4305-9227-4fca3bdbb835**

---

## Other

| File | Description |
|------|-------------|
| `run_health.json` | Scoring run health check (199 queries in DB, 3 missing GT) |
