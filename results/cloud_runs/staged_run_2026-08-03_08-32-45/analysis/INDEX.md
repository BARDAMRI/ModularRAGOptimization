# Analysis Index — staged_run_2026-08-03_08-32-45

**Run completed:** 2026-08-08 00:17 UTC  
**Scored pairs:** 1,000,174 · **Pool:** 5,035 docs · **Queries:** 196  
**Scoring model:** gcp/google/gemini-2.5-flash-lite

---

## Directory structure

```
analysis/
├── 01_run_overview/        Base statistics + full run reports
├── 02_group_analysis/      G1/G2/G3 group geometry analysis
├── 03_centroid_geometry/   Centroid geometry experiment
├── 04_circle_bfs/          Circle & BFS experiments
├── 05_extreme_examples/    Extreme failure/success case deep-dive
├── 06_triangle_geometry/   Triangle Q–Centroid–GT geometric analysis
├── csv/                    Raw experiment output CSVs
├── INDEX.md                This file
└── run_health.json         Scoring run health metadata
```

---

## 01 — Run Overview

*Do LLM scores reflect cosine-distance proximity to GT? Answer: yes (Spearman −0.290, ROC-AUC 0.9974, GT recovery 99.5%).*

| File | Description |
|------|-------------|
| [`01_run_overview/report_full.html`](01_run_overview/report_full.html) / [`.pdf`](01_run_overview/report_full.pdf) | Complete run report — correlation analysis + all group geometry |
| [`01_run_overview/report_summary.html`](01_run_overview/report_summary.html) / [`.pdf`](01_run_overview/report_summary.pdf) | Summary of main correlation experiment |
| [`01_run_overview/charts/`](01_run_overview/charts/) | Base statistical charts (scatter, histograms, ROC, etc.) |

**CSV data:** `csv/global/`, `csv/per_query/`

---

## 02 — Group Analysis (G1 / G2 / G3)

*Why does the centroid circle (G2) capture 3× more docs than the query circle (G1)?*

| Group | Center | Radius | Meaning |
|-------|--------|--------|---------|
| G1 | Query Q | dist(Q, GT) | docs within query neighbourhood |
| G2 | G3 centroid | dist(centroid, GT) | docs within centroid neighbourhood |
| G3 | — | — | LLM score = 1.0 (oracle) |

**Key results:**
- G2 mean 9.72 docs vs G1 mean 3.13 (77.6% of queries)
- Halfpoint random anchors: only 0.82 docs — GT neighbourhood is not inherently dense
- Pool-doc centroid: 10.12 docs (91.8% beat G1)

| File | Description |
|------|-------------|
| [`02_group_analysis/group_analysis_extended.html`](02_group_analysis/group_analysis_extended.html) / [`.pdf`](02_group_analysis/group_analysis_extended.pdf) | Extended G1/G2/G3 report with charts |
| [`02_group_analysis/charts_extended/`](02_group_analysis/charts_extended/) | Extended analysis charts |
| [`02_group_analysis/summaries/group_analysis_summary.md`](02_group_analysis/summaries/group_analysis_summary.md) | Narrative findings summary |

**CSV data:** `csv/group_analysis/`, `csv/halfpoint_groups/`, `csv/pooldoc_centroid/`

---

## 03 — Centroid Geometry Experiment

*How does the centroid relate geometrically to the query and pool documents?*

| File | Description |
|------|-------------|
| [`03_centroid_geometry/experiment_centroid_geometry.html`](03_centroid_geometry/experiment_centroid_geometry.html) / [`.pdf`](03_centroid_geometry/experiment_centroid_geometry.pdf) | Full centroid geometry report |
| [`03_centroid_geometry/experiment_centroid_geometry.md`](03_centroid_geometry/experiment_centroid_geometry.md) | Markdown version |
| [`03_centroid_geometry/charts_vector_space/`](03_centroid_geometry/charts_vector_space/) | Vector space visualisations (UMAP, trilateration, radial, interactive Plotly) |

**How to open interactive visualisations:**
```bash
# Trilateration (recommended — exact cosine distances)
open 03_centroid_geometry/charts_vector_space/geo_interactive/browse.html

# dist(Q) vs dist(GT) scatter
open 03_centroid_geometry/charts_vector_space/dqdgt_interactive/browse_dqdgt.html
```

---

## 04 — Circle & BFS Experiments

*Do query embeddings sit further from document space than GT embeddings? Does a centroid-guided BFS improve retrieval?*

**Key results:**

| Strategy | Mean rank | Top-1 | Top-10 |
|---|---|---|---|
| Baseline 1 — dist(Q) | 3.3 | 70.4% | 91.8% |
| Hybrid 1 α=0.93 (best) | 3.3 | **71.4%** | 90.8% |
| Baseline 2 — dist(Centroid) | 9.9 | 14.8% | 62.8% |

Conclusion: Q_CIRCLE always empty (query is geometric outlier). Oracle centroid adds only +1% top-1.

| File | Description |
|------|-------------|
| [`04_circle_bfs/circle_bfs_report.html`](04_circle_bfs/circle_bfs_report.html) | Full Circle & BFS visual report |

**CSV data:** `csv/circle_bfs/`

---

## 05 — Extreme Examples

*Deep-dive into the 10 queries where G2 most exceeds G1.*

Two mechanisms identified:
- **A** — centroid displaced (r2 ≥ r1, lands in dense zone)
- **B** — tight G3 cluster (r2 < r1, coherent centroid)

| File | Description |
|------|-------------|
| [`05_extreme_examples/extreme_examples_abstract.html`](05_extreme_examples/extreme_examples_abstract.html) / [`.pdf`](05_extreme_examples/extreme_examples_abstract.pdf) | Extreme examples analysis report |

**CSV data:** `csv/extreme_examples/`

---

## 06 — Triangle Geometry (Q – Centroid – GT)

*What is the geometric relationship between Q, the G3 centroid, and GT on the unit hypersphere?*

**Key results (196 queries, Cosine/Spherical):**

| Vertex | Mean angle | Interpretation |
|--------|-----------|----------------|
| ∠Q (apex) | 45.7° | Centroid and GT diverge ~46° as seen from Q |
| ∠C | 70.8° | GT is ~71° away from Q as seen from C |
| ∠GT | 70.6° | C is ~71° away from Q as seen from GT |
| Sum | 187.0° | Spherical excess of 7° |

Q→GT ≈ Q→C (both ~0.17 cosine), but C→GT is only 0.10 — the triangle is nearly isosceles with Q at the apex. This explains geometrically why mixing centroid into retrieval deflects away from GT.

| File | Description |
|------|-------------|
| [`06_triangle_geometry/triangle_geometry.html`](06_triangle_geometry/triangle_geometry.html) | Full interactive report (angle distributions + heatmap) |
| [`06_triangle_geometry/triangle_geometry_web.pdf`](06_triangle_geometry/triangle_geometry_web.pdf) | PDF — ready to share |
| [`06_triangle_geometry/triangle_geometry_page1.png`](06_triangle_geometry/triangle_geometry_page1.png) | PNG — ready to share |
| [`06_triangle_geometry/triangle_geometry.tex`](06_triangle_geometry/triangle_geometry.tex) | LaTeX source for academic use |

---

## CSV Data

| Folder | Experiment | Contents |
|--------|-----------|----------|
| `csv/global/` | Run overview | Correlation summary, ROC, MRR, top-K |
| `csv/per_query/` | Run overview | Per-query correlation, GT rank, score gap |
| `csv/group_analysis/` | Group analysis | G1/G2/G3 sizes, overlaps, Jaccard |
| `csv/halfpoint_groups/` | Group analysis | Random anchor halfpoint statistics |
| `csv/pooldoc_centroid/` | Group analysis | Pool-doc centroid statistics |
| `csv/circle_bfs/` | Circle & BFS | Circle doc counts + BFS rank comparison |
| `csv/extreme_examples/` | Extreme examples | Top-10 extreme G2−G1 gap queries |
| `csv/llm_rerank/` | LLM re-ranking | Re-rank k=5..100 vs Baseline 1 |
| `csv/iterative_prf/` | Iterative PRF | PRF rounds k=5..50, thresholds g2/g3 |
| `csv/adaptive_search/` | Adaptive search | Sequential search, α sweep |

---

## Infrastructure

| Component | Location | Description |
|-----------|----------|-------------|
| Data loader | `experiments/global_correlation/load_experiment_data.py` | Single entry point for all experiments |
| Embedding factory | `experiments/global_correlation/embedding_factory.py` | Model, batch size, encode/save/load |
| Embedding cache | `embeddings/` (run root) | Pre-computed vectors, ~16MB, loads in 0.05s |
