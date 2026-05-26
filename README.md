# Modular RAG Optimization

Research codebase for evaluating retrieval-augmented generation (RAG) pipelines: vector search, LLM relevance scoring, and correlation experiments over PubMedQA.

## Requirements

- Python 3.10+
- See project dependencies (PyTorch, LlamaIndex, Chroma, etc.)
- API keys in `.env` as needed (`GEMINI_API_KEY`, `INFERENCE_API_KEY`, …)

## Quick start

```bash
git clone https://github.com/BARDAMRI/ModularRAGOptimization.git
cd ModularRAGOptimization
cp .env.example .env   # fill in API keys
python main.py
```

With `use_config_file: true` in `run_config.json`, the app runs one pre-configured action and exits (no interactive menus).

---

## `run_config.json` — non-interactive runs

File location: **`run_config.json`** (project root).

Set `"use_config_file": true` to skip menu prompts and drive the app from JSON. Set it to `false` for the classic interactive flow.

### Top-level keys

| Key | Purpose |
|-----|---------|
| `use_config_file` | `true` = read all choices from JSON; `false` = interactive menus |
| `setup` | Vector DB / dataset bootstrap (`dataset_key`, `storing_method`, `distance_metric`) |
| `main_menu_choice` | Main menu option (`1`–`10`). `7` = Experiments submenu |
| `experiments_menu_choice` | Experiments submenu (`1`–`9`). `8` = Global correlation |
| `global_correlation` | Global correlation experiment (see below) |
| `performance_summary` | Whether to print session performance stats on exit |

### Global correlation — `active_mode`

Inside `global_correlation`, set **`active_mode`** to one of the keys under **`modes`**. The loader merges that mode’s fields into the flat `global_correlation.*` namespace and sets `run_mode`.

| Mode | Menu | What it does |
|------|------|----------------|
| `staged` | `4` | **Recommended.** Incrementally grows the doc pool and scores in stages (resumable) |
| `pilot` | `1` | Small test run (few queries) to validate plumbing |
| `full` | `2` | One-shot full run via Gemini Batch API |
| `sync` | `3` | Ingest downloaded Batch JSONL into SQLite |
| `analyze` | `5` | **Post-run analysis only** — no scoring |

### Example: run staged experiment

```json
{
  "use_config_file": true,
  "main_menu_choice": "7",
  "experiments_menu_choice": "8",
  "global_correlation": {
    "active_mode": "staged",
    "modes": {
      "staged": {
        "scoring_provider": "inference_api",
        "queries_to_load": 200,
        "staging_stride": 10,
        "staging_max_ranked_per_gt": 100,
        "staging_max_ranked_per_query": 100,
        "auto_run_post_analysis": false
      }
    }
  }
}
```

Run: `python main.py`

### Example: analyze latest run (no experiment)

```json
{
  "use_config_file": true,
  "main_menu_choice": "7",
  "experiments_menu_choice": "8",
  "global_correlation": {
    "active_mode": "analyze",
    "modes": {
      "analyze": {
        "run_dir": "",
        "database_path": ""
      }
    }
  }
}
```

Empty `run_dir` / `database_path` → auto-detect the latest run under `results/global_exp/`.

To target a specific run:

```json
"analyze": {
  "run_dir": "results/global_exp/staged_run_2026-05-17_18-18-42",
  "database_path": ""
}
```

### Example: auto-analyze after staged run completes

Add to the **staged** (or `full` / `pilot` / `sync`) mode block:

```json
"auto_run_post_analysis": true
```

When the experiment finishes, the pipeline runs automatically:

`analyze_experiment.py` → `analyze_experiment_extended.py` → `export_report.py`

### Common `global_correlation` fields

| Field | Modes | Description |
|-------|-------|-------------|
| `scoring_provider` | pilot, staged, full | `inference_api`, `gemini`, or `ollama` |
| `queries_to_load` | pilot, staged, full | Number of QA queries (default 200) |
| `staging_stride` | staged | Docs advanced per pool position per stage |
| `staging_max_ranked_per_gt` | staged | GT-embedding neighbours per query |
| `staging_max_ranked_per_query` | staged | Query-embedding neighbours per query |
| `sync_database_path` | sync | Path to `experiment_results.db` |
| `sync_output_path` | sync | Batch output JSONL file or folder |
| `run_dir` | analyze | Run directory (optional) |
| `database_path` | analyze | Path to `experiment_results.db` (optional) |
| `auto_run_post_analysis` | staged, full, pilot, sync | Run analysis pipeline when experiment ends |

---

## Global correlation experiment

**Menu path:** Main menu `7` → Experiments `8` → choose mode.

**Outputs** (under `results/global_exp/<run_dir>/`):

- `experiment_results.db` — scored (query, doc) pairs
- `correlation_pool_manifest.json` — pool / staging state
- `experiment_config.json` — run parameters

### Staged mode (recommended)

Builds a global document pool incrementally and scores every query against new docs each stage. Supports resume after failures.

---

## Post-run analysis

After scoring (or on a partial run), generate CSVs, charts, and a self-contained HTML report.

### Option A — unified script (recommended)

```bash
python run_global_correlation_analysis.py
python run_global_correlation_analysis.py --run-dir results/global_exp/staged_run_YYYY-MM-DD_HH-MM-SS
python run_global_correlation_analysis.py --steps extended,report   # skip base if already run
```

Runs, in order:

1. **`analyze_experiment.py`** — base metrics (correlation, recovery, ROC/PR, …)
2. **`analyze_experiment_extended.py`** — tie-corrected MRR, difficulty tiers, hybrid tiebreaker
3. **`export_report.py`** — `analysis/report.html`

**Re-running:** overwrites files at the same paths (no versioning). Safe after a partial run — use `--steps` to run only missing stages.

### Output layout (`<run_dir>/analysis/`)

| Path | Scope | Purpose |
|------|-------|---------|
| `INDEX.md` | — | Catalog: which files are global vs per-query |
| `csv/global/` | **Global** | Pool-level aggregates (MRR, Recovery@K, ROC, …) |
| `csv/per_query/` | **Per-query** | One row per `query_idx` |
| `csv/per_query/per_query_overview.csv` | **Per-query** | **Master merge** — start here for query-level review |
| `charts/base/` | — | Charts from step 1 |
| `charts/extended/` | — | Charts from step 2 |
| `summaries/` | — | `analysis_summary.txt`, `extended_summary.txt` |
| `report.html` | Both | Interactive report with global + per-query sections |

**Global** = one number summarises all queries/doc pairs. **Per-query** = filter/sort by `query_idx` in Excel to inspect individual queries.

### Option B — interactive menu

Main menu `7` → Experiments `8` → **`analyze` / `5`**

### Option C — `run_config.json`

Set `"active_mode": "analyze"` (see example above).

### Analysis outputs

All files land in `<run_dir>/analysis/`:

| File | Description |
|------|-------------|
| `report.html` | Primary deliverable — self-contained HTML report |
| `run_health.json` | Run completeness, NULL scores, skipped queries, pool coverage |
| `analysis_summary.txt` | Base text summary |
| `extended_summary.txt` | Corrected conclusions (tie-aware) |
| `*.csv` / `*.png` | Per-metric tables and charts |

---

## Understanding the analysis results

### Central question

> **When the retriever fails to rank the ground-truth document (GT) at position 1, can an LLM relevance score rescue it?**

The experiment scores every `(query, document)` pair with an LLM (integer 1–10, stored as 0.1–1.0). Each query has one GT document. The DB records, for every doc: its LLM score, its cosine distance to the GT embedding (`dist_to_gt`), and whether it is the GT (`is_gt`).

The analysis pipeline has three parts that run in sequence.

---

### Part 1 — Base analysis (`analyze_experiment.py`)

15 sections. Answers: *does the LLM score correlate with proximity to GT, and how often does LLM put GT near the top?*

**Section 1 — Correlation**
Runs 4 methods **per query** between `llm_score` and `dist_to_gt` / `is_gt`:

| Method | Measures | Expected sign |
|--------|----------|---------------|
| Spearman r | Rank-based monotone: does higher score → lower distance? | Negative |
| Pearson r | Linear component of same | Negative |
| Kendall τ-b | Same as Spearman but most robust to the heavy tie structure in scores | Negative |
| Point-biserial | Does the GT doc score higher than the pool average? | Positive |

`correlation_summary.csv` reports median/mean of each method across all queries, plus % of queries where the sign is correct and % statistically significant.

**Section 2 — GT Rank**
Ranks all docs by `llm_score` DESC per query and records where GT lands. Produces P@1, P@3, P@5, P@10. The histogram and CDF show that GT almost always gets rank 1 (~P@1 ≈ 98%). *This is misleading — it is inflated by ties — see Part 2.*

**Section 3 — Recovery Rate**
Same as P@K framed as recovery: the retriever already failed, so if the LLM puts GT in its top-K it *recovers* the retrieval. Tells you: as a re-ranker returning K results, the LLM almost never loses GT entirely.

**Section 4 — Score Distributions**
Overlapping histogram, box plot, and violin of `llm_score` for GT vs non-GT docs. Distribution is heavily bimodal: ~83% of non-GT docs score 0.1, ~98% of GT docs score 1.0. Excellent macro separation, but the tie problem prevents precise single-doc identification.

**Section 5 — GT Score Percentile**
Per query, what percentile is the GT doc's score within its query's pool? Confirms the LLM reliably elevates GT — but many other docs share the same top percentile.

**Section 6 — ROC / PR Curves**
Treats GT detection as binary classification globally across all 1M pairs. ROC-AUC ≈ 0.996 means near-perfect coarse separation. Average Precision (~8.7%) is lower because at threshold 1.0 only ~1-in-11 retained docs is GT. The gap between the two numbers is the story: great at separating GT from random noise, poor at pinpointing GT among close neighbours.

**Section 7 — Scatter**
LLM score (x) vs `dist_to_gt` (y) for a 10 000 doc sample + all GTs. Shows 10 vertical stripes (the 10 discrete score levels). Non-GT docs at score=1.0 cluster at dist ≈ 0.3, not 0.85 like 0.1-scoring docs — confirming the score carries real distance signal.

**Section 8 — Distance Quantile Analysis**
Non-GT docs split into 10 deciles by `dist_to_gt`. Mean/median LLM score per decile. Expect a declining curve (docs closer to GT should score higher). Flat median at 0.1 with a rising mean confirms signal exists but only for the outliers above 0.1.

**Section 9 — Top-K Jaccard**
Per query and K ∈ {5, 10, 20, 50, 100}: Jaccard similarity between the top-K by LLM score and the top-K by `dist_to_gt` ranking. Moderate overlap (~0.3–0.4 at K=20) justifies combining both signals.

**Section 10 — Score Gap**
Per query: `GT_score − best_nonGT_score`. Positive = GT uniquely highest. Zero = tied. Negative = GT outscored. Histogram shows how many queries fall into each category and how severely.

**Section 11–12 — Visual diagnostics**
Scatter of per-query GT rank coloured by normalised rank, and a 4-panel overview relating correlation strength to rank, score gap, and pool size. Use to spot outlier queries.

**Section 13 — MRR and P@K**
Global summary table: MRR and P@1 through P@100. Optimistic MRR ≈ 0.98. The companion `corrected_mrr_at_k.csv` from Part 2 gives the honest number (0.26).

**Sections 14–15 — Sanity checks**
`dist_to_gt` for GT docs should be ≈ 0 (GT compared against its own embedding). Any GT doc with large dist_to_gt indicates a data integrity issue.

**Text output:** `summaries/analysis_summary.txt` — all key numbers from sections 1–15 in one place with a conclusion paragraph.

---

### Part 2 — Extended / corrective analysis (`analyze_experiment_extended.py`)

10 sections (A–J). Fixes misleading conclusions from Part 1. Answers: *the LLM puts GT in the top tier, but can it identify GT uniquely? What happens inside the tied group?*

**Section A — Tie Structure**
Per query: how many docs share the exact same score as GT (`n_tied_at_gt`), how many score *above* GT (`n_above_gt`), and `tier_precision` = 1 / n_tied (expected accuracy of a random pick from the tied group).

In a typical run, 195/199 queries have GT in the top tier but the median tie size is 9 — GT shares first place with 9 other docs on average. Mean tier precision = 0.17 (1-in-6 chance of picking GT at random). **This is the core finding that explains why optimistic P@1 = 98% is not the right metric.**

**Section B — Score Discretization**
Breaks down every discrete score level (0.1, 0.2, … 1.0): doc counts, GT fraction, mean/median `dist_to_gt`. Mean dist_to_gt decreases monotonically as score increases — intermediate scores carry real signal even though they are sparsely used.

**Section C — Threshold Filter**
For each threshold: if you keep only docs scoring ≥ threshold, what is precision, recall, F1, and average pool size? At threshold = 1.0: pool drops from ~6000 to ~11 docs, recall stays ~98%, precision reaches ~9%. **This is the optimal operating point for a two-stage pipeline: LLM filter → secondary ranker.**

**Section D — Query Difficulty**
Classifies every query into four tiers:

| Category | Definition |
|----------|------------|
| **EASY** | GT is the *only* doc at score 1.0 |
| **MEDIUM** | GT at 1.0, 2–10 others also tied |
| **HARD** | GT at 1.0, 11+ others also tied |
| **FAILED** | GT does not reach score 1.0 at all |

Typical split: ~3% EASY, ~50% MEDIUM, ~45% HARD, ~2% FAILED. 95% of queries need a tiebreaker.

**Section E — Failure Deep-Dive**
Detailed look at FAILED queries: GT's actual score, its rank, how many docs beat it. These are true LLM failures — e.g., a query where GT scored 0.1 and was ranked 281st. Useful for diagnosing edge cases (unusual phrasing, short GT docs, jargon mismatch).

**Section F — False Positives**
Compares `dist_to_gt` for GT docs at score=1.0, non-GT docs at score=1.0, and non-GT docs at score=0.1. Non-GT docs scoring 1.0 have median dist_to_gt ≈ 0.3 — they are genuine semantic neighbours of GT, not random noise. This explains why the ~11-doc pool at score=1.0 is high-quality, not noisy.

**Section G — Tie-Corrected MRR / P@K**
Computes three rank variants per query:

| Variant | How ties are ranked | When to use |
|---------|--------------------|----|
| `rank_min` (optimistic) | All tied docs share the lowest rank in their group | Upper bound |
| `rank_avg` (tie-corrected) | Tied docs share the average rank | **Honest estimate** |
| `rank_max` (pessimistic) | Tied docs get the highest rank in their group | Lower bound |

Typical result: MRR(min)=0.98 → MRR(avg)=0.26 → MRR(max)=0.17. **Always report the tie-corrected number when discussing LLM-as-ranker performance.**

**Section H — Within-Tier Distance Tiebreaker**
Among docs tied at score=1.0 per query: does GT have the lowest `dist_to_gt`? In the typical run this is true in **100% of applicable queries**. This is the key actionable finding: a two-stage pipeline — (1) filter to score=1.0, (2) return the doc with lowest embedding distance — achieves near-perfect retrieval.

**Section I — Marginal Value of Score Levels**
For non-GT docs: does score=0.6 predict lower dist_to_gt than score=0.2, relative to the 0.1 baseline? Positive improvement at every level confirms that intermediate scores carry distance signal and could be used in a more nuanced filter.

**Text output:** `summaries/extended_summary.txt` — tie problem explained, corrected MRR values, query difficulty breakdown, and the revised bottom-line verdict.

---

### Part 3 — HTML report (`export_report.py`)

Reads all CSVs and charts from Parts 1 and 2, embeds all 24 charts as base64, and writes a single `analysis/report.html`. The sidebar has one-click navigation to 12 sections. Open this in a browser; nothing else is needed.

---

### Complete file reference

```
analysis/
├── report.html                         ← open this first
├── INDEX.md                            ← full file catalog
├── run_health.json                     ← DB completeness, NULL scores, pool coverage
│
├── summaries/
│   ├── analysis_summary.txt            ← Part 1 numbers in one place
│   └── extended_summary.txt            ← corrected, tie-aware conclusions
│
├── csv/global/                         ← one number per experiment
│   ├── correlation_summary.csv         ← 4-method correlation across all queries
│   ├── roc_pr_summary.csv              ← ROC-AUC, average precision
│   ├── mrr_and_precision_at_k.csv      ← optimistic MRR + P@K
│   ├── corrected_mrr_at_k.csv          ← tie-corrected MRR + P@K (use this one)
│   ├── llm_recovery_at_k.csv           ← % queries with GT in LLM top-K
│   ├── score_distribution_summary.csv  ← GT vs non-GT score statistics
│   ├── distance_quantile_analysis.csv  ← LLM score by dist_to_gt decile
│   ├── topk_jaccard_summary.csv        ← LLM vs dist top-K agreement
│   ├── threshold_filter_analysis.csv   ← precision / recall at each score threshold
│   ├── score_level_breakdown.csv       ← stats per 0.1–1.0 score level
│   ├── false_positive_analysis.csv     ← non-GT at score=1.0 vs score=0.1
│   └── marginal_score_value.csv        ← dist_to_gt improvement per score level
│
├── csv/per_query/                      ← one row per query — filter in Excel
│   ├── per_query_overview.csv          ← MASTER MERGE, start here
│   ├── gt_llm_rank.csv                 ← GT rank, score, percentile per query
│   ├── per_query_correlation.csv       ← 4 correlation values per query
│   ├── tie_structure.csv               ← tie size, tier precision per query
│   ├── tie_corrected_ranks.csv         ← min / avg / max rank per query
│   ├── query_difficulty.csv            ← EASY / MEDIUM / HARD / FAILED label
│   ├── score_gap.csv                   ← GT score − best non-GT per query
│   ├── within_tier_dist.csv            ← hybrid tiebreaker result per query
│   ├── topk_jaccard.csv                ← Jaccard per query × K
│   ├── gt_score_percentile.csv         ← GT percentile rank per query
│   └── failure_cases.csv               ← only the FAILED queries (typically 2–5 rows)
│
├── charts/base/                        ← 15 PNGs from analyze_experiment.py
└── charts/extended/                    ← 9 PNGs from analyze_experiment_extended.py
```

---

### Key findings and verdict

| Finding | Metric | Value |
|---------|--------|-------|
| LLM assigns score=1.0 to GT | Recovery@1 | ~98% of queries |
| But GT shares that score with others | Median tie size | ~9 other docs |
| Honest rank of GT (tie-corrected) | MRR (rank_avg) | ~0.26 |
| Inflated rank of GT (optimistic) | MRR (rank_min) | ~0.98 |
| Non-GT docs at score=1.0 are real neighbours | Median dist_to_gt | ~0.30 (vs 0.85 for score=0.1) |
| LLM as binary GT detector | ROC-AUC | ~0.996 |
| Hybrid: LLM filter + dist tiebreaker | Win rate within top tier | 100% |

**Bottom line:** The LLM is an excellent *coarse filter* but a limited *precise ranker*. At threshold 1.0 it reduces the candidate pool from ~6000 to ~11 docs while retaining GT in 98% of queries. Within that pool, GT consistently has the lowest embedding distance. A two-stage pipeline — (1) keep all docs at score=1.0, (2) return the one with lowest cosine distance — achieves near-perfect retrieval and is the primary actionable recommendation from this experiment.

---

## Project layout (selected)

```
main.py                              # Application entry point
run_config.json                      # Non-interactive run configuration
run_global_correlation_analysis.py   # Unified post-run analysis flow
analyze_experiment.py                # Base analysis (step 1)
analyze_experiment_extended.py       # Extended / tie-corrected analysis (step 2)
export_report.py                     # HTML report (step 3)
analysis_common.py                   # Shared DB health + path helpers
experiments/global_correlation_experiment.py
results/global_exp/<run_dir>/        # Experiment + analysis outputs
configurations/config.py
utility/run_config.py                # run_config.json loader
```

---

## Interactive menu map

| Main | Action |
|------|--------|
| `7` | Experiments submenu |
| `4` | Results analysis (legacy ResultsLogger) |

| Experiments | Action |
|-------------|--------|
| `8` | Global correlation (staged / pilot / full / sync / **analyze**) |

Within global correlation:

| Choice | Mode |
|--------|------|
| `4` / `staged` | Run staged experiment |
| `1` / `pilot` | Pilot test |
| `2` / `full` | Full batch experiment |
| `3` / `sync` | Sync Batch JSONL → SQLite |
| `5` / `analyze` | Post-run analysis pipeline |
| `0` | Back |
