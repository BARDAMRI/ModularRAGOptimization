# Analysis Output Index

Run analysis folder. **199 queries** in DB (see `run_health.json`).

## How to read these files

| Scope | Meaning | Example |
|-------|---------|---------|
| **Global** | One metric summarising the **entire pool** (all queries × all docs combined, or aggregated across queries) | `MRR`, `Recovery@K`, ROC-AUC |
| **Per-query** | One **row per `query_idx`** — filter/sort in Excel to inspect individual queries | `gt_llm_rank`, `difficulty` |

> **Start here for per-query review:** `csv/per_query/per_query_overview.csv`
> (merged view of rank, correlation, difficulty, ties, hybrid win).

## Summaries (text)

| File | Description |
|------|-------------|
| `summaries/analysis_summary.txt` | Base run summary + run health |
| `summaries/extended_summary.txt` | Tie-corrected conclusions |
| `report.html` | Full interactive report (global + per-query sections) |

## CSV — global (pool-level)

| File | Description |
|------|-------------|
| `correlation_summary.csv` | Median/mean correlation across queries |
| `llm_recovery_at_k.csv` | GT in top-K by LLM (all queries) |
| `mrr_and_precision_at_k.csv` | Optimistic MRR / P@K |
| `corrected_mrr_at_k.csv` | Tie-corrected MRR / P@K (min/avg/max rank) |
| `roc_pr_summary.csv` | Global ROC-AUC / average precision |
| `score_distribution_summary.csv` | GT vs non-GT score stats |
| `distance_quantile_analysis.csv` | LLM score by dist decile (non-GT) |
| `topk_jaccard_summary.csv` | LLM vs dist top-K agreement (aggregated) |
| `threshold_filter_analysis.csv` | Precision/recall at each score threshold |
| `score_level_breakdown.csv` | Stats per discrete LLM score level |
| `false_positive_analysis.csv` | Non-GT docs at score=1.0 vs 0.1 |
| `marginal_score_value.csv` | dist improvement per score level |

## CSV — per-query (one row per query_idx)

| File | Key columns |
|------|-------------|
| **`per_query_overview.csv`** | **Master merge — start here** |
| `per_query_correlation.csv` | spearman_r, pearson_r, kendall_tau |
| `gt_llm_rank.csv` | gt_llm_rank, gt_llm_score, gt_rank_pct |
| `query_difficulty.csv` | EASY / MEDIUM / HARD / FAILED |
| `tie_structure.csv` | n_tied_at_gt, tier_precision |
| `tie_corrected_ranks.csv` | rank_min, rank_avg, rank_max |
| `score_gap.csv` | GT score − best non-GT |
| `within_tier_dist.csv` | Hybrid dist tiebreaker per query |
| `failure_cases.csv` | Queries where GT did not score 1.0 |
| `topk_jaccard.csv` | Jaccard vs dist ranking per query × K |

## Charts

| Folder | Source script |
|--------|---------------|
| `charts/base/` | analyze_experiment.py |
| `charts/extended/` | analyze_experiment_extended.py |

## Re-running analysis

- Re-running **replaces** files in this folder (same paths, no versioning).
- Safe to run after a partial run — each step overwrites only its own outputs.
- Use `python run_global_correlation_analysis.py --steps extended,report` to skip base.
