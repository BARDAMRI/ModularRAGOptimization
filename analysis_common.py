"""
Shared helpers for global-correlation experiment analysis scripts.

Used by analyze_experiment.py, analyze_experiment_extended.py, and export_report.py.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path("results/global_exp")
ANALYSIS_DIR = "analysis"
META_KEY_MAIN_LOOP_COMPLETED = "main_loop_completed"


def find_latest_db(results_root: Path = RESULTS_ROOT) -> Path:
    candidates = sorted(results_root.glob("staged_run_*/experiment_results.db"))
    if not candidates:
        candidates = sorted(results_root.glob("*/experiment_results.db"))
    if not candidates:
        raise FileNotFoundError(f"No experiment_results.db found under {results_root}")
    return candidates[-1]


def find_latest_run_dir(results_root: Path = RESULTS_ROOT) -> Path:
    candidates = sorted(results_root.glob("staged_run_*"))
    if not candidates:
        candidates = sorted(p.parent for p in results_root.glob("*/experiment_results.db"))
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {results_root}")
    return candidates[-1]


def load_scored_dataframe(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT query_idx, doc_id, llm_score, dist_to_gt, is_gt, rag_failed "
        "FROM results WHERE llm_score IS NOT NULL",
        conn,
    )
    conn.close()
    return df


def _read_meta(conn: sqlite3.Connection) -> dict[str, str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='experiment_meta' LIMIT 1"
    )
    if cur.fetchone() is None:
        return {}
    cur.execute("SELECT key, value FROM experiment_meta")
    return {str(k): str(v) for k, v in cur.fetchall()}


def _read_skipped(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='skipped_queries' LIMIT 1"
    )
    if cur.fetchone() is None:
        return []
    cur.execute("SELECT query_idx, reason FROM skipped_queries ORDER BY query_idx")
    return [(int(r[0]), str(r[1])) for r in cur.fetchall()]


def collect_run_health(db_path: Path) -> dict:
    """Inspect experiment_results.db and return structured run-health metadata."""
    conn = sqlite3.connect(db_path)
    meta = _read_meta(conn)
    skipped = _read_skipped(conn)

    total_rows = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    scored_rows = conn.execute(
        "SELECT COUNT(*) FROM results WHERE llm_score IS NOT NULL"
    ).fetchone()[0]
    null_rows = total_rows - scored_rows

    per_query = pd.read_sql_query(
        """
        SELECT query_idx,
               COUNT(*) AS n_rows,
               SUM(llm_score IS NULL) AS null_scores,
               SUM(is_gt) AS gt_rows
        FROM results
        GROUP BY query_idx
        ORDER BY query_idx
        """,
        conn,
    )
    unique_docs = int(conn.execute("SELECT COUNT(DISTINCT doc_id) FROM results").fetchone()[0])
    conn.close()

    n_queries_in_db = len(per_query)
    n_queries_scored = int((per_query["null_scores"] < per_query["n_rows"]).sum())

    pool_sizes = per_query["n_rows"]
    max_pool = int(pool_sizes.max()) if len(pool_sizes) else 0
    n_full = int((pool_sizes == max_pool).sum()) if max_pool else 0
    n_partial = n_queries_in_db - n_full

    null_by_query = [
        (int(r.query_idx), int(r.null_scores))
        for r in per_query.itertuples()
        if int(r.null_scores) > 0
    ]

    loop_completed = meta.get(META_KEY_MAIN_LOOP_COMPLETED, "").strip().lower() in ("1", "true", "yes")
    fatal_error = meta.get("fatal_llm_error")
    fatal_q = meta.get("fatal_llm_error_query_idx")
    staging_stage = meta.get("staging_last_completed_stage")

    sync_pct = (scored_rows / total_rows * 100) if total_rows else 100.0

    warnings: list[str] = []
    if not loop_completed:
        warnings.append("Experiment main loop did not complete (experiment_meta.main_loop_completed absent).")
    if fatal_error:
        q_note = f" at query {fatal_q}" if fatal_q else ""
        warnings.append(f"Fatal LLM error recorded{q_note}: {fatal_error[:120]}…"
                          if len(fatal_error or "") > 120 else
                          f"Fatal LLM error recorded{q_note}: {fatal_error}")
    if null_rows > 0:
        warnings.append(f"{null_rows:,} rows have NULL llm_score ({len(null_by_query)} queries affected).")
    if skipped:
        warnings.append(f"{len(skipped)} query/queries skipped during experiment setup/scoring.")
    if n_partial > 0 and max_pool > 0:
        warnings.append(
            f"{n_partial}/{n_queries_in_db} queries have a partial doc pool "
            f"(max {max_pool:,} docs/query in DB; not all queries reached full pool size)."
        )

    is_complete = loop_completed and null_rows == 0 and not skipped and n_partial == 0

    return {
        "db_path": str(db_path),
        "loop_completed": loop_completed,
        "is_complete": is_complete,
        "fatal_error": fatal_error,
        "fatal_error_query_idx": int(fatal_q) if fatal_q and fatal_q.isdigit() else None,
        "staging_last_completed_stage": int(staging_stage) if staging_stage and staging_stage.isdigit() else None,
        "total_rows": int(total_rows),
        "scored_rows": int(scored_rows),
        "null_score_rows": int(null_rows),
        "sync_pct": float(sync_pct),
        "n_queries_in_db": n_queries_in_db,
        "n_queries_scored": n_queries_scored,
        "n_queries_analyzed": n_queries_scored,  # alias for summaries
        "n_skipped": len(skipped),
        "skipped_queries": skipped,
        "unique_docs_in_pool": unique_docs,
        "pool_size_min": int(pool_sizes.min()) if len(pool_sizes) else 0,
        "pool_size_max": int(pool_sizes.max()) if len(pool_sizes) else 0,
        "pool_size_median": float(pool_sizes.median()) if len(pool_sizes) else 0.0,
        "n_queries_full_pool": n_full,
        "n_queries_partial_pool": n_partial,
        "queries_with_null_scores": null_by_query,
        "warnings": warnings,
    }


def save_run_health(health: dict, out_dir: Path) -> Path:
    path = out_dir / "run_health.json"
    path.write_text(json.dumps(health, indent=2), encoding="utf-8")
    return path


def load_run_health(out_dir: Path, db_path: Path | None = None) -> dict:
    path = out_dir / "run_health.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if db_path is not None:
        return collect_run_health(db_path)
    raise FileNotFoundError(f"run_health.json not found in {out_dir} and no db_path provided")


def format_health_text_lines(health: dict, width: int = 70) -> list[str]:
    """Plain-text run-health banner for summary .txt files."""
    import textwrap

    status = "COMPLETE" if health.get("is_complete") else "INCOMPLETE — interpret metrics with caution"
    lines = [
        "RUN HEALTH",
        f"  Status              : {status}",
        f"  Scored / total rows : {health['scored_rows']:,} / {health['total_rows']:,} "
        f"({health['sync_pct']:.1f}% synced)",
        f"  Queries analyzed    : {health['n_queries_analyzed']} "
        f"(in DB: {health['n_queries_in_db']}, skipped: {health['n_skipped']})",
        f"  Doc pool per query  : {health['pool_size_min']:,}–{health['pool_size_max']:,} "
        f"(median {health['pool_size_median']:,.0f}; "
        f"{health['n_queries_full_pool']} full, {health['n_queries_partial_pool']} partial)",
        f"  Unique docs in pool : {health['unique_docs_in_pool']:,}",
    ]
    if health.get("staging_last_completed_stage") is not None:
        lines.append(f"  Last completed stage: {health['staging_last_completed_stage']}")
    if health.get("fatal_error"):
        q = health.get("fatal_error_query_idx")
        q_txt = f" (query {q})" if q is not None else ""
        err = health["fatal_error"]
        if len(err) > 100:
            err = err[:97] + "…"
        lines.append(f"  Fatal error{q_txt}     : {err}")
    if health.get("skipped_queries"):
        for qidx, reason in health["skipped_queries"]:
            lines.append(f"  Skipped query {qidx}  : {reason}")
    if health.get("queries_with_null_scores"):
        for qidx, n_null in health["queries_with_null_scores"]:
            lines.append(f"  Query {qidx} NULL scores: {n_null:,} rows excluded from analysis")
    if health.get("warnings"):
        lines.append("  Warnings:")
        for w in health["warnings"]:
            lines.extend(textwrap.wrap(f"    • {w}", width=width))
    return lines


def format_health_html(health: dict) -> str:
    """HTML callout for report.html."""
    if health.get("is_complete"):
        kind = "good"
        title = "Run complete — all rows scored, no skipped queries."
    else:
        kind = "warn"
        title = "Partial run — metrics reflect scored data only; see details below."

    items = [
        f"<b>Status:</b> {'Complete' if health.get('is_complete') else 'Incomplete'}",
        f"<b>Synced:</b> {health['scored_rows']:,} / {health['total_rows']:,} rows "
        f"({health['sync_pct']:.1f}%)",
        f"<b>Queries:</b> {health['n_queries_analyzed']} analyzed "
        f"({health['n_skipped']} skipped, {health['n_queries_in_db']} in DB)",
        f"<b>Doc pool:</b> {health['pool_size_min']:,}–{health['pool_size_max']:,} docs/query "
        f"({health['n_queries_partial_pool']} queries partial)",
    ]
    if health.get("staging_last_completed_stage") is not None:
        items.append(f"<b>Last stage:</b> {health['staging_last_completed_stage']}")
    if health.get("fatal_error"):
        q = health.get("fatal_error_query_idx")
        q_txt = f" (query {q})" if q is not None else ""
        err = health["fatal_error"]
        if len(err) > 160:
            err = err[:157] + "…"
        items.append(f"<b>Fatal error{q_txt}:</b> {err}")

    body = "<br>".join(items)
    if health.get("warnings"):
        body += "<ul style='margin:8px 0 0 18px;font-size:12px'>"
        for w in health["warnings"]:
            body += f"<li>{w}</li>"
        body += "</ul>"

    icon = {"good": "✓", "warn": "⚠", "info": "ℹ"}.get(kind, "•")
    return (
        f'<div class="callout {kind}"><span class="icon">{icon}</span>'
        f"<b>{title}</b><br>{body}</div>"
    )


# ── Structured output layout ──────────────────────────────────────────────────

class AnalysisLayout:
    """
    Canonical folder layout under ``<run_dir>/analysis/``:

        analysis/
          INDEX.md                 ← catalog (global vs per-query)
          run_health.json
          report.html
          summaries/               ← text summaries
          csv/global/              ← pool-level aggregates (one table per file)
          csv/per_query/           ← one row per query_idx
          charts/base/             ← charts from analyze_experiment.py
          charts/extended/         ← charts from analyze_experiment_extended.py
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.summaries = self.root / "summaries"
        self.csv_global = self.root / "csv" / "global"
        self.csv_per_query = self.root / "csv" / "per_query"
        self.charts_base = self.root / "charts" / "base"
        self.charts_extended = self.root / "charts" / "extended"

    def ensure(self) -> "AnalysisLayout":
        for d in (
            self.root,
            self.summaries,
            self.csv_global,
            self.csv_per_query,
            self.charts_base,
            self.charts_extended,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def global_csv(self, name: str) -> Path:
        return self.csv_global / name

    def per_query_csv(self, name: str) -> Path:
        return self.csv_per_query / name

    def chart_base(self, name: str) -> Path:
        return self.charts_base / name

    def chart_extended(self, name: str) -> Path:
        return self.charts_extended / name

    def summary_txt(self, name: str) -> Path:
        return self.summaries / name

    def report_html(self) -> Path:
        return self.root / "report.html"

    def run_health_json(self) -> Path:
        return self.root / "run_health.json"

    def index_md(self) -> Path:
        return self.root / "INDEX.md"


# Legacy flat paths (pre-subfolder layout) — checked by resolve_artifact()
_LEGACY_FLAT = {
    "correlation_summary.csv", "llm_recovery_at_k.csv", "mrr_and_precision_at_k.csv",
    "roc_pr_summary.csv", "score_distribution_summary.csv", "distance_quantile_analysis.csv",
    "topk_jaccard_summary.csv", "threshold_filter_analysis.csv", "score_level_breakdown.csv",
    "false_positive_analysis.csv", "marginal_score_value.csv", "corrected_mrr_at_k.csv",
    "per_query_correlation.csv", "gt_llm_rank.csv", "gt_score_percentile.csv", "score_gap.csv",
    "topk_jaccard.csv", "tie_structure.csv", "query_difficulty.csv", "tie_corrected_ranks.csv",
    "within_tier_dist.csv", "failure_cases.csv", "per_query_overview.csv",
    "analysis_summary.txt", "extended_summary.txt",
}


def resolve_artifact(root: Path, filename: str) -> Path:
    """
    Resolve an analysis artifact path. Prefer the structured layout; fall back to
    legacy flat ``analysis/<filename>`` from older runs.
    """
    layout = AnalysisLayout(root)
    candidates: list[Path] = []

    if filename.endswith(".csv"):
        if filename in {
            "per_query_correlation.csv", "gt_llm_rank.csv", "gt_score_percentile.csv",
            "score_gap.csv", "topk_jaccard.csv", "tie_structure.csv", "query_difficulty.csv",
            "tie_corrected_ranks.csv", "within_tier_dist.csv", "failure_cases.csv",
            "per_query_overview.csv",
        }:
            candidates.append(layout.per_query_csv(filename))
        else:
            candidates.append(layout.global_csv(filename))
    elif filename.endswith(".txt"):
        candidates.append(layout.summary_txt(filename))
    elif filename.endswith(".png"):
        # try both chart dirs
        candidates.append(layout.charts_base / filename)
        candidates.append(layout.charts_extended / filename)
    elif filename == "report.html":
        candidates.append(layout.report_html())
    elif filename == "run_health.json":
        candidates.append(layout.run_health_json())

    candidates.append(root / filename)

    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


_PER_QUERY_OVERVIEW_JOINS: tuple[tuple[str, str, list[str]], ...] = (
    ("gt_llm_rank.csv", "csv/per_query", [
        "gt_llm_rank", "gt_llm_score", "gt_rank_pct", "n_docs", "rag_failed",
    ]),
    ("per_query_correlation.csv", "csv/per_query", ["spearman_r", "pearson_r", "kendall_tau"]),
    ("query_difficulty.csv", "csv/per_query", ["difficulty", "gt_score", "n_at_top"]),
    ("tie_structure.csv", "csv/per_query", ["n_tied_at_gt", "tier_precision", "gt_uniquely_top"]),
    ("tie_corrected_ranks.csv", "csv/per_query", ["rank_min", "rank_avg", "rank_max"]),
    ("score_gap.csv", "csv/per_query", ["score_gap", "best_ngt_score"]),
    ("within_tier_dist.csv", "csv/per_query", ["pct_ngt_further", "gt_would_win_dist"]),
)


def build_per_query_overview(root: Path) -> Path | None:
    """Merge key per-query CSVs into one master table for human review."""
    layout = AnalysisLayout(root)
    layout.ensure()

    base_path = resolve_artifact(root, "gt_llm_rank.csv")
    if not base_path.exists():
        return None

    overview = pd.read_csv(base_path)
    if "query_idx" not in overview.columns:
        return None

    for filename, _subdir, cols in _PER_QUERY_OVERVIEW_JOINS[1:]:
        path = resolve_artifact(root, filename)
        if not path.exists():
            continue
        extra = pd.read_csv(path)
        if "query_idx" not in extra.columns:
            continue
        use_cols = ["query_idx"] + [c for c in cols if c in extra.columns and c not in overview.columns]
        if len(use_cols) > 1:
            overview = overview.merge(extra[use_cols], on="query_idx", how="left")

    out = layout.per_query_csv("per_query_overview.csv")
    overview.sort_values("query_idx").to_csv(out, index=False)
    return out


def write_analysis_index(root: Path, health: dict | None = None) -> Path:
    """Write INDEX.md explaining global vs per-query artifacts."""
    layout = AnalysisLayout(root)
    layout.ensure()

    n_q = health.get("n_queries_analyzed", "?") if health else "?"
    lines = [
        "# Analysis Output Index",
        "",
        f"Run analysis folder. **{n_q} queries** in DB (see `run_health.json`).",
        "",
        "## How to read these files",
        "",
        "| Scope | Meaning | Example |",
        "|-------|---------|---------|",
        "| **Global** | One metric summarising the **entire pool** (all queries × all docs combined, or aggregated across queries) | `MRR`, `Recovery@K`, ROC-AUC |",
        "| **Per-query** | One **row per `query_idx`** — filter/sort in Excel to inspect individual queries | `gt_llm_rank`, `difficulty` |",
        "",
        "> **Start here for per-query review:** `csv/per_query/per_query_overview.csv`",
        "> (merged view of rank, correlation, difficulty, ties, hybrid win).",
        "",
        "## Summaries (text)",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `summaries/analysis_summary.txt` | Base run summary + run health |",
        "| `summaries/extended_summary.txt` | Tie-corrected conclusions |",
        "| `report.html` | Full interactive report (global + per-query sections) |",
        "",
        "## CSV — global (pool-level)",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `correlation_summary.csv` | Median/mean correlation across queries |",
        "| `llm_recovery_at_k.csv` | GT in top-K by LLM (all queries) |",
        "| `mrr_and_precision_at_k.csv` | Optimistic MRR / P@K |",
        "| `corrected_mrr_at_k.csv` | Tie-corrected MRR / P@K (min/avg/max rank) |",
        "| `roc_pr_summary.csv` | Global ROC-AUC / average precision |",
        "| `score_distribution_summary.csv` | GT vs non-GT score stats |",
        "| `distance_quantile_analysis.csv` | LLM score by dist decile (non-GT) |",
        "| `topk_jaccard_summary.csv` | LLM vs dist top-K agreement (aggregated) |",
        "| `threshold_filter_analysis.csv` | Precision/recall at each score threshold |",
        "| `score_level_breakdown.csv` | Stats per discrete LLM score level |",
        "| `false_positive_analysis.csv` | Non-GT docs at score=1.0 vs 0.1 |",
        "| `marginal_score_value.csv` | dist improvement per score level |",
        "",
        "## CSV — per-query (one row per query_idx)",
        "",
        "| File | Key columns |",
        "|------|-------------|",
        "| **`per_query_overview.csv`** | **Master merge — start here** |",
        "| `per_query_correlation.csv` | spearman_r, pearson_r, kendall_tau |",
        "| `gt_llm_rank.csv` | gt_llm_rank, gt_llm_score, gt_rank_pct |",
        "| `query_difficulty.csv` | EASY / MEDIUM / HARD / FAILED |",
        "| `tie_structure.csv` | n_tied_at_gt, tier_precision |",
        "| `tie_corrected_ranks.csv` | rank_min, rank_avg, rank_max |",
        "| `score_gap.csv` | GT score − best non-GT |",
        "| `within_tier_dist.csv` | Hybrid dist tiebreaker per query |",
        "| `failure_cases.csv` | Queries where GT did not score 1.0 |",
        "| `topk_jaccard.csv` | Jaccard vs dist ranking per query × K |",
        "",
        "## Charts",
        "",
        "| Folder | Source script |",
        "|--------|---------------|",
        "| `charts/base/` | analyze_experiment.py |",
        "| `charts/extended/` | analyze_experiment_extended.py |",
        "",
        "## Re-running analysis",
        "",
        "- Re-running **replaces** files in this folder (same paths, no versioning).",
        "- Safe to run after a partial run — each step overwrites only its own outputs.",
        "- Use `python run_global_correlation_analysis.py --steps extended,report` to skip base.",
        "",
    ]
    path = layout.index_md()
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
