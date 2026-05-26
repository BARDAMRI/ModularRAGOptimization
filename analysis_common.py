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
    conn.close()

    n_queries_in_db = len(per_query)
    n_queries_scored = int((per_query["null_scores"] < per_query["n_rows"]).sum())
    unique_docs = int(
        pd.read_sql_query("SELECT COUNT(DISTINCT doc_id) AS n FROM results", sqlite3.connect(db_path))["n"].iloc[0]
    )

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
