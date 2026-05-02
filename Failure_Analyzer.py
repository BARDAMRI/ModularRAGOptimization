"""
Failure_Analyzer.py — Forensic analysis of the pipeline experiment results.

What it shows
─────────────
• Per-chunk submission summary  : which chunks succeeded / errored
• Skipped queries               : GT-missing queries and their reasons
• Non-GT imposters              : docs that scored higher than their query's GT
• GT score breakdown            : distribution of GT scores (was GT found?)
• Zero-score hot-spots          : which queries have the most blocked responses
• Spearman correlation overview : best / worst correlating queries
• RAG-failure breakdown         : queries where top-1 retrieval missed GT

Run modes
─────────
  python Failure_Analyzer.py              → auto-detect latest run, print to terminal
  python Failure_Analyzer.py path/to/run  → specific run directory
  python Failure_Analyzer.py path/to/db   → specific DB file

Output file
───────────
<run_dir>/failure_report.txt  (overwritten on each run — latest state only)
"""

import os
import sqlite3
import sys
from datetime import datetime

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

PROJECT_PATH        = os.path.dirname(os.path.abspath(__file__))
BASE_DIR            = os.path.join(PROJECT_PATH, "results", "global_exp")
REPORT_FILE         = os.path.join(BASE_DIR, "failure_report.txt")
MIN_IMPOSTER_SCORE  = 0.5   # (on 0-1 scale = 5/10) threshold for an "imposter"
TOP_IMPOSTERS       = 30    # max rows in the imposter table

console = Console()


# ─── helpers ────────────────────────────────────────────────────────────────

def get_latest_db() -> str | None:
    try:
        runs = [
            os.path.join(BASE_DIR, d)
            for d in os.listdir(BASE_DIR)
            if os.path.isdir(os.path.join(BASE_DIR, d))
        ]
        if not runs:
            return None
        # Pick latest run folder that actually has experiment_results.db.
        for run_dir in sorted(runs, key=os.path.getmtime, reverse=True):
            candidate = os.path.join(run_dir, "experiment_results.db")
            if os.path.exists(candidate):
                return candidate
        return None
    except Exception:
        return None


def resolve_db_path(user_input: str | None) -> str | None:
    """
    Resolve user-provided path to experiment_results.db.
    Required fallback behavior: invalid/missing path -> latest run DB.
    """
    raw = (user_input or "").strip()
    if not raw:
        return get_latest_db()

    p = os.path.expanduser(raw)
    if os.path.isfile(p):
        return p if os.path.basename(p) == "experiment_results.db" else get_latest_db()
    if os.path.isdir(p):
        candidate = os.path.join(p, "experiment_results.db")
        return candidate if os.path.exists(candidate) else get_latest_db()
    return get_latest_db()


def _report_file_for_db(db_path: str) -> str:
    """Write report next to the monitored run."""
    return os.path.join(os.path.dirname(db_path), "failure_report.txt")


def _read_meta(conn) -> dict:
    meta = {}
    try:
        for row in conn.execute("SELECT key, value FROM experiment_meta").fetchall():
            meta[str(row[0])] = str(row[1])
    except Exception:
        pass
    return meta


def _read_run_mode(db_path: str) -> str:
    try:
        cfg_path = os.path.join(os.path.dirname(db_path), "experiment_config.json")
        if not os.path.exists(cfg_path):
            return ""
        import json
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return str(cfg.get("mode", "")).strip().lower()
    except Exception:
        return ""


# ─── individual analyses ─────────────────────────────────────────────────────

def section_pipeline_summary(conn, meta: dict, report_lines: list[str], run_mode: str = "") -> None:
    chunk_jobs   = {k: v for k, v in meta.items() if k.startswith("chunk_") and k.endswith("_job_id")}
    chunk_errors = {k: v for k, v in meta.items() if k.startswith("chunk_") and k.endswith("_error")}
    loop_done    = meta.get("main_loop_completed", "0") in ("1", "true", "yes")
    all_ids      = meta.get("all_job_ids", "")
    fatal_llm_error = meta.get("fatal_llm_error", "")
    fatal_q_idx = meta.get("fatal_llm_error_query_idx", "")

    is_ollama = "ollama" in str(run_mode).lower()
    table_title = "📦 Pipeline Submission Summary" if not is_ollama else "📦 Ollama Run Summary"
    table = Table(title=table_title, show_lines=True)
    table.add_column("Item",  style="dim", min_width=28)
    table.add_column("Value", min_width=40)

    table.add_row("Pipeline loop completed", "✅ Yes" if loop_done else "⏳ Still running")
    if is_ollama:
        table.add_row("Provider", "Ollama (cluster)")
        if fatal_llm_error:
            table.add_row("Fatal LLM error", f"[red]q{fatal_q_idx}: {fatal_llm_error[:100]}[/red]")
    else:
        table.add_row("Chunks with job IDs",     str(len(chunk_jobs)))
        table.add_row("Chunks with errors",      f"[red]{len(chunk_errors)}[/red]")

    n_submitted = 0
    try:
        n_submitted = conn.execute("SELECT COUNT(DISTINCT custom_id) FROM batch_request_map").fetchone()[0]
    except Exception:
        pass
    if is_ollama:
        table.add_row("Batch requests submitted", "N/A (live scoring)")
        table.add_row("All job IDs (comma list)", "N/A")
    else:
        table.add_row("Batch requests submitted", f"{n_submitted:,}")
        table.add_row("All job IDs (comma list)",  all_ids[:80] + ("…" if len(all_ids) > 80 else ""))

    console.print(table)

    report_lines += [
        "[PIPELINE SUMMARY]",
        f"  Loop completed     : {loop_done}",
        f"  Chunks w/ job IDs  : {len(chunk_jobs)}",
        f"  Chunks w/ errors   : {len(chunk_errors)}",
        f"  Batch requests     : {n_submitted:,}",
        f"  All job IDs        : {all_ids[:120]}",
    ]

    if chunk_errors and not is_ollama:
        err_table = Table(title="⚠️  Submission Errors by Chunk", show_lines=True)
        err_table.add_column("Chunk", style="cyan", justify="right")
        err_table.add_column("Error", style="red")
        report_lines.append("\n[SUBMISSION ERRORS]")
        for k in sorted(chunk_errors):
            chunk_num = k.replace("chunk_", "").replace("_error", "")
            err_table.add_row(chunk_num, chunk_errors[k][:100])
            report_lines.append(f"  Chunk {chunk_num}: {chunk_errors[k][:120]}")
        console.print(err_table)
    report_lines.append("")


def section_skipped_queries(conn, report_lines: list[str]) -> None:
    try:
        df = pd.read_sql(
            "SELECT query_idx, reason, created_at FROM skipped_queries ORDER BY query_idx",
            conn,
        )
    except Exception:
        df = pd.DataFrame()

    table = Table(title=f"🚫 Skipped Queries ({len(df)} total)", show_lines=True)
    table.add_column("Q idx", justify="right", style="cyan")
    table.add_column("Reason", style="yellow")
    table.add_column("At", style="dim")

    report_lines.append(f"[SKIPPED QUERIES]  total={len(df)}")
    for _, row in df.iterrows():
        table.add_row(str(int(row["query_idx"])), str(row["reason"]), str(row.get("created_at", "")))
        report_lines.append(f"  q{int(row['query_idx'])}: {row['reason']}")

    if df.empty:
        table.add_row("[dim]none[/dim]", "", "")
    console.print(table)
    report_lines.append("")


def section_score_overview(conn, report_lines: list[str]) -> None:
    try:
        total_docs = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        synced     = conn.execute(
            "SELECT COUNT(*) FROM results WHERE llm_score IS NOT NULL"
        ).fetchone()[0]
        zero_score = conn.execute(
            "SELECT COUNT(*) FROM results WHERE llm_score = 0.0"
        ).fetchone()[0]
        null_score = conn.execute(
            "SELECT COUNT(*) FROM results WHERE llm_score IS NULL"
        ).fetchone()[0]
        total_q    = conn.execute("SELECT COUNT(DISTINCT query_idx) FROM results").fetchone()[0]
        synced_q   = conn.execute(
            "SELECT COUNT(DISTINCT query_idx) FROM results WHERE llm_score IS NOT NULL"
        ).fetchone()[0]
    except Exception:
        console.print("[red]Could not read results table[/red]")
        return

    sync_pct = f"{synced/total_docs*100:.1f}%" if total_docs else "—"

    table = Table(title="📊 DB Score Overview", show_lines=False)
    table.add_column("Metric",      style="dim", min_width=28)
    table.add_column("Value",       min_width=20)

    table.add_row("Total docs in DB",     f"{total_docs:,}")
    table.add_row("Synced (score ≠ NULL)",f"[bold cyan]{synced:,}[/bold cyan]  ({sync_pct})")
    table.add_row("Pending (NULL score)", f"[yellow]{null_score:,}[/yellow]")
    table.add_row("Blocked/Zero (0.0)",   f"[bold red]{zero_score:,}[/bold red]")
    table.add_row("Queries total",        str(total_q))
    table.add_row("Queries synced",       str(synced_q))

    console.print(table)
    report_lines += [
        "[DB SCORE OVERVIEW]",
        f"  Total docs      : {total_docs:,}",
        f"  Synced          : {synced:,}  ({sync_pct})",
        f"  Pending (NULL)  : {null_score:,}",
        f"  Blocked (0.0)   : {zero_score:,}",
        f"  Queries total   : {total_q}",
        f"  Queries synced  : {synced_q}",
        "",
    ]

    # score distribution
    try:
        sdf = pd.read_sql(
            "SELECT llm_score, COUNT(*) as cnt FROM results "
            "WHERE llm_score IS NOT NULL GROUP BY llm_score ORDER BY llm_score",
            conn,
        )
        if not sdf.empty:
            total_s = sdf["cnt"].sum()
            max_c   = sdf["cnt"].max()
            dist_table = Table(title="📈 Score Distribution (0–1 scale)", show_lines=False)
            dist_table.add_column("Score", justify="right", style="bold")
            dist_table.add_column("Bar (relative)",  min_width=34)
            dist_table.add_column("Count",  justify="right")
            dist_table.add_column("Pct",    justify="right", style="dim")
            report_lines.append("[SCORE DISTRIBUTION]")
            for _, row in sdf.iterrows():
                raw = int(round(row["llm_score"] * 10))
                cnt = int(row["cnt"])
                pct = cnt / total_s * 100
                bar = "█" * int((cnt / max_c) * 30)
                dist_table.add_row(str(raw), bar, f"{cnt:,}", f"{pct:.1f}%")
                report_lines.append(f"  {raw:>2}  {cnt:>7,}  ({pct:4.1f}%)")
            console.print(dist_table)
            report_lines.append("")
    except Exception:
        pass


def section_gt_scores(conn, report_lines: list[str]) -> None:
    """Distribution of GT-doc scores — tells us whether GT was found well."""
    try:
        gt_df = pd.read_sql(
            "SELECT llm_score FROM results WHERE is_gt = 1 AND llm_score IS NOT NULL",
            conn,
        )
    except Exception:
        gt_df = pd.DataFrame()
    if gt_df.empty:
        console.print("[dim]No GT scores available yet.[/dim]")
        return

    scores = (gt_df["llm_score"] * 10).round().astype(int)
    counts = scores.value_counts().sort_index()
    total  = len(scores)
    mean_s = gt_df["llm_score"].mean() * 10
    hi     = (gt_df["llm_score"] >= 0.8).sum()   # 8+
    lo     = (gt_df["llm_score"] <= 0.3).sum()   # 3-

    table = Table(title=f"🎯 GT Document Score Distribution  (n={total})", show_lines=False)
    table.add_column("Score", justify="right", style="bold green")
    table.add_column("Count", justify="right")
    table.add_column("Pct",   justify="right", style="dim")
    report_lines.append(f"[GT SCORE DISTRIBUTION]  n={total}  mean={mean_s:.2f}")
    for sc, cnt in counts.items():
        pct = cnt / total * 100
        table.add_row(str(sc), f"{cnt:,}", f"{pct:.1f}%")
        report_lines.append(f"  score {sc:>2}: {cnt:>5,}  ({pct:.1f}%)")
    console.print(table)
    console.print(
        f"  Mean GT score: [bold]{mean_s:.2f}/10[/bold]  |  "
        f"High (≥8): [green]{hi}[/green]  |  Low (≤3): [red]{lo}[/red]"
    )
    report_lines += [
        f"  Mean={mean_s:.2f}  High(>=8)={hi}  Low(<=3)={lo}",
        "",
    ]


def section_imposters(conn, report_lines: list[str]) -> None:
    """Non-GT docs that outscored their query's GT document."""
    try:
        imp_df = pd.read_sql(
            f"""
            WITH gt AS (
                SELECT query_idx,
                       llm_score  AS gt_score,
                       dist_to_gt AS gt_dist
                FROM   results
                WHERE  is_gt = 1
            )
            SELECT
                r.query_idx,
                r.doc_id,
                r.llm_score                        AS imp_score,
                g.gt_score,
                r.llm_score - COALESCE(g.gt_score, 0) AS score_diff,
                r.dist_to_gt                       AS imp_dist,
                g.gt_dist,
                r.rag_failed
            FROM results r
            LEFT JOIN gt g ON r.query_idx = g.query_idx
            WHERE r.is_gt = 0
              AND r.llm_score IS NOT NULL
              AND r.llm_score >= {MIN_IMPOSTER_SCORE}
            ORDER BY score_diff DESC, r.llm_score DESC
            LIMIT {TOP_IMPOSTERS}
            """,
            conn,
        )
    except Exception as exc:
        console.print(f"[red]Imposter query failed: {exc}[/red]")
        return

    n_total_imposters = 0
    try:
        n_total_imposters = conn.execute(
            f"SELECT COUNT(*) FROM results r "
            f"JOIN (SELECT query_idx, llm_score as gt_score FROM results WHERE is_gt=1) g "
            f"  ON r.query_idx=g.query_idx "
            f"WHERE r.is_gt=0 AND r.llm_score IS NOT NULL AND r.llm_score>=g.gt_score"
        ).fetchone()[0]
    except Exception:
        pass

    table = Table(
        title=f"🕵️  Imposters — Non-GT scored ≥ {int(MIN_IMPOSTER_SCORE*10)}/10  "
              f"(showing top {TOP_IMPOSTERS}, total ≥ GT: {n_total_imposters})",
        show_lines=True,
    )
    table.add_column("Q",            justify="right", style="cyan")
    table.add_column("Doc ID",       style="magenta")
    table.add_column("Imp Score",    justify="right", style="bold red")
    table.add_column("GT Score",     justify="right", style="bold green")
    table.add_column("Δ (imp-gt)",   justify="right", style="yellow")
    table.add_column("Dist(Imp)",    justify="right", style="dim")
    table.add_column("Dist(GT)",     justify="right", style="dim")
    table.add_column("RAG fail",     justify="center")

    report_lines.append(
        f"[IMPOSTERS]  score>={int(MIN_IMPOSTER_SCORE*10)}  total_beating_gt={n_total_imposters}"
    )
    for _, row in imp_df.iterrows():
        gt_s   = f"{row['gt_score']:.2f}"    if pd.notnull(row["gt_score"])  else "[red]MISS[/red]"
        gt_d   = f"{row['gt_dist']:.4f}"     if pd.notnull(row["gt_dist"])   else "—"
        delta  = f"{row['score_diff']:+.2f}" if pd.notnull(row["score_diff"]) else "—"
        rag_f  = "❌" if row.get("rag_failed") else "✓"
        short_id = str(row["doc_id"])[:22]
        table.add_row(
            str(int(row["query_idx"])),
            short_id,
            f"{row['imp_score']:.2f}",
            gt_s,
            delta,
            f"{row['imp_dist']:.4f}",
            gt_d,
            rag_f,
        )
        report_lines.append(
            f"  q{int(row['query_idx'])}  {short_id:<24}  imp={row['imp_score']:.2f}  "
            f"gt={row.get('gt_score', 'MISS')}  delta={delta}"
        )
    if imp_df.empty:
        table.add_row("[dim]none yet[/dim]", "", "", "", "", "", "", "")
    console.print(table)
    report_lines.append("")


def section_zero_score_hotspots(conn, report_lines: list[str]) -> None:
    """Queries with the most blocked/zero-score docs."""
    try:
        z_df = pd.read_sql(
            "SELECT query_idx, COUNT(*) AS zeros "
            "FROM results WHERE llm_score = 0.0 "
            "GROUP BY query_idx ORDER BY zeros DESC LIMIT 15",
            conn,
        )
    except Exception:
        z_df = pd.DataFrame()
    if z_df.empty:
        return

    table = Table(title="🔴 Zero-Score Hot-Spots (blocked/safety-filtered)", show_lines=False)
    table.add_column("Query idx", justify="right", style="cyan")
    table.add_column("Zero-score docs", justify="right", style="bold red")
    report_lines.append("[ZERO SCORE HOTSPOTS]")
    for _, row in z_df.iterrows():
        table.add_row(str(int(row["query_idx"])), str(int(row["zeros"])))
        report_lines.append(f"  q{int(row['query_idx'])}: {int(row['zeros'])} zeros")
    console.print(table)
    report_lines.append("")


def section_spearman(db_path: str, report_lines: list[str]) -> None:
    """Spearman overview from summary_stats.csv."""
    csv_path = os.path.join(os.path.dirname(db_path), "summary_stats.csv")
    if not os.path.exists(csv_path):
        console.print("[dim]No summary_stats.csv yet — run sync_batch_results first.[/dim]")
        return
    try:
        sdf = pd.read_csv(csv_path)
    except Exception as exc:
        console.print(f"[red]Could not read summary_stats.csv: {exc}[/red]")
        return
    if sdf.empty:
        return

    mean_r   = sdf["correlation"].mean()
    median_r = sdf["correlation"].median()
    std_r    = sdf["correlation"].std()
    pos      = (sdf["correlation"] > 0).sum()
    n        = len(sdf)
    sig      = (sdf["p_value"] < 0.05).sum() if "p_value" in sdf else "—"
    rag_fail = sdf["rag_failed"].sum() if "rag_failed" in sdf else "—"

    overview = Table(title="📈 Spearman Correlation Summary", show_lines=False)
    overview.add_column("Metric", style="dim", min_width=26)
    overview.add_column("Value",  min_width=20)
    overview.add_row("Queries analysed",     str(n))
    overview.add_row("Mean r",               f"{mean_r:+.4f}")
    overview.add_row("Median r",             f"{median_r:+.4f}")
    overview.add_row("Std r",                f"{std_r:.4f}")
    overview.add_row("Positive correlation", f"{pos} / {n}  ({pos/n*100:.1f}%)")
    overview.add_row("Significant (p<.05)",  str(sig))
    overview.add_row("RAG-failed queries",   str(rag_fail))
    console.print(overview)

    report_lines += [
        "[SPEARMAN SUMMARY]",
        f"  n={n}  mean={mean_r:+.4f}  median={median_r:+.4f}  std={std_r:.4f}",
        f"  positive={pos}/{n}  significant={sig}  rag_failed={rag_fail}",
    ]

    # ── worst & best 5 ──
    best  = sdf.nlargest(5, "correlation")[["query_idx", "correlation", "p_value", "rag_failed"]]
    worst = sdf.nsmallest(5, "correlation")[["query_idx", "correlation", "p_value", "rag_failed"]]

    for label, subset in [("🏆 Best 5", best), ("💀 Worst 5", worst)]:
        t = Table(title=label, show_lines=False)
        t.add_column("Q idx",   justify="right", style="cyan")
        t.add_column("r",       justify="right", style="bold")
        t.add_column("p-val",   justify="right", style="dim")
        t.add_column("RAG fail",justify="center")
        for _, row in subset.iterrows():
            t.add_row(
                str(int(row["query_idx"])),
                f"{row['correlation']:+.4f}",
                f"{row['p_value']:.4f}" if pd.notnull(row.get("p_value")) else "—",
                "❌" if row.get("rag_failed") else "✓",
            )
        console.print(t)
    report_lines.append("")


# ─── main report ────────────────────────────────────────────────────────────

def analyze_failures(db_path: str, report_file: str | None = None) -> None:
    if not db_path or not os.path.exists(db_path):
        console.print(f"[red]DB not found: {db_path}[/red]")
        return

    run_name = os.path.basename(os.path.dirname(db_path))
    now_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        f"# Failure Analysis Report",
        f"# Generated : {now_str}",
        f"# Run       : {run_name}",
        "",
    ]

    console.print(Panel(
        Text(f"Failure Analyzer\nRun: {run_name}\n{now_str}", justify="center"),
        border_style="bold magenta",
        title="🔬 ModularRAG — Experiment Post-Mortem",
    ))

    run_mode = _read_run_mode(db_path)
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        meta = _read_meta(conn)

        section_pipeline_summary(conn, meta, report_lines, run_mode=run_mode)
        section_skipped_queries(conn, report_lines)
        section_score_overview(conn, report_lines)
        section_gt_scores(conn, report_lines)
        section_imposters(conn, report_lines)
        section_zero_score_hotspots(conn, report_lines)

        conn.close()
    except Exception as exc:
        console.print(f"[bold red]Fatal DB error: {exc}[/bold red]")
        return

    section_spearman(db_path, report_lines)

    # ── write report file (overwrite — latest state only) ──
    target_report = report_file or _report_file_for_db(db_path)
    os.makedirs(os.path.dirname(target_report), exist_ok=True)
    try:
        with open(target_report, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")
        console.print(
            Panel(
                f"[bold green]Report saved →[/bold green] {target_report}",
                border_style="green",
            )
        )
    except Exception as exc:
        console.print(f"[red]Could not write report file: {exc}[/red]")


if __name__ == "__main__":
    db_path = resolve_db_path(sys.argv[1] if len(sys.argv) > 1 else "")
    analyze_failures(db_path)
