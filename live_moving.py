"""
live_moving.py — Real-time dashboard for the chunk-based pipeline experiment.

What it shows
─────────────
• Pipeline progress  : chunks produced / submitted / completed
• Queue state        : how many Google jobs are PENDING / RUNNING / DONE / FAILED
• Local DB sync      : placeholder rows vs synced (llm_score filled) rows
• Score distribution : histogram of all synced scores so far
• Skipped / errors   : queries that were permanently skipped or failed submission
• ETA estimate       : based on elapsed time and remaining chunks

A plain-text snapshot (``results/global_exp/live_status.txt``) is overwritten
every refresh cycle so the latest state is always available without scrolling.

Run:
    python live_moving.py
    python live_moving.py results/global_exp/pilot_run_ollama_2026-04-29_16-24-59
    python live_moving.py results/global_exp/pilot_run_ollama_2026-04-29_16-24-59/experiment_results.db
"""

import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ─── Google Batch client (optional) ─────────────────────────────────────────
try:
    from configurations.config import GEMINI_API_KEY
    from utility.llm_gateway import get_gemini_client
    _genai_client = get_gemini_client(GEMINI_API_KEY)
except Exception:
    _genai_client = None

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(PROJECT_PATH, "results", "global_exp")
SNAPSHOT_FILE = os.path.join(BASE_DIR, "live_status.txt")
REFRESH_SECONDS = 10          # Interval between full refreshes
GOOGLE_POLL_EVERY = 3         # Poll Google only every N refreshes (avoid rate limits)

console = Console()

# ─── helpers ────────────────────────────────────────────────────────────────

def _fmt_duration(secs: float) -> str:
    td = timedelta(seconds=int(secs))
    h, rem = divmod(td.seconds, 3600)
    m, s = divmod(rem, 60)
    if td.days:
        return f"{td.days}d {h:02d}h {m:02d}m"
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


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
    Resolve an explicit DB/run path to experiment_results.db.
    Fallback behavior (required): if missing/invalid input -> latest run DB.
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


def _snapshot_file_for_db(db_path: str) -> str:
    """Keep snapshot next to the monitored run DB."""
    return os.path.join(os.path.dirname(db_path), "live_status.txt")


# ─── data gathering ─────────────────────────────────────────────────────────

def _read_db(db_path: str) -> dict:
    """Read ALL relevant state from SQLite in one pass (read-only)."""
    data = {
        # meta
        "run_name": os.path.basename(os.path.dirname(db_path)),
        "main_loop_done": False,
        "all_job_ids_raw": None,
        "chunk_job_meta": {},    # chunk_XXXX_job_id -> job_id
        "chunk_err_meta": {},    # chunk_XXXX_error  -> error
        "fatal_llm_error": "",
        "fatal_llm_error_query_idx": "",
        # results
        "total_placeholder_rows": 0,
        "synced_rows": 0,
        "zero_score_rows": 0,
        "total_queries_with_rows": 0,
        "synced_queries": 0,
        "skipped_queries": 0,
        "total_requests": 0,
        "scores_df": pd.DataFrame(),
        "spearman_df": pd.DataFrame(),
        "run_mode": "",
        # staged experiment totals (from manifest)
        "staged_total_pairs": 0,
        "staged_ranked_pool_size": 0,
        "staged_n_queries": 0,
        "staged_current_stage": 0,
        "staged_total_stages": 0,
        "staged_ranked_cursor": 0,
    }
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            # experiment_meta table
            for row in conn.execute("SELECT key, value FROM experiment_meta").fetchall():
                k, v = row
                if k == "main_loop_completed":
                    data["main_loop_done"] = str(v).strip() in ("1", "true", "yes")
                elif k == "all_job_ids":
                    data["all_job_ids_raw"] = str(v)
                elif k.startswith("chunk_") and k.endswith("_job_id") and "upload_part" not in k:
                    data["chunk_job_meta"][k] = str(v)
                elif k.startswith("chunk_") and k.endswith("_error"):
                    data["chunk_err_meta"][k] = str(v)
                elif k == "fatal_llm_error":
                    data["fatal_llm_error"] = str(v)
                elif k == "fatal_llm_error_query_idx":
                    data["fatal_llm_error_query_idx"] = str(v)
        except Exception:
            pass

        try:
            data["total_placeholder_rows"] = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            data["synced_rows"] = conn.execute(
                "SELECT COUNT(*) FROM results WHERE llm_score IS NOT NULL"
            ).fetchone()[0]
            data["zero_score_rows"] = conn.execute(
                "SELECT COUNT(*) FROM results WHERE llm_score = 0.0"
            ).fetchone()[0]
            data["total_queries_with_rows"] = conn.execute(
                "SELECT COUNT(DISTINCT query_idx) FROM results"
            ).fetchone()[0]
            data["synced_queries"] = conn.execute(
                "SELECT COUNT(DISTINCT query_idx) FROM results WHERE llm_score IS NOT NULL"
            ).fetchone()[0]
        except Exception:
            pass

        try:
            data["skipped_queries"] = conn.execute("SELECT COUNT(*) FROM skipped_queries").fetchone()[0]
        except Exception:
            pass

        try:
            data["total_requests"] = conn.execute("SELECT COUNT(*) FROM batch_request_map").fetchone()[0]
        except Exception:
            pass

        try:
            data["scores_df"] = pd.read_sql(
                "SELECT llm_score, COUNT(*) as cnt FROM results "
                "WHERE llm_score IS NOT NULL GROUP BY llm_score",
                conn,
            )
        except Exception:
            pass

        try:
            # Spearman correlations already computed by analytics
            data["spearman_df"] = pd.read_sql(
                "SELECT query_idx, correlation, p_value, rag_failed FROM results "
                "WHERE 1=0",   # placeholder – filled from CSV if present
                conn,
            )
        except Exception:
            pass

        conn.close()
    except Exception:
        pass

    # Read run mode from config if available
    try:
        import json
        cfg_path = os.path.join(os.path.dirname(db_path), "experiment_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            data["run_mode"] = str(cfg.get("mode", "")).strip().lower()
    except Exception:
        pass

    # Read staged manifest if present (correlation_pool_manifest.json)
    run_dir = os.path.dirname(db_path)
    manifest_path = os.path.join(run_dir, "correlation_pool_manifest.json")
    if os.path.exists(manifest_path):
        try:
            import json as _json
            with open(manifest_path, "r", encoding="utf-8") as _f:
                _m = _json.load(_f)
            _pool_size = len(_m.get("ranked_pool") or []) or int(_m.get("staging_total_stages_expected", 0)) * 10 * 200
            _n_q = len(_m.get("gt_ids") or {})
            _cursor = int(_m.get("ranked_cursor") or 0)
            _last_stage = int(_m.get("staging_last_completed_stage") or 0)
            _total_stages = int(_m.get("staging_total_stages_expected") or 0)
            # total expected pairs = (pool docs + GT docs) × n_queries
            _total_pairs = (_pool_size + _n_q) * _n_q if _n_q else 0
            data["staged_ranked_pool_size"] = _pool_size
            data["staged_n_queries"] = _n_q
            data["staged_ranked_cursor"] = _cursor
            data["staged_current_stage"] = _last_stage
            data["staged_total_stages"] = _total_stages
            data["staged_total_pairs"] = _total_pairs
        except Exception:
            pass

    # Try loading pre-computed summary CSV if it exists
    run_dir = os.path.dirname(db_path)
    csv_path = os.path.join(run_dir, "summary_stats.csv")
    if os.path.exists(csv_path):
        try:
            data["spearman_df"] = pd.read_csv(csv_path)
        except Exception:
            pass

    return data


def _poll_google_jobs(job_ids: list[str]) -> list[dict]:
    """
    Poll Google Batch API for each job_id.
    Returns list of dicts with keys: id, state, successful, total.
    """
    results = []
    if not _genai_client:
        return results
    for jid in job_ids:
        jid = jid.strip()
        if not jid:
            continue
        try:
            job = _genai_client.batches.get(name=jid)
            state = str(getattr(job, "state", "UNKNOWN")).replace("JobState.", "")
            succ, total = 0, 0
            cs = getattr(job, "completion_stats", None)
            if cs:
                total = getattr(cs, "total_requests", 0) or 0
                succ = getattr(cs, "successful_requests", 0) or 0
            results.append({"id": jid, "state": state, "successful": succ, "total": total})
        except Exception as exc:
            results.append({"id": jid, "state": f"ERR({str(exc)[:30]})", "successful": 0, "total": 0})
    return results


# ─── rendering ──────────────────────────────────────────────────────────────

_STATE_COLOR = {
    "SUCCEEDED": "bold green",
    "RUNNING": "bold cyan",
    "PENDING": "yellow",
    "FAILED": "bold red",
    "CANCELLED": "red",
}

def _state_colored(state: str) -> str:
    color = next((v for k, v in _STATE_COLOR.items() if k in state.upper()), "white")
    return f"[{color}]{state}[/{color}]"


def _build_pipeline_table(data: dict, google_rows: list[dict]) -> Table:
    """Top section: pipeline chunk / job overview."""
    t = Table(show_header=True, header_style="bold magenta", expand=True, show_lines=False)
    t.add_column("Metric", style="dim", min_width=26)
    t.add_column("Value", min_width=50)

    # ── overall counts ──
    n_jobs_submitted = len(data["chunk_job_meta"])
    n_errors         = len(data["chunk_err_meta"])
    n_succeeded      = sum(1 for r in google_rows if "SUCCEEDED" in r["state"].upper())
    n_running        = sum(1 for r in google_rows if "RUNNING"   in r["state"].upper())
    n_pending        = sum(1 for r in google_rows if "PENDING"   in r["state"].upper())
    n_failed_google  = sum(1 for r in google_rows if "FAILED"    in r["state"].upper())

    total_ph   = data["total_placeholder_rows"]
    synced     = data["synced_rows"]
    sync_pct   = f"{synced/total_ph*100:.1f}%" if total_ph else "—"
    loop_done  = "✅ Done" if data["main_loop_done"] else "⏳ Running…"

    t.add_row("Run Name",               f"[bold]{data['run_name']}[/bold]")
    t.add_row("Pipeline Loop",           loop_done)

    # Staged experiment progress
    if data["staged_total_pairs"] > 0:
        _tp   = data["staged_total_pairs"]
        _sc   = data["synced_rows"]
        _pct  = f"{_sc / _tp * 100:.2f}%" if _tp else "—"
        _pool = data["staged_ranked_pool_size"]
        _cur  = data["staged_ranked_cursor"]
        _pool_pct = f"{_cur / _pool * 100:.1f}%" if _pool else "—"
        _stg  = data["staged_current_stage"]
        _tot  = data["staged_total_stages"]
        t.add_row("Total comparisons (scored)",
                  f"[bold cyan]{_sc:,}[/bold cyan] / {_tp:,}  ({_pct})")
        t.add_row("Ranked pool progress",
                  f"{_cur:,} / {_pool:,} docs consumed  ({_pool_pct})")
        t.add_row("Stage progress",
                  f"{_stg} / {_tot} stages complete")
    t.add_row("Chunks Submitted",        f"{n_jobs_submitted} chunks  |  {n_errors} submission errors")
    is_ollama = "ollama" in str(data.get("run_mode", "")).lower()
    if is_ollama:
        t.add_row("Provider", "Ollama (cluster live scoring)")
        t.add_row("Google Job States", "[dim]N/A in ollama mode[/dim]")
        t.add_row("Batch Requests Registered", "N/A")
    else:
        t.add_row("Google Job States",
                  f"🟢 {n_succeeded} SUCCEEDED  |  🔵 {n_running} RUNNING  |  🟡 {n_pending} PENDING  |  🔴 {n_failed_google} FAILED"
                  if google_rows else "[dim]not polled yet[/dim]")
        t.add_row("Batch Requests Registered", f"{data['total_requests']:,}")
    t.add_row("Local Placeholder Rows",  f"{total_ph:,}")
    t.add_row("Synced (llm_score filled)", f"[bold cyan]{synced:,}[/bold cyan] / {total_ph:,}  ({sync_pct})")
    t.add_row("Queries synced",          f"{data['synced_queries']} / {data['total_queries_with_rows']}")
    t.add_row("Queries skipped (GT miss)",f"[red]{data['skipped_queries']}[/red]")
    t.add_row("Zero/Blocked scores",     f"[bold red]{data['zero_score_rows']:,}[/bold red]")
    return t


def _build_jobs_table(google_rows: list[dict]) -> str:
    """Per-job status lines."""
    if not google_rows:
        return "[dim]Google Batch status not available (not polled yet or SDK missing)[/dim]"
    lines = []
    for r in google_rows:
        short_id = r["id"].split("/")[-1]
        state_str = _state_colored(r["state"])
        if r["total"]:
            pct = r["successful"] / r["total"] * 100
            prog = f"  {r['successful']:,}/{r['total']:,} ({pct:.0f}%)"
        else:
            prog = ""
        lines.append(f"  {short_id[:18]:<20} {state_str}{prog}")
    return "\n".join(lines)


def _build_score_histogram(scores_df: pd.DataFrame) -> str:
    if scores_df.empty:
        return "Waiting for Sync (0 local scores yet)…"
    total = scores_df["cnt"].sum()
    max_c = scores_df["cnt"].max()
    lines = [f"Total scored: {total:,}"]
    for _, row in scores_df.sort_values("llm_score").iterrows():
        raw = int(round(row["llm_score"] * 10))
        cnt = int(row["cnt"])
        pct = cnt / total * 100
        bar = "█" * int((cnt / max_c) * 32)
        lines.append(f"[{raw:>2}] {bar:<32} {cnt:>6,}  ({pct:4.1f}%)")
    return "\n".join(lines)


def _build_spearman_summary(spearman_df: pd.DataFrame) -> str:
    if spearman_df.empty:
        return "[dim]No Spearman stats yet (summaries computed after sync completes)[/dim]"
    mean_r  = spearman_df["correlation"].mean()
    median_r = spearman_df["correlation"].median()
    positive = (spearman_df["correlation"] > 0).sum()
    n       = len(spearman_df)
    sig     = (spearman_df["p_value"] < 0.05).sum() if "p_value" in spearman_df else "—"
    failed  = spearman_df["rag_failed"].sum() if "rag_failed" in spearman_df else "—"
    return (
        f"Queries analysed : {n}\n"
        f"Mean Spearman r  : {mean_r:+.4f}   Median: {median_r:+.4f}\n"
        f"Positive corr    : {positive} / {n}\n"
        f"Significant (p<.05): {sig}\n"
        f"RAG-failed queries : {failed}"
    )


def _current_phase(data: dict, google_rows: list[dict]) -> str:
    """Human-readable description of what the system is doing RIGHT NOW."""
    if data.get("fatal_llm_error"):
        q_idx = data.get("fatal_llm_error_query_idx") or "?"
        return f"🛑 Fatal LLM connectivity error at query {q_idx}: {data['fatal_llm_error'][:140]}"

    if "ollama" in str(data.get("run_mode", "")).lower():
        total_ph = data["total_placeholder_rows"]
        synced = data["synced_rows"]
        if not data["main_loop_done"]:
            return (
                f"🤖 Ollama live scoring in progress: synced {synced:,}/{total_ph:,} rows."
                if total_ph else "🤖 Ollama live scoring in progress."
            )
        if total_ph and synced < total_ph:
            pct = synced / total_ph * 100
            return f"🤖 Ollama run finished main loop, DB sync at {pct:.1f}%."
        return "✅ Ollama run complete. Analytics ready."

    if not data["main_loop_done"]:
        if data["total_requests"] == 0:
            return "🔄 Phase 1 — Retrieval & JSONL production (no chunks submitted yet)"
        n_sub = len(data["chunk_job_meta"])
        n_run = sum(1 for r in google_rows if "RUNNING" in r["state"].upper())
        n_pen = sum(1 for r in google_rows if "PENDING" in r["state"].upper())
        if n_run or n_pen:
            return (
                f"🚀 Phase 2 — Pipeline active: producing + submitting chunks  "
                f"({n_sub} submitted, {n_run} RUNNING, {n_pen} PENDING)"
            )
        return f"⏳ Phase 2 — Waiting for a free slot ({n_sub} chunks submitted so far)"

    total_ph = data["total_placeholder_rows"]
    synced   = data["synced_rows"]
    if synced < total_ph:
        pct = synced / total_ph * 100 if total_ph else 0
        return f"📥 Phase 3 — Sync & Harvest: ({pct:.1f}% of rows synced). Run sync_batch_results.py."
    return "✅ Phase 4 — Complete. All rows synced. Run Failure_Analyzer.py for final report."


def generate_dashboard(db_path: str, google_rows: list[dict], start_time: float, snapshot_file: str) -> Layout:
    data    = _read_db(db_path)
    elapsed = time.time() - start_time

    phase_text = _current_phase(data, google_rows)

    # ── pipeline overview table ──
    pipeline_table = _build_pipeline_table(data, google_rows)

    # ── per-job status ──
    jobs_text = _build_jobs_table(google_rows)

    # ── score histogram ──
    hist_text = _build_score_histogram(data["scores_df"])

    # ── spearman summary ──
    spearman_text = _build_spearman_summary(data["spearman_df"])

    # ── submission errors ──
    err_lines = []
    for k, v in sorted(data["chunk_err_meta"].items()):
        chunk_num = k.replace("chunk_", "").replace("_error", "")
        err_lines.append(f"  Chunk {chunk_num}: {v[:80]}")
    err_text = "\n".join(err_lines) if err_lines else "[dim]None[/dim]"

    # ── assemble layout ──
    layout = Layout()
    layout.split_column(
        Layout(name="phase",    size=3),
        Layout(name="middle",   ratio=2),
        Layout(name="bottom",   ratio=3),
    )
    layout["middle"].split_row(
        Layout(name="pipeline", ratio=2),
        Layout(name="jobs",     ratio=1),
    )
    layout["bottom"].split_row(
        Layout(name="hist",     ratio=2),
        Layout(name="right",    ratio=1),
    )
    layout["right"].split_column(
        Layout(name="spearman", ratio=1),
        Layout(name="errors",   ratio=1),
    )

    title = (
        f"⭐ {data['run_name']}   "
        f"│  ⏱ {_fmt_duration(elapsed)} elapsed   "
        f"│  🕐 {datetime.now().strftime('%H:%M:%S')}"
    )

    layout["phase"].update(
        Panel(Text(phase_text, overflow="fold"), border_style="bold yellow", title=title)
    )
    layout["pipeline"].update(
        Panel(pipeline_table, title="📋 Pipeline & DB State", border_style="blue")
    )
    layout["jobs"].update(
        Panel(jobs_text, title="☁️  Google Batch Jobs", border_style="cyan")
    )
    layout["hist"].update(
        Panel(hist_text, title="📊 LLM Score Distribution (1–10 scale)", border_style="green")
    )
    layout["spearman"].update(
        Panel(spearman_text, title="📈 Spearman Correlations", border_style="magenta")
    )
    layout["errors"].update(
        Panel(err_text, title="⚠️  Submission Errors", border_style="red")
    )

    # ── write plain-text snapshot (overwrite, not append) ──
    _write_snapshot(data, google_rows, elapsed, phase_text, err_lines, snapshot_file)

    return layout


# ─── snapshot file ──────────────────────────────────────────────────────────

def _write_snapshot(
    data: dict,
    google_rows: list[dict],
    elapsed: float,
    phase_text: str,
    err_lines: list[str],
    snapshot_file: str,
) -> None:
    """
    Overwrite snapshot_file with the latest plain-text summary.
    Only the current state is written — this is a live status file, not a log.
    """
    os.makedirs(os.path.dirname(snapshot_file), exist_ok=True)
    try:
        total_ph = data["total_placeholder_rows"]
        synced   = data["synced_rows"]
        sync_pct = f"{synced/total_ph*100:.1f}%" if total_ph else "—"
        now_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"# Live Pipeline Status Snapshot",
            f"# Generated  : {now_str}",
            f"# Elapsed    : {_fmt_duration(elapsed)}",
            f"# Run        : {data['run_name']}",
            "",
            f"[PHASE]",
            f"  {phase_text}",
            "",
            f"[PIPELINE OVERVIEW]",
            f"  Loop completed        : {data['main_loop_done']}",
            f"  Chunks submitted      : {len(data['chunk_job_meta'])}",
            f"  Submission errors     : {len(data['chunk_err_meta'])}",
            f"  Batch requests reg.   : {data['total_requests']:,}",
            f"  Placeholder rows (DB) : {total_ph:,}",
            f"  Synced rows           : {synced:,} / {total_ph:,}  ({sync_pct})",
            f"  Queries with rows     : {data['total_queries_with_rows']}",
            f"  Queries synced        : {data['synced_queries']}",
            f"  Queries skipped       : {data['skipped_queries']}",
            f"  Zero/blocked scores   : {data['zero_score_rows']:,}",
            "",
            f"[GOOGLE BATCH JOBS]",
        ]
        if google_rows:
            for r in google_rows:
                short = r["id"].split("/")[-1]
                prog  = f"  {r['successful']:,}/{r['total']:,}" if r["total"] else ""
                lines.append(f"  {short:<22} {r['state']}{prog}")
        else:
            lines.append("  (not polled yet)")
        lines.append("")

        lines.append("[SCORE DISTRIBUTION]")
        if not data["scores_df"].empty:
            total_s = data["scores_df"]["cnt"].sum()
            for _, row in data["scores_df"].sort_values("llm_score").iterrows():
                raw = int(round(row["llm_score"] * 10))
                cnt = int(row["cnt"])
                pct = cnt / total_s * 100
                lines.append(f"  Score {raw:>2} : {cnt:>6,} docs  ({pct:4.1f}%)")
            lines.append(f"  Total   : {total_s:,}")
        else:
            lines.append("  (no scores yet)")
        lines.append("")

        sdf = data["spearman_df"]
        lines.append("[SPEARMAN SUMMARY]")
        if not sdf.empty:
            lines.append(f"  Queries analysed : {len(sdf)}")
            lines.append(f"  Mean r           : {sdf['correlation'].mean():+.4f}")
            lines.append(f"  Median r         : {sdf['correlation'].median():+.4f}")
            lines.append(f"  Positive corr    : {(sdf['correlation'] > 0).sum()} / {len(sdf)}")
            if "p_value" in sdf:
                lines.append(f"  Significant      : {(sdf['p_value'] < 0.05).sum()}")
        else:
            lines.append("  (not yet computed)")
        lines.append("")

        lines.append("[SUBMISSION ERRORS]")
        if err_lines:
            lines.extend(err_lines)
        else:
            lines.append("  None")
        lines.append("")

        with open(snapshot_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass


# ─── main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    db_path = resolve_db_path(sys.argv[1] if len(sys.argv) > 1 else "")
    if not db_path or not os.path.exists(db_path):
        console.print("[red]No experiment database found. Pass path as argument or run inside the project.[/red]")
        sys.exit(1)

    snapshot_file = _snapshot_file_for_db(db_path)
    console.print(f"[bold green]Monitoring:[/bold green] {db_path}")
    console.print(f"[dim]Snapshot file: {snapshot_file}[/dim]")

    start_time   = time.time()
    google_rows  = []
    poll_counter = 0

    # Initial data to get job IDs
    _initial = _read_db(db_path)
    raw_ids  = _initial.get("all_job_ids_raw") or ""
    job_ids  = [j.strip() for j in raw_ids.split(",") if j.strip()]

    with Live(
        generate_dashboard(db_path, google_rows, start_time, snapshot_file),
        console=console,
        refresh_per_second=0.5,
        screen=True,
    ) as live:
        while True:
            time.sleep(REFRESH_SECONDS)
            poll_counter += 1

            # Re-read job IDs every cycle (new chunks may have been submitted)
            fresh = _read_db(db_path)
            raw_ids = fresh.get("all_job_ids_raw") or ""
            job_ids = [j.strip() for j in raw_ids.split(",") if j.strip()]

            # Poll Google only every GOOGLE_POLL_EVERY refreshes
            if poll_counter % GOOGLE_POLL_EVERY == 0 and job_ids:
                google_rows = _poll_google_jobs(job_ids)

            live.update(generate_dashboard(db_path, google_rows, start_time, snapshot_file))
