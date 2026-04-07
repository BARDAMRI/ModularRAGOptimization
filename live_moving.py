import os
import sqlite3
import time
from datetime import timedelta

import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

BASE_DIR = "results/global_exp"
TOTAL_QUERIES = 200
DOCS_PER_QUERY = 13362
TOTAL_EXPECTED = TOTAL_QUERIES * DOCS_PER_QUERY

console = Console()


def get_latest_db():
    """מוצא אוטומטית את ה-DB של הריצה האחרונה."""
    try:
        runs = [os.path.join(BASE_DIR, d) for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
        latest_run = max(runs, key=os.path.getmtime)
        return os.path.join(latest_run, "experiment_results.db")
    except:
        return None


def get_stats(db_path):
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        processed = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        comp_q = conn.execute("SELECT COUNT(DISTINCT query_idx) FROM results").fetchone()[0]
        skip_q = conn.execute("SELECT COUNT(*) FROM skipped_queries").fetchone()[0]

        # בדיקת חסימות (ציוני 0.0)
        invalid_docs = conn.execute("SELECT COUNT(*) FROM results WHERE llm_score = 0.0").fetchone()[0]

        scores_df = pd.read_sql("SELECT llm_score, COUNT(*) as count FROM results GROUP BY llm_score", conn)
        conn.close()
        return processed, comp_q, skip_q, invalid_docs, scores_df
    except:
        return None, None, None, None, None


def generate_dashboard(start_time, db_path):
    processed, comp_q, skip_q, invalid, scores = get_stats(db_path)
    if processed is None:
        return Panel(f"Waiting for DB at: {db_path}...")

    elapsed = time.time() - start_time
    docs_per_sec = processed / elapsed if elapsed > 0 else 0
    docs_per_min = docs_per_sec * 60
    eta = str(timedelta(seconds=int((TOTAL_EXPECTED - processed) / docs_per_sec))) if docs_per_sec > 0 else "N/A"

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_row("Processed Docs", f"{processed:,} / {TOTAL_EXPECTED:,} ({(processed / TOTAL_EXPECTED) * 100:.1f}%)")
    table.add_row("Completed Queries", f"{comp_q} / {TOTAL_QUERIES}")
    table.add_row("Skipped (GT Missing)", f"[red]{skip_q}[/red]")
    table.add_row("Invalid/Blocked (0.0)", f"[bold red]{invalid}[/bold red]")  # השורה החדשה
    table.add_row("Speed (Docs/Min)", f"[green]{docs_per_min:.1f}[/green]")
    table.add_row("ETA", f"[yellow]{eta}[/yellow]")

    score_summary = ""
    if not scores.empty:
        max_c = scores['count'].max()
        for _, row in scores.sort_values('llm_score').iterrows():
            bar = "█" * int((row['count'] / max_c) * 20)
            score_summary += f"{row['llm_score']:.1f}: {bar} ({row['count']:,})\n"

    layout = Table.grid(expand=True)
    layout.add_row(Panel(table, title=f"🚀 Run: {os.path.basename(os.path.dirname(db_path))}", border_style="blue"))
    layout.add_row(Panel(score_summary, title="📊 Score Distribution", border_style="green"))
    return layout


db_path = get_latest_db()
start_time = time.time()
with Live(generate_dashboard(start_time, db_path), refresh_per_second=1) as live:
    while True:
        time.sleep(2)
        live.update(generate_dashboard(start_time, db_path))
