import os
import sqlite3

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

BASE_DIR = "results/global_exp"
MIN_SCORE_THRESHOLD = 0.8

console = Console()


def get_latest_db():
    """auto-detects the DB of the latest run."""
    try:
        runs = [os.path.join(BASE_DIR, d) for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
        latest_run = max(runs, key=os.path.getmtime)
        return os.path.join(latest_run, "experiment_results.db")
    except:
        return None


def analyze_failures(db_base_path):
    try:
        conn = sqlite3.connect(f"file:{db_base_path}?mode=ro", uri=True)
        query = f"""
            WITH GT_Info AS (
                SELECT query_idx, llm_score as gt_score, dist_to_gt as gt_dist
                FROM results 
                WHERE is_gt = 1
            )
            SELECT r.query_idx, r.doc_id, r.llm_score, r.dist_to_gt, 
                   g.gt_score, g.gt_dist
            FROM results r
            LEFT JOIN GT_Info g ON r.query_idx = g.query_idx
            WHERE r.is_gt = 0 AND r.llm_score >= {MIN_SCORE_THRESHOLD}
            ORDER BY r.query_idx ASC, r.llm_score DESC
            LIMIT 30
        """
        report = pd.read_sql(query, conn)
        conn.close()

        if report.empty:
            console.print("[yellow]No high-scoring failures found yet.[/yellow]")
            return

        table = Table(title="🕵️ Failure Analysis: Why non-GT got high scores?", show_lines=True)
        table.add_column("Q", style="cyan", justify="center")
        table.add_column("Doc ID (Failure)", style="magenta")
        table.add_column("Score", style="bold red")
        table.add_column("GT Score", style="bold green")
        table.add_column("Dist (Fail)", style="dim")
        table.add_column("Dist (GT)", style="dim")

        for _, row in report.iterrows():
            table.add_row(
                str(int(row['query_idx'])),
                str(row['doc_id'][:18]) + "...",
                f"{row['llm_score']:.2f}",
                f"{row['gt_score']:.2f}" if pd.notnull(row['gt_score']) else "[red]MISSING[/red]",
                f"{row['dist_to_gt']:.4f}",
                f"{row['gt_dist']:.4f}" if pd.notnull(row['gt_dist']) else "N/A"
            )

        console.print(table)

        console.print(Panel(
            "💡 [bold blue]Logic Check:[/bold blue]\n"
            "1. If 'GT Score' is MISSING: The ground truth was never evaluated or ID mismatch is severe.\n"
            "2. If 'Dist (Fail)' is 0.000: This IS the Ground Truth, but it was mislabeled as is_gt=0.\n"
            "3. If 'Score' > 'GT Score': The LLM officially preferred a wrong document."
        ))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    db_path = get_latest_db()
    analyze_failures(db_path)
