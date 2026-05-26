#!/usr/bin/env python3
"""
Unified post-run analysis for the global correlation experiment.

Runs, in order:
  1. analyze_experiment.py       — base metrics, CSVs, charts
  2. analyze_experiment_extended.py — tie-corrected metrics, extended CSVs
  3. export_report.py            — self-contained HTML report

Usage:
    python run_global_correlation_analysis.py
    python run_global_correlation_analysis.py --run-dir results/global_exp/staged_run_YYYY-MM-DD_HH-MM-SS
    python run_global_correlation_analysis.py path/to/experiment_results.db

Also invoked from main.py (Global Correlation menu → Analyze) and optionally
after an experiment when ``global_correlation.auto_run_post_analysis`` is true
in run_config.json.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from analysis_common import find_latest_db

PROJECT_ROOT = Path(__file__).resolve().parent

_ANALYSIS_SCRIPTS: tuple[tuple[str, str], ...] = (
    ("Base analysis", "analyze_experiment.py"),
    ("Extended analysis", "analyze_experiment_extended.py"),
    ("HTML report", "export_report.py"),
)


def resolve_run_paths(
    db_path: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return (run_dir, db_path), resolving defaults from the latest staged run."""
    if run_dir:
        run = Path(run_dir).expanduser().resolve()
        db = run / "experiment_results.db"
    elif db_path:
        db = Path(db_path).expanduser().resolve()
        run = db.parent
    else:
        db = find_latest_db().resolve()
        run = db.parent

    if not db.exists():
        raise FileNotFoundError(f"experiment_results.db not found: {db}")
    return run, db


def run_global_correlation_analysis(
    db_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    *,
    quiet: bool = False,
) -> dict[str, Path]:
    """
    Execute the full analysis pipeline for one experiment run.

    Returns paths to run_dir, db_path, analysis_dir, and report_html.
    """
    run, db = resolve_run_paths(db_path=db_path, run_dir=run_dir)
    analysis_dir = run / "analysis"

    if not quiet:
        print(f"Run dir : {run}")
        print(f"DB      : {db}")
        print(f"Output  : {analysis_dir}\n")

    for step_label, script_name in _ANALYSIS_SCRIPTS:
        if not quiet:
            print(f"{'=' * 60}")
            print(f"  {step_label} — {script_name}")
            print(f"{'=' * 60}")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / script_name),
            "--run-dir",
            str(run),
        ]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

    report_html = analysis_dir / "report.html"
    if not report_html.exists():
        raise FileNotFoundError(f"Expected report not found: {report_html}")

    if not quiet:
        size_kb = report_html.stat().st_size / 1024
        print(f"\n✅ Analysis complete.")
        print(f"   Report : {report_html}")
        print(f"   Size   : {size_kb:.0f} KB")

    return {
        "run_dir": run,
        "db_path": db,
        "analysis_dir": analysis_dir,
        "report_html": report_html,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full global-correlation post-run analysis (base + extended + HTML report)",
    )
    parser.add_argument(
        "db_path",
        nargs="?",
        help="Path to experiment_results.db (optional; defaults to latest run)",
    )
    parser.add_argument(
        "--run-dir",
        help="Path to staged run directory (DB auto-located inside)",
    )
    args = parser.parse_args()

    try:
        run_global_correlation_analysis(db_path=args.db_path, run_dir=args.run_dir)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Analysis step failed (exit {exc.returncode})", file=sys.stderr)
        sys.exit(exc.returncode or 1)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
