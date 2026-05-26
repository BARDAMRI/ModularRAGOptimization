#!/usr/bin/env python3
"""
Unified post-run analysis for the global correlation experiment.

Runs, in order:
  1. analyze_experiment.py       — base metrics, CSVs, charts
  2. analyze_experiment_extended.py — tie-corrected metrics, extended CSVs
  3. export_report.py            — self-contained HTML report

Outputs are written to a structured ``analysis/`` tree (see ``analysis/INDEX.md``):
  csv/global/      pool-level aggregates
  csv/per_query/   one row per query_idx (+ per_query_overview.csv master merge)
  charts/base/     charts from step 1
  charts/extended/ charts from step 2
  summaries/       text summaries
  report.html      primary HTML deliverable

Re-running replaces files at the same paths (no versioning). Safe after partial runs —
use ``--steps`` to run only missing stages.

Usage:
    python run_global_correlation_analysis.py
    python run_global_correlation_analysis.py --run-dir results/global_exp/staged_run_YYYY-MM-DD
    python run_global_correlation_analysis.py --steps extended,report
    python run_global_correlation_analysis.py path/to/experiment_results.db
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from analysis_common import find_latest_db

PROJECT_ROOT = Path(__file__).resolve().parent

ALL_STEPS: tuple[str, ...] = ("base", "extended", "report")

_STEP_SCRIPTS: dict[str, tuple[str, str]] = {
    "base": ("Base analysis", "analyze_experiment.py"),
    "extended": ("Extended analysis", "analyze_experiment_extended.py"),
    "report": ("HTML report", "export_report.py"),
}


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


def _parse_steps(raw: str | None) -> tuple[str, ...]:
    if not raw or raw.strip().lower() in ("all", "*"):
        return ALL_STEPS
    steps = []
    for part in raw.split(","):
        s = part.strip().lower()
        if not s:
            continue
        if s not in _STEP_SCRIPTS:
            valid = ", ".join(ALL_STEPS)
            raise ValueError(f"Unknown step {s!r}. Valid: {valid}")
        if s not in steps:
            steps.append(s)
    if not steps:
        return ALL_STEPS
    return tuple(steps)


def run_global_correlation_analysis(
    db_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    *,
    steps: str | tuple[str, ...] | None = None,
    quiet: bool = False,
) -> dict[str, Path]:
    """
    Execute the analysis pipeline for one experiment run.

    Parameters
    ----------
    steps : Which stages to run — ``base``, ``extended``, ``report``, or comma-separated.
            Default: all three. Re-running a step **overwrites** its outputs.

    Returns paths to run_dir, db_path, analysis_dir, and report_html.
    """
    run, db = resolve_run_paths(db_path=db_path, run_dir=run_dir)
    analysis_dir = run / "analysis"
    step_list = _parse_steps(steps if isinstance(steps, str) or steps is None else ",".join(steps))

    if not quiet:
        print(f"Run dir : {run}")
        print(f"DB      : {db}")
        print(f"Output  : {analysis_dir}")
        print(f"Steps   : {', '.join(step_list)}  (re-runs overwrite existing files)\n")

    n = len(step_list)
    for i, step in enumerate(step_list, start=1):
        step_label, script_name = _STEP_SCRIPTS[step]
        if not quiet:
            print(f"\n{'─' * 60}", flush=True)
            print(f"  [{i}/{n}] {step_label}  ({script_name})", flush=True)
            print(f"{'─' * 60}", flush=True)
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / script_name),
            "--run-dir",
            str(run),
        ]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)

    report_html = analysis_dir / "report.html"
    if "report" in step_list and not report_html.exists():
        raise FileNotFoundError(f"Expected report not found: {report_html}")

    if not quiet:
        if report_html.exists():
            size_kb = report_html.stat().st_size / 1024
            report_pdf = report_html.with_suffix(".pdf")
            print(f"\n✅ Analysis complete.")
            print(f"   Report : {report_html}  ({size_kb:.0f} KB)")
            if report_pdf.exists():
                print(f"   PDF    : {report_pdf}  ({report_pdf.stat().st_size / 1024:.0f} KB)")
            print(f"   Index  : {analysis_dir / 'INDEX.md'}")
            print(f"   Per-Q  : {analysis_dir / 'csv' / 'per_query' / 'per_query_overview.csv'}")
        else:
            print(f"\n✅ Steps {', '.join(step_list)} complete (report not generated).")

    return {
        "run_dir": run,
        "db_path": db,
        "analysis_dir": analysis_dir,
        "report_html": report_html,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run global-correlation post-run analysis (base + extended + HTML report)",
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
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma-separated steps: base, extended, report (default: all). "
             "Example: --steps extended,report to skip base after partial run.",
    )
    args = parser.parse_args()

    try:
        run_global_correlation_analysis(
            db_path=args.db_path,
            run_dir=args.run_dir,
            steps=args.steps,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Analysis step failed (exit {exc.returncode})", file=sys.stderr)
        sys.exit(exc.returncode or 1)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
