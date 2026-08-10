#!/usr/bin/env python3
"""
fix_rag_failed.py — Patch the rag_failed column in an existing experiment DB.

The original _check_rag_baseline() compared Chroma UUID IDs against PMID
strings, which always mismatched → rag_failed=1 for every query.

This script derives the correct rag_failed value from the retrieval-base
experiment CSV (which compares PMIDs correctly) and writes it back to the DB.

Usage:
    python fix_rag_failed.py                          # auto-detect latest run
    python fix_rag_failed.py --run-dir <path>
    python fix_rag_failed.py --db <path/to/experiment_results.db>
"""

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

RETRIEVAL_BASE_CSV = Path(
    "results/retrieval_base_experiment/"
    "chroma_cosine_top_1000_results_retrieval_base_19-11-2025_14-30-45.csv"
)


def _find_latest_run() -> Path | None:
    base = Path("results/global_exp")
    if not base.exists():
        return None
    runs = sorted(base.glob("staged_run_*"), reverse=True)
    return runs[0] if runs else None


def fix_rag_failed(db_path: Path, run_dir: Path) -> None:
    queries_path = run_dir / "queries.json"
    if not queries_path.exists():
        raise FileNotFoundError(f"queries.json not found in {run_dir}")

    with open(queries_path) as f:
        queries = json.load(f)

    # Build PMID → query_idx map
    pmid_to_qidx: dict[str, int] = {}
    for q in queries:
        pmid = str(q.get("pubid", "")).strip()
        qidx = q.get("query_idx") or queries.index(q) + 1
        if pmid:
            pmid_to_qidx[pmid] = qidx

    # Load retrieval base results
    if not RETRIEVAL_BASE_CSV.exists():
        raise FileNotFoundError(f"Retrieval base CSV not found: {RETRIEVAL_BASE_CSV}")
    ret_df = pd.read_csv(RETRIEVAL_BASE_CSV)
    ret_df["gt_str"] = ret_df["ground_truth_id"].astype(str).str.strip()

    # Build query_idx → rag_failed map
    updates: list[tuple[int, int]] = []
    missing: list[str] = []
    for pmid, qidx in pmid_to_qidx.items():
        row = ret_df[ret_df["gt_str"] == pmid]
        if row.empty:
            missing.append(pmid)
            continue
        naive_hit = int(row.iloc[0]["naive_hit"])
        rag_failed = 0 if naive_hit == 1 else 1
        updates.append((rag_failed, qidx))

    print(f"Queries matched: {len(updates)}")
    print(f"Queries not found in retrieval base: {len(missing)}")
    if missing:
        print(f"  Missing PMIDs: {missing[:10]}")

    rag_failed_counts = {v: sum(1 for rf, _ in updates if rf == v) for v in (0, 1)}
    print(f"rag_failed=0 (retriever succeeded): {rag_failed_counts.get(0, 0)}")
    print(f"rag_failed=1 (retriever failed):    {rag_failed_counts.get(1, 0)}")

    # Apply updates
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executemany(
        "UPDATE results SET rag_failed = ? WHERE query_idx = ?",
        updates,
    )
    conn.commit()
    rows_updated = cur.rowcount
    conn.close()
    print(f"\nUpdated {rows_updated:,} rows in {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch rag_failed column in experiment DB")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    db_path = args.db

    if run_dir is None and db_path is None:
        run_dir = _find_latest_run()
        if run_dir is None:
            raise RuntimeError("No staged_run_* folder found under results/global_exp/")
        print(f"Auto-detected run: {run_dir}")

    if db_path is None:
        db_path = run_dir / "experiment_results.db"

    if run_dir is None:
        run_dir = db_path.parent

    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    fix_rag_failed(db_path, run_dir)


if __name__ == "__main__":
    main()
