#!/usr/bin/env python3
"""
score_from_queue.py — Phase 2 (Local / no GPU)
================================================
Reads the scoring_queue.parquet produced by build_scoring_queue.py (Phase 1)
and scores every (query, doc) pair with the configured LLM.

No Chroma / vector DB access required — all distances are pre-computed.

Usage
-----
    python score_from_queue.py --queue scoring_queue.parquet
    python score_from_queue.py --queue results/global_exp/staged_run_XXX/scoring_queue.parquet
    python score_from_queue.py --queue <path> --batch-size 20 --concurrency 4
    python score_from_queue.py --queue <path> --skip-scoring  # analysis only
"""

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from configurations.config import correlation_live_model_name
from utility.logger import logger
from utility.llm_gateway import (
    await_relevance_scores_for_documents,
    LLMGatewayError,
)
from experiments.global_correlation_experiment import (
    _init_sqlite,
    _normalize_id,
    _normalize_live_provider,
    _extract_contexts_text,
)

BATCH_SIZE   = 20    # docs per LLM call
CONCURRENCY  = 4    # parallel LLM calls per query
COMMIT_EVERY = 200  # rows before SQLite commit
RUN_STATE_FILE = "run_state.json"


def _load_run_state(run_dir: Path) -> dict:
    p = run_dir / RUN_STATE_FILE
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_run_state(run_dir: Path, updates: dict) -> None:
    p = run_dir / RUN_STATE_FILE
    state = _load_run_state(run_dir)
    state.update(updates)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(p)


def _init_db_from_queue(db_path: Path, queue_df: pd.DataFrame) -> sqlite3.Connection:
    """Create DB and pre-populate all rows with NULL llm_score (resume-safe)."""
    conn = _init_sqlite(str(db_path))
    cur = conn.cursor()
    cur.executemany(
        """INSERT OR IGNORE INTO results
           (query_idx, doc_id, llm_score, dist_to_gt, is_gt, rag_failed)
           VALUES (?, ?, NULL, ?, ?, ?)""",
        [
            (
                int(row.query_idx),
                str(row.doc_id),
                float(row.dist_to_gt),
                int(row.is_gt),
                int(row.rag_failed),
            )
            for row in queue_df.itertuples(index=False)
        ],
    )
    conn.commit()
    logger.info(f"[PHASE2] DB initialised: {db_path}")
    return conn


def _get_unscored_pairs(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT query_idx, doc_id FROM results WHERE llm_score IS NULL ORDER BY query_idx, doc_id"
    )
    return cur.fetchall()


def _save_scores(conn: sqlite3.Connection, updates: list[tuple[float | None, int, str]]) -> None:
    cur = conn.cursor()
    cur.executemany(
        "UPDATE results SET llm_score = ? WHERE query_idx = ? AND doc_id = ?",
        [(s, q, d) for s, q, d in updates if s is not None],
    )
    conn.commit()


async def _score_query_docs(
    query_text: str,
    doc_ids: list[str],
    doc_texts: list[str],
    *,
    batch_size: int,
    concurrency: int,
    provider: str,
    model: str,
) -> list[tuple[str, float]]:
    """Score all docs for one query; returns [(doc_id, score_0_to_1), ...]."""
    results: list[tuple[str, float]] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def _score_chunk(chunk_ids: list[str], chunk_texts: list[str]) -> list[tuple[str, float]]:
        async with semaphore:
            try:
                ints = await await_relevance_scores_for_documents(
                    provider=provider,
                    model=model,
                    query=query_text,
                    doc_contexts=chunk_texts,
                )
            except LLMGatewayError as exc:
                logger.warning(f"[PHASE2] LLM error: {exc} — leaving batch as NULL (will retry on resume)")
                return [(did, None) for did in chunk_ids]
            return [(did, float(v) / 10.0) for did, v in zip(chunk_ids, ints)]

    tasks = []
    for i in range(0, len(doc_ids), batch_size):
        tasks.append(_score_chunk(doc_ids[i: i + batch_size], doc_texts[i: i + batch_size]))

    for coro in asyncio.as_completed(tasks):
        chunk_result = await coro
        results.extend(chunk_result)

    return results


async def _run_scoring(
    conn: sqlite3.Connection,
    queue_df: pd.DataFrame,
    run_dir: Path,
    *,
    batch_size: int,
    concurrency: int,
) -> None:
    provider = _normalize_live_provider()
    model = correlation_live_model_name()
    logger.info(f"[PHASE2] Provider: {provider} | Model: {model}")

    # Build lookup: (query_idx, doc_id) → (query_text, plain_doc_text)
    lookup = {
        (_normalize_id(str(row.doc_id)), int(row.query_idx)): (
            str(row.query_text),
            _extract_contexts_text(str(row.doc_text)),
        )
        for row in queue_df.itertuples(index=False)
    }

    unscored = _get_unscored_pairs(conn)
    if not unscored:
        logger.info("[PHASE2] All pairs already scored.")
        return

    # Group by query
    per_query: dict[int, list[str]] = {}
    for q_idx, doc_id in unscored:
        per_query.setdefault(int(q_idx), []).append(_normalize_id(str(doc_id)))

    total = len(unscored)
    done = 0
    t0 = time.time()

    for q_idx, doc_ids in per_query.items():
        # Build parallel lists of texts
        query_text = ""
        doc_texts: list[str] = []
        valid_ids: list[str] = []
        for did in doc_ids:
            entry = lookup.get((did, q_idx))
            if entry is None:
                continue
            qt, dt = entry
            if not query_text:
                query_text = qt
            doc_texts.append(dt)
            valid_ids.append(did)

        if not valid_ids or not query_text:
            continue

        scored = await _score_query_docs(
            query_text,
            valid_ids,
            doc_texts,
            batch_size=batch_size,
            concurrency=concurrency,
            provider=provider,
            model=model,
        )

        updates = [(score, q_idx, did) for did, score in scored]
        _save_scores(conn, updates)

        done += len(valid_ids)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta_s = (total - done) / rate if rate > 0 else 0
        logger.info(
            f"[PHASE2] q{q_idx:03d} done | {done:,}/{total:,} "
            f"({100*done/total:.1f}%) | {rate:.0f} pairs/s | ETA {eta_s/60:.0f}m"
        )

    logger.info(f"[PHASE2] Scoring complete. Total pairs: {total:,}")
    _save_run_state(run_dir, {
        "phase2_complete":     True,
        "phase2_completed_at": datetime.now(timezone.utc).isoformat(),
        "phase2_scored_pairs": total,
        "phase2_db_path":      str(run_dir / "experiment_results.db"),
    })


def _run_analysis(run_dir: Path, db_path: Path) -> None:
    """Run correlation analysis using the experiment's own analysis pipeline."""
    logger.info(f"[PHASE2] Running analysis on {db_path}")
    try:
        from analyze_experiment import run_analysis
        run_analysis(run_dir=str(run_dir), db_path=str(db_path))
        logger.info("[PHASE2] Analysis complete.")
    except ImportError:
        logger.warning("[PHASE2] analyze_experiment not found — skipping analysis")
    except Exception as exc:
        logger.error(f"[PHASE2] Analysis failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: LLM scoring from pre-built queue")
    parser.add_argument("--queue", type=Path, required=True, help="Path to scoring_queue.parquet")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--skip-scoring",  action="store_true", help="Run analysis only (all pairs must be scored)")
    parser.add_argument("--skip-analysis", action="store_true", help="Score only, skip analysis")
    parser.add_argument(
        "--fresh", action="store_true",
        help=(
            "Delete the existing experiment_results.db and re-score every pair from scratch. "
            "Without this flag the script resumes from the last scored pair."
        ),
    )
    args = parser.parse_args()

    queue_path = args.queue.resolve()
    if not queue_path.exists():
        print(f"ERROR: Queue file not found: {queue_path}", file=sys.stderr)
        sys.exit(1)

    run_dir = queue_path.parent
    db_path = run_dir / "experiment_results.db"

    if args.fresh and db_path.exists():
        db_path.unlink()
        logger.info(f"[PHASE2] --fresh: deleted {db_path}, starting from scratch")
        _save_run_state(run_dir, {"phase2_complete": False, "analysis_complete": False})

    logger.info(f"[PHASE2] Queue: {queue_path}")
    logger.info(f"[PHASE2] DB:    {db_path}")

    queue_df = pd.read_parquet(queue_path)
    logger.info(
        f"[PHASE2] Queue loaded: {len(queue_df):,} rows | "
        f"{queue_df['query_idx'].nunique()} queries | "
        f"{queue_df['doc_id'].nunique():,} unique docs"
    )

    state = _load_run_state(run_dir)
    if state.get("phase2_complete") and not args.fresh and not args.skip_scoring:
        logger.info(
            "[PHASE2] All pairs already scored (run_state.json: phase2_complete=True). "
            "Skipping scoring. Pass --fresh to re-score from scratch."
        )
        args.skip_scoring = True

    conn = _init_db_from_queue(db_path, queue_df)

    if not args.skip_scoring:
        asyncio.run(
            _run_scoring(
                conn,
                queue_df,
                run_dir,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
            )
        )

    conn.close()

    if not args.skip_analysis:
        _run_analysis(run_dir, db_path)
        _save_run_state(run_dir, {
            "analysis_complete":     True,
            "analysis_completed_at": datetime.now(timezone.utc).isoformat(),
        })

    print(f"\nDone. Results in: {db_path}")


if __name__ == "__main__":
    main()
