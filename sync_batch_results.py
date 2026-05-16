import argparse
import json
import os
import sqlite3
import sys

from experiments.global_correlation_experiment import (
    _calculate_final_stats_and_plot_sqlite,
    _snap_to_anchor,
    BATCH_STATUS_SUCCEEDED,
    BATCH_STATUS_MISSING,
)
from utility.llm_gateway import parse_relevance_scores


def _normalize_id(x) -> str:
    return str(x).strip().lower()


def _get_doc_ids_for_custom_id(conn, custom_id: str) -> tuple[int, list[str]] | None:
    cur = conn.cursor()
    cur.execute(
        "SELECT query_idx, doc_ids_json FROM batch_request_map WHERE custom_id = ?",
        (str(custom_id),),
    )
    row = cur.fetchone()
    if not row:
        return None
    q_idx, doc_ids_json = row
    try:
        doc_ids = json.loads(doc_ids_json)
    except Exception:
        doc_ids = []
    return int(q_idx), [_normalize_id(d) for d in doc_ids]


def _extract_response_text(obj: dict) -> str | None:
    """
    Best-effort: handle different Batch output shapes.
    """
    # Common shapes (Batch output):
    # response -> body -> candidates -> content -> parts -> text
    resp = obj.get("response") if isinstance(obj, dict) else None
    if resp is None:
        return None
    if isinstance(resp, dict):
        body = resp.get("body") if isinstance(resp.get("body"), dict) else resp
        if not isinstance(body, dict):
            return None
        # direct text fallbacks
        if isinstance(body.get("text"), str):
            return body.get("text")
        
        # New OpenAI format (from ChatCompletions)
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            msg = (choices[0] or {}).get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]

        candidates = body.get("candidates")
        if isinstance(candidates, list) and candidates:
            # join all candidate part texts
            texts: list[str] = []
            for cand in candidates:
                content = (cand or {}).get("content")
                if not isinstance(content, dict):
                    continue
                parts = content.get("parts")
                if not isinstance(parts, list):
                    continue
                for p in parts:
                    if isinstance(p, dict) and isinstance(p.get("text"), str):
                        texts.append(p["text"])
            if texts:
                return "\n".join(texts)
    return None


def _is_safety_block(obj: dict) -> bool:
    """
    Batch output can include safety blocks. We treat them as score 0.0.
    """
    try:
        resp = obj.get("response") if isinstance(obj, dict) else None
        if not isinstance(resp, dict):
            return True
        # direct / nested body
        body = resp.get("body") if isinstance(resp.get("body"), dict) else resp
        pf = body.get("prompt_feedback") if isinstance(body, dict) else None
        if isinstance(pf, dict):
            br = pf.get("block_reason")
            if br and str(br).upper() != "BLOCK_REASON_UNSPECIFIED":
                return True
        cands = body.get("candidates") if isinstance(body, dict) else None
        if isinstance(cands, list):
            for c in cands:
                fr = (c or {}).get("finish_reason")
                if fr and ("SAFETY" in str(fr).upper() or "BLOCK" in str(fr).upper()):
                    return True
    except Exception:
        return True
    return False


def sync_batch_results(db_path: str, output_paths_input: str, force_overwrite: bool = False,
                       commit_every: int = 200, job_id: str | None = None) -> None:
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)
    
    # Expand glob or handle comma-separated list
    import glob
    all_paths = []
    for part in output_paths_input.split(","):
        part = part.strip()
        if os.path.isdir(part):
            all_paths.extend(glob.glob(os.path.join(part, "*.jsonl")))
        elif "*" in part:
            all_paths.extend(glob.glob(part))
        elif os.path.exists(part):
            all_paths.append(part)
    
    if not all_paths:
        raise FileNotFoundError(f"No JSONL files found matching input: {output_paths_input}")

    conn = sqlite3.connect(db_path)
    try:
        # Optional: infer last job id from experiment_meta
        if not job_id:
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT value FROM experiment_meta WHERE key = ?",
                    ("last_batch_job_id",),
                )
                row = cur.fetchone()
                if row and row[0]:
                    job_id = str(row[0])
            except Exception:
                job_id = None
        if job_id:
            print(f"Associated Job ID(s): {job_id}")

        # Track which custom_ids actually received a valid scored response so that
        # batch_request_map.status can be updated precisely.
        succeeded_custom_ids: set[str] = set()
        bad_response_custom_ids: set[str] = set()

        for output_jsonl_path in all_paths:
            print(f"Processing: {os.path.basename(output_jsonl_path)}")
            total_lines = 0
            with open(output_jsonl_path, "r", encoding="utf-8") as f:
                for _ in f:
                    total_lines += 1

            updated_rows = 0
            skipped_already_set = 0
            bad_lines = 0
            line_idx = 0
            with open(output_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_idx += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        bad_lines += 1
                        continue

                    if not isinstance(obj, dict) or "response" not in obj:
                        bad_lines += 1
                        continue
                    # Native Gemini Batch output uses `key`; the OpenAI-compat layer uses `custom_id`.
                    custom_id = obj.get("custom_id") or obj.get("key") or obj.get("id")
                    if not custom_id:
                        bad_lines += 1
                        continue

                    mapping = _get_doc_ids_for_custom_id(conn, custom_id)
                    if mapping is None:
                        continue
                    q_idx, doc_ids = mapping
                    if not doc_ids:
                        continue

                    # skip if already has a non-NULL score (unless force overwrite)
                    if not force_overwrite:
                        cur = conn.cursor()
                        cur.execute(
                            "SELECT COUNT(*) FROM results WHERE query_idx = ? AND doc_id IN (%s) AND llm_score IS NOT NULL"
                            % (",".join(["?"] * len(doc_ids))),
                            [int(q_idx), *doc_ids],
                        )
                        already = int(cur.fetchone()[0])
                        if already == len(doc_ids):
                            skipped_already_set += 1
                            succeeded_custom_ids.add(str(custom_id))
                            continue

                    if _is_safety_block(obj):
                        ints = [0] * len(doc_ids)
                        bad_response_custom_ids.add(str(custom_id))
                    else:
                        text = _extract_response_text(obj)
                        ints = parse_relevance_scores(text, expected_n=len(doc_ids))
                        ints = [_snap_to_anchor(v) if v != 0 else 0 for v in ints]
                        # If the parser returned all zeros it usually means the response
                        # was unparseable (e.g. truncated or non-JSON). Mark for retry.
                        if all(v == 0 for v in ints):
                            bad_response_custom_ids.add(str(custom_id))
                        else:
                            succeeded_custom_ids.add(str(custom_id))
                    scores = [float(v) / 10.0 for v in ints]

                    cur = conn.cursor()
                    cur.executemany(
                        "UPDATE results SET llm_score = ? WHERE query_idx = ? AND doc_id = ?",
                        [(scores[i], q_idx, doc_ids[i]) for i in range(len(doc_ids))],
                    )
                    updated_rows += cur.rowcount
                    if line_idx % int(commit_every) == 0:
                        conn.commit()
                        print(f"Progress: {line_idx}/{total_lines} lines | updated_rows={updated_rows}")

        conn.commit()

        print(f"✅ Sync complete. Updated rows: {updated_rows}")
        print(f"Skipped already-set batches: {skipped_already_set}")
        print(f"Bad/invalid JSONL lines: {bad_lines}")

        # Update per-request status: 'succeeded' for everyone we actually scored,
        # 'missing' for any 'submitted' custom_id that didn't show up (whether the
        # response was absent, unparseable, or safety-blocked).
        try:
            cur = conn.cursor()
            if succeeded_custom_ids:
                cur.executemany(
                    "UPDATE batch_request_map SET status = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE custom_id = ?",
                    [(BATCH_STATUS_SUCCEEDED, cid) for cid in succeeded_custom_ids],
                )
            if bad_response_custom_ids:
                cur.executemany(
                    "UPDATE batch_request_map SET status = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE custom_id = ?",
                    [(BATCH_STATUS_MISSING, cid) for cid in bad_response_custom_ids],
                )
            # Anything still 'submitted' after sync = response simply did not arrive.
            cur.execute(
                "UPDATE batch_request_map SET status = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE status = 'submitted'",
                (BATCH_STATUS_MISSING,),
            )
            conn.commit()
        except Exception as exc:
            print(f"⚠️  Could not update batch_request_map.status: {exc}")

        # Recompute final stats & plots
        run_dir = os.path.dirname(os.path.abspath(db_path))
        _calculate_final_stats_and_plot_sqlite(conn, run_dir)
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("db_path", help="Path to experiment_results.db")
    ap.add_argument("output_paths", help="Path to Batch output JSONL file, folder, or comma-separated list/glob")
    ap.add_argument("--job-id", default=None, help="Optional job id(s)")
    ap.add_argument("--force-overwrite", action="store_true", help="Overwrite existing scores")
    ap.add_argument("--commit-every", type=int, default=200, help="Commit every N output lines")
    args = ap.parse_args()

    sync_batch_results(
        db_path=args.db_path,
        output_paths_input=args.output_paths,
        job_id=str(args.job_id) if args.job_id else None,
        force_overwrite=bool(args.force_overwrite),
        commit_every=int(args.commit_every),
    )

