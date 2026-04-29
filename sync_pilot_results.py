import json
import os
import sqlite3
import sys

from Failure_Analyzer import analyze_failures


def _normalize_id(x) -> str:
    return str(x).strip().lower()


def _parse_batch_scores(text: str, expected_n: int) -> list[int]:
    if text is None:
        return [0] * expected_n
    t = str(text).strip()
    if not t:
        return [0] * expected_n
    import re as _re

    m = _re.search(r"\[[\s\S]*?\]", t)
    candidate = m.group(0).strip() if m else t
    try:
        arr = json.loads(candidate)
        if isinstance(arr, list):
            out = []
            for x in arr:
                try:
                    v = int(x)
                except Exception:
                    v = 0
                out.append(max(0, min(10, v)))
            if len(out) < expected_n:
                out.extend([0] * (expected_n - len(out)))
            return out[:expected_n]
    except Exception:
        pass
    nums = _re.findall(r"\b(10|[1-9])\b", t)
    out = [int(n) for n in nums[:expected_n]]
    if len(out) < expected_n:
        out.extend([0] * (expected_n - len(out)))
    return out


def _extract_response_text(obj: dict) -> str | None:
    resp = obj.get("response") if isinstance(obj, dict) else None
    if resp is None:
        return None
    if isinstance(resp, dict):
        body = resp.get("body") if isinstance(resp.get("body"), dict) else resp
        if not isinstance(body, dict):
            return None
        if isinstance(body.get("text"), str):
            return body.get("text")
        candidates = body.get("candidates")
        if isinstance(candidates, list) and candidates:
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


def _lookup_mapping(conn, custom_id: str) -> tuple[int, str] | None:
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
        arr = json.loads(doc_ids_json)
        doc_id = arr[0] if isinstance(arr, list) and arr else None
    except Exception:
        doc_id = None
    if not doc_id:
        return None
    return int(q_idx), _normalize_id(doc_id)


def sync_pilot_results(db_path: str, output_jsonl_path: str) -> None:
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)
    if not os.path.exists(output_jsonl_path):
        raise FileNotFoundError(output_jsonl_path)

    conn = sqlite3.connect(db_path)
    try:
        updated = 0
        with open(output_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                custom_id = obj.get("custom_id")
                if not custom_id:
                    continue
                mapping = _lookup_mapping(conn, custom_id)
                if mapping is None:
                    continue
                q_idx, doc_id = mapping
                text = _extract_response_text(obj)
                ints = _parse_batch_scores(text, expected_n=1)
                score = float(ints[0]) / 10.0
                cur = conn.cursor()
                cur.execute(
                    "UPDATE results SET llm_score = ? WHERE query_idx = ? AND doc_id = ?",
                    (score, int(q_idx), doc_id),
                )
                updated += cur.rowcount
        conn.commit()
        print(f"✅ Pilot sync complete. Updated rows: {updated}")
    finally:
        conn.close()

    # Auto-run Failure Analyzer on this pilot DB
    print("\n🚩 Running Failure Analyzer on pilot DB (top winners)...\n")
    analyze_failures(db_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sync_pilot_results.py <experiment_results.db> <pilot_output.jsonl>")
        sys.exit(2)
    sync_pilot_results(sys.argv[1], sys.argv[2])

