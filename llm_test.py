#!/usr/bin/env python3
# ollama_cluster_check.py

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from utility.llm_gateway import complete_prompt, get_ollama_client

OLLAMA_HOST = "https://cis-ollama.auth.ad.bgu.ac.il"
DEFAULT_MODEL = "Qwen3.5:4B"
REQUEST_TIMEOUT_S = 180

# Requested high-load sanity test:
PRIMARY_TEST_PARALLEL_REQUESTS = 50
PRIMARY_TEST_DOCS_PER_BATCH = 5

# Focused benchmark cases for fast decision-making (~10-15 minutes).
# Format: (parallel_requests, docs_per_batch)
FOCUSED_BENCHMARK_CASES = [
    (5, 5),
    (8, 5),
    (10, 5),
    (6, 10),
    (8, 10),
]

# If True, prints per-call details for each combination (very noisy).
VERBOSE_PER_CALL = False


def print_active_models(client: Client) -> None:
    ps_obj = client.ps()
    print("Currently active models:")
    if not ps_obj.models:
        print("No active models.")
        return

    for model in ps_obj.models:
        size_gb = model.size_vram / (1024 ** 3) if model.size_vram else 0.0
        print(f"Model: {model.model}, Size: {size_gb:.2f} GB")


def print_available_models_sorted(client: Client) -> None:
    models_list = json.loads(client.list().model_dump_json())["models"]
    if not models_list:
        print("No available models returned by server.")
        return

    print("\nAvailable models (sorted by size):")
    first = True
    for model in sorted(models_list, key=lambda x: int(x.get("size", 0))):
        if not first:
            print("-" * 20)
        first = False

        model_name = model.get("model", "unknown")
        size_gb = int(model.get("size", 0)) / (1024 ** 3)
        print(f"Model: {model_name}")
        print(f"Size: {size_gb:.2f} GB")


def _build_batch_prompt(call_idx: int, docs_per_batch: int) -> str:
    query = f"What are key biomarkers for disease category #{call_idx}?"
    docs = [f"Document {i + 1}: clinical summary text for batch call {call_idx}." for i in range(docs_per_batch)]
    docs_json = json.dumps(docs, ensure_ascii=False)
    return (
        "You are a relevance scorer.\n"
        f'Query: "{query}"\n'
        f"Documents: {docs_json}\n"
        'Return strictly JSON only: {"scores":[<int 1-10 for each document>]}'
    )


def _extract_text(response: str) -> str:
    return str(response or "").strip()


def _parse_scores_json(text: str, expected_n: int) -> bool:
    """Best-effort validation that response contains expected count of scores."""
    if not text:
        return False
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            arr = obj.get("scores")
        elif isinstance(obj, list):
            arr = obj
        else:
            return False
        if not isinstance(arr, list):
            return False
        if len(arr) != expected_n:
            return False
        for x in arr:
            if not isinstance(x, int):
                return False
            if x < 1 or x > 10:
                return False
        return True
    except Exception:
        return False


def _run_one_call(host: str, model: str, call_idx: int, docs_per_batch: int) -> dict[str, Any]:
    prompt = _build_batch_prompt(call_idx, docs_per_batch)
    t0 = time.time()
    try:
        response = complete_prompt(
            provider="ollama",
            model=model,
            prompt=prompt,
            system_prompt="Return strictly JSON only.",
            metadata={"host": host, "verify_ssl": False, "timeout_s": REQUEST_TIMEOUT_S},
            timeout_s=REQUEST_TIMEOUT_S,
        )
        elapsed = time.time() - t0
        text = _extract_text(response)
        ok = _parse_scores_json(text, expected_n=docs_per_batch)
        return {
            "call_idx": call_idx,
            "ok": ok,
            "elapsed_s": elapsed,
            "response_snippet": text[:200],
            "error": "",
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "call_idx": call_idx,
            "ok": False,
            "elapsed_s": elapsed,
            "response_snippet": "",
            "error": str(e),
        }


def run_parallel_batch_test(
    host: str,
    model: str,
    parallel_requests: int,
    docs_per_batch: int,
    verbose_per_call: bool = False,
) -> dict[str, Any]:
    print(
        f"\nRunning stress test: {parallel_requests} parallel requests, "
        f"{docs_per_batch} docs per request, model={model}"
    )
    start = time.time()
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=parallel_requests) as pool:
        futures = [
            pool.submit(_run_one_call, host, model, i + 1, docs_per_batch)
            for i in range(parallel_requests)
        ]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if verbose_per_call:
                status = "OK" if res["ok"] else "BAD"
                if res.get("error"):
                    status = "ERROR"
                print(
                    f"[call {res['call_idx']:02d}] {status} "
                    f"time={res['elapsed_s']:.2f}s "
                    f"snippet={res['response_snippet']!r} "
                    f"error={res.get('error', '')[:120]!r}"
                )

    total = len(results)
    success = sum(1 for r in results if r.get("ok"))
    elapsed_total = time.time() - start
    times = [r.get("elapsed_s", 0.0) for r in results if r.get("elapsed_s", 0.0) > 0]
    avg_time = (sum(times) / len(times)) if times else 0.0
    print("\n=== Parallel Stress Test Summary ===")
    print(f"Total calls     : {total}")
    print(f"Successful calls: {success}")
    print(f"Failed/empty    : {total - success}")
    print(f"Total wall time : {elapsed_total:.2f}s")
    print(f"Avg call time   : {avg_time:.2f}s")
    throughput = total / elapsed_total if elapsed_total > 0 else 0.0
    success_rate = (success / total * 100.0) if total else 0.0
    print(f"Throughput      : {throughput:.2f} calls/sec")
    print(f"Success rate    : {success_rate:.1f}%")

    return {
        "parallel_requests": parallel_requests,
        "docs_per_batch": docs_per_batch,
        "total_calls": total,
        "success_calls": success,
        "failed_calls": total - success,
        "wall_time_s": elapsed_total,
        "avg_call_s": avg_time,
        "throughput_calls_per_sec": throughput,
        "success_rate_pct": success_rate,
    }


def run_combinations_benchmark(host: str, model: str) -> None:
    combinations: list[tuple[int, int]] = list(FOCUSED_BENCHMARK_CASES)

    print("\n=== Combinations Benchmark ===")
    print(f"Total combinations: {len(combinations)}")
    summary_rows: list[dict[str, Any]] = []

    for idx, (p, d) in enumerate(combinations, start=1):
        print(f"\n[{idx}/{len(combinations)}] Benchmarking parallel={p}, docs={d}")
        row = run_parallel_batch_test(
            host=host,
            model=model,
            parallel_requests=p,
            docs_per_batch=d,
            verbose_per_call=VERBOSE_PER_CALL,
        )
        summary_rows.append(row)

    # Rank first by success-rate then by throughput.
    ranked = sorted(
        summary_rows,
        key=lambda r: (r["success_rate_pct"], r["throughput_calls_per_sec"]),
        reverse=True,
    )

    print("\n=== Top 15 Combinations (success + throughput) ===")
    print(
        "rank | parallel | docs | success% | throughput(calls/s) | wall_time_s | "
        "avg_call_s | ok/total"
    )
    for i, r in enumerate(ranked[:15], start=1):
        print(
            f"{i:>4} | {r['parallel_requests']:>8} | {r['docs_per_batch']:>4} | "
            f"{r['success_rate_pct']:>7.1f} | {r['throughput_calls_per_sec']:>19.2f} | "
            f"{r['wall_time_s']:>10.2f} | {r['avg_call_s']:>10.2f} | "
            f"{r['success_calls']}/{r['total_calls']}"
        )


def main() -> None:
    client = get_ollama_client(OLLAMA_HOST, verify_ssl=False, timeout_s=REQUEST_TIMEOUT_S)
    print(f"Connected host: {OLLAMA_HOST}")
    print_active_models(client)
    print_available_models_sorted(client)

    # Mandatory high-load test requested by user.
    run_parallel_batch_test(
        host=OLLAMA_HOST,
        model=DEFAULT_MODEL,
        parallel_requests=PRIMARY_TEST_PARALLEL_REQUESTS,
        docs_per_batch=PRIMARY_TEST_DOCS_PER_BATCH,
        verbose_per_call=True,
    )
    # Multi-combination comparison across range.
    run_combinations_benchmark(host=OLLAMA_HOST, model=DEFAULT_MODEL)


if __name__ == "__main__":
    main()