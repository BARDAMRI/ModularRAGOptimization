"""
Central LLM gateway: async dispatch service + per-provider senders + rate limits.

Experiments submit prompts via :func:`await_prompt` or :func:`complete_prompt` and
never implement provider HTTP/SDK details themselves.

Architecture
------------
* A background **dispatcher** thread pulls requests from a queue.
* Before each outbound call the dispatcher **acquires** a rate-limit slot (RPM/RPS).
* Work runs in a **thread pool**; results are delivered to ``asyncio.Future`` or
  ``concurrent.futures.Future`` callers.
* Each provider has a dedicated **send** function that returns **parsed text**.

Rate limits: ``config.LLM_GATEWAY_RATE_LIMITS`` (missing entry ⇒ no throttle).
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import threading
import time
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import requests

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class LLMRequestKind(str, Enum):
    REGULAR = "regular"
    BATCH = "batch"


@dataclass(frozen=True)
class LLMRequest:
    """Provider-agnostic outbound prompt request."""

    provider: str
    model: str
    prompt: str
    system_prompt: str | None = None
    request_kind: LLMRequestKind = LLMRequestKind.REGULAR
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMGatewayError(RuntimeError):
    """Raised when the gateway or a provider returns no usable text."""


# ---------------------------------------------------------------------------
# Provider id normalization
# ---------------------------------------------------------------------------

_PROVIDER_ALIASES: dict[str, str] = {
    "ollama": "ollama",
    "nvidia_ih": "nvidia_ih",
    "nvidia": "nvidia_ih",
    "inference_hub": "nvidia_ih",
    "gemini": "gemini",
    "google": "gemini",
}


def normalize_provider(provider: str) -> str:
    raw = str(provider).strip().lower()
    return _PROVIDER_ALIASES.get(raw, raw)


# ---------------------------------------------------------------------------
# Rate limiting (internal)
# ---------------------------------------------------------------------------


def _norm_provider(p: str) -> str:
    return normalize_provider(p)


class _MinIntervalLimiter:
    __slots__ = ("min_interval_s", "_last_mono")

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = max(0.0, float(min_interval_s))
        self._last_mono = 0.0

    def wait_and_stamp(self) -> None:
        now = time.monotonic()
        if self.min_interval_s <= 0:
            self._last_mono = now
            return
        wait = self.min_interval_s - (now - self._last_mono)
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        self._last_mono = now


class _RateLimiterRegistry:
    """Thread-safe per-(provider, model, kind) minimum-interval throttling."""

    def __init__(self) -> None:
        self._locks: dict[tuple[str, str, str], threading.Lock] = {}
        self._limiters: dict[tuple[str, str, str], _MinIntervalLimiter] = {}

    def _key(self, provider: str, model: str, request_kind: LLMRequestKind | str) -> tuple[str, str, str]:
        rk = request_kind.value if isinstance(request_kind, LLMRequestKind) else str(request_kind)
        return (_norm_provider(provider), str(model).strip(), rk)

    def acquire(
        self,
        provider: str,
        model: str,
        request_kind: LLMRequestKind | str = LLMRequestKind.REGULAR,
    ) -> None:
        k = self._key(provider, model, request_kind)
        if k not in self._limiters:
            interval = _resolve_min_interval_s(provider, model, request_kind)
            self._limiters[k] = _MinIntervalLimiter(interval)
            self._locks[k] = threading.Lock()
        lim = self._limiters[k]
        lock = self._locks[k]
        if lim.min_interval_s <= 0:
            return
        with lock:
            lim.wait_and_stamp()


def _resolve_min_interval_s(
    provider: str,
    model: str,
    request_kind: LLMRequestKind | str,
) -> float:
    from configurations import config as cfg

    limits_root: dict[str, Any] = getattr(cfg, "LLM_GATEWAY_RATE_LIMITS", None) or {}
    prov = _norm_provider(provider)
    by_model = limits_root.get(prov)
    if not isinstance(by_model, dict):
        return 0.0

    m = str(model).strip()
    rk = request_kind.value if isinstance(request_kind, LLMRequestKind) else str(request_kind)

    entry: Any = None
    if m and m in by_model:
        entry = by_model[m]
    elif f"{rk}:{m}" in by_model:
        entry = by_model[f"{rk}:{m}"]
    elif f"{rk}:*" in by_model:
        entry = by_model[f"{rk}:*"]
    elif "__default__" in by_model:
        entry = by_model["__default__"]

    if entry is None or not isinstance(entry, dict):
        return 0.0

    mis = entry.get("min_interval_s")
    if mis is not None:
        try:
            return max(0.0, float(mis))
        except (TypeError, ValueError):
            pass

    rps = entry.get("rps")
    if rps is not None:
        try:
            r = float(rps)
            if r > 0:
                return 1.0 / r
        except (TypeError, ValueError):
            pass

    rpm = entry.get("rpm")
    if rpm is None:
        return 0.0
    try:
        rpi = int(rpm)
        if rpi <= 0:
            return 0.0
        return 60.0 / float(rpi)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Provider-specific senders (sync; return parsed text)
# ---------------------------------------------------------------------------


def _response_to_dict(resp: Any) -> dict:
    if isinstance(resp, dict):
        return resp
    if resp is None:
        return {}
    if hasattr(resp, "model_dump"):
        try:
            out = resp.model_dump()
            return out if isinstance(out, dict) else {}
        except Exception:
            return {}
    return {}


def _parse_ollama_text(resp: Any) -> str | None:
    payload = _response_to_dict(resp)
    direct = payload.get("response")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    msg = payload.get("message", {})
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, str) and content.strip():
        return content.strip()
    return None


def _send_ollama(req: LLMRequest) -> str:
    from configurations.config import OLLAMA_HOST, OLLAMA_TIMEOUT_S, OLLAMA_VERIFY_SSL

    host = str(req.metadata.get("host") or OLLAMA_HOST)
    verify_ssl = bool(req.metadata.get("verify_ssl", OLLAMA_VERIFY_SSL))
    timeout_s = float(req.metadata.get("timeout_s", OLLAMA_TIMEOUT_S))
    system = (req.system_prompt or "Return strictly JSON only.").strip()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": req.prompt},
    ]
    client = get_ollama_client(host, verify_ssl=verify_ssl, timeout_s=timeout_s)

    gen_resp = client.generate(
        model=req.model,
        prompt=req.prompt,
        stream=False,
        format="json",
    )
    text = _parse_ollama_text(gen_resp)
    if text:
        return text

    try:
        chat_resp = client.chat(model=req.model, messages=messages, think="low")
    except TypeError:
        chat_resp = client.chat(model=req.model, messages=messages)
    text = _parse_ollama_text(chat_resp)
    if text:
        return text
    raise LLMGatewayError("Ollama returned empty content from generate and chat")


def _send_nvidia_ih(req: LLMRequest) -> str:
    from configurations.config import (
        NVIDIA_IH_API_KEY,
        NVIDIA_IH_GENERATE_URL_TEMPLATE,
        NVIDIA_IH_TIMEOUT_S,
    )

    api_key = str(req.metadata.get("api_key") or NVIDIA_IH_API_KEY or "").strip()
    if not api_key:
        raise LLMGatewayError(
            "NVIDIA Inference Hub: set IH_API_KEY or NVIDIA_IH_API_KEY in the environment."
        )
    url_template = str(
        req.metadata.get("url_template") or NVIDIA_IH_GENERATE_URL_TEMPLATE
    )
    timeout_s = float(req.metadata.get("timeout_s", NVIDIA_IH_TIMEOUT_S))
    temperature = float(req.metadata.get("temperature", 0.0))
    max_output_tokens = int(req.metadata.get("max_output_tokens", 512))
    top_p = float(req.metadata.get("top_p", 0.1))
    top_k = int(req.metadata.get("top_k", 1))

    url = url_template.format(model=req.model)
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": req.prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "topP": top_p,
            "topK": top_k,
        },
    }
    si = (req.system_prompt or "").strip()
    if si:
        payload["systemInstruction"] = {"parts": [{"text": si}]}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    cands = data.get("candidates") or []
    if not cands:
        raise LLMGatewayError(
            f"NVIDIA IH: no candidates in response: {json.dumps(data)[:800]}"
        )
    parts = (cands[0].get("content") or {}).get("parts") or []
    if not parts:
        raise LLMGatewayError("NVIDIA IH: empty content.parts")
    text = parts[0].get("text")
    if text is None or not str(text).strip():
        raise LLMGatewayError("NVIDIA IH: empty text in first part")
    return str(text).strip()


def _send_gemini(req: LLMRequest) -> str:
    from configurations.config import GEMINI_API_KEY

    api_key = str(req.metadata.get("api_key") or GEMINI_API_KEY or "").strip()
    if not api_key:
        raise LLMGatewayError("Gemini: GEMINI_API_KEY is not configured.")
    client = get_gemini_client(api_key)
    response = client.models.generate_content(model=req.model, contents=req.prompt)
    text = getattr(response, "text", None)
    if text is None or not str(text).strip():
        raise LLMGatewayError("Gemini returned empty text")
    return str(text).strip()


_PROVIDER_SENDERS: dict[str, Callable[[LLMRequest], str]] = {
    "ollama": _send_ollama,
    "nvidia_ih": _send_nvidia_ih,
    "gemini": _send_gemini,
}


def _execute_request(req: LLMRequest) -> str:
    prov = normalize_provider(req.provider)
    sender = _PROVIDER_SENDERS.get(prov)
    if sender is None:
        raise LLMGatewayError(f"Unsupported LLM provider: {req.provider!r}")
    return sender(req)


# ---------------------------------------------------------------------------
# Async gateway service (background dispatcher + worker pool)
# ---------------------------------------------------------------------------


@dataclass
class _QueuedWork:
    request: LLMRequest
    future: ConcurrentFuture[str]


class LLMGatewayService:
    """
    Background service: queue → rate limit → thread-pool provider send → future.

    Start once via :func:`get_llm_gateway_service`; experiments only submit prompts.
    """

    def __init__(self, *, max_workers: int = 32) -> None:
        self._rate_limiter = _RateLimiterRegistry()
        self._queue: queue.Queue[_QueuedWork | None] = queue.Queue()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="llm-gateway-worker",
        )
        self._dispatcher: threading.Thread | None = None
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._dispatcher = threading.Thread(
                target=self._dispatch_loop,
                name="llm-gateway-dispatcher",
                daemon=True,
            )
            self._dispatcher.start()
            self._started = True

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            if not self._started:
                return
            self._queue.put(None)
            if self._dispatcher is not None:
                self._dispatcher.join(timeout=30.0 if wait else 0.5)
            self._executor.shutdown(wait=wait, cancel_futures=False)
            self._started = False

    def _dispatch_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                self._rate_limiter.acquire(
                    item.request.provider,
                    item.request.model,
                    item.request.request_kind,
                )
                result = self._executor.submit(_execute_request, item.request).result()
                item.future.set_result(result)
            except Exception as exc:
                item.future.set_exception(exc)

    def submit(self, request: LLMRequest) -> ConcurrentFuture[str]:
        """Enqueue a request; returns a concurrent Future with parsed text."""
        self.start()
        cfut: ConcurrentFuture[str] = ConcurrentFuture()
        self._queue.put(_QueuedWork(request=request, future=cfut))
        return cfut

    def submit_async(self, request: LLMRequest) -> asyncio.Future[str]:
        """Enqueue from async code; returns an asyncio Future with parsed text."""
        self.start()
        loop = asyncio.get_running_loop()
        afut: asyncio.Future[str] = loop.create_future()
        cfut = self.submit(request)

        def _bridge_done(done: ConcurrentFuture[str]) -> None:
            try:
                loop.call_soon_threadsafe(afut.set_result, done.result())
            except Exception as exc:
                loop.call_soon_threadsafe(afut.set_exception, exc)

        cfut.add_done_callback(_bridge_done)
        return afut


_service: LLMGatewayService | None = None
_service_lock = threading.Lock()


def get_llm_gateway_service() -> LLMGatewayService:
    global _service
    with _service_lock:
        if _service is None:
            _service = LLMGatewayService()
            _service.start()
        return _service


# ---------------------------------------------------------------------------
# High-level API for experiments
# ---------------------------------------------------------------------------


def complete_prompt(
    *,
    provider: str,
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    request_kind: LLMRequestKind | str = LLMRequestKind.REGULAR,
    timeout_s: float | None = 180.0,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Submit a prompt and block until parsed text is returned."""
    req = LLMRequest(
        provider=provider,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        request_kind=request_kind,
        metadata=dict(metadata or {}),
    )
    fut = get_llm_gateway_service().submit(req)
    return fut.result(timeout=timeout_s)


async def await_prompt(
    *,
    provider: str,
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    request_kind: LLMRequestKind | str = LLMRequestKind.REGULAR,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Submit a prompt without blocking the event loop; await parsed text."""
    req = LLMRequest(
        provider=provider,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        request_kind=request_kind,
        metadata=dict(metadata or {}),
    )
    return await get_llm_gateway_service().submit_async(req)


# ---------------------------------------------------------------------------
# Relevance scoring (prompt build, outbound call, parse — for experiments)
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)

RELEVANCE_JSON_SYSTEM_PROMPT = "Return strictly JSON only."


def build_relevance_scoring_prompt(query: str, doc_contexts: list[str]) -> str:
    """Build the multi-document relevance scoring user prompt (1–10 scale, JSON array output)."""
    docs_block = ""
    for i, context in enumerate(doc_contexts, start=1):
        docs_block += f"--- ITEM {i} ---\n{str(context)[:4000]}\n\n"
    n = len(doc_contexts)
    return f"""
### ROLE
Expert Scientific Relevance Assessor.

### TASK
Rate the relevance of EACH ITEM (DOCUMENT excerpt) to the QUERY on a scale of 1 to 10.
Return ONLY a JSON array of integers (length {n}). No extra text.

### SCORING DEFINITIONS (STRICT - NO OVERLAP)
- 10: [PERFECT] Document contains the exact answer or specific evidence needed.
- 8: [STRONG] Document is directly on-topic and provides significant information, but no direct answer.
- 6: [SPECIFIC FIELD] Document is in the same sub-field and discusses relevant entities, but not the specific question.
- 4: [GENERAL DOMAIN] Document is in the same general scientific field but answers a different problem.
- 2: [TANGENTIAL] Only shares keywords; the context is entirely different.
- 1: [IRRELEVANT] No connection at all.

*For values 3, 5, 7, 9: Use only if the document falls exactly between two definitions.*

### OUTPUT RULES (MANDATORY)
- Output must be valid JSON in ONE of these exact shapes only:
  1) [8, 2, 10, 4, ...]
  2) {{"scores":[8, 2, 10, 4, ...]}}
- Each value must be an integer in 1..10 (0 is not allowed)
- Return exactly {n} integers, in the same order as the ITEMS.
- No explanations, no markdown, no keys besides "scores".

---
QUERY:
{query}

        {docs_block}

Return ONLY the JSON array of {n} integers.
    """


def parse_relevance_scores(text: str | None, expected_n: int) -> list[int]:
    """
    Parse model text into ``expected_n`` integer scores in 0..10 (0 = parse failure).
    Used for live responses and Gemini Batch harvest output.
    """
    if text is None:
        return [0] * expected_n
    t = str(text).strip()
    if not t:
        return [0] * expected_n

    def _normalize_out(vals: list[int]) -> list[int]:
        out2 = [max(0, min(10, int(v))) for v in vals]
        if len(out2) < expected_n:
            out2.extend([0] * (expected_n - len(out2)))
        return out2[:expected_n]

    def _extract_from_obj(obj: Any) -> list[int] | None:
        if isinstance(obj, list):
            vals: list[int] = []
            for x in obj:
                try:
                    vals.append(int(x))
                except Exception:
                    pass
            return _normalize_out(vals) if vals else None
        if isinstance(obj, dict):
            for k in ("scores", "answer", "answers", "result", "results", "values", "output"):
                if k in obj:
                    got = _extract_from_obj(obj.get(k))
                    if got is not None:
                        return got
        return None

    try:
        full_obj = json.loads(t)
        got = _extract_from_obj(full_obj)
        if got is not None:
            return got
    except Exception:
        pass

    m = re.search(r"\[[\s\S]*?\]", t)
    candidate = m.group(0).strip() if m else t
    try:
        arr = json.loads(candidate)
        got = _extract_from_obj(arr)
        if got is not None:
            return got
    except Exception:
        pass

    nums = re.findall(r"\b(10|[1-9])\b", t)
    return _normalize_out([int(n) for n in nums[:expected_n]])


def normalize_relevance_scores(scores: list[int], expected_n: int) -> list[int]:
    """Clamp live scores to 1..10; pad or truncate to ``expected_n``."""
    out: list[int] = []
    changes = 0
    for i in range(expected_n):
        raw = scores[i] if i < len(scores) else 1
        try:
            v = int(raw)
        except Exception:
            v = 1
            changes += 1
        if v < 1:
            v = 1
            changes += 1
        elif v > 10:
            v = 10
            changes += 1
        out.append(v)
    if changes > 0:
        _logger.warning(
            "Relevance score validation adjusted %s value(s); expected_n=%s.",
            changes,
            expected_n,
        )
    return out


def relevance_scores_from_text(text: str | None, expected_n: int) -> list[int]:
    """Parse then normalize to valid 1..10 live scores."""
    return normalize_relevance_scores(parse_relevance_scores(text, expected_n), expected_n)


async def await_relevance_scores_for_documents(
    *,
    provider: str,
    model: str,
    query: str,
    doc_contexts: list[str],
    system_prompt: str | None = RELEVANCE_JSON_SYSTEM_PROMPT,
    request_kind: LLMRequestKind | str = LLMRequestKind.REGULAR,
    metadata: dict[str, Any] | None = None,
) -> list[int]:
    """
    Score multiple document excerpts against one query (live providers).

    Builds the scoring prompt internally, calls the provider, parses scores.
    """
    n = len(doc_contexts)
    if n < 1:
        return []
    prompt = build_relevance_scoring_prompt(query, doc_contexts)
    return await await_relevance_scores(
        provider=provider,
        model=model,
        prompt=prompt,
        expected_n=n,
        system_prompt=system_prompt,
        request_kind=request_kind,
        metadata=metadata,
    )


async def await_relevance_scores(
    *,
    provider: str,
    model: str,
    prompt: str,
    expected_n: int,
    system_prompt: str | None = RELEVANCE_JSON_SYSTEM_PROMPT,
    request_kind: LLMRequestKind | str = LLMRequestKind.REGULAR,
    metadata: dict[str, Any] | None = None,
) -> list[int]:
    """
    Send a scoring prompt to the provider and return integer relevance scores (1..10).

  All HTTP/SDK, rate limiting, and response parsing happen inside the gateway.
    """
    if expected_n < 1:
        return []
    text = await await_prompt(
        provider=provider,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        request_kind=request_kind,
        metadata=metadata,
    )
    if not str(text).strip():
        raise LLMGatewayError(f"Empty response from provider {provider!r}")
    return relevance_scores_from_text(text, expected_n)


def complete_relevance_scores(
    *,
    provider: str,
    model: str,
    prompt: str,
    expected_n: int,
    system_prompt: str | None = RELEVANCE_JSON_SYSTEM_PROMPT,
    request_kind: LLMRequestKind | str = LLMRequestKind.REGULAR,
    timeout_s: float | None = 180.0,
    metadata: dict[str, Any] | None = None,
) -> list[int]:
    """Blocking variant of :func:`await_relevance_scores`."""
    if expected_n < 1:
        return []
    text = complete_prompt(
        provider=provider,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        request_kind=request_kind,
        timeout_s=timeout_s,
        metadata=metadata,
    )
    if not str(text).strip():
        raise LLMGatewayError(f"Empty response from provider {provider!r}")
    return relevance_scores_from_text(text, expected_n)


# Legacy singleton rate gate (batch job pacing outside the async queue)
_gateway: _RateLimiterRegistry | None = None


def get_llm_gateway() -> _RateLimiterRegistry:
    """Acquire-only limiter for code paths that manage their own HTTP (e.g. batch submit)."""
    global _gateway
    if _gateway is None:
        _gateway = _RateLimiterRegistry()
    return _gateway


# ---------------------------------------------------------------------------
# Client singletons (batch / listing — not live prompt queue)
# ---------------------------------------------------------------------------

_ollama_clients: dict[str, Any] = {}
_ollama_clients_lock = threading.Lock()
_gemini_clients: dict[str, Any] = {}
_gemini_clients_lock = threading.Lock()


def get_ollama_client(
    host: str,
    *,
    verify_ssl: bool = False,
    timeout_s: float = 180.0,
) -> Any:
    with _ollama_clients_lock:
        if host not in _ollama_clients:
            from ollama import Client as _OllamaClient
            try:
                _ollama_clients[host] = _OllamaClient(
                    host=host,
                    verify=verify_ssl,
                    timeout=timeout_s,
                )
            except TypeError:
                _ollama_clients[host] = _OllamaClient(host=host, verify=verify_ssl)
    return _ollama_clients[host]


def get_gemini_client(api_key: str) -> Any:
    with _gemini_clients_lock:
        if api_key not in _gemini_clients:
            from google import genai as _genai
            _gemini_clients[api_key] = _genai.Client(api_key=api_key)
    return _gemini_clients[api_key]


# ---------------------------------------------------------------------------
# Thin backward-compatible wrappers (delegate to gateway service)
# ---------------------------------------------------------------------------


def ollama_generate(
    prompt: str,
    *,
    model: str,
    host: str,
    verify_ssl: bool = False,
    timeout_s: float = 180.0,
    stream: bool = False,
    response_format: str | None = None,
) -> str:
    del stream, response_format  # live path always returns parsed text via service
    return complete_prompt(
        provider="ollama",
        model=model,
        prompt=prompt,
        system_prompt="Return strictly JSON only.",
        metadata={"host": host, "verify_ssl": verify_ssl, "timeout_s": timeout_s},
        timeout_s=timeout_s,
    )


def ollama_chat(
    messages: list[dict],
    *,
    model: str,
    host: str,
    verify_ssl: bool = False,
    timeout_s: float = 180.0,
    **chat_kwargs: Any,
) -> str:
    del chat_kwargs
    system = next(
        (m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "system"),
        "Return strictly JSON only.",
    )
    user = next(
        (m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "user"),
        "",
    )
    return complete_prompt(
        provider="ollama",
        model=model,
        prompt=str(user or ""),
        system_prompt=str(system or ""),
        metadata={"host": host, "verify_ssl": verify_ssl, "timeout_s": timeout_s},
        timeout_s=timeout_s,
    )


def nvidia_inference_hub_generate(
    *,
    model: str,
    prompt: str,
    api_key: str,
    url_template: str,
    timeout_s: float,
    temperature: float = 0.0,
    max_output_tokens: int = 512,
    top_p: float = 0.1,
    top_k: int = 1,
    system_instruction: str | None = None,
) -> str:
    return complete_prompt(
        provider="nvidia_ih",
        model=model,
        prompt=prompt,
        system_prompt=system_instruction,
        metadata={
            "api_key": api_key,
            "url_template": url_template,
            "timeout_s": timeout_s,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "top_p": top_p,
            "top_k": top_k,
        },
        timeout_s=timeout_s,
    )


def gemini_generate_content(
    prompt: str,
    *,
    model: str,
    api_key: str,
) -> str:
    return complete_prompt(
        provider="gemini",
        model=model,
        prompt=prompt,
        metadata={"api_key": api_key},
    )
