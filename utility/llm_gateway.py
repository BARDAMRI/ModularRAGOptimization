"""
Central LLM gateway: rate limiting + dispatch for all outbound LLM calls.

All experiment code must go through this module to send prompts to any
provider (Ollama, Gemini live, Gemini Batch, NVIDIA Inference Hub).
The gateway handles:
  - per-(provider, model, kind) rate limiting via :meth:`LLMGateway.acquire`
  - provider client singletons (Ollama, Gemini)
  - outbound call wrappers that acquire a slot then make the HTTP/SDK request

Rate limits are read from ``config.LLM_GATEWAY_RATE_LIMITS``. Missing model
entry, missing provider entry, or ``None``/omitted RPM/RPS ⇒ **no throttle**.
"""
from __future__ import annotations

import json
import threading
import time
from enum import Enum
from typing import Any

import requests


class LLMRequestKind(str, Enum):
    REGULAR = "regular"
    BATCH = "batch"


def _norm_provider(p: str) -> str:
    return str(p).strip().lower()


class LLMGateway:
    """
    Thread-safe per-(provider, model, kind) throttling using a minimum interval
    between outbound calls (derived from RPM when configured).
    """

    def __init__(self) -> None:
        self._locks: dict[tuple[str, str, str], threading.Lock] = {}
        self._limiters: dict[tuple[str, str, str], _MinIntervalLimiter] = {}

    def _key(
        self,
        provider: str,
        model: str,
        request_kind: LLMRequestKind | str,
    ) -> tuple[str, str, str]:
        rk = request_kind.value if isinstance(request_kind, LLMRequestKind) else str(request_kind)
        return (_norm_provider(provider), str(model).strip(), rk)

    def _get_limiter(self, provider: str, model: str, request_kind: LLMRequestKind | str) -> _MinIntervalLimiter:
        k = self._key(provider, model, request_kind)
        if k not in self._limiters:
            interval = _resolve_min_interval_s(provider, model, request_kind)
            self._limiters[k] = _MinIntervalLimiter(interval)
            self._locks[k] = threading.Lock()
        return self._limiters[k]

    def acquire(
        self,
        provider: str,
        model: str,
        request_kind: LLMRequestKind | str = LLMRequestKind.REGULAR,
    ) -> None:
        """Block until a request may be sent according to configured rate limits."""
        k = self._key(provider, model, request_kind)
        if k not in self._limiters:
            self._get_limiter(provider, model, request_kind)
        lim = self._limiters[k]
        lock = self._locks[k]
        if lim.min_interval_s <= 0:
            return
        with lock:
            lim.wait_and_stamp()


# --- module singleton ----------------------------------------------------------

_gateway: LLMGateway | None = None


def get_llm_gateway() -> LLMGateway:
    global _gateway
    if _gateway is None:
        _gateway = LLMGateway()
    return _gateway


# --- rate limit resolution ------------------------------------------------------


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

    if entry is None:
        return 0.0
    if not isinstance(entry, dict):
        return 0.0

    # Explicit seconds between requests (highest precedence if set)
    mis = entry.get("min_interval_s")
    if mis is not None:
        try:
            v = float(mis)
            return max(0.0, v)
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


# --- NVIDIA Inference Hub (Vertex-style generateContent over HTTP) ------------


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
) -> str:
    """
    Acquire a rate-limit slot then POST to NVIDIA IH ``generateContent``.

    Returns raw model text from the first candidate.
    """
    get_llm_gateway().acquire("nvidia_ih", model, LLMRequestKind.REGULAR)
    url = url_template.format(model=model)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "topP": top_p,
            "topK": top_k,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    cands = data.get("candidates") or []
    if not cands:
        raise RuntimeError(f"NVIDIA IH: no candidates in response: {json.dumps(data)[:800]}")
    parts = (cands[0].get("content") or {}).get("parts") or []
    if not parts:
        raise RuntimeError("NVIDIA IH: empty content.parts")
    text = parts[0].get("text")
    if text is None or not str(text).strip():
        raise RuntimeError("NVIDIA IH: empty text in first part")
    return str(text).strip()


# --- Ollama client + call wrappers -------------------------------------------

_ollama_clients: dict[str, Any] = {}
_ollama_clients_lock = threading.Lock()


def get_ollama_client(
    host: str,
    *,
    verify_ssl: bool = False,
    timeout_s: float = 180.0,
) -> Any:
    """
    Return a per-host Ollama ``Client`` singleton (thread-safe).

    Uses the first ``verify_ssl`` / ``timeout_s`` values seen for each host.
    All experiment code should obtain clients here rather than constructing them
    directly so the gateway owns the connection lifecycle.
    """
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
                # Older SDK versions may not accept timeout= in the constructor.
                _ollama_clients[host] = _OllamaClient(host=host, verify=verify_ssl)
    return _ollama_clients[host]


def ollama_generate(
    prompt: str,
    *,
    model: str,
    host: str,
    verify_ssl: bool = False,
    timeout_s: float = 180.0,
    stream: bool = False,
    response_format: str | None = None,
) -> Any:
    """
    Acquire a rate-limit slot then call ``ollama.Client.generate()``.

    Returns the raw SDK response object so callers can inspect ``response``
    or ``message.content`` as needed.
    """
    get_llm_gateway().acquire("ollama", model, LLMRequestKind.REGULAR)
    client = get_ollama_client(host, verify_ssl=verify_ssl, timeout_s=timeout_s)
    kwargs: dict[str, Any] = {"model": model, "prompt": prompt, "stream": stream}
    if response_format is not None:
        kwargs["format"] = response_format
    return client.generate(**kwargs)


def ollama_chat(
    messages: list[dict],
    *,
    model: str,
    host: str,
    verify_ssl: bool = False,
    timeout_s: float = 180.0,
    **chat_kwargs: Any,
) -> Any:
    """
    Acquire a rate-limit slot then call ``ollama.Client.chat()``.

    Extra keyword arguments (e.g. ``think="low"``) are forwarded to the SDK;
    callers are responsible for catching ``TypeError`` when the installed SDK
    version does not support a given parameter.

    Returns the raw SDK response object.
    """
    get_llm_gateway().acquire("ollama", model, LLMRequestKind.REGULAR)
    client = get_ollama_client(host, verify_ssl=verify_ssl, timeout_s=timeout_s)
    return client.chat(model=model, messages=messages, **chat_kwargs)


# --- Gemini client + call wrappers -------------------------------------------

_gemini_clients: dict[str, Any] = {}
_gemini_clients_lock = threading.Lock()


def get_gemini_client(api_key: str) -> Any:
    """
    Return a per-key ``google.genai.Client`` singleton (thread-safe).

    All code that needs a Gemini client for batch, file, or live operations
    should call this rather than constructing ``genai.Client`` directly.
    """
    with _gemini_clients_lock:
        if api_key not in _gemini_clients:
            from google import genai as _genai
            _gemini_clients[api_key] = _genai.Client(api_key=api_key)
    return _gemini_clients[api_key]


def gemini_generate_content(
    prompt: str,
    *,
    model: str,
    api_key: str,
) -> str:
    """
    Acquire a rate-limit slot then call Gemini live ``generate_content``.

    Returns ``response.text`` stripped of leading/trailing whitespace.
    Raises whatever the SDK raises (callers should implement retry logic).
    """
    get_llm_gateway().acquire("gemini", model, LLMRequestKind.REGULAR)
    client = get_gemini_client(api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text.strip()
