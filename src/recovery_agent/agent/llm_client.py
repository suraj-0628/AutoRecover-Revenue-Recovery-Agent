"""Shared LLM client — single entry point for all LLM calls.

Uses Nemotron via OmniRoute local API (OpenAI-compatible).
Provides structured output parsing with Pydantic validation.

When the LLM server is unavailable or returning invalid responses,
all calls return None immediately so downstream code falls back
to rule-based logic.

TIMEOUT ARCHITECTURE:
  - LLM_TIMEOUT (env var, default 25s): Total budget for ALL models combined.
  - Per-model cap: LLM_TIMEOUT / len(candidates), min 5s.
  - If primary model doesn't respond within per-model cap, immediately
    try one fallback, then fall back to heuristic. Total never exceeds LLM_TIMEOUT.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# --- Arize Phoenix Observability (LangChain) ---

def _ensure_phoenix():
    """Tracing init is shared and idempotent — observability.py owns it.

    This used to register its OWN Phoenix provider lazily on the first
    invoke_llm call, racing frontend's homegrown provider for OTel's single
    global slot. One init point, or no init point.
    """
    from recovery_agent.observability import init_observability
    init_observability("llm-client")

# Total budget for ALL LLM fallback models combined.
# Override via LLM_TIMEOUT env var (e.g., LLM_TIMEOUT=15).
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "25"))

_llm_available: bool | None = None
_LLM_FAIL_THRESHOLD = 2  # After N failures, stop trying for this session

# Thread-local storage for per-thread consecutive failure tracking
_thread_local = threading.local()


def _get_consecutive_failures() -> int:
    """Get consecutive failure count for current thread."""
    return getattr(_thread_local, "consecutive_failures", 0)


def _set_consecutive_failures(count: int) -> None:
    """Set consecutive failure count for current thread."""
    _thread_local.consecutive_failures = count


def _check_llm_reachable() -> bool:
    """Quick TCP check if the LLM server is reachable."""
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:20128/v1")
    try:
        parsed = urlparse(base_url)
        host = parsed.hostname or "localhost"
        # A remote provider (https://api.anthropic.com/v1) carries no explicit
        # port, and defaulting to the local proxy's 20128 made this probe dial
        # a port nothing listens on. The probe short-circuits get_llm_response,
        # so every remote endpoint looked unreachable and the agent dropped to
        # heuristics without ever placing a call — a silent downgrade that
        # reads, from the outside, exactly like a model that answered badly.
        port = parsed.port or (443 if parsed.scheme == "https" else 20128)
        sock = socket.create_connection((host, port), timeout=1)
        sock.close()
        return True
    except (socket.timeout, OSError, ConnectionRefusedError):
        return False


DEFAULT_MODELS = [
    "antigravity/gemini-2.5-flash",
    "no-think/antigravity/claude-sonnet-4-6",
    "antigravity/gemini-3.6-flash-high",
    "auto/best-reasoning",
]

def _fallback_models() -> list[str]:
    """Alternates to try when the primary model fails — from THIS endpoint.

    DEFAULT_MODELS are ids the local router invents (`no-think/...`, `auto/...`);
    they mean nothing to a first-party API. Pointed at Anthropic, a single
    hiccup on the primary spent the rest of the budget on three guaranteed-404
    model names and then returned None, so a working key still produced
    heuristics. A remote endpoint gets one honest attempt instead of three
    fictional ones.
    """
    extra = [m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",")
             if m.strip()]
    if extra:
        # Explicit config wins, on any endpoint. This used to be ignored on
        # localhost while graph._fallback_chain honoured it, so the same env
        # var produced two different chains in one process — the sort of split
        # that is invisible until a fallback silently does nothing.
        return extra
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:20128/v1")
    host = (urlparse(base_url).hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1"):
        return list(DEFAULT_MODELS)
    return []


def get_llm(
    temperature: float = 0,
    max_tokens: int = 512,
    model_name: str | None = None,
) -> ChatOpenAI:
    """Get or create the shared LLM client instance.

    MANDATE 1: Uses ChatOpenAI(cache=True) — langchain_core.caches.InMemoryCache
    to avoid redundant LLM calls for identical prompts.
    """
    from langchain_core.globals import set_llm_cache
    from langchain_core.caches import InMemoryCache
    set_llm_cache(InMemoryCache())  # safe to call multiple times

    base_url = os.getenv("LLM_BASE_URL", "http://localhost:20128/v1")
    model = model_name or os.getenv("LLM_MODEL", "antigravity/gemini-2.5-flash")
    api_key = os.getenv("LLM_API_KEY", "dummy")
    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=LLM_TIMEOUT,
        cache=True,  # MANDATE 1: langchain_core.caches.InMemoryCache
    )


def _invoke_with_timeout(llm: ChatOpenAI, messages: list, deadline: float) -> Any | None:
    """Run llm.invoke in a daemon thread with a hard timeout based on a shared deadline.

    Returns response on success, None on timeout/error.
    The deadline is an absolute timestamp (time.monotonic() + seconds).
    """
    result = [None]
    exc = [None]

    def _target():
        try:
            result[0] = llm.invoke(messages)
        except Exception as e:
            exc[0] = e

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=remaining)
    if t.is_alive():
        print(f"[LLM TIMEOUT] Model exceeded deadline ({remaining:.1f}s remaining)")
        return None
    if exc[0] is not None:
        raise exc[0]
    return result[0]


def invoke_llm(
    prompt: str,
    system: str = "",
    temperature: float = 0,
    max_tokens: int = 512,
) -> str | None:
    """Invoke the LLM with budget-capped multi-model fallback.

    Total time across ALL models is capped at LLM_TIMEOUT seconds.
    Per-model timeout scales down with number of candidates so the total
    never exceeds the budget. Falls back to heuristic immediately on timeout.

    Architecture:
      - 4 candidates, 25s budget → ~6s per model cap
      - 4 candidates, 15s budget → ~3s per model cap
      - Primary model always gets first shot
    """
    _ensure_phoenix()
    if not _check_llm_reachable():
        return None

    primary_model = os.getenv("LLM_MODEL", "antigravity/gemini-2.5-flash")
    candidate_models = [primary_model] + [
        m for m in _fallback_models() if m != primary_model]

    deadline = time.monotonic() + LLM_TIMEOUT
    per_model_timeout = max(5, LLM_TIMEOUT // max(1, len(candidate_models)))

    for model in candidate_models:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"[LLM BUDGET EXHAUSTED] Total {LLM_TIMEOUT}s budget spent. Falling back to heuristic.")
            break

        model_timeout = min(per_model_timeout, remaining)

        for attempt in range(2):
            try:
                llm = get_llm(temperature=temperature, max_tokens=max_tokens, model_name=model)
                messages = []
                if system:
                    messages.append(SystemMessage(content=system))
                messages.append(HumanMessage(content=prompt))
                response = _invoke_with_timeout(llm, messages, deadline)
                if response is not None and response.content:
                    return response.content.strip()
            except Exception as e:
                err_str = str(e).lower()
                if attempt == 0 and ("400" in str(e) or "bad request" in err_str or "invalid_argument" in err_str):
                    time.sleep(1)
                    continue
                print(f"[LLM ERROR with model {model}]: {e}")
                break

    return None


def invoke_llm_json(
    prompt: str,
    system: str = "",
    temperature: float = 0,
    max_tokens: int = 512,
) -> dict[str, Any] | None:
    """Invoke the LLM and parse the response as JSON.

    Tracks consecutive JSON parse failures — after 2 unparseable responses,
    stops trying for this thread (the LLM may be reachable but not
    producing structured output). Thread-safe via thread-local storage.
    """
    if _get_consecutive_failures() >= _LLM_FAIL_THRESHOLD:
        return None

    json_system = (system + "\n\nCRITICAL: Respond ONLY with a valid JSON object. Do NOT include any intro text, preamble, or markdown code blocks.").strip()
    raw = invoke_llm(
        prompt=prompt,
        system=json_system,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if raw is None:
        return None

    cleaned = raw.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(cleaned)
        _set_consecutive_failures(0)  # Reset on success
        return result
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(cleaned[start:end])
                _set_consecutive_failures(0)
                return result
            except json.JSONDecodeError:
                pass
        _set_consecutive_failures(_get_consecutive_failures() + 1)
        return None
