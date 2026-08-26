"""Shared LLM client — single entry point for all LLM calls.

Uses Nemotron via OmniRoute local API (OpenAI-compatible).
Provides structured output parsing with Pydantic validation.

When the LLM server is unavailable or returning invalid responses,
all calls return None immediately so downstream code falls back
to rule-based logic.
"""
from __future__ import annotations

import json
import os
import socket
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Timeout for LLM calls — allow time for complete dynamic reasoning completions
# Override via LLM_TIMEOUT env var for production use (e.g., LLM_TIMEOUT=30)
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "25"))

_llm_available: bool | None = None
_llm_consecutive_failures: int = 0
_LLM_FAIL_THRESHOLD = 2  # After N failures, stop trying for this session


def _check_llm_reachable() -> bool:
    """Quick TCP check if the LLM server is reachable."""
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:20128/v1")
    try:
        parsed = urlparse(base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 20128
        sock = socket.create_connection((host, port), timeout=1)
        sock.close()
        return True
    except (socket.timeout, OSError, ConnectionRefusedError):
        return False


DEFAULT_MODELS = [
    "antigravity/gemini-2.5-flash",
    "auto/best-reasoning",
    "antigravity/gemini-3.6-flash-high",
    "no-think/antigravity/claude-sonnet-4-6",
]

def get_llm(
    temperature: float = 0,
    max_tokens: int = 512,
    model_name: str | None = None,
) -> ChatOpenAI:
    """Get or create the shared LLM client instance."""
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
    )


def invoke_llm(
    prompt: str,
    system: str = "",
    temperature: float = 0,
    max_tokens: int = 512,
) -> str | None:
    """Invoke the LLM with multi-model fallback.

    Returns the raw string response, or None on failure.
    """
    if not _check_llm_reachable():
        return None

    primary_model = os.getenv("LLM_MODEL", "antigravity/gemini-2.5-flash")
    candidate_models = [primary_model] + [m for m in DEFAULT_MODELS if m != primary_model]

    for model in candidate_models:
        try:
            llm = get_llm(temperature=temperature, max_tokens=max_tokens, model_name=model)
            messages = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(HumanMessage(content=prompt))
            response = llm.invoke(messages)
            if response and response.content:
                return response.content.strip()
        except Exception as e:
            print(f"[LLM ERROR with model {model}]: {e}")
            continue

    return None


def invoke_llm_json(
    prompt: str,
    system: str = "",
    temperature: float = 0,
    max_tokens: int = 512,
) -> dict[str, Any] | None:
    """Invoke the LLM and parse the response as JSON.

    Tracks consecutive JSON parse failures — after 2 unparseable responses,
    stops trying for this session (the LLM may be reachable but not
    producing structured output).
    """
    global _llm_consecutive_failures
    if _llm_consecutive_failures >= _LLM_FAIL_THRESHOLD:
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
        _llm_consecutive_failures = 0  # Reset on success
        return result
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(cleaned[start:end])
                _llm_consecutive_failures = 0
                return result
            except json.JSONDecodeError:
                pass
        _llm_consecutive_failures += 1
        return None
