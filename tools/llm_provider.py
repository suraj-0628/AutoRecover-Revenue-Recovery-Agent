#!/usr/bin/env python3
"""Switch the agent between LLM providers, all four settings at once.

Base URL, model, key and fallback chain have to agree. Changing one by hand
leaves the others describing a different provider: point LLM_BASE_URL back at
the local router while a previous provider's names are still in
LLM_FALLBACK_MODELS and the chain becomes model ids that endpoint has never
heard of — every fallback 404s, and the only visible symptom is an agent that
gives up early.

Keys are kept per provider (GROQ_API_KEY, GOOGLE_AI_API_KEY) so switching back
and forth never loses one. Nothing is printed but the resulting model names.

    python tools/llm_provider.py            # show what is active
    python tools/llm_provider.py proxy      # local OmniRoute
    python tools/llm_provider.py groq
    python tools/llm_provider.py google
"""
import pathlib
import re
import sys

ENV = pathlib.Path(__file__).resolve().parent.parent / ".env"

PROVIDERS = {
    "proxy": {
        "LLM_BASE_URL": "http://localhost:20128/v1",
        "LLM_MODEL": "no-think/antigravity/claude-sonnet-4-6",
        # Empty: let graph._fallback_chain use its own capability-ordered list.
        "LLM_FALLBACK_MODELS": "",
        "_key_from": None,          # the router holds the real credentials
        "LLM_CALLS_PER_MINUTE": "7.5",
    },
    "groq": {
        "LLM_BASE_URL": "https://api.groq.com/openai/v1",
        "LLM_MODEL": "openai/gpt-oss-120b",
        "LLM_FALLBACK_MODELS": "qwen/qwen3.8-27b,openai/gpt-oss-20b",
        "_key_from": "GROQ_API_KEY",
        # 8000 TPM per model against ~7k-token turns is ~1 turn/model/minute.
        "LLM_CALLS_PER_MINUTE": "3",
    },
    "google": {
        "LLM_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "LLM_MODEL": "gemini-3.6-flash",
        "LLM_FALLBACK_MODELS": "gemini-3.5-flash,gemini-3-flash-preview,gemini-3.1-flash-lite",
        "_key_from": "GOOGLE_AI_API_KEY",
        "LLM_CALLS_PER_MINUTE": "5",
    },
}


def read() -> str:
    return ENV.read_text() if ENV.exists() else ""


def get(text: str, name: str) -> str:
    m = re.search(rf"^{re.escape(name)}=(.*)$", text, flags=re.M)
    return m.group(1).strip() if m else ""


def put(text: str, name: str, value: str) -> str:
    line = f"{name}={value}"
    if re.search(rf"^{re.escape(name)}=", text, flags=re.M):
        return re.sub(rf"^{re.escape(name)}=.*$", line, text, count=1, flags=re.M)
    return text.rstrip("\n") + f"\n{line}\n"


def main() -> int:
    text = read()
    if not text:
        print(f"no .env at {ENV}")
        return 1

    if len(sys.argv) < 2:
        url = get(text, "LLM_BASE_URL")
        active = next((n for n, c in PROVIDERS.items()
                       if c["LLM_BASE_URL"] == url), "custom")
        print(f"active   : {active}")
        print(f"base_url : {url}")
        print(f"model    : {get(text, 'LLM_MODEL')}")
        print(f"fallbacks: {get(text, 'LLM_FALLBACK_MODELS') or '(provider default)'}")
        print(f"available: {', '.join(PROVIDERS)}")
        return 0

    name = sys.argv[1].lower()
    cfg = PROVIDERS.get(name)
    if cfg is None:
        print(f"unknown provider {name!r}; choose from {', '.join(PROVIDERS)}")
        return 1

    # Preserve whichever key is currently live before overwriting it, so a
    # switch away from a provider does not discard the credential.
    cur_url = get(text, "LLM_BASE_URL")
    cur = next((c for c in PROVIDERS.values() if c["LLM_BASE_URL"] == cur_url), None)
    if cur and cur["_key_from"]:
        live = get(text, "LLM_API_KEY")
        if live and live != "dummy":
            text = put(text, cur["_key_from"], live)

    for field in ("LLM_BASE_URL", "LLM_MODEL", "LLM_FALLBACK_MODELS",
                  "LLM_CALLS_PER_MINUTE"):
        text = put(text, field, cfg[field])
    key = get(text, cfg["_key_from"]) if cfg["_key_from"] else ""
    text = put(text, "LLM_API_KEY", key or "dummy")

    ENV.write_text(text)
    print(f"switched to {name}")
    print(f"  model    : {cfg['LLM_MODEL']}")
    print(f"  fallbacks: {cfg['LLM_FALLBACK_MODELS'] or '(provider default)'}")
    if cfg["_key_from"] and not key:
        print(f"  WARNING  : {cfg['_key_from']} is empty — set it in .env")
    print("  restart the stack for this to take effect: ./start.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
