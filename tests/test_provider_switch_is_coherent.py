"""Switching providers must move every setting that depends on the provider.

Base URL, model, key and fallback chain describe ONE endpoint between them.
Editing the URL alone leaves the previous provider's model names in
LLM_FALLBACK_MODELS, and those ids mean nothing to the new endpoint: every
fallback 404s and the only symptom is an agent that stops trying early.
"""
import importlib.util
import pathlib

import pytest

SPEC = importlib.util.spec_from_file_location(
    "llm_provider",
    pathlib.Path(__file__).resolve().parent.parent / "tools" / "llm_provider.py")
LP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LP)


@pytest.fixture
def env(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("LLM_BASE_URL=https://api.groq.com/openai/v1\n"
                 "LLM_MODEL=openai/gpt-oss-120b\n"
                 "LLM_FALLBACK_MODELS=qwen/qwen3.8-27b\n"
                 "LLM_API_KEY=gsk_secret\n")
    monkeypatch.setattr(LP, "ENV", p)
    return p


def _switch(monkeypatch, name):
    monkeypatch.setattr("sys.argv", ["llm_provider.py", name])
    assert LP.main() == 0


def test_switching_to_the_proxy_clears_the_other_providers_models(env, monkeypatch):
    _switch(monkeypatch, "proxy")
    text = env.read_text()
    assert LP.get(text, "LLM_BASE_URL") == "http://localhost:20128/v1"
    fb = LP.get(text, "LLM_FALLBACK_MODELS")
    assert "qwen" not in fb and "gpt-oss" not in fb, \
        "Groq model ids would be sent to the local router and 404"
    assert all(m.startswith(("antigravity/", "auto/", "gemini", "no-think/"))
               for m in fb.split(",") if m), f"not router models: {fb}"


def test_the_outgoing_key_is_kept_not_discarded(env, monkeypatch):
    _switch(monkeypatch, "proxy")
    assert LP.get(env.read_text(), "GROQ_API_KEY") == "gsk_secret"


def test_switching_back_restores_that_key(env, monkeypatch):
    _switch(monkeypatch, "proxy")
    _switch(monkeypatch, "groq")
    assert LP.get(env.read_text(), "LLM_API_KEY") == "gsk_secret"


def test_pacing_follows_the_provider(env, monkeypatch):
    """Groq's 8000 TPM against ~7k-token turns needs slower pacing than the
    router does; carrying one provider's rate to another guarantees 429s."""
    _switch(monkeypatch, "proxy")
    proxy_rate = float(LP.get(env.read_text(), "LLM_CALLS_PER_MINUTE"))
    _switch(monkeypatch, "groq")
    groq_rate = float(LP.get(env.read_text(), "LLM_CALLS_PER_MINUTE"))
    # Groq's 8000 TPM against ~7k-token turns is the tighter cap of the two.
    assert groq_rate < proxy_rate, (groq_rate, proxy_rate)


def test_every_provider_is_internally_consistent():
    for name, cfg in LP.PROVIDERS.items():
        assert cfg["LLM_BASE_URL"].startswith("http"), name
        assert cfg["LLM_MODEL"], name
        assert "LLM_CALLS_PER_MINUTE" in cfg, name
