"""One tracing init point, or no tracing at all.

Before observability.py, frontend.py and llm_client.py each registered their
own tracer provider and raced for OTel's single global slot — whichever lost,
its spans silently degraded, and nothing carried a case id for Phoenix to
group costs by. These tests pin the contract: init is idempotent and safe
with the collector down, every entry path delegates to it, case_session
stamps the session id Phoenix aggregates by, and the price-seeding and
backfill tooling's pure parts do their arithmetic.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from recovery_agent import observability as obs

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = (ROOT / "src" / "recovery_agent" / "frontend.py").read_text()
LLM_CLIENT = (ROOT / "src" / "recovery_agent" / "agent" / "llm_client.py").read_text()
GRAPH = (ROOT / "src" / "recovery_agent" / "agent" / "graph.py").read_text()
START_SH = (ROOT / "start.sh").read_text()


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    obs.reset_for_tests()
    monkeypatch.delenv("PHOENIX_DISABLED", raising=False)
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    yield
    obs.reset_for_tests()


# ── init behaviour ──────────────────────────────────────────────────────

def test_init_is_a_clean_noop_when_the_collector_is_down(monkeypatch):
    """The agent must never fail to recover money because the observability
    stack is off. A dead collector means disabled, not an exception."""
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT",
                       "http://localhost:1/v1/traces")   # nothing listens here
    assert obs.init_observability("test") is False
    assert obs.is_enabled() is False


def test_init_runs_once_and_later_calls_are_free(monkeypatch):
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT",
                       "http://localhost:1/v1/traces")
    calls = {"n": 0}
    real = obs._collector_reachable

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)
    monkeypatch.setattr(obs, "_collector_reachable", counting)
    obs.init_observability("a")
    obs.init_observability("b")
    obs.init_observability("c")
    assert calls["n"] == 1


def test_the_kill_switch_wins(monkeypatch):
    monkeypatch.setenv("PHOENIX_DISABLED", "1")
    assert obs.init_observability("test") is False


def test_get_tracer_never_raises_even_disabled(monkeypatch):
    monkeypatch.setenv("PHOENIX_DISABLED", "1")
    tracer = obs.get_tracer()
    with tracer.start_as_current_span("free-span"):
        pass  # no-op tracer: spans cost nothing and crash nothing


# ── the case session ────────────────────────────────────────────────────

def test_case_session_is_transparent_when_tracing_is_off():
    ran = []
    with obs.case_session("pay_x", failure_code="51"):
        ran.append(True)
    assert ran == [True]


def test_case_session_stamps_openinference_context_when_enabled(monkeypatch):
    """Inside the block, the OpenInference context carries session.id —
    which is exactly what the LangChain instrumentor copies onto every LLM
    span, and what Phoenix rolls cost up by."""
    obs._state["initialized"] = True
    obs._state["enabled"] = True
    from openinference.instrumentation import get_attributes_from_context
    with obs.case_session("pay_ctx", failure_code="51", amount=100):
        attrs = dict(get_attributes_from_context())
    assert attrs.get("session.id") == "case:pay_ctx"
    assert attrs.get("metadata.failure_code") == "51" or "metadata" in str(attrs)
    outside = dict(get_attributes_from_context())
    assert outside.get("session.id") is None


def test_session_id_shape_matches_the_thread_id_convention():
    assert obs.session_id_for("pay_1") == "case:pay_1"


# ── every entry path goes through the one init ──────────────────────────

def test_frontend_delegates_instead_of_building_its_own_provider():
    assert "TracerProvider()" not in FRONTEND, (
        "frontend must not construct its own provider — that is the race "
        "this module exists to end")
    assert "from recovery_agent.observability import get_tracer" in FRONTEND


def test_frontend_wraps_the_run_in_a_case_session():
    assert "case_session(payment_id" in FRONTEND
    assert 'parent_attrs["session.id"] = session_id_for(payment_id)' in FRONTEND


def test_llm_client_delegates():
    assert "phoenix.otel" not in LLM_CLIENT
    assert "init_observability" in LLM_CLIENT


def test_graph_tracer_initialises_observability():
    assert "init_observability(\"agent-graph\")" in GRAPH


def test_phoenix_data_survives_restarts():
    assert "PHOENIX_WORKING_DIR" in START_SH


# ── span kinds: every manual span says what it is ───────────────────────

def _recording_span(name="s", kind=obs.KIND_CHAIN, body=None):
    """Run traced_span against a real in-memory SDK tracer and return the
    finished span, so the attributes and status are the real ones."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    try:
        with obs.traced_span(name, kind=kind, tracer=tracer) as span:
            if body:
                body(span)
    except ValueError:
        pass
    return exporter.get_finished_spans()[0]


def test_a_manual_span_declares_its_openinference_kind():
    """Without this attribute Phoenix files every hand-rolled span under
    'unknown' — which is what the whole safety layer looked like."""
    span = _recording_span("stopping_check", kind=obs.KIND_GUARDRAIL)
    assert span.attributes["openinference.span.kind"] == "GUARDRAIL"


def test_a_clean_span_ends_OK_not_unset():
    from opentelemetry.trace import StatusCode
    assert _recording_span().status.status_code == StatusCode.OK


def test_a_raising_body_marks_the_span_errored():
    from opentelemetry.trace import StatusCode

    def boom(_):
        raise ValueError("kaboom")
    span = _recording_span(body=boom)
    assert span.status.status_code == StatusCode.ERROR
    assert "kaboom" in (span.status.description or "")


def test_a_status_the_body_set_is_never_overwritten():
    """A node that deliberately recorded ERROR keeps it — stamping OK on the
    way out would erase real failures."""
    from opentelemetry.trace import Status, StatusCode

    def marks_error(span):
        span.set_status(Status(StatusCode.ERROR, "the model refused"))
    span = _recording_span(body=marks_error)
    assert span.status.status_code == StatusCode.ERROR


def test_the_safety_layer_is_traced_as_guardrails_and_evaluators():
    """The kinds are not decoration: Phoenix groups and filters by them, so
    the guards read as guards instead of a flat list of unknowns."""
    assert 'kind=KIND_GUARDRAIL' in GRAPH  # repetition guard, gates, stopping
    assert GRAPH.count("kind=KIND_GUARDRAIL") == 4
    assert 'traced_span("self_critique", kind=KIND_EVALUATOR' in GRAPH
    assert 'kind=KIND_AGENT' in FRONTEND   # the case root is the agent itself


def test_no_raw_span_creation_survives_outside_the_helper():
    for source, label in ((FRONTEND, "frontend"), (GRAPH, "graph")):
        assert "start_as_current_span" not in source, (
            f"{label} still creates a raw span — it would report as 'unknown'")


# ── price seeder (pure parts) ───────────────────────────────────────────

def test_seeder_skips_models_phoenix_already_prices():
    from recovery_agent.scripts.seed_phoenix_model_costs import already_covered
    existing = [{"name": "gemini-2.5-flash",
                 "namePattern": r".*gemini-2\.5-flash.*"}]
    dupe = {"name": "mine", "name_pattern": "x",
            "probe": "antigravity/gemini-2.5-flash"}
    fresh = {"name": "gemma-4 (proxy)", "name_pattern": "x",
             "probe": "gemma-4-31b-it"}
    assert already_covered(dupe, existing)
    assert not already_covered(fresh, existing)


def test_seeder_survives_a_broken_existing_pattern():
    from recovery_agent.scripts.seed_phoenix_model_costs import already_covered
    existing = [{"name": "bad", "namePattern": "("}]
    entry = {"name": "n", "name_pattern": "x", "probe": "m"}
    assert already_covered(entry, existing) == ""


# ── backfill (pure parts) ───────────────────────────────────────────────

def _ai(tokens_in=0, tokens_out=0, model="gemma"):
    from langchain_core.messages import AIMessage
    m = AIMessage(content="x")
    m.usage_metadata = {"input_tokens": tokens_in, "output_tokens": tokens_out,
                        "total_tokens": tokens_in + tokens_out}
    m.response_metadata = {"model_name": model}
    return m


def test_backfill_reads_history_from_checkpointed_messages():
    from langchain_core.messages import HumanMessage, ToolMessage
    from recovery_agent.scripts.backfill_llm_usage import usage_from_messages
    msgs = [HumanMessage(content="case"),
            _ai(1000, 50, "gemma-4-31b-it"),
            ToolMessage(content="r", tool_call_id="t1"),
            _ai(2000, 100, "gemini-3.1-flash-lite")]
    u = usage_from_messages(msgs)
    assert u == {"calls": 2, "input_tokens": 3000, "output_tokens": 150,
                 "by_model": {"gemma-4-31b-it": 1, "gemini-3.1-flash-lite": 1}}


def test_a_response_without_usage_still_counts_as_a_call():
    from langchain_core.messages import AIMessage
    from recovery_agent.scripts.backfill_llm_usage import usage_from_messages
    u = usage_from_messages([AIMessage(content="x")])
    assert u["calls"] == 1 and u["input_tokens"] == 0


def test_backfill_never_shrinks_a_live_captured_record():
    from recovery_agent.scripts.backfill_llm_usage import should_backfill
    derived = {"calls": 3, "input_tokens": 900, "output_tokens": 40}
    assert should_backfill(None, derived)
    assert should_backfill({"calls": 2, "input_tokens": 500}, derived)
    assert not should_backfill({"calls": 3, "input_tokens": 900}, derived)
    assert not should_backfill({"calls": 9, "input_tokens": 9000}, derived)
    assert not should_backfill({}, {"calls": 0, "input_tokens": 0,
                                    "output_tokens": 0})
