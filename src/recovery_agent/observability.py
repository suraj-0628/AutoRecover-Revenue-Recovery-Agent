"""One init point for tracing. Everything else was fragments fighting.

Before this module, three places touched OpenTelemetry and none of them knew
about the others: frontend.py built its own bare TracerProvider (no
OpenInference conventions, no project name), llm_client.py lazily registered
Phoenix only when the diagnosis path happened to run first, and OTel honours
exactly ONE global provider — so whichever init won the race, the loser's
spans degraded silently. No span carried a case identifier, so a case's runs
scattered across unrelated traces, and Phoenix's cost engine had nothing to
group by.

Now: every process calls init_observability() once at startup (idempotent,
cheap, safe with the collector down), LangChain auto-instrumentation captures
every LLM call with the token counts and model name Phoenix's cost table
keys on, and case_session() stamps `session.id = case:{payment_id}` onto
everything inside a run — so Phoenix rolls tokens AND rupees up per case,
per session, per project, on its own.

Deliberately NOT instrumented: the OpenAI SDK instrumentor. LangChain already
emits the LLM span for every ChatOpenAI call; instrumenting the underlying
client as well would double-count every token in Phoenix's cost rollup.
"""
from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from typing import Any, Iterator
from urllib.parse import urlparse

_state: dict[str, Any] = {"initialized": False, "enabled": False}

DEFAULT_COLLECTOR = "http://localhost:6006/v1/traces"


def collector_endpoint() -> str:
    return os.getenv("PHOENIX_COLLECTOR_ENDPOINT", DEFAULT_COLLECTOR)


def phoenix_base_url() -> str:
    """The Phoenix UI/API origin, derived from the collector endpoint."""
    parsed = urlparse(collector_endpoint())
    return f"{parsed.scheme}://{parsed.netloc}"


def project_name() -> str:
    return os.getenv("PHOENIX_PROJECT_NAME", "autorecover")


def _collector_reachable(timeout: float = 1.0) -> bool:
    parsed = urlparse(collector_endpoint())
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def init_observability(service: str = "recovery-agent") -> bool:
    """Register Phoenix tracing for this process. Call early, call anywhere —
    the first call decides, every later call is free.

    Returns True when spans will actually reach a collector. With the
    collector down or PHOENIX_DISABLED=1 this is a clean no-op: the agent must
    never fail to recover money because the observability stack is off.
    """
    if _state["initialized"]:
        return _state["enabled"]
    _state["initialized"] = True

    if os.getenv("PHOENIX_DISABLED", "").strip().lower() in ("1", "true", "yes"):
        print(f"[obs] tracing disabled by PHOENIX_DISABLED ({service})", flush=True)
        return False
    if not _collector_reachable():
        print(f"[obs] no collector at {collector_endpoint()} — tracing off "
              f"({service})", flush=True)
        return False

    try:
        from phoenix.otel import register
        provider = register(
            endpoint=collector_endpoint(),
            project_name=project_name(),
            # Batch export: never block an agent turn on a span flush.
            batch=True,
            set_global_tracer_provider=True,
            verbose=False,
        )
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument(tracer_provider=provider)
        try:
            # Only when llama-index is actually importable — the instrumentor
            # prints a dependency warning otherwise, on every process start.
            import importlib.util
            if importlib.util.find_spec("llama_index.core") is not None:
                from openinference.instrumentation.llama_index import (
                    LlamaIndexInstrumentor,
                )
                LlamaIndexInstrumentor().instrument(tracer_provider=provider)
        except Exception:
            pass  # the RAG path just goes untraced
        _state["enabled"] = True
        print(f"[obs] tracing → {collector_endpoint()} project="
              f"{project_name()!r} ({service})", flush=True)
        return True
    except Exception as exc:
        print(f"[obs] tracing unavailable ({service}): {exc}", flush=True)
        return False


def is_enabled() -> bool:
    return bool(_state["enabled"])


def get_tracer(name: str = "recovery-agent"):
    """The global tracer, after ensuring init ran. Never raises; with tracing
    off, OTel hands back a no-op tracer and every span becomes free."""
    init_observability()
    from opentelemetry import trace
    return trace.get_tracer(name)


def session_id_for(payment_id: str) -> str:
    return f"case:{payment_id}"


#: OpenInference span kinds we use. Anything the auto-instrumentation does not
#: create has to declare its own kind, or Phoenix files it under "unknown" —
#: which is what every hand-rolled span in this codebase was doing: the whole
#: safety layer, the case root, the recovery phases, all shapeless.
#:
#: The vocabulary maps onto this agent almost exactly. GUARDRAIL is what the
#: repetition guard, the policy gate, the approval gate, PII masking and the
#: stopping rules ARE. EVALUATOR is what self-critique is. Naming them so is
#: not decoration: Phoenix groups, colours and filters by kind, so a trace
#: becomes readable as "agent → guardrails → tools" instead of a flat list of
#: unknowns.
KIND_AGENT = "AGENT"
KIND_CHAIN = "CHAIN"
KIND_TOOL = "TOOL"
KIND_GUARDRAIL = "GUARDRAIL"
KIND_EVALUATOR = "EVALUATOR"

_SPAN_KIND_ATTR = "openinference.span.kind"


@contextmanager
def traced_span(name: str, kind: str = KIND_CHAIN, tracer: Any = None,
                attributes: dict | None = None) -> Iterator[Any]:
    """A manual span that declares what it is, and how it ended.

    Two things every hand-rolled span here was missing. Without the kind
    attribute Phoenix shows "unknown"; without a status it shows nothing at
    all, so a span that failed looked exactly like one that succeeded.

    The status is only set when the body left it UNSET — a node that
    deliberately recorded its own ERROR keeps it (OTel treats OK as final, so
    blindly stamping OK on the way out would erase real failures).
    """
    tracer = tracer or get_tracer()
    attrs = dict(attributes or {})
    attrs[_SPAN_KIND_ATTR] = kind
    with tracer.start_as_current_span(name, attributes=attrs) as span:
        try:
            yield span
        except Exception as exc:
            try:
                from opentelemetry.trace import StatusCode
                span.set_status(StatusCode.ERROR, str(exc)[:300])
            except Exception:
                pass
            raise
        try:
            from opentelemetry.trace import StatusCode
            current = getattr(getattr(span, "status", None), "status_code", None)
            if current in (None, StatusCode.UNSET):
                span.set_status(StatusCode.OK)
        except Exception:
            pass


@contextmanager
def case_session(payment_id: str, **metadata: Any) -> Iterator[None]:
    """Everything traced inside belongs to this case.

    Stamps OpenInference session/metadata context so every instrumented span
    (each LLM turn, each tool call) carries `session.id = case:{payment_id}` —
    which is exactly the key Phoenix aggregates cost by in its Sessions view.
    A case worked across five separate runs is still ONE session there.
    """
    if not payment_id or not _state["enabled"]:
        yield
        return
    try:
        from openinference.instrumentation import using_attributes
        clean = {k: v for k, v in metadata.items() if v not in (None, "")}
        with using_attributes(session_id=session_id_for(payment_id),
                              metadata=clean or None):
            yield
    except Exception:
        yield


def phoenix_project_info(timeout: float = 3.0) -> dict | None:
    """The autorecover project's id and UI URLs, for deep links from the ops
    view. Cached after the first success; None while Phoenix is down or the
    project has no traces yet. Never raises."""
    if _state.get("project_info"):
        return _state["project_info"]
    try:
        import json as _json
        import urllib.request
        body = _json.dumps({"query": """
            query { projects(first: 50) { edges { node { id name } } } }"""})
        req = urllib.request.Request(
            phoenix_base_url() + "/graphql", data=body.encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = _json.loads(r.read().decode())
        for edge in data["data"]["projects"]["edges"]:
            if edge["node"]["name"] == project_name():
                pid = edge["node"]["id"]
                info = {
                    "base_url": phoenix_base_url(),
                    "project_id": pid,
                    "project_url": f"{phoenix_base_url()}/projects/{pid}",
                    "sessions_url": f"{phoenix_base_url()}/projects/{pid}/sessions",
                }
                _state["project_info"] = info
                return info
    except Exception:
        pass
    return None


def reset_for_tests() -> None:
    _state["initialized"] = False
    _state["enabled"] = False
    _state.pop("project_info", None)
