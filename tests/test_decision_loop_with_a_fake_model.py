"""Drive the REAL graph with a scripted model.

Every other test in this suite exercises the scaffolding around the agent —
tools, ladders, guards — or greps the source. Nothing ran the decision loop
itself, so a change to the graph's wiring or the prompt could break the agent
in production and the suite would stay green. 810 tests and none of them could
tell you the loop still turns.

These drive `build_graph()` end to end with a fake chat model that returns
scripted tool calls, so the assertions are about what the GRAPH does with a
decision: which node runs next, whether a refusal comes back as something the
model can read, whether a decided action survives the turn cap, and whether a
settled case can still be acted on.

No network, no LLM, no Razorpay: the fake model replaces the only thing that
would have made a call.
"""
import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage

import recovery_agent.state_store as state_store
from recovery_agent.agent import graph as G
from recovery_agent.agent.tools import RecoveryContext
from recovery_agent.models import Case, CaseStatus, PaymentEvent
from recovery_agent.state_store import StateStore

PID = "pay_loop1"


class ScriptedModel:
    """Returns pre-written turns, then falls silent — which is how a real run
    ends. Records what it was bound to, so a test can assert on the tools the
    agent was actually offered."""

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.calls = 0
        self.bound_tools: list[str] = []

    def bind_tools(self, tools):
        self.bound_tools = [getattr(t, "name", str(t)) for t in (tools or [])]
        return self

    def invoke(self, messages, **kw):
        self.seen = messages
        if not self.script:
            return AIMessage(content="Nothing further this turn.")
        self.calls += 1
        nxt = self.script.pop(0)
        if isinstance(nxt, AIMessage):
            return nxt
        return AIMessage(content="", tool_calls=[
            {"name": n, "args": a, "id": f"t{self.calls}_{i}"}
            for i, (n, a) in enumerate(nxt)])


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    StateStore.reset_instances()
    s = StateStore()
    s.save_payment(PID, {"payment_id": PID, "amount": 2499.0,
                         "status": "recovering", "failure_code": "customer_cancelled",
                         "failure_reason": "Payment cancelled by customer",
                         "customer": {"email": "a@b.com", "contact": "9000000000"},
                         "customer_email": "a@b.com", "ladder": {}, "trail": []})
    s.flush()
    yield s
    StateStore.reset_instances()


def run(monkeypatch, script, *, status=CaseStatus.OPEN, pid=PID):
    """Run the real compiled graph with a scripted model."""
    model = ScriptedModel(script)
    monkeypatch.setattr(G, "_build_model_for_task", lambda task, tools=None:
                        model.bind_tools(tools))
    monkeypatch.setattr(G, "_build_model", lambda tools=None, model_name=None:
                        model.bind_tools(tools))
    case = Case(payment=PaymentEvent(payment_id=pid, customer_id="a@b.com",
                                     amount=2499.0,
                                     failure_code="customer_cancelled"))
    case.status = status
    graph = G.build_graph()
    final = graph.invoke(
        G.build_initial_state(case),
        config={"configurable": {"thread_id": f"t:{pid}:{id(model)}",
                                 "payment_id": pid, "customer_id": "a_b_com"}},
        context=RecoveryContext(case=case, guardrail_engine=None))
    return model, final, case


def tool_results(final, name):
    out = []
    for m in final["messages"]:
        if getattr(m, "name", None) == name and hasattr(m, "tool_call_id"):
            try:
                out.append(json.loads(str(m.content)[str(m.content).index("{"):]))
            except (ValueError, TypeError):
                out.append({"raw": str(m.content)})
    return out


# ── the loop actually turns ─────────────────────────────────────────────────

def test_a_tool_call_executes_and_comes_back_to_the_model(env, monkeypatch):
    model, final, _ = run(monkeypatch, [[("check_payment_status", {"payment_id": PID})]])
    assert model.calls >= 1
    results = tool_results(final, "check_payment_status")
    assert results and "briefing" in results[0], \
        "the tool ran and its result returned to the loop"


def test_the_model_is_offered_the_real_toolset(env, monkeypatch):
    model, _, _ = run(monkeypatch, [[("check_payment_status", {"payment_id": PID})]])
    assert "send_recovery_notification" in model.bound_tools
    assert "close_case" in model.bound_tools
    assert "discover_recovery_rail" not in model.bound_tools, \
        "the decorative discovery tool must not be offered"


def test_perception_is_in_front_of_the_model_every_turn(env, monkeypatch):
    model, _, _ = run(monkeypatch, [[("check_payment_status", {"payment_id": PID})]])
    briefings = [m for m in model.seen
                 if "WHAT IS TRUE RIGHT NOW" in str(getattr(m, "content", ""))]
    assert briefings, "the agent must be shown the ground truth, not just told a story"


# ── a refusal is something the model can act on ─────────────────────────────

def test_a_refused_call_returns_a_reason_and_an_instruction(env, monkeypatch):
    """A settled case refuses contact. The model must get both the refusal AND
    a directive it cannot read as payload."""
    env.update_payment(PID, status="recovered", recovered_amount=2499.0,
                       recovered_payment_id="pay_real")
    env.flush()
    _, final, _ = run(monkeypatch, [
        [("send_recovery_notification",
          {"payment_id": PID, "customer_email": "a@b.com", "customer_phone": "",
           "message": "pay now", "payment_link": "https://rzp.io/x"})]])
    blocked = [r for r in tool_results(final, "send_recovery_notification")
               if r.get("status") == "blocked"]
    assert blocked, "a settled case must refuse a further contact"
    assert "already" in blocked[0]["reason"]
    directives = [m for m in final["messages"]
                  if type(m).__name__ == "SystemMessage"
                  and "[Guardrail]" in str(m.content)]
    assert directives, "the refusal must also reach the instruction channel"


def test_the_ladder_refuses_a_premature_escalation(env, monkeypatch):
    _, final, _ = run(monkeypatch, [
        [("escalate_to_human", {"payment_id": PID, "reason": "give up"})]])
    results = tool_results(final, "escalate_to_human")
    assert results and results[0]["status"] == "blocked"
    assert results[0]["next_rung"], "a refusal must name the move to make instead"


# ── the money invariants hold inside the real loop ──────────────────────────

def test_a_discount_below_the_floor_is_refused_in_the_loop(env, monkeypatch):
    _, final, _ = run(monkeypatch, [
        [("generate_recovery_payment_link",
          {"payment_id": PID, "amount": 100.0, "customer_email": "a@b.com"})]])
    r = tool_results(final, "generate_recovery_payment_link")
    assert r and r[0]["status"] == "error"
    assert "below the policy floor" in r[0]["message"]


def test_overcharging_is_refused_in_the_loop(env, monkeypatch):
    _, final, _ = run(monkeypatch, [
        [("generate_recovery_payment_link",
          {"payment_id": PID, "amount": 249900.0, "customer_email": "a@b.com"})]])
    r = tool_results(final, "generate_recovery_payment_link")
    assert r and r[0]["status"] == "error"
    assert "MORE than" in r[0]["message"], "the 100x paise bug must stay dead"


# ── endings ─────────────────────────────────────────────────────────────────

def test_closing_without_the_money_is_refused(env, monkeypatch):
    _, final, _ = run(monkeypatch, [
        [("close_case", {"payment_id": PID, "outcome": "recovered",
                         "what_happened": "claiming a win"})]])
    r = tool_results(final, "close_case")
    assert r and r[0]["status"] == "blocked"
    assert "no payment is recorded" in r[0]["reason"]


def test_waiting_ends_the_turn_rather_than_looping(env, monkeypatch):
    model, final, _ = run(monkeypatch, [
        [("wait_for_customer", {"payment_id": PID, "waiting_for": "the customer",
                                "expected_within_minutes": 15})],
        [("send_page_push", {"payment_id": PID, "headline": "h", "body": "b"})]])
    assert model.calls == 1, "wait_for_customer ends the turn; it must not think again"
    assert tool_results(final, "wait_for_customer")[0]["wake_at"], \
        "the wait must be registered, not merely noted"


def test_a_settled_case_cannot_be_contacted_again(env, monkeypatch):
    env.update_payment(PID, status="recovered", recovered_amount=2499.0)
    env.flush()
    model, _, _ = run(monkeypatch, [[("close_case", {
        "payment_id": PID, "outcome": "recovered", "what_happened": "done"})]],
        status=CaseStatus.RECOVERED)
    assert "send_recovery_notification" not in model.bound_tools, \
        "a settled case must not even be offered a way to contact the customer"
    assert "close_case" in model.bound_tools, "but it must still be closable"
