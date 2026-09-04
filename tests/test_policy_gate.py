"""The GuardrailEngine must sit on the live tool path, not beside it.

For months `validate_action` had zero callers: quiet hours, the frequency cap,
opt-out, the double-debit lock, the monetary cap and hard-decline protection
were constructed, displayed, tested — and never consulted before a tool ran.
These tests pin the policy_gate node that closes that hole, and the contract
that a refusal is something the agent can perceive and adapt to: a ToolMessage
with a reason and an alternative, a `refusals` entry on the durable record,
and a line in the verdict audit log.
"""
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from recovery_agent.agent.guardrails import GuardrailEngine, QuietHourGuardrail
from recovery_agent.agent.memory import CustomerMemoryStore
from recovery_agent.agent.policy_gate import policy_gate
from recovery_agent.agent.tools import RecoveryContext
from recovery_agent.models import Case, PaymentEvent
from recovery_agent.state_store import StateStore

PID = "pay_gate1"
EMAIL = "gate@test.com"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from recovery_agent import state_store
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.delenv("GUARDRAILS_ENFORCE", raising=False)
    monkeypatch.delenv("GUARDRAIL_QUIET_DISABLED", raising=False)
    state_store.StateStore.reset_instances()
    CustomerMemoryStore.reset_live()
    yield
    state_store.StateStore.reset_instances()
    CustomerMemoryStore.reset_live()


def _seed_record(**over):
    rec = {"payment_id": PID, "amount": 2499.0, "status": "recovering",
           "recovered_amount": 0, "customer": {"email": EMAIL},
           "customer_email": EMAIL, "ladder": {}}
    rec.update(over)
    s = StateStore()
    s.save_payment(PID, rec)
    s.flush()


def _case(failure_code: str = "51", amount: float = 2499.0) -> Case:
    return Case(payment=PaymentEvent(
        payment_id=PID, customer_id="c1", amount=amount,
        failure_code=failure_code,
    ))


def _engine(quiet: bool) -> GuardrailEngine:
    e = GuardrailEngine()
    # (0, 24): every hour satisfies "hour >= start"; (24, 0): no hour does.
    e.quiet_hours = QuietHourGuardrail(0, 24) if quiet else QuietHourGuardrail(24, 0)
    return e


def _invoke(tool_calls, engine, case=None):
    state = {"messages": [AIMessage(content="", tool_calls=tool_calls)]}
    ctx = RecoveryContext(case=case or _case(), guardrail_engine=engine)
    config = {"configurable": {"__pregel_runtime": SimpleNamespace(context=ctx)}}
    return policy_gate(state, config)


def _tc(name, args=None, id="tc1"):
    return {"name": name, "args": args or {}, "id": id, "type": "tool_call"}


def _refusal_for(result, tc_id):
    for m in result["messages"]:
        if isinstance(m, ToolMessage) and m.tool_call_id == tc_id:
            return json.loads(m.content)
    raise AssertionError(f"no ToolMessage answered {tc_id}")


# ── refusals ────────────────────────────────────────────────────────────

def test_voice_call_refused_in_quiet_hours():
    _seed_record()
    out = _invoke([_tc("initiate_voice_call", {"payment_id": PID})], _engine(quiet=True))
    refusal = _refusal_for(out, "tc1")
    assert refusal["status"] == "blocked"
    assert refusal["guardrail"] == "quiet_hours"
    assert "retry_in_hours" in refusal["guidance"]
    assert any(isinstance(m, SystemMessage) and "[Guardrail]" in m.content
               for m in out["messages"])


def test_refusal_lands_on_the_durable_record():
    _seed_record()
    _invoke([_tc("initiate_voice_call", {"payment_id": PID})], _engine(quiet=True))
    rec = StateStore().get_payment(PID)
    assert rec["refusals"] == {"initiate_voice_call: quiet_hours": 1}


def test_repeated_refusal_reaches_the_next_briefing():
    """The gate's refusals feed the same perception loop as every other
    refusal — twice refused, the next briefing says so out loud."""
    _seed_record()
    for _ in range(2):
        _invoke([_tc("initiate_voice_call", {"payment_id": PID})], _engine(quiet=True))
    from recovery_agent.agent.perception import ground_truth, as_briefing
    briefing = as_briefing(ground_truth(PID))
    assert "already been refused" in briefing
    assert "initiate_voice_call" in briefing


def test_frequency_cap_blocks_the_contact_past_the_ceiling():
    """Seeded from the CAP rather than a hardcoded 3.

    This asserted that the fourth contact was refused, which silently encoded
    a cap of 3. Raising the ceiling (it is operator-tunable now, and blocked a
    real bank-decline recovery at "4/3 contacts in 24h") turned a correct
    policy change into a red test about an unrelated number."""
    engine = _engine(quiet=False)
    cap = engine.frequency_cap.max_contacts
    _seed_record()
    store = CustomerMemoryStore.live()
    for _ in range(cap):
        store.record_contact(EMAIL, "email", payment_id=PID)
    out = _invoke([_tc("send_recovery_notification", {"payment_id": PID})], engine)
    refusal = _refusal_for(out, "tc1")
    assert refusal["guardrail"] == "frequency_cap"
    assert f"/{cap}" in refusal["reason"], "the refusal names the ceiling it enforces"


def test_hard_decline_blocks_scheduled_retry():
    _seed_record(failure_code="54")
    out = _invoke([_tc("retry_in_hours", {"payment_id": PID})],
                  _engine(quiet=False), case=_case(failure_code="54"))
    refusal = _refusal_for(out, "tc1")
    assert refusal["status"] == "blocked"
    assert refusal["guardrail"] == "hard_decline"
    assert "escalate_to_human" in refusal["guidance"]


def test_opted_out_customer_gets_no_link():
    _seed_record()
    profile = CustomerMemoryStore.live().get_or_create_profile(EMAIL)
    profile.opt_out = True
    CustomerMemoryStore.live().save_profile(profile)
    out = _invoke([_tc("generate_recovery_payment_link",
                       {"payment_id": PID, "amount": 2499.0})],
                  _engine(quiet=False))
    refusal = _refusal_for(out, "tc1")
    assert refusal["guardrail"] == "opt_out"


# ── pass-throughs ───────────────────────────────────────────────────────

def test_diagnostics_are_not_the_engines_business():
    """Even in quiet hours, reading state is always allowed."""
    _seed_record()
    out = _invoke([_tc("check_payment_status", {"payment_id": PID}),
                   _tc("search_memory", {"query": "x"}, id="tc2")],
                  _engine(quiet=True))
    assert out == {}


def test_clean_contact_passes_by_day():
    _seed_record()
    out = _invoke([_tc("send_recovery_notification", {"payment_id": PID})],
                  _engine(quiet=False))
    assert out == {}


def test_page_push_is_not_a_contact():
    """The push renders inside a page the customer already has open — quiet
    hours have nothing to say about it."""
    _seed_record()
    out = _invoke([_tc("send_page_push", {"payment_id": PID})], _engine(quiet=True))
    assert out == {}


def test_kill_switch_disables_enforcement(monkeypatch):
    _seed_record()
    monkeypatch.setenv("GUARDRAILS_ENFORCE", "0")
    out = _invoke([_tc("initiate_voice_call", {"payment_id": PID})], _engine(quiet=True))
    assert out == {}


# ── batch semantics ─────────────────────────────────────────────────────

def test_one_refusal_does_not_execute_the_rest_silently():
    _seed_record()
    out = _invoke([_tc("initiate_voice_call", {"payment_id": PID}),
                   _tc("check_payment_status", {"payment_id": PID}, id="tc2")],
                  _engine(quiet=True))
    blocked = _refusal_for(out, "tc1")
    bystander = _refusal_for(out, "tc2")
    assert blocked["status"] == "blocked"
    assert bystander["status"] == "not_executed"
    assert "on its own" in bystander["guidance"]


# ── audit trail ─────────────────────────────────────────────────────────

def test_every_evaluation_is_logged(tmp_path):
    _seed_record()
    _invoke([_tc("send_recovery_notification", {"payment_id": PID})],
            _engine(quiet=False))                                   # pass
    _invoke([_tc("initiate_voice_call", {"payment_id": PID})],
            _engine(quiet=True))                                    # blocked
    log = tmp_path / "audit_logs" / "guardrail_verdicts.jsonl"
    entries = [json.loads(l) for l in log.read_text().splitlines()]
    outcomes = {(e["tool"], e["outcome"]) for e in entries}
    assert ("send_recovery_notification", "pass") in outcomes
    assert ("initiate_voice_call", "blocked") in outcomes
    assert all(e["agent_version"] for e in entries)


# ── graph wiring ────────────────────────────────────────────────────────

def test_policy_gate_is_a_graph_node():
    from recovery_agent.agent.graph import build_graph
    nodes = set(build_graph().get_graph().nodes)
    assert "policy_gate" in nodes


def test_repetition_guard_routes_into_policy_gate_not_around_it():
    from recovery_agent.agent import graph as g
    state = {"messages": [AIMessage(content="", tool_calls=[_tc("send_page_push")])],
             "phase": ""}
    assert g._route_after_guard(state) == "policy_gate"


def test_policy_refusal_routes_back_to_agent():
    from recovery_agent.agent import graph as g
    refused = {"messages": [SystemMessage(content="[Guardrail] Refused by policy: x")]}
    clean = {"messages": [AIMessage(content="", tool_calls=[_tc("send_page_push")])]}
    assert g._route_after_policy(refused) == "agent"
    assert g._route_after_policy(clean) == "human_approval_gate"
