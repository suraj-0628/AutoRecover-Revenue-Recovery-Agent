"""Ending a case must be something the agent does, not something it stops doing.

Stopping used to be the absence of a tool call — indistinguishable, from the
outside, from running out of ideas or hitting a turn limit. Nothing recorded
that the agent had *decided* the case was over, and the outcome had to be
inferred from whichever tools happened to run last (which is why a closing run
reported "Primary action: none").
"""
import json
from pathlib import Path

import pytest

from recovery_agent.agent.tools import TOOLS_BY_NAME
from recovery_agent.state_store import StateStore

GRAPH = (Path(__file__).resolve().parents[1] / "src" / "recovery_agent" / "agent"
         / "graph.py").read_text()
FRONTEND = (Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
            / "frontend.py").read_text()


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    from recovery_agent import state_store
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    state_store.StateStore.reset_instances()
    yield
    state_store.StateStore.reset_instances()


def _case(**over):
    rec = {"payment_id": "p", "amount": 1299.0, "status": "recovering",
           "recovered_amount": 0, "customer": {"email": "a@b.com"},
           "customer_email": "a@b.com", "ladder": {}}
    rec.update(over)
    s = StateStore()
    s.save_payment("p", rec)
    s.flush()


def _close(**kw):
    args = {"payment_id": "p", "outcome": "recovered",
            "what_happened": "x", "lesson": ""}
    args.update(kw)
    return json.loads(TOOLS_BY_NAME["close_case"].invoke(args))


# ── a claim of victory is checked against the money ─────────────────────

def test_it_refuses_to_close_as_recovered_with_no_payment():
    """The agent writes the summary; whether the payment happened is not its to
    assert."""
    _case()
    r = _close()
    assert r["status"] == "blocked"
    assert "no payment is recorded" in r["reason"]


def test_it_closes_when_the_money_really_is_in():
    _case(status="recovered", recovered_amount=1234.05)
    r = _close(what_happened="Paid after a 5% offer.")
    assert r["status"] == "closed"
    assert r["amount_recovered"] == 1234.05
    assert StateStore().get_payment("p")["closed"]["outcome"] == "recovered"


def test_a_discounted_recovery_closes_cleanly():
    _case(status="recovered", recovered_amount=1234.05)
    assert _close()["status"] == "closed"


# ── giving up is also gated ─────────────────────────────────────────────

def test_it_refuses_to_give_up_while_rungs_remain():
    _case(ladder={"page_push": {"at": "x"}})
    r = _close(outcome="unrecoverable", what_happened="giving up")
    assert r["status"] == "blocked"
    # This case carries no failure code, so it climbs the fail-closed ladder:
    # after the silent push comes a FULL-PRICE attempt on another rail, and a
    # discount only if that fails too. It used to say "offer" here because one
    # global ladder sent every failure to a discount as rung 2, whatever had
    # actually gone wrong.
    assert r["next_rung"] == "rail_switch"


def test_the_next_rung_depends_on_what_actually_broke():
    """One ladder for every failure was the generic-agent problem itself."""
    from recovery_agent.agent import ladder

    def nxt(**over):
        rec = {"payment_id": "p", "amount": 1299.0,
               "customer": {"email": "a@b.com", "contact": "9000000000"},
               "ladder": {}}
        rec.update(over)
        n = ladder.next_rung(rec)
        return n["rung"] if n else None

    # The account was short: a different DAY, not a different message.
    assert nxt(failure_code="insufficient_funds") == "silent_retry"
    # Our own plumbing dropped it: retry, never apologise with money.
    assert nxt(failure_code="gateway_timeout") == "silent_retry"
    # The bank refused the instrument: another rail at the same price.
    assert nxt(failure_code="bank_declined", ladder={"page_push": {"at": "x"}}) \
        == "rail_switch"
    # They simply left: price is the honest lever here, and only here.
    assert nxt(failure_code="customer_cancelled",
               ladder={"page_push": {"at": "x"}}) == "offer"


def test_a_transient_failure_has_no_discount_rung_at_all():
    """Discounting a gateway timeout pays the customer to forgive our outage."""
    from recovery_agent.agent import ladder
    rungs = [k for k, _ in ladder.rungs_for({"failure_code": "network_timeout"})]
    assert "offer" not in rungs
    assert rungs[0] == "silent_retry"


def test_a_pending_retry_means_the_case_is_waiting_not_exhausted():
    """C1, 2026-09-03: a funds case was handed to a human while retries were
    scheduled for the next day and three days out — and the closing note
    claimed both had failed when neither had run."""
    from datetime import datetime, timedelta, timezone
    from recovery_agent.agent import ladder

    soon = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    rec = {"payment_id": "p", "amount": 1299.0, "failure_code": "insufficient_funds",
           "customer": {}, "ladder": {"silent_retry": {"at": "x"},
                                      "page_push": {"at": "x"}},
           "scheduled_job": {"status": "scheduled", "target_timestamp": soon}}
    assert not ladder.exhausted(rec), "a retry on the clock is not exhaustion"

    rec["scheduled_job"]["status"] = "completed"
    assert ladder.exhausted(rec), "once it has run, the ladder is done"


def test_it_accepts_unrecoverable_once_nothing_is_left():
    _case(customer={}, customer_email="", ladder={"page_push": {"at": "x"}})
    assert _close(outcome="unrecoverable",
                  what_happened="no way to reach them")["status"] == "closed"


def test_an_unknown_outcome_is_rejected():
    _case(status="recovered", recovered_amount=100)
    assert _close(outcome="done")["status"] == "error"


# ── closing records the outcome AND the lesson in one act ───────────────

def test_the_lesson_is_stored_with_the_closure():
    """A lesson that needs a follow-up call is a lesson lost when the turn ends
    first."""
    _case(status="recovered", recovered_amount=1234.05)
    r = _close(lesson="On-page offers convert this customer.")
    assert r["lesson_stored"] and "not stored" not in str(r["lesson_stored"])


def test_closing_without_a_lesson_is_allowed():
    _case(status="recovered", recovered_amount=1234.05)
    assert _close(lesson="")["status"] == "closed"


# ── a closed case stays closed ──────────────────────────────────────────

def test_a_closed_case_is_not_reopened_by_a_later_signal():
    i = FRONTEND.index("def _handoff_to_agent(")
    body = FRONTEND[i:i + 2400]
    assert 'p.get("closed")' in body, (
        "a closure the agent declared is a decision, not a side effect of "
        "running out of tools; a late timer must not restart it"
    )


# ── it is a real ending, wired like one ─────────────────────────────────

def test_close_case_ends_the_turn():
    i = GRAPH.index("def _route_after_tools")
    body = GRAPH[i:i + 900]
    assert '"close_case"' in body


def test_a_settled_case_can_always_close_itself():
    i = GRAPH.index("A finished case gets bookkeeping tools ONLY")
    assert '"close_case"' in GRAPH[i:i + 2800]


def test_the_hud_names_the_ending_instead_of_reporting_none():
    from recovery_agent.frontend import _select_primary_action
    action, receipt, _ = _select_primary_action({
        "close_case": {"args": {}, "result": {"status": "closed",
                                              "outcome": "recovered"}},
    })
    assert action == "close_case"
    assert receipt["outcome"] == "recovered"
    assert "Case closed:" in FRONTEND


def test_the_prompt_says_not_to_end_by_falling_silent():
    i = GRAPH.index("call close_case. That is the ending")
    body = GRAPH[i - 400:i + 700]
    assert "indistinguishable from running out of ideas" in body


# ── escalation is an ending too ─────────────────────────────────────────

def test_it_refuses_to_close_as_escalated_with_no_ticket(monkeypatch):
    """Symmetry with "recovered": the agent may not claim a person has the case
    unless a person actually does."""
    _case(customer={}, customer_email="", ladder={"page_push": {"at": "x"}})
    monkeypatch.setattr("recovery_agent.escalation_queue.list_tickets",
                        lambda status=None, limit=200: [])
    r = _close(outcome="escalated", what_happened="handed over")
    assert r["status"] == "blocked"
    assert "not with a human" in r["reason"]


def test_it_closes_as_escalated_once_a_ticket_exists(monkeypatch):
    _case(customer={}, customer_email="", ladder={"page_push": {"at": "x"}})
    monkeypatch.setattr("recovery_agent.escalation_queue.list_tickets",
                        lambda status=None, limit=200: [{"payment_id": "p"}])
    r = _close(outcome="escalated", what_happened="Ladder exhausted; ticket raised.")
    assert r["status"] == "closed"
    assert StateStore().get_payment("p")["status"] == "escalated"


def test_a_queue_read_failure_never_blocks_a_closure(monkeypatch):
    """Bookkeeping must not be able to strand a case."""
    def boom(*a, **k):
        raise RuntimeError("queue unavailable")
    _case(customer={}, customer_email="", ladder={"page_push": {"at": "x"}})
    monkeypatch.setattr("recovery_agent.escalation_queue.list_tickets", boom)
    assert _close(outcome="escalated", what_happened="x")["status"] == "closed"


def test_escalating_tells_the_agent_the_case_is_not_finished_yet():
    """Filing the ticket used to be where the run stopped — the same implicit
    ending a successful recovery had, with the same consequence: nothing showed
    the agent had decided anything."""
    TOOLS = (Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
             / "agent" / "tools.py").read_text()
    i = TOOLS.index('"queued_for_human": True')
    body = TOOLS[i:i + 500]
    assert "close_case" in body
    assert "not finished until you" in body, (
        "the tool must say the case is not over merely because a ticket exists"
    )


def test_the_prompt_pairs_escalation_with_the_ending():
    i = GRAPH.index("6. escalate_to_human")
    body = GRAPH[i:i + 700]
    assert "close_case" in body
    assert "abandoned, not delegated" in body


def test_the_executed_tool_line_names_the_action_not_the_last_call():
    """A run whose agent closed the case reported "Agent executed:
    manage_memory" — the self-critique node's own reflection write, which the
    agent did not do. It sat directly above "Primary action: close_case" and
    contradicted it."""
    i = FRONTEND.index("_ACTION_TOOL = {")
    body = FRONTEND[i:i + 600]
    assert '"close_case": "close_case"' in body
    assert '"send_notification": "send_recovery_notification"' in body


def test_the_critique_node_cannot_ask_for_tools_it_cannot_run():
    """It reflects and stores a lesson; critique_tools can run nothing else. It
    is bound to manage_memory alone and still asked for check_payment_status —
    the conversation it reads is full of other tool names. The ToolNode then
    answered "check_payment_status is not a valid tool", which reads in the
    trace as the agent being refused a perfectly sensible call."""
    i = GRAPH.index("def self_critique")
    body = GRAPH[i:i + 3200]
    assert 'tc.get("name") == "manage_memory"' in body
    assert "model_copy" in body, (
        "drop the call from the AIMessage rather than answering it with an "
        "error, or an orphan tool_use is left for the next turn"
    )
