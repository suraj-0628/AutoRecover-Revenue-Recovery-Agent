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
    assert r["next_rung"] == "offer"


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
    assert 'action_val, sdk_res = "close_case", close_res' in FRONTEND
    assert "Case closed:" in FRONTEND


def test_the_prompt_says_not_to_end_by_falling_silent():
    i = GRAPH.index("call close_case. That is the ending")
    body = GRAPH[i - 400:i + 700]
    assert "indistinguishable from running out of ideas" in body
