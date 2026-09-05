"""Money in the bank must end the case even when the model forgets to.

Live (pay_gyx5fd5dv): the closing run's model had already stored its memory,
read "then stop" in the settled briefing, and ended with zero tool calls. The
case sat at status "recovered" with closed: null — no closure marker, no
episode — and nothing would ever run it again. Closing on verified money is
bookkeeping; the harness now does it whenever a run ends settled-but-unclosed,
and the briefing itself now names close_case instead of stopping short of it.
"""
from __future__ import annotations

import pytest

import recovery_agent.state_store as state_store
from recovery_agent.state_store import StateStore


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    StateStore.reset_instances()
    yield
    StateStore.reset_instances()


def _recovered_but_unclosed():
    s = StateStore()
    s.save_payment("p", {"payment_id": "p", "amount": 3798.0,
                         "status": "recovered", "recovered_amount": 3798.0,
                         "customer": {"email": "a@b.com"},
                         "customer_email": "a@b.com", "ladder": {}})
    s.flush()
    return s


# ── the harness backstop ────────────────────────────────────────────────

def test_the_harness_closes_what_the_model_left_open():
    from recovery_agent.frontend import _close_if_settled_and_unclosed
    s = _recovered_but_unclosed()
    assert _close_if_settled_and_unclosed("p") is True
    closed = s.get_payment("p")["closed"]
    assert closed["outcome"] == "recovered"
    assert closed["amount_recovered"] == 3798.0


def test_an_already_closed_case_is_left_alone():
    from recovery_agent.frontend import _close_if_settled_and_unclosed
    s = _recovered_but_unclosed()
    s.update_payment("p", closed={"outcome": "recovered", "at": "earlier"})
    assert _close_if_settled_and_unclosed("p") is False
    assert s.get_payment("p")["closed"]["at"] == "earlier"


def test_no_money_means_no_manufactured_closure():
    """close_case re-verifies the money; the backstop cannot close a case
    that has not actually been paid."""
    from recovery_agent.frontend import _close_if_settled_and_unclosed
    s = StateStore()
    s.save_payment("p", {"payment_id": "p", "amount": 3798.0,
                         "status": "recovering", "recovered_amount": 0,
                         "customer_email": "a@b.com", "ladder": {}})
    s.flush()
    assert _close_if_settled_and_unclosed("p") is False
    assert not s.get_payment("p").get("closed")


# ── the briefing names the tool it expects to be used ───────────────────

def test_the_settled_briefing_names_close_case():
    from recovery_agent.agent.perception import as_briefing
    text = as_briefing({"known": True, "settled": True, "owed": 3798.0,
                        "received": 3798.0, "outstanding": 0.0,
                        "captured_payment_id": "pay_x"})
    assert "close_case" in text
    assert "manage_memory" in text


# ── a fresh send is pending, not failed ─────────────────────────────────

def _open_facts(**over):
    facts = {"known": True, "settled": False, "owed": 3798.0,
             "received": 0.0, "outstanding": 3798.0,
             "actions_tried": ["link:card+upi+wallet:3798.00",
                               "notify:email:https://rzp.io/x"]}
    facts.update(over)
    return facts


def test_the_briefing_does_not_call_a_fresh_send_a_failure():
    from recovery_agent.agent.perception import as_briefing
    text = as_briefing(_open_facts(minutes_since_last_rung=0.5))
    assert "did not work" not in text
    assert "has not had time to work" in text
    assert "wait_for_customer" in text


def test_the_wait_nudge_expires_once_the_send_is_stale():
    from recovery_agent.agent.perception import as_briefing
    text = as_briefing(_open_facts(minutes_since_last_rung=45.0))
    assert "has not had time to work" not in text
    assert "a repeat is not a new attempt" in text


def test_a_settled_run_reads_only_the_newest_handoff():
    """The closing run is bookkeeping. Handed the whole recovery transcript,
    the model (live, pay_xtog5tut1) answered by quoting the briefing back —
    zero tool calls. The transcript is noise for the close-and-record job."""
    from langchain_core.messages import (AIMessage, HumanMessage,
                                         SystemMessage, ToolMessage)
    from recovery_agent.agent.graph import _assemble_llm_messages
    history = [
        HumanMessage(content="A payment has failed and needs recovery."),
        AIMessage(content="", tool_calls=[{"name": "wait_for_customer",
                                           "args": {}, "id": "t1"}]),
        ToolMessage(content="Turn ended. Do not call any more tools.",
                    tool_call_id="t1"),
        HumanMessage(content="RECOVERED. Close the case."),
    ]
    out = _assemble_llm_messages([SystemMessage(content="briefing")], history,
                                 settled=True)
    texts = [str(m.content) for m in out]
    assert any("RECOVERED. Close the case." in t for t in texts)
    assert not any("needs recovery" in t for t in texts)
    assert not any("Do not call any more tools" in t for t in texts)


def test_the_closing_runs_own_turns_survive_the_trim():
    """Turn 2 of a closing run must still see the memory it stored on turn 1."""
    from langchain_core.messages import (AIMessage, HumanMessage,
                                         SystemMessage, ToolMessage)
    from recovery_agent.agent.graph import _assemble_llm_messages
    history = [
        HumanMessage(content="A payment has failed and needs recovery."),
        HumanMessage(content="RECOVERED. Close the case."),
        AIMessage(content="", tool_calls=[{"name": "manage_memory",
                                           "args": {}, "id": "t2"}]),
        ToolMessage(content="created memory abc", tool_call_id="t2"),
    ]
    out = _assemble_llm_messages([SystemMessage(content="briefing")], history,
                                 settled=True)
    texts = [str(m.content) for m in out]
    assert any("created memory abc" in t for t in texts)
    assert not any("needs recovery" in t for t in texts)


def test_an_open_run_still_reads_its_full_history():
    from langchain_core.messages import HumanMessage, SystemMessage
    from recovery_agent.agent.graph import _assemble_llm_messages
    history = [HumanMessage(content="A payment has failed and needs recovery."),
               HumanMessage(content="the customer answered")]
    out = _assemble_llm_messages([SystemMessage(content="briefing")], history,
                                 settled=False)
    texts = [str(m.content) for m in out]
    assert any("needs recovery" in t for t in texts)


def test_ground_truth_reports_how_fresh_the_last_rung_is():
    from datetime import datetime, timedelta, timezone
    from recovery_agent.agent.perception import ground_truth
    s = StateStore()
    at = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    s.save_payment("p", {"payment_id": "p", "amount": 100.0,
                         "status": "recovering",
                         "customer": {"email": "a@b.com"},
                         "ladder": {"rail_switch": {"at": at}}})
    s.flush()
    age = ground_truth("p").get("minutes_since_last_rung")
    assert age is not None and 4.5 <= age <= 6.0
