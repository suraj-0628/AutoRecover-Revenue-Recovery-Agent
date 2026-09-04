"""Self-critique ran on every turn, cost a call, and hurt when it failed.

Three problems, found on pay_qssihjc5z (INR 12,495):

1. UNCONDITIONAL. `stopping_check -> self_critique` was a plain edge, so every
   run paid for a reflection LLM call — including a run whose only act was a
   page push and which is now simply waiting on the customer.

2. REDUNDANT AND DILUTING. `close_case` already stores the agent's considered
   lesson at the moment it declares the ending, and those read as rules ("For
   customer_cancelled failures, a silent page push is highly effective
   when..."). The critique's read as case files ("SELF-CRITIQUE — Case pay_x.
   Rung 1: ..."). Both land in the same memory namespace, so the narration
   diluted retrieval of the signal.

3. DANGEROUS ON FAILURE. When the critique's LLM call failed the node returned
   no message, and `_route_after_critique` then inspected the AGENT's last
   turn — routing its pending `send_recovery_notification` into a ToolNode that
   holds only `manage_memory`. The live result: "send_recovery_notification is
   not a valid tool, try one of [manage_memory]", and a customer who had
   correctly been decided a full-price rail switch got nothing at all.
"""
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from types import SimpleNamespace

import pytest

from recovery_agent.agent.graph import (SELF_CRITIQUE_PROMPT,
                                        _closed_deliberately,
                                        _route_after_critique,
                                        _route_after_stopping)
from recovery_agent.models import Case, CaseStatus, PaymentEvent


def case_at(status):
    c = Case(payment=PaymentEvent(payment_id="p", customer_id="a@b.com",
                                  amount=2499.0))
    c.status = status
    return c


def cfg(case):
    """A config carrying the runtime the graph nodes read their Case from."""
    return {"configurable": {"__pregel_runtime": SimpleNamespace(
        context=SimpleNamespace(case=case, guardrail_engine=None))}}


def closed_run():
    return [HumanMessage(content="recover"),
            AIMessage(content="", tool_calls=[
                {"name": "close_case", "args": {}, "id": "c1"}]),
            ToolMessage(content=json.dumps({"status": "closed",
                                            "outcome": "recovered"}),
                        tool_call_id="c1", name="close_case")]


def pushed_run():
    return [HumanMessage(content="recover"),
            AIMessage(content="", tool_calls=[
                {"name": "send_page_push", "args": {}, "id": "p1"}]),
            ToolMessage(content=json.dumps({"status": "delivered"}),
                        tool_call_id="p1", name="send_page_push")]


# ── it no longer fires on every run ─────────────────────────────────────────

def test_a_run_still_in_flight_is_not_critiqued():
    """A push went out and the customer has not answered. There is no lesson
    yet, and the call costs real money."""
    assert _route_after_stopping({"messages": pushed_run()},
                                 cfg(case_at(CaseStatus.AWAITING_CUSTOMER))) == "__end__"


def test_a_case_the_agent_closed_itself_is_not_critiqued():
    """close_case already stored the better lesson."""
    assert _route_after_stopping({"messages": closed_run()},
                                 cfg(case_at(CaseStatus.RECOVERED))) == "__end__"


def test_an_ending_nobody_recorded_IS_critiqued():
    """Hit the turn cap, ran out of moves, or fell silent — nothing else will
    write down why this case ended."""
    assert _route_after_stopping({"messages": pushed_run()},
                                 cfg(case_at(CaseStatus.STOPPED))) == "self_critique"


@pytest.mark.parametrize("status", [CaseStatus.RECOVERED, CaseStatus.ESCALATED,
                                    CaseStatus.STOPPED])
def test_every_terminal_state_without_a_close_is_worth_a_lesson(status):
    assert _route_after_stopping({"messages": pushed_run()},
                                 cfg(case_at(status))) == "self_critique"


def test_no_case_in_context_ends_quietly():
    assert _route_after_stopping({"messages": pushed_run()}, None) == "__end__"


def test_close_case_is_only_counted_for_THIS_run():
    """An earlier run's close must not silence this run's ending."""
    msgs = closed_run() + [HumanMessage(content="new signal")] + pushed_run()[1:]
    assert not _closed_deliberately(msgs)


# ── a failed critique can no longer hijack the agent's tool calls ───────────

def test_a_stale_agent_tool_call_is_never_fed_to_the_memory_node():
    """The live bug, in one assertion: the critique failed, left no message,
    and the agent's pending email was routed into a manage_memory-only node."""
    stranded = [HumanMessage(content="recover"),
                AIMessage(content="", tool_calls=[
                    {"name": "send_recovery_notification", "args": {}, "id": "n1"}])]
    assert _route_after_critique({"messages": stranded}) == "__end__"


def test_a_mixed_batch_is_not_routed_either():
    mixed = [AIMessage(content="", tool_calls=[
        {"name": "manage_memory", "args": {}, "id": "m1"},
        {"name": "escalate_to_human", "args": {}, "id": "e1"}])]
    assert _route_after_critique({"messages": mixed}) == "__end__"


def test_a_genuine_memory_write_still_runs():
    ok = [AIMessage(content="", tool_calls=[
        {"name": "manage_memory", "args": {"content": "lesson"}, "id": "m1"}])]
    assert _route_after_critique({"messages": ok}) == "critique_tools"


def test_a_critique_with_no_tool_call_ends():
    assert _route_after_critique({"messages": [AIMessage(content="nothing to add")]}) \
        == "__end__"


# ── it asks for a rule now, not a case file ─────────────────────────────────

def test_the_prompt_demands_a_generalisable_lesson():
    assert "not summarising the case" in SELF_CRITIQUE_PROMPT
    assert "For <kind of failure>" in SELF_CRITIQUE_PROMPT
    assert "60 words" in SELF_CRITIQUE_PROMPT


def test_the_prompt_shows_the_bad_shape_it_used_to_produce():
    """The old output really did start 'SELF-CRITIQUE — Case pay_...'."""
    assert "SELF-CRITIQUE" in SELF_CRITIQUE_PROMPT.split("Bad:")[1]


def test_storing_nothing_is_an_allowed_outcome():
    assert "store nothing" in SELF_CRITIQUE_PROMPT
