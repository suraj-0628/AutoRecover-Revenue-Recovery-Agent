"""The agent needs a way to say "I am done for now".

It invented one three times in a single trace — `wait_for_page_push_response`,
`wait_for_payment_recovery`, `monitor_push_response` — because ending a turn with
plain text is, from the model's side, indistinguishable from abandoning the case.
Two rounds of prompt instruction did not stop it. It was reaching for a verb the
tool surface did not have.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from recovery_agent.agent.governance import get_allowed_tools
from recovery_agent.agent.graph import SYSTEM_PROMPT, _route_after_tools
from recovery_agent.agent.tools import TOOLS_BY_NAME


def test_the_verb_exists_now():
    assert "wait_for_customer" in TOOLS_BY_NAME


def test_it_is_available_from_the_very_first_turn():
    """The silent tier is where the model first wanted it."""
    assert "wait_for_customer" in get_allowed_tools("silent")


def test_calling_it_records_what_is_being_waited_on():
    out = json.loads(TOOLS_BY_NAME["wait_for_customer"].invoke({
        "payment_id": "pay_x", "waiting_for": "customer to act on the in-page offer",
        "expected_within_minutes": 5,
    }))
    assert out["status"] == "ok"
    assert "in-page offer" in out["waiting_for"]
    assert "Turn ended" in out["note"]


def test_calling_it_actually_ends_the_turn():
    """If it did not, the model would learn the verb means nothing."""
    msgs = [
        HumanMessage(content="a payment failed"),
        AIMessage(content="", tool_calls=[{"name": "wait_for_customer", "args": {},
                                           "id": "w1", "type": "tool_call"}]),
        ToolMessage(content='{"status":"ok"}', tool_call_id="w1",
                    name="wait_for_customer"),
    ]
    assert _route_after_tools({"messages": msgs}) == "stopping_check"


def test_any_other_tool_still_loops_back_to_think_again():
    msgs = [
        AIMessage(content="", tool_calls=[{"name": "diagnose_payment_failure", "args": {},
                                           "id": "d1", "type": "tool_call"}]),
        ToolMessage(content='{"root_cause":"user_dropoff"}', tool_call_id="d1",
                    name="diagnose_payment_failure"),
    ]
    assert _route_after_tools({"messages": msgs}) == "agent"


def test_a_batch_ending_in_another_tool_keeps_going():
    """Only the last result decides; a wait earlier in a batch must not end it."""
    msgs = [
        AIMessage(content="", tool_calls=[
            {"name": "wait_for_customer", "args": {}, "id": "w1", "type": "tool_call"},
            {"name": "send_page_push", "args": {}, "id": "p1", "type": "tool_call"}]),
        ToolMessage(content='{"status":"ok"}', tool_call_id="w1", name="wait_for_customer"),
        ToolMessage(content='{"status":"delivered"}', tool_call_id="p1", name="send_page_push"),
    ]
    assert _route_after_tools({"messages": msgs}) == "agent"


def test_the_prompt_names_it_and_forbids_inventing_one():
    flat = " ".join(SYSTEM_PROMPT.split())      # the prompt is hard-wrapped
    assert "wait_for_customer" in flat
    assert "Never invent a tool to wait" in flat


def test_the_invented_names_are_not_real_tools():
    """Pinned so nobody 'fixes' this by adding the hallucinated names as aliases."""
    for invented in ("wait_for_page_push_response", "wait_for_payment_recovery",
                     "monitor_push_response"):
        assert invented not in TOOLS_BY_NAME
