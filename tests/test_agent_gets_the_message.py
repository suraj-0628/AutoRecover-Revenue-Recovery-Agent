"""Three ways the agent was being made to look dumber than it is.

Rated 6.5/10 on its own judgment, and each of these was a case where the
scaffolding — not the model — produced the bad behaviour:

1. A refusal arrived only as a tool RESULT, in the same channel as ordinary
   data, and the model read it as data. Live (pay_l477mhjef) the link tool
   said "the customer is mid-payment; call wait_for_customer and let them
   finish"; the very next turn re-proposed the identical call.
2. The turn cap stopped a run that had already DECIDED. Live (pay_qssihjc5z)
   the agent worked out the right move — a full-price rail switch by email —
   and hit the cap with that call pending. The one correct decision of the run
   was discarded and the customer got nothing.
3. `discover_recovery_rail` accepted `customer_id` and never used it, falling
   back to a fixed list — so it recommended `card_gateway` regardless of who
   the customer was, while `get_customer_payment_history` already returned
   their real per-rail success rates. A turn spent on discovery theatre.
"""
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from recovery_agent.agent.graph import should_continue, MAX_TURNS_PER_RUN as CAP
from recovery_agent.agent.governance import get_allowed_tools
from recovery_agent.agent.tools import TOOLS_BY_NAME


def ai(*names):
    return AIMessage(content="", tool_calls=[
        {"name": n, "args": {}, "id": f"c{i}"} for i, n in enumerate(names)])


def run_of(turns, *pending):
    """A run that has already taken `turns` tool-calling turns."""
    msgs = [HumanMessage(content="recover")]
    for i in range(turns - 1):
        msgs.append(ai("check_payment_status"))
        msgs.append(ToolMessage(content="{}", tool_call_id="c0",
                                name="check_payment_status"))
    msgs.append(ai(*pending) if pending else AIMessage(content="done"))
    return {"messages": msgs, "tool_call_history": []}


# ── 2. a decided action survives the turn cap ───────────────────────────────

def test_the_cap_lets_a_decided_action_finish():
    """The cap exists to stop a runaway loop, not to discard a conclusion."""
    out = should_continue(run_of(CAP, "send_recovery_notification"))
    assert out == "tool_repetition_guard", \
        "the agent had decided to send; the cap must not eat it"


def test_every_finishing_action_gets_the_grace():
    for tool in ("send_recovery_notification", "show_page_offer", "retry_in_hours",
                 "escalate_to_human", "close_case", "wait_for_customer"):
        assert should_continue(run_of(CAP, tool)) == "tool_repetition_guard", tool


def test_more_exploring_at_the_cap_still_stops():
    """A diagnostic call is not a conclusion — it is the loop the cap is for."""
    assert should_continue(run_of(CAP, "check_payment_status")) == "stopping_check"
    assert should_continue(run_of(CAP, "get_customer_payment_history")) == "stopping_check"


def test_the_grace_is_granted_exactly_once():
    """Otherwise the escape hatch becomes the loop."""
    assert should_continue(run_of(CAP + 1, "send_recovery_notification")) == "stopping_check"


def test_a_mixed_batch_does_not_buy_extra_turns():
    assert should_continue(
        run_of(CAP, "send_recovery_notification", "check_payment_status")) \
        == "stopping_check"


def test_under_the_cap_nothing_changes():
    assert should_continue(run_of(3, "send_recovery_notification")) \
        == "tool_repetition_guard"


# ── 1. a refusal reaches the instruction channel ────────────────────────────

def _mask(msgs):
    from recovery_agent.agent.graph import mask_tool_outputs_node
    return mask_tool_outputs_node({"messages": msgs}, None)["messages"]


def test_a_refusal_is_restated_as_an_instruction():
    """Same words, a channel the model cannot treat as payload."""
    blocked = json.dumps({
        "status": "blocked",
        "reason": "the customer has just acted and is in the middle of paying",
        "guidance": "Call wait_for_customer and let them finish."})
    out = _mask([HumanMessage(content="go"),
                 ai("generate_recovery_payment_link"),
                 ToolMessage(content=blocked, tool_call_id="c0",
                             name="generate_recovery_payment_link")])
    directive = [m for m in out if isinstance(m, SystemMessage)]
    assert directive, "a refusal must also arrive as an instruction"
    body = directive[-1].content
    assert "generate_recovery_payment_link" in body and "REFUSED" in body
    assert "wait_for_customer" in body, "it must name the move to make instead"
    assert "Do not re-propose a refused call" in body


def test_an_ordinary_result_produces_no_directive():
    ok = json.dumps({"status": "ok", "link_url": "https://rzp.io/x"})
    out = _mask([HumanMessage(content="go"),
                 ai("generate_recovery_payment_link"),
                 ToolMessage(content=ok, tool_call_id="c0",
                             name="generate_recovery_payment_link")])
    assert not [m for m in out if isinstance(m, SystemMessage)
                and "[Guardrail]" in m.content]


def test_a_refusal_with_no_guidance_is_not_amplified():
    """Repeating "no" louder teaches nothing."""
    bare = json.dumps({"status": "blocked", "reason": "no"})
    out = _mask([HumanMessage(content="go"), ai("show_page_offer"),
                 ToolMessage(content=bare, tool_call_id="c0",
                             name="show_page_offer")])
    assert not [m for m in out if isinstance(m, SystemMessage)
                and "[Guardrail]" in m.content]


# ── 3. the decorative tool is gone ──────────────────────────────────────────

def test_rail_discovery_theatre_is_not_offered_to_the_agent():
    assert "discover_recovery_rail" not in TOOLS_BY_NAME
    for tier in ("silent", "active", "escalated"):
        assert "discover_recovery_rail" not in get_allowed_tools(tier)


def test_the_real_source_of_rail_advice_remains():
    """It returns the customer's OWN per-rail success rates, which is what the
    discovery tool only pretended to do."""
    assert "get_customer_payment_history" in TOOLS_BY_NAME
    assert "get_customer_payment_history" in get_allowed_tools("silent")


def test_the_prompt_no_longer_advertises_it():
    from recovery_agent.agent.graph import SYSTEM_PROMPT
    assert "discover_recovery_rail" not in SYSTEM_PROMPT
