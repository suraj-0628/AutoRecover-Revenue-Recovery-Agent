"""pay_lohj3x6gt, INR 34,990, 2026-09-04 — recovered and left open.

The customer paid the recovery link fifteen seconds after the offer email. The
watcher caught it, the record shows recovered_amount 33,240.50, and the next
turn ran with settled=True, the SETTLED_PROMPT ("store the lesson, then
close_case") and only four tools bound.

It called NOTHING. Turn 8 of the decision log: chose=NOTHING.

The reason was in its own history. `wait_for_customer` ends its result with
"Do not call any more tools." — true for the turn it was issued in, false for
every turn afterwards, and the most recent concrete instruction in context. The
model obeyed the expired order over the live prompt.

SETTLED_PROMPT already neutralises stale "FOLLOW-UP" framing by name; this is
the same defect one layer down.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from recovery_agent.agent.graph import (_EXPIRED_TURN_ORDER,
                                        _defuse_expired_instructions)


def _wait_result() -> ToolMessage:
    return ToolMessage(
        content=('{"status": "ok", "payment_id": "pay_lohj3x6gt", '
                 '"note": "Turn ended. You will be started again when this '
                 f'resolves. {_EXPIRED_TURN_ORDER}"}}'),
        tool_call_id="tc_wait", name="wait_for_customer")


def test_the_expired_order_does_not_survive_into_the_next_turn():
    """The live failure, exactly: money in, case open, because history said
    'do not call any more tools'."""
    out = _defuse_expired_instructions([HumanMessage(content="go"),
                                        _wait_result()])
    body = str(out[-1].content)
    assert _EXPIRED_TURN_ORDER not in body, (
        "the expired order still outranks the settled prompt")
    assert "that turn only" in body, "it must read as a past fact, not an order"


def test_the_rest_of_the_result_is_untouched():
    """Only the expired sentence is rewritten; the record of what happened
    stays intact, because the agent still needs to know it waited."""
    out = _defuse_expired_instructions([_wait_result()])
    body = str(out[-1].content)
    assert "pay_lohj3x6gt" in body and "Turn ended." in body


def test_messages_without_the_order_are_passed_through_unchanged():
    msgs = [HumanMessage(content="hi"), AIMessage(content="ok"),
            ToolMessage(content='{"status":"ok"}', tool_call_id="t1",
                        name="check_payment_status")]
    out = _defuse_expired_instructions(msgs)
    assert [str(m.content) for m in out] == [str(m.content) for m in msgs]


def test_the_defuser_runs_on_the_messages_the_model_actually_sees():
    """It has to sit in the assembly path, not merely exist."""
    import inspect
    from recovery_agent.agent import graph as G
    src = inspect.getsource(G.agent_node)
    assert "_defuse_expired_instructions(" in src, \
        "the defuser is never applied to the LLM's message list"
