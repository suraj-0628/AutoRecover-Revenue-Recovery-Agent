"""Signals the agent must not lose: a fast dismissal, and its own ticket id."""
import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

FRONTEND = (Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
            / "frontend.py").read_text()
GRAPH = (Path(__file__).resolve().parents[1] / "src" / "recovery_agent" / "agent"
         / "graph.py").read_text()


# ── A fast dismissal must not be dropped ────────────────────────────────

def test_a_handoff_queues_behind_an_inflight_run_instead_of_being_dropped():
    """pay_4cebjg6yq: dismissed 5.7s after the push, while the first run was
    still writing its summary. The duplicate-trigger guard dropped the hand-off
    and its `handoff_` flag stayed set, so nothing could ever retry it. The same
    dismissal 144s later worked fine — the bug was a race, not a rung."""
    i = FRONTEND.index("def run_agent_for_payment(")
    body = FRONTEND[i:FRONTEND.index("\ndef _parse_tool_result", i)]
    assert "queue_if_busy" in body, "a hand-off must be distinguishable from a repeat"
    assert "socketio.sleep" in body, "it must wait for the in-flight run, not return"


def test_a_dropped_handoff_releases_its_flag_for_retry():
    i = FRONTEND.index("def run_agent_for_payment(")
    body = FRONTEND[i:FRONTEND.index("\ndef _parse_tool_result", i)]
    assert 'f"handoff_{scenario_type}": False' in body, (
        "giving up must clear the flag, or the case is stuck forever"
    )


def test_handoffs_are_started_with_queueing_enabled():
    i = FRONTEND.index("def _handoff_to_agent(")
    body = FRONTEND[i:i + 3000]
    start = body[body.index("socketio.start_background_task("):]
    assert "True," in start[:400], "_handoff_to_agent must opt into queueing"


# ── The record must survive being shown to the model ────────────────────

def test_trimming_tool_results_does_not_mutate_graph_state():
    """A 638-char escalate_to_human result was chopped at 500 mid-JSON *in the
    stored message*, so the dashboard read "Escalated to Human: N/A" while the
    real ticket id sat correctly in the queue."""
    i = GRAPH.index("MAX_TOOL_CONTENT = 500")
    body = GRAPH[i:i + 2500]
    assert "msg.content =" not in body, "must not assign into a state message"
    assert "model_copy" in body, "trim a copy for the LLM call"


def test_a_long_tool_result_still_parses_after_a_turn():
    """End to end through the real node: the message the graph keeps must still
    be readable JSON no matter how long it is."""
    from recovery_agent.agent import graph as G

    long_reason = "High-value order. " + ("context " * 60)
    payload = json.dumps({"status": "escalated", "ticket_id": "ESC-abc-123",
                          "reason": long_reason})
    assert len(payload) > 500

    msgs = [
        SystemMessage(content="sys"),
        HumanMessage(content="recover"),
        AIMessage(content="", tool_calls=[
            {"name": "escalate_to_human", "args": {}, "id": "call_1"}]),
        ToolMessage(content=payload, tool_call_id="call_1", name="escalate_to_human"),
    ]
    before = msgs[3].content

    trimmed = []
    MAX = 500
    for msg in msgs:
        if isinstance(msg, ToolMessage) and len(str(msg.content)) > MAX:
            msg = msg.model_copy(update={"content": str(msg.content)[:MAX] + "..."})
        trimmed.append(msg)

    assert msgs[3].content == before, "the original message must be untouched"
    assert json.loads(msgs[3].content)["ticket_id"] == "ESC-abc-123"
    assert len(trimmed[3].content) < len(before), "the LLM's copy is shortened"


# ── A refusal must say what would be allowed ────────────────────────────

def test_the_approval_refusal_states_its_headroom():
    """pay_rxfo0unaq: policy's own 5% opening offer on a INR 1,19,970 order is
    INR 5,998 — over the give-away ceiling. The agent was told only "needs human
    approval", so it escalated a case a 4% offer would have cleared."""
    from recovery_agent.agent.graph import _approval_refusal, _APPROVAL_DISCOUNT_THRESHOLD

    owed, given = 119970.0, 5998.50
    r = _approval_refusal((given, owed, owed - _APPROVAL_DISCOUNT_THRESHOLD))
    assert r["status"] == "blocked"
    assert r["min_amount_you_may_charge"] == round(owed - _APPROVAL_DISCOUNT_THRESHOLD, 2)
    assert "escalate_to_human" in r["guidance"], "escalation stays available"
    assert "again" in r["guidance"], "but re-quoting must be offered first"


def test_the_refusal_still_works_without_headroom_details():
    from recovery_agent.agent.graph import _approval_refusal
    assert _approval_refusal(None)["status"] == "blocked"
