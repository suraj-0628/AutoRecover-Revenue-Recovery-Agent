"""Perception must survive context management.

The "WHAT IS TRUE RIGHT NOW" briefing is recomputed every turn precisely so the
agent cannot act on stale beliefs. Truncation used to run over the assembled
message list keeping index 0 and the first HumanMessage — the briefing sat at
index 1, so on any case longer than ~30 messages it silently fell out of the
context. The cases long enough to lose it are exactly the multi-run cases most
at risk of re-contacting a customer who already paid.

Same principle for the on-demand check: check_payment_status output IS the
perception, and chopping it at 500 chars mid-JSON is induced half-blindness.
"""
import json

from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)

from recovery_agent.agent.graph import (SYSTEM_PROMPT, _assemble_llm_messages,
                                        _trim_tool_results_for_llm)

BRIEFING = SystemMessage(content="WHAT IS TRUE RIGHT NOW: sentinel-briefing")


def _long_history(n_pairs: int) -> list:
    history = [HumanMessage(content="case facts: pay_x failed")]
    for i in range(n_pairs):
        history.append(AIMessage(content="", tool_calls=[
            {"name": "check_payment_status", "args": {}, "id": f"c{i}"}]))
        history.append(ToolMessage(content='{"status": "open"}',
                                   tool_call_id=f"c{i}", name="check_payment_status"))
    return history


def test_the_briefing_survives_a_long_session():
    history = _long_history(40)          # 81 messages, far past the cap
    out = _assemble_llm_messages([BRIEFING], history, max_messages=30)

    assert out[0].content == SYSTEM_PROMPT
    assert out[1] is BRIEFING, "the briefing is the head, not the history"
    assert out[2] is history[0], "the case facts survive too"
    assert len(out) <= 3 + 30


def test_a_short_session_passes_through_untouched():
    history = _long_history(3)
    out = _assemble_llm_messages([BRIEFING], history, max_messages=30)
    assert out == [out[0], BRIEFING] + history


def test_the_trimmed_tail_never_starts_with_an_orphan():
    out = _assemble_llm_messages([BRIEFING], _long_history(40), max_messages=30)
    tail = out[3:]
    assert tail, "something of the history must remain"
    assert not isinstance(tail[0], ToolMessage), (
        "a tool result with no matching call is a guaranteed 400")


def test_no_briefing_still_assembles():
    out = _assemble_llm_messages([], _long_history(40), max_messages=30)
    assert out[0].content == SYSTEM_PROMPT
    assert out[1] is not None


# ── the on-demand perception must arrive whole ──────────────────────────────

def test_check_payment_status_output_is_never_trimmed():
    briefing_json = json.dumps({"status": "settled",
                                "briefing": "WHAT IS TRUE RIGHT NOW " * 40})
    assert len(briefing_json) > 500
    msgs = [ToolMessage(content=briefing_json, tool_call_id="c1",
                        name="check_payment_status")]
    out = _trim_tool_results_for_llm(msgs, max_chars=500)
    assert out[0].content == briefing_json
    assert json.loads(out[0].content)["status"] == "settled"


def test_other_long_results_are_trimmed_on_a_copy():
    payload = json.dumps({"status": "escalated", "reason": "x" * 700})
    original = ToolMessage(content=payload, tool_call_id="c1",
                           name="escalate_to_human")
    out = _trim_tool_results_for_llm([original], max_chars=500)
    assert len(out[0].content) < len(payload), "the LLM's copy is shortened"
    assert original.content == payload, "the graph-state message is untouched"
