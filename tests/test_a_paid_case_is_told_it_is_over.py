"""A settled case must not be handed the recovery manual.

pay_nam2g0aj7: INR 1,234.05 arrived 15 seconds after the offer email. The
briefing said so plainly — "RECOVERED... Close the case... Do NOT contact the
customer again" — and the model opened its reply with "FOLLOW-UP: ... 15
minutes have passed. The customer has NOT paid the recovery link."

It had invented that, because the operating prompt opens with "FIRST, check
the context for FOLLOW-UP" and that was the loudest instruction present. The
rung tools were already withheld, so it could not act on the belief; it simply
called nothing, and the case stayed open with the money banked.

Withholding tools stops the action. Only replacing the prompt stops the belief.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from recovery_agent.agent import graph as G


def _head(settled: bool) -> str:
    msgs = G._assemble_llm_messages(
        [SystemMessage(content="WHAT IS TRUE RIGHT NOW: the money is in.")],
        [HumanMessage(content="A payment has failed."), AIMessage(content="ok")],
        settled=settled)
    assert isinstance(msgs[0], SystemMessage)
    return msgs[0].content


def test_an_open_case_still_gets_the_recovery_manual():
    head = _head(settled=False)
    assert "FOLLOW-UP" in head
    assert head == G.SYSTEM_PROMPT


def test_a_settled_case_is_not_told_to_look_for_a_follow_up():
    head = _head(settled=True)
    assert head == G.SETTLED_PROMPT
    assert "FIRST, check the context for" not in head


def test_the_closing_prompt_names_the_only_two_moves():
    head = _head(settled=True)
    assert "manage_memory" in head and "close_case" in head


def test_the_closing_prompt_disowns_a_remembered_follow_up():
    """The history is replayed, so the earlier FOLLOW-UP text is still in the
    context window. The prompt has to say it no longer describes anything."""
    head = _head(settled=True)
    assert "no longer exists" in head


def test_the_briefing_still_follows_the_prompt_in_both_modes():
    for settled in (True, False):
        msgs = G._assemble_llm_messages(
            [SystemMessage(content="WHAT IS TRUE RIGHT NOW: x")],
            [HumanMessage(content="h")], settled=settled)
        assert "WHAT IS TRUE RIGHT NOW" in msgs[1].content


def test_the_head_is_never_trimmed_away_on_a_long_settled_case():
    history = [HumanMessage(content="A payment has failed.")]
    history += [AIMessage(content=f"turn {i}") for i in range(80)]
    msgs = G._assemble_llm_messages(
        [SystemMessage(content="WHAT IS TRUE RIGHT NOW: the money is in.")],
        history, settled=True)
    assert msgs[0].content == G.SETTLED_PROMPT
    assert "WHAT IS TRUE RIGHT NOW" in msgs[1].content
