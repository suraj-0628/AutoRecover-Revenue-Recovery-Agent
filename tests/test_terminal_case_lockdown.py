"""A finished case must not be able to contact or charge the customer.

Sessions are continuous per payment, so the closing run after a successful
payment replays the earlier turns — including the instruction to recover it.
Told to stop, the model recorded the lesson and then resumed the older task: it
pushed a fresh notification and re-offered 5% to a customer who had already paid
INR 37,990.50.

Instruction is not a control. These tests pin the tools away.
"""
from __future__ import annotations

import pytest

from recovery_agent.agent.governance import get_allowed_tools
from recovery_agent.agent.tools import TOOLS_BY_NAME
from recovery_agent.models import CaseStatus

CONTACT_OR_CHARGE = {
    "send_page_push", "show_page_offer", "send_recovery_notification",
    "initiate_voice_call", "generate_recovery_payment_link", "retry_in_hours",
}
TERMINAL = (CaseStatus.RECOVERED, CaseStatus.ESCALATED, CaseStatus.STOPPED)


def _bound_tools(status: CaseStatus, tier: str = "active") -> set[str]:
    """Mirrors the restriction applied in agent_node."""
    allowed = set(get_allowed_tools(tier)) & set(TOOLS_BY_NAME)
    if status in TERMINAL:
        allowed = {n for n in allowed if n in ("manage_memory", "search_memory")}
    return allowed


@pytest.mark.parametrize("status", TERMINAL)
def test_a_finished_case_cannot_reach_the_customer(status):
    assert not (_bound_tools(status) & CONTACT_OR_CHARGE)


@pytest.mark.parametrize("status", TERMINAL)
def test_a_finished_case_can_still_record_what_it_learned(status):
    """Stopping must not stop it learning — that is the point of the closing run."""
    assert "manage_memory" in _bound_tools(status)


@pytest.mark.parametrize("status", [CaseStatus.OPEN, CaseStatus.ACTING,
                                    CaseStatus.AWAITING_CUSTOMER])
def test_a_live_case_keeps_its_full_toolset(status):
    tools = _bound_tools(status)
    assert CONTACT_OR_CHARGE & tools, "a live case must still be recoverable"
    assert len(tools) > 5


def test_the_restriction_is_applied_in_agent_node():
    """Pin the call site so this cannot regress to prompt-only guidance."""
    import inspect
    from recovery_agent.agent import graph
    src = inspect.getsource(graph.agent_node)
    assert "CaseStatus.RECOVERED" in src
    assert '"manage_memory", "search_memory"' in src
