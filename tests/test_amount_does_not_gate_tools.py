"""Charging what is owed is never the risk; the give-away is.

`get_allowed_tools` used to strip `generate_recovery_payment_link` above an
"escalation threshold" of 5,000,000 paise — but the caller passed RUPEES, so
the rule slept until ₹50,00,000 and, had it ever woken, would have refused a
full-price recovery link to exactly the orders most worth recovering (the
₹79,980 incident the approval gate's comments record). The rule is gone: tool
access is tier-gated only, and the money control is the discount ceiling in
`human_approval_gate`.
"""
import inspect

from recovery_agent.agent import governance
from recovery_agent.agent.governance import get_allowed_tools


def test_the_link_tool_is_available_at_every_tier():
    for tier in ("silent", "active", "escalated"):
        assert "generate_recovery_payment_link" in get_allowed_tools(tier), tier


def test_no_amount_parameter_exists_to_misuse():
    """The bug was a units mismatch at the call site; the parameter that made
    it possible must not linger to be misused again."""
    params = inspect.signature(get_allowed_tools).parameters
    assert "amount_paise" not in params
    assert not any("amount" in name for name in params)


def test_no_paise_threshold_survives_in_the_policy():
    from recovery_agent.agent.governance import ToolAccessPolicy
    assert "escalation_threshold_paise" not in ToolAccessPolicy.model_fields


def test_the_money_control_is_the_discount_ceiling():
    """What replaced it: the approval gate caps what is GIVEN AWAY."""
    from recovery_agent.agent.graph import (_APPROVAL_DISCOUNT_THRESHOLD,
                                            _MONEY_MOVING_TOOLS)
    assert _APPROVAL_DISCOUNT_THRESHOLD > 0
    assert "generate_recovery_payment_link" in _MONEY_MOVING_TOOLS


def test_governance_module_carries_no_dead_amount_logic():
    src = inspect.getsource(governance)
    assert "escalation_threshold_paise" not in src
