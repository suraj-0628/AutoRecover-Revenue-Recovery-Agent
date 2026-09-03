"""One agent session per payment — never per run, never shared between payments.

`thread_id` was `case.id`, and a fresh `Case` (new uuid) is built on every
invocation. A single payment therefore ran as several unrelated sessions —
pay_fw1l0oppo had three — each starting from an empty thread and knowing only
what the frontend re-injected. Two different payments could equally have been
handled with no structural guarantee of separation.
"""
from __future__ import annotations

from recovery_agent.models import Case, PaymentEvent


def _event(payment_id: str, amount: float = 1000.0) -> PaymentEvent:
    return PaymentEvent(payment_id=payment_id, amount=amount, currency="INR",
                        failure_code="card_expired", failure_reason="expired",
                        customer_id="same@customer.com")


def _thread_for(case: Case) -> str:
    """Mirrors how RecoveryAgent.run builds its config."""
    return f"case:{case.payment.payment_id}"


def test_two_runs_of_one_payment_share_a_session():
    """Follow-ups must continue the same conversation, not start a blank one."""
    first = Case(payment=_event("pay_abc"))
    followup = Case(payment=_event("pay_abc"))          # new Case object, same payment
    assert first.id != followup.id, "precondition: a fresh Case each run"
    assert _thread_for(first) == _thread_for(followup)


def test_two_payments_never_share_a_session():
    a = Case(payment=_event("pay_one"))
    b = Case(payment=_event("pay_two"))
    assert _thread_for(a) != _thread_for(b)


def test_same_customer_two_orders_stay_separate():
    """The confusion risk: one customer, two live orders at once."""
    order1 = Case(payment=_event("pay_order1", 2499.0))
    order2 = Case(payment=_event("pay_order2", 49980.0))
    assert order1.payment.customer_id == order2.payment.customer_id
    assert _thread_for(order1) != _thread_for(order2)


def test_the_agent_builds_the_session_key_this_way():
    """Pin the real call site so this cannot silently regress to case.id."""
    import inspect
    from recovery_agent import agent as agent_pkg
    src = inspect.getsource(agent_pkg.RecoveryAgent.run)
    assert 'f"case:{case.payment.payment_id}"' in src
    assert '"thread_id": case.id' not in src


def test_the_frontend_builds_the_session_key_this_way():
    import inspect
    from recovery_agent import frontend
    src = inspect.getsource(frontend._run_agent_for_payment_inner)
    assert 'f"case:{payment_id}"' in src
    assert '"thread_id": case.id' not in src


def test_the_agent_is_told_it_owns_one_payment():
    from recovery_agent.agent.graph import SYSTEM_PROMPT
    assert "ONE payment" in SYSTEM_PROMPT
    assert "DIFFERENT payment" in SYSTEM_PROMPT
