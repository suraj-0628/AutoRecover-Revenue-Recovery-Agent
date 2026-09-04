"""Two defects from one live case: pay_p9oxiasll, 2026-09-04.

The customer paid. INR 1,709.05 was captured on plink_TXwDValFzzEGgV, the link
that was emailed to them. The agent escalated the case to a human at zero
recovered, and its closing summary said the customer "failed repeatedly".

Nothing about that is a model mistake. Two instruments lied to it:

1. The recovery watcher polled ONE link id, read out of `order_id`, which holds
   only the most recently minted link. The rail-switch rung had minted a second
   link and overwritten the first, so the watcher spent the whole window
   polling a link nobody would ever pay.

2. The double-debit lock reported "payment already succeeded" because a
   NOTIFICATION had been delivered -- `Attempt.result` is the result of the
   action, not of a payment.

Asking a customer to pay again after they have paid is the single worst thing
this system can do, so both get a test.
"""
from __future__ import annotations

import pytest

from recovery_agent.models import (ActionType, Attempt, Case, PaymentEvent,
                                   RecoveryTier)


# ── 1. every minted link is watched, not just the last ──────────────────

def _links(*ids):
    return [{"link_id": i, "amount": 1709.05} for i in ids]


def test_the_watcher_collects_every_link_the_case_has_minted(monkeypatch):
    """A case that climbs the ladder mints more than one link. The customer
    pays whichever one reached them -- usually the emailed one, which is the
    FIRST, not the last."""
    import recovery_agent.frontend as fe

    record = {
        "payment_id": "pay_p9oxiasll",
        "recovery_links": _links("plink_FIRST_emailed", "plink_SECOND_railswitch"),
        # order_id holds only the most recent mint -- this is the trap.
        "order_id": "plink_SECOND_railswitch",
        "recovery_link_id": "plink_SECOND_railswitch",
    }
    fetched: list[str] = []

    class _Links:
        def fetch(self, link_id):
            fetched.append(link_id)
            if link_id == "plink_FIRST_emailed":
                return {"status": "paid", "payments": [
                    {"payment_id": "pay_TXwFDdnYDGSKPf", "status": "captured",
                     "amount": 170905}]}
            return {"status": "created", "payments": []}

    class _Client:
        payment_link = _Links()

    class _RZP:
        is_configured = True
        client = _Client()

    marked: dict = {}
    monkeypatch.setattr("recovery_agent.razorpay_client.RazorpayClient",
                        lambda *a, **k: _RZP())
    monkeypatch.setattr(fe.store, "has_payment", lambda pid: True)
    monkeypatch.setattr(fe.store, "get_payment", lambda pid: record)
    monkeypatch.setattr(fe.store, "has_pending", lambda pid: False)
    monkeypatch.setattr(fe, "_mark_recovered",
                        lambda pid, amt, cap, secs, how: marked.update(
                            payment_id=pid, amount=amt, captured=cap, how=how))
    monkeypatch.setattr("time.sleep", lambda *_: None)

    fe._watch_for_recovery("pay_p9oxiasll", max_wait_seconds=20)

    assert "plink_FIRST_emailed" in fetched, (
        "the emailed link was never polled -- this is the live failure: the "
        "watcher only ever looked at the most recently minted link")
    assert marked.get("amount") == pytest.approx(1709.05)
    assert marked.get("captured") == "pay_TXwFDdnYDGSKPf"


def test_an_unreadable_link_does_not_hide_a_paid_one(monkeypatch):
    """One link failing to fetch must not abandon the rest. The old code fell
    out of the check entirely on the first exception."""
    import recovery_agent.frontend as fe

    record = {"payment_id": "pay_x",
              "recovery_links": _links("plink_BROKEN", "plink_PAID"),
              "order_id": "plink_BROKEN"}

    class _Links:
        def fetch(self, link_id):
            if link_id == "plink_BROKEN":
                raise RuntimeError("Razorpay 500")
            return {"status": "paid", "payments": [
                {"payment_id": "pay_ok", "status": "captured", "amount": 170905}]}

    class _RZP:
        is_configured = True
        client = type("C", (), {"payment_link": _Links()})()

    marked: dict = {}
    monkeypatch.setattr("recovery_agent.razorpay_client.RazorpayClient",
                        lambda *a, **k: _RZP())
    monkeypatch.setattr(fe.store, "has_payment", lambda pid: True)
    monkeypatch.setattr(fe.store, "get_payment", lambda pid: record)
    monkeypatch.setattr(fe.store, "has_pending", lambda pid: False)
    monkeypatch.setattr(fe, "_mark_recovered",
                        lambda pid, amt, cap, secs, how: marked.update(amount=amt))
    monkeypatch.setattr("time.sleep", lambda *_: None)

    fe._watch_for_recovery("pay_x", max_wait_seconds=20)
    assert marked.get("amount") == pytest.approx(1709.05)


# ── 2. only a debit can double-debit ────────────────────────────────────

def _case_with(attempt: Attempt) -> Case:
    return Case(payment=PaymentEvent(payment_id="pay_p9oxiasll",
                                     customer_id="cust1", amount=1799.0),
                attempts=[attempt])


def _check(case: Case):
    from recovery_agent.agent.guardrails import (DoubleDebitLockGuardrail,
                                                 GuardrailVerdict)
    return DoubleDebitLockGuardrail().check(
        ActionType.RETRY_PAYMENT, case=case), GuardrailVerdict


def test_a_delivered_notification_is_not_a_successful_payment():
    """The live false positive. `Attempt.result` is the ACTION's result, so a
    sent email recorded "success" and the lock announced "payment already
    succeeded" -- contradicting the same turn's briefing, which correctly said
    the money was not back."""
    case = _case_with(Attempt(action_type=ActionType.SEND_NOTIFICATION,
                              result="success", tier=RecoveryTier.ACTIVE))
    result, Verdict = _check(case)
    assert result.verdict is Verdict.PASS, (
        f"a delivered notification blocked a retry: {result.reason}")


def test_a_successful_retry_still_blocks_a_second_debit():
    """The guardrail must keep doing its real job."""
    case = _case_with(Attempt(action_type=ActionType.RETRY_PAYMENT,
                              result="success", tier=RecoveryTier.ACTIVE))
    result, Verdict = _check(case)
    assert result.verdict is Verdict.BLOCKED
    assert "retry already succeeded" in result.reason


# ── 3. the suite must not mail real people ──────────────────────────────

def test_the_test_suite_cannot_send_real_email():
    """A live incident, not a hypothetical.

    `frontend.py` calls load_dotenv() at import, so as soon as any test
    imported it SMTP_HOST became the real Brevo relay for the whole session.
    One suite run delivered five emails carrying live payment links to a real
    inbox and spent five of a 300/day allowance. The .eml file is still
    written — that path is what the outbox is for — but nothing leaves the
    machine.
    """
    import os
    import recovery_agent.frontend  # noqa: F401  — the import that armed SMTP
    assert os.environ.get("SMTP_HOST") == "", (
        "SMTP is live during tests; load_dotenv() has re-armed the real relay")


def test_the_suite_holds_no_licence_to_spend_paid_quota():
    """Payment links are a 30-per-account LIFETIME quota and voice calls cost
    real credits. Both are service-only capabilities that start.sh grants; a
    test process must never hold them."""
    import os
    for cap in ("RAZORPAY_WRITES_OK", "SUPERU_CALLS_OK"):
        assert cap not in os.environ, f"tests are authorised to spend {cap}"
