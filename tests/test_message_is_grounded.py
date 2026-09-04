"""The agent writes the words; the case has to back the facts in them.

Two problems at once, and they pull in opposite directions:

  - GENERIC. Every recovery email carried the same subject — "Payment
    Recovery: Complete Your Pending Payment" — for a bank decline, an
    abandoned cart and a payday retry alike, and the agent's own prose was
    injected as a `Reason:` line inside a fixed template.
  - UNGROUNDED. The obvious fix (let it write freely) lets it promise things
    the link will not honour. That already happened once: the agent created a
    link at full price, then obtained a 5% offer, and emailed "you only pay
    INR 2,374.05" against a link charging INR 2,499.

So the prose is free and the claims are checked: money, percentages, the
language of a discount, and any payment method it tells the customer to use.
"""
import json

import pytest

import recovery_agent.notifications as notifications
import recovery_agent.state_store as state_store
from recovery_agent.agent.tools import _ungrounded_claims
from recovery_agent.state_store import StateStore

# A ₹2,499 order with a 5% offer: the link charges ₹2,374.05, saving ₹124.95.
OFFER = {"owed": 2499.0, "charged": 2374.05, "discount_pct": 5.0,
         "rails": ["upi", "card", "netbanking"]}
FULL = {"owed": 2499.0, "charged": 2499.0, "discount_pct": 0,
        "rails": ["upi", "netbanking"]}


# ── money ───────────────────────────────────────────────────────────────────

def test_the_price_the_link_charges_is_allowed():
    assert not _ungrounded_claims("Pay just ₹2,374.05 to finish.", OFFER)


def test_the_amount_owed_is_allowed():
    assert not _ungrounded_claims("Your ₹2,499.00 order is waiting.", OFFER)


def test_the_saving_between_them_is_allowed():
    assert not _ungrounded_claims("You save ₹124.95 today.", OFFER)


def test_a_price_the_link_will_not_charge_is_refused():
    """The live incident, in one assertion."""
    why = _ungrounded_claims("You only pay INR 1,999.00.", OFFER)
    assert "1,999" in why and "neither the amount owed" in why


def test_a_duration_is_not_read_as_a_price():
    """'16 minutes' and '24 hours' are not money and must not trip the check."""
    assert not _ungrounded_claims(
        "This link expires in 16 minutes. We retried after 24 hours.", OFFER)


# ── percentages ─────────────────────────────────────────────────────────────

def test_the_authorised_discount_may_be_stated():
    assert not _ungrounded_claims("Here's 5% off to finish your order.", OFFER)


def test_a_bigger_discount_than_authorised_is_refused():
    why = _ungrounded_claims("Here's 20% off!", OFFER)
    assert "20% off" in why and "authorised discount is 5%" in why


def test_a_discount_claimed_when_none_exists_is_refused():
    why = _ungrounded_claims("Here's 5% off to finish.", FULL)
    assert "no discount is authorised" in why


def test_discount_language_without_a_discount_is_refused():
    """No number, but the same broken promise."""
    why = _ungrounded_claims("Complete now at a special price.", FULL)
    assert "promises a discount" in why


# ── payment rails: "bank issue, use UPI" must be true of the link ───────────

def test_recommending_a_rail_the_link_accepts_is_fine():
    assert not _ungrounded_claims(
        "Your bank declined the card. Try UPI on this link instead.", FULL)


def test_recommending_a_rail_the_link_refuses_is_refused():
    why = _ungrounded_claims(
        "Your payment failed — pay with your wallet instead.", FULL)
    assert "wallet" in why and "does not accept" in why


def test_naming_the_failed_method_is_not_a_recommendation():
    """"Your card was declined" mentions a rail the link may not take. That is
    describing what happened, not promising a route."""
    assert not _ungrounded_claims(
        "Your card was declined by the bank. Use netbanking on this link.", FULL)


def test_rails_are_unchecked_when_the_link_did_not_report_any():
    assert not _ungrounded_claims("Try UPI.", {**FULL, "rails": []})


# ── end to end through the tool ─────────────────────────────────────────────

@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    StateStore.reset_instances()
    s = StateStore()
    s.save_payment("p", {"payment_id": "p", "amount": 2499.0,
                         "status": "failed", "failure_code": "bank_declined",
                         "trail": []})
    s.flush()
    yield s
    StateStore.reset_instances()


class _Dispatched:
    sent = {}

    def __init__(self, *a, **k):
        pass

    def dispatch(self, **kw):
        _Dispatched.sent = dict(kw)
        return {"status": "dispatched", "channels": ["email"],
                "attempted": ["email"], "undelivered": {}, "results": []}


def _send(store, monkeypatch, **over):
    from recovery_agent.agent.tools import send_recovery_notification
    monkeypatch.setattr(notifications, "NotificationDispatcher", _Dispatched)
    args = {"payment_id": "p", "customer_email": "c@example.com",
            "customer_phone": "+919999999999",
            "subject": "Your bank declined — try UPI",
            "message": "Your bank declined the card. This link takes UPI too.",
            "payment_link": "https://rzp.io/x", "amount": 2499.0}
    args.update(over)
    return json.loads(send_recovery_notification.invoke(args))


def test_a_grounded_message_is_sent_with_the_agents_own_subject(store, monkeypatch):
    r = _send(store, monkeypatch)
    assert r["status"] == "ok"
    assert _Dispatched.sent["subject"] == "Your bank declined — try UPI"
    assert "declined the card" in _Dispatched.sent["body"], \
        "the agent's words are the body, not a Reason: line in a template"


def test_an_ungrounded_price_is_refused_before_it_is_sent(store, monkeypatch):
    _Dispatched.sent = {}
    r = _send(store, monkeypatch, message="Pay only ₹99.00 now.")
    assert r["status"] == "error"
    assert "₹99.00" in r["message"]
    assert _Dispatched.sent == {}, "nothing may go out"


def test_a_claim_hidden_in_the_subject_is_caught_too(store, monkeypatch):
    r = _send(store, monkeypatch, subject="Get 50% off today")
    assert r["status"] == "error"
    assert "50% off" in r["message"]


def test_an_over_long_subject_is_refused():
    """Anything past 60 characters is cut off on a phone, which is where most
    of these are read."""
    import recovery_agent.notifications as n
    from recovery_agent.agent.tools import send_recovery_notification
    long = "A" * 61
    r = json.loads(send_recovery_notification.invoke({
        "payment_id": "nope", "customer_email": "c@e.com", "customer_phone": "",
        "subject": long, "message": "hello", "payment_link": "https://rzp.io/x"}))
    assert r["status"] == "error" and "61 characters" in r["message"]


def test_the_refusal_says_what_to_do_about_it(store, monkeypatch):
    r = _send(store, monkeypatch, message="Pay only ₹99.00 now.")
    assert "Rewrite it" in r["guidance"]
    assert "Do not remove the link" in r["guidance"], \
        "a wording problem must not cost the customer their offer"
