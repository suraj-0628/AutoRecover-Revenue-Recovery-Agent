"""Closing the checkout after a decline must not be thrown away.

`/api/payment-failed` returned early whenever the case was already
`recovering`, before the block that reconciles a synthetic page signal against
the gateway's diagnosis. Once the gateway failure began deferring the agent,
the case was ALWAYS `recovering` by the time the customer closed the modal —
so `abandoned_after_failure` was computed and discarded on every live decline.

That flag is what makes a discount defensible on a method failure: they were
declined, Razorpay put its other rails in front of them, and they left anyway.
Losing it silently changes what the agent may offer.
"""
import pytest

import recovery_agent.frontend as F

CUST = {"email": "s@example.com", "contact": "9363064502"}
DECLINE = ("Your payment didn't go through as it was declined by the bank. "
           "Try another payment method or contact your bank.")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from recovery_agent import state_store
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    state_store.StateStore.reset_instances()
    monkeypatch.setattr(F, "store", state_store.StateStore())
    monkeypatch.setattr(F.socketio, "start_background_task", lambda *a, **k: None)
    yield
    state_store.StateStore.reset_instances()


def _decline(c, pid):
    return c.post("/api/payment-failed", json={
        "payment_id": pid, "amount": 1299.0, "failure_code": "BAD_REQUEST_ERROR",
        "failure_reason": DECLINE, "error_source": "bank",
        "error_step": "payment_authorization", "customer": CUST,
        "defer_agent": True})


def _dismiss(c, pid):
    return c.post("/api/payment-failed", json={
        "payment_id": pid, "amount": 1299.0, "failure_code": "customer_cancelled",
        "failure_reason": "Closed the checkout after a failed attempt",
        "customer": CUST, "defer_agent": True})


def test_walking_away_after_a_decline_is_recorded():
    c = F.app.test_client()
    _decline(c, "pay_l1")
    _dismiss(c, "pay_l1")
    rec = F.store.get_payment("pay_l1") or {}
    assert rec.get("abandoned_after_failure") is True, \
        "the demonstrated reluctance that justifies a discount was lost"


def test_the_gateway_diagnosis_survives_the_dismissal():
    """A synthetic customer_cancelled must not overwrite a real bank decline."""
    c = F.app.test_client()
    _decline(c, "pay_l2")
    _dismiss(c, "pay_l2")
    assert (F.store.get_payment("pay_l2") or {}).get("failure_code") == \
        "BAD_REQUEST_ERROR"


def test_the_gateways_own_account_is_persisted():
    """error_step is what tells an auth failure from a network one, and every
    later run rebuilds from the record."""
    c = F.app.test_client()
    _decline(c, "pay_l3")
    rec = F.store.get_payment("pay_l3") or {}
    assert rec.get("error_source") == "bank"
    assert rec.get("error_step") == "payment_authorization"


def test_a_dismissal_without_contact_does_not_erase_the_source():
    c = F.app.test_client()
    _decline(c, "pay_l4")
    _dismiss(c, "pay_l4")
    rec = F.store.get_payment("pay_l4") or {}
    assert rec.get("error_source") == "bank"


def test_an_in_flight_case_is_not_given_a_second_agent():
    started = []
    F.socketio.start_background_task = lambda *a, **k: started.append(a)
    c = F.app.test_client()
    _decline(c, "pay_l5")
    body = _dismiss(c, "pay_l5").get_json()
    assert body["status"] == "already_recovering"
    assert body["signal_recorded"] is True
    assert started == [], "two agent runs on one case"
