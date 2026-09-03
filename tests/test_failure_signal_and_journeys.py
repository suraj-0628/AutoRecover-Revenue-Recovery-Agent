"""The live-checkout findings of 2026-09-03, locked down.

Case 3/5: a netbanking decline became a "drop-off" the moment the customer
closed the Razorpay modal — the synthetic `customer_cancelled` overwrote the
gateway's diagnosis, which made a 5% discount legal for a payment the customer
had been trying to MAKE (pay_fotw1e7b6, ₹59,976). Case 4: a clean first-try
sale flowed into the recovery machinery via the `pending` record that
create-order opens for every checkout.

Mitigations under test: signal precedence at ingress, allowlisted (fail-closed)
incentives, the rail-switch surface for method failures, and the clean-sale
guard. The modal-hold JS is inline in PAY_PAGE and gets a source check only.
"""
import json

import pytest

import recovery_agent.push_bus as push_bus
import recovery_agent.state_store as state_store
import recovery_agent.frontend as F
from recovery_agent.state_store import StateStore


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    StateStore.reset_instances()
    monkeypatch.setattr(F, "store", StateStore())
    yield
    StateStore.reset_instances()


@pytest.fixture()
def agent_runs(monkeypatch):
    """Capture agent launches instead of running them (no LLM in tests)."""
    calls = []
    monkeypatch.setattr(F.socketio, "start_background_task",
                        lambda fn, *a, **k: calls.append(a))
    return calls


def _fail(payment_id, code, reason="failed"):
    return F.app.test_client().post("/api/payment-failed", json={
        "payment_id": payment_id, "amount": 59976.0,
        "failure_code": code, "failure_reason": reason,
        "customer": {"email": "c@example.com"},
    })


# ── A: signal precedence ────────────────────────────────────────────────────

def test_modal_dismissal_does_not_overwrite_a_bank_decline(agent_runs):
    _fail("pay_a", "BAD_REQUEST_ERROR", "Netbanking payment failed at the bank")
    F.store.update_payment("pay_a", status="awaiting_customer")  # run over

    _fail("pay_a", "customer_cancelled", "Payment cancelled by customer")

    rec = F.store.get_payment("pay_a")
    assert rec["failure_code"] == "BAD_REQUEST_ERROR", \
        "a synthetic page event downgraded the gateway diagnosis"
    assert any(t.get("step") == "signal_precedence" for t in rec["trail"]), \
        "the dismissal must still be visible on the trail"
    # The agent run that follows must be briefed with the REAL failure.
    assert agent_runs[-1][5] == "BAD_REQUEST_ERROR"


def test_a_real_code_still_upgrades_a_synthetic_one(agent_runs):
    _fail("pay_b", "customer_cancelled")
    F.store.update_payment("pay_b", status="awaiting_customer")

    _fail("pay_b", "bank_declined", "do_not_honor")
    assert F.store.get_payment("pay_b")["failure_code"] == "bank_declined"


# ── B: incentives are allowlisted ───────────────────────────────────────────

def _offer(payment_id, amount=59976.0):
    from recovery_agent.agent.tools import get_recovery_offer
    return json.loads(get_recovery_offer.invoke({
        "amount": amount, "stage": "email_offer", "payment_id": payment_id}))


def _case(payment_id, code, **extra):
    F.store.save_payment(payment_id, {"payment_id": payment_id,
                                      "amount": 59976.0, "status": "recovering",
                                      "failure_code": code, "trail": [], **extra})


def test_an_unclassified_failure_gets_no_discount():
    _case("pay_u", "")
    r = _offer("pay_u")
    assert r["allowed"] is False
    assert "unclassified" in r["reason"]


def test_a_confirmed_dropoff_still_gets_the_offer():
    _case("pay_d", "customer_cancelled")
    r = _offer("pay_d")
    assert r["allowed"] is True and r["discount_pct"] > 0


def test_insufficient_funds_gets_timing_not_price():
    _case("pay_f", "insufficient_funds")
    r = _offer("pay_f")
    assert r["allowed"] is False
    assert "timing" in r["reason"]


def test_a_failed_full_price_attempt_unlocks_the_offer():
    _case("pay_u2", "", actions_tried=["link:card+upi:59976.00"])
    assert _offer("pay_u2")["allowed"] is True


def test_risk_is_never_incentivised():
    _case("pay_r", "fraud_suspected")
    r = _offer("pay_r")
    assert r["allowed"] is False and "escalate" in r["do_this_instead"]


# ── D: a method failure gets a rail switch, not promotion dressing ──────────

def test_zero_discount_on_a_bank_decline_renders_as_rail_switch(monkeypatch):
    delivered = []
    monkeypatch.setattr(push_bus, "deliver",
                        lambda payload: delivered.append(payload) or
                        {"status": "delivered"})
    _case("pay_m", "bank_declined")

    from recovery_agent.agent.tools import show_page_offer
    r = json.loads(show_page_offer.invoke({
        "payment_id": "pay_m", "headline": "Try another way to pay",
        "body": "Your bank could not process this. Same amount, another method.",
        "payable_amount": 59976.0, "discount_pct": 0.0,
        "payment_link": "https://rzp.io/x"}))
    assert r["status"] == "delivered"
    assert delivered[0]["mode"] == "rail_switch"
    assert "another payment method" in delivered[0]["offer_text"]
    assert "% off" not in delivered[0]["offer_text"]


def test_a_real_discount_still_renders_as_an_offer(monkeypatch):
    delivered = []
    monkeypatch.setattr(push_bus, "deliver",
                        lambda payload: delivered.append(payload) or
                        {"status": "delivered"})
    _case("pay_o", "customer_cancelled")

    from recovery_agent.agent.tools import show_page_offer
    show_page_offer.invoke({
        "payment_id": "pay_o", "headline": "5% off to finish",
        "body": "b", "payable_amount": 56977.20, "discount_pct": 5.0,
        "payment_link": "https://rzp.io/x"})
    assert delivered[0]["mode"] == "offer"
    assert "5% off" in delivered[0]["offer_text"]


# ── E: a clean first-try sale is not a recovery ─────────────────────────────

class _StubRzp:
    is_configured = True

    class client:
        class payment:
            @staticmethod
            def fetch(pid):
                return {"status": "captured", "amount": 5997600,
                        "order_id": "order_1"}


def _succeed(payment_id):
    return F.app.test_client().post("/api/payment-succeeded", json={
        "payment_id": payment_id, "razorpay_payment_id": "pay_rzp1",
        "razorpay_order_id": "order_1"})


def test_a_first_try_success_never_enters_the_recovery_books(monkeypatch):
    monkeypatch.setattr(F, "razorpay_client", _StubRzp())
    recovered = []
    monkeypatch.setattr(F, "_mark_recovered",
                        lambda *a, **k: recovered.append(a) or True)

    F.store.save_payment("pay_c", {"payment_id": "pay_c", "amount": 59976.0,
                                   "status": "pending", "order_id": "order_1",
                                   "trail": []})
    resp = _succeed("pay_c")
    assert resp.get_json()["status"] == "paid_clean"
    assert recovered == [], "a sale must not be recorded as a rescue"
    rec = F.store.get_payment("pay_c")
    assert rec["status"] == "paid"
    assert not rec.get("recovered_amount")


def test_a_case_that_failed_first_is_still_a_recovery(monkeypatch):
    monkeypatch.setattr(F, "razorpay_client", _StubRzp())
    recovered = []
    monkeypatch.setattr(F, "_mark_recovered",
                        lambda *a, **k: recovered.append(a) or True)

    F.store.save_payment("pay_x", {"payment_id": "pay_x", "amount": 59976.0,
                                   "status": "recovering", "order_id": "order_1",
                                   "failure_code": "bank_declined", "trail": []})
    resp = _succeed("pay_x")
    assert resp.get_json()["status"] == "recovered"
    assert recovered, "a real recovery must still go through the recorder"


# ── C: the modal-hold JS exists (inline JS — source check only) ─────────────

def test_the_page_holds_pushes_while_the_razorpay_modal_is_open():
    from pathlib import Path
    text = Path(F.__file__).read_text()
    assert "_rzpModalOpen = true;\n    rzp.open();" in text
    assert "if (_rzpModalOpen) { _heldAgentPush = d; return; }" in text
    assert "_flushHeldPush();" in text          # delivered on dismiss
    assert "_rzpModalOpen = false; _heldAgentPush = null;" in text  # dropped on success
