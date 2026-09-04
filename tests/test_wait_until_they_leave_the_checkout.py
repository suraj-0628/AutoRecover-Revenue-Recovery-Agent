"""The agent decides after the customer leaves the modal, not during.

A bank decline leaves the customer standing in Razorpay's own screen with its
other rails in front of them — the cheapest recovery there is, costing no
link, no discount and no message. Starting the agent on the decline meant it
reasoned to a conclusion and pushed a notification underneath the iframe they
were still using (pay_glpfpyq90), having never asked why they stopped.

Holding the notification was not enough. A decision taken before the reason is
known cannot use the reason.
"""
import json
import re
from pathlib import Path

import pytest

import recovery_agent.frontend as F

PAY_PAGE = F.PAY_PAGE


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from recovery_agent import state_store
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    state_store.StateStore.reset_instances()
    monkeypatch.setattr(F, "store", state_store.StateStore())
    yield
    state_store.StateStore.reset_instances()


# ── the checkout holds the agent while the modal is open ────────────────

def _handler(name: str) -> str:
    i = PAY_PAGE.index(name)
    return PAY_PAGE[i:i + 1400]


def test_a_decline_does_not_start_the_agent():
    """triggerRecovery's second argument is deferAgent; it must be true here."""
    body = _handler('rzp.on("payment.failed"')
    call = re.search(r"triggerRecovery\(\{[^}]*\}\s*,\s*(\w+)\)", body)
    assert call, "payment.failed no longer calls triggerRecovery"
    assert call.group(1) == "true", \
        "the agent must wait for the customer to leave the checkout"


def test_closing_after_a_failure_is_what_asks_why():
    body = _handler("_closedAfterFailure) {")
    assert "askWhyTheyStopped()" in body
    assert re.search(r"triggerRecovery\(\{[^}]*\}\s*,\s*true\)", body)


# ── a hold must never be permanent ──────────────────────────────────────

def test_the_hold_is_stamped_so_it_can_expire(monkeypatch):
    monkeypatch.setattr(F.socketio, "start_background_task", lambda *a, **k: None)
    F.app.test_client().post("/api/payment-failed", json={
        "payment_id": "pay_h1", "amount": 2499.0,
        "failure_code": "bank_declined", "failure_reason": "declined",
        "customer": {"email": "s@example.com"}, "defer_agent": True})
    rec = F.store.get_payment("pay_h1") or {}
    assert rec.get("drop_reason_pending") is True
    assert rec.get("drop_reason_pending_at"), \
        "an unstamped hold cannot be swept, so a closed tab strands the case"


def test_the_daemon_releases_a_hold_the_customer_never_answered(monkeypatch):
    """Closing the tab killed the browser timer that was the ONLY release."""
    from datetime import datetime, timedelta, timezone

    from recovery_agent import daemon_worker as D

    stale = (datetime.now(timezone.utc)
             - timedelta(seconds=D.DROP_REASON_GRACE + 60)).isoformat()
    F.store.save_payment("pay_h2", {
        "payment_id": "pay_h2", "amount": 999.0, "status": "recovering",
        "drop_reason_pending": True, "drop_reason_pending_at": stale})
    F.store.flush()

    posted = []
    monkeypatch.setattr(D, "StateStore", None, raising=False)
    monkeypatch.setattr(D.urllib.request, "urlopen",
                        lambda req, timeout=None: posted.append(
                            json.loads(req.data.decode())) or _Resp())
    assert D._release_stranded_holds() == 1
    assert posted == [{"payment_id": "pay_h2"}]


def test_a_fresh_hold_is_left_alone(monkeypatch):
    """The customer is still reading the question; do not answer it for them."""
    from datetime import datetime, timezone

    from recovery_agent import daemon_worker as D

    F.store.save_payment("pay_h3", {
        "payment_id": "pay_h3", "amount": 999.0, "status": "recovering",
        "drop_reason_pending": True,
        "drop_reason_pending_at": datetime.now(timezone.utc).isoformat()})
    F.store.flush()
    monkeypatch.setattr(D.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("swept a live hold"))
    assert D._release_stranded_holds() == 0


class _Resp:
    def __enter__(self): return self
    def __exit__(self, *a): return False
