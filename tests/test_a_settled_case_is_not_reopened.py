"""A settled case must not be reopened by a late page signal.

Live (pay_alobc8wt6): the customer paid the recovery link, the case closed as
recovered, and then the stale "Pay Now" / retry button still on the checkout
re-fired /api/payment-failed for the SAME payment_id. The handler guarded only
against a case still `recovering`, so a recovered/closed case fell through: it
reset the status, re-emitted [INIT] and the "why did you stop?" hold, and
started a fresh agent run on a case that was already done.

Asking a customer to pay again after they have paid is the worst thing this
system can do, so the failure and drop-reason ingestion points now refuse a
settled case the same way the push/hand-off path already did. The late signal
is still recorded on the trail for the audit; nothing else fires.
"""
import pytest

import recovery_agent.frontend as F

CUST = {"email": "s@example.com", "contact": "9363064502"}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from recovery_agent import state_store
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    state_store.StateStore.reset_instances()
    monkeypatch.setattr(F, "store", state_store.StateStore())
    yield
    state_store.StateStore.reset_instances()


def _settle(pid, outcome="recovered", status="recovered"):
    F.store.save_payment(pid, {
        "payment_id": pid, "amount": 2499.0, "status": status,
        "closed": {"outcome": outcome}, "customer": CUST, "trail": []})


def _capture_agent_starts(monkeypatch):
    started = []
    monkeypatch.setattr(F.socketio, "start_background_task",
                        lambda *a, **k: started.append(a))
    return started


def _fail(c, pid):
    return c.post("/api/payment-failed", json={
        "payment_id": pid, "amount": 2499.0, "failure_code": "BAD_REQUEST_ERROR",
        "failure_reason": "declined by the bank", "customer": CUST,
        "defer_agent": True})


def test_a_late_failure_does_not_reopen_a_recovered_case(monkeypatch):
    started = _capture_agent_starts(monkeypatch)
    events = []
    monkeypatch.setattr(F, "push_event",
                        lambda pid, ev, data: events.append((ev, data.get("step"))))
    _settle("pay_s1")

    body = _fail(F.app.test_client(), "pay_s1").get_json()

    assert body["status"] == "already_settled"
    assert F.store.get_payment("pay_s1")["status"] == "recovered", \
        "a recovered case was reset to recovering by a late failure"
    assert started == [], "a fresh agent run was started on a settled case"
    assert not any(step in ("init", "awaiting_reason") for _ev, step in events), \
        "the settled case re-emitted [INIT] / the 'why did you stop?' hold"


def test_the_late_signal_is_kept_on_the_trail(monkeypatch):
    """Refusing to reopen is not the same as dropping the signal: the append-only
    audit still records that a late failure arrived on a settled case."""
    _capture_agent_starts(monkeypatch)
    _settle("pay_s1b")
    _fail(F.app.test_client(), "pay_s1b")
    trail = F.store.get_payment("pay_s1b").get("trail", [])
    assert any(e.get("step") == "settled_no_reopen" for e in trail)


def test_a_closed_case_is_not_reopened_whatever_the_outcome(monkeypatch):
    """`closed` is terminal whatever the outcome: an escalated or unrecoverable
    case the agent explicitly finished is not reopened by a retry either."""
    started = _capture_agent_starts(monkeypatch)
    _settle("pay_s2", outcome="escalated", status="escalated")
    body = _fail(F.app.test_client(), "pay_s2").get_json()
    assert body["status"] == "already_settled"
    assert started == []


def test_a_late_drop_reason_does_not_restart_a_settled_case(monkeypatch):
    started = _capture_agent_starts(monkeypatch)
    _settle("pay_s3")
    body = F.app.test_client().post(
        "/api/drop-reason",
        json={"payment_id": "pay_s3", "code": "payment_kept_failing"}).get_json()
    assert body["status"] == "already_settled"
    assert started == []


def test_a_late_skip_does_not_restart_a_settled_case(monkeypatch):
    started = _capture_agent_starts(monkeypatch)
    _settle("pay_s4")
    F.store.update_payment("pay_s4", drop_reason_pending=True)
    body = F.app.test_client().post(
        "/api/drop-reason/skip", json={"payment_id": "pay_s4"}).get_json()
    assert body["status"] == "already_settled"
    assert started == []


def test_a_genuinely_new_failure_still_opens_a_case(monkeypatch):
    """The guard must catch only SETTLED cases. A first-time failure on an
    unknown payment is not refused; it opens a case as before."""
    _capture_agent_starts(monkeypatch)
    body = _fail(F.app.test_client(), "pay_new").get_json()
    assert body["status"] != "already_settled"
    assert F.store.get_payment("pay_new")["status"] == "recovering"
