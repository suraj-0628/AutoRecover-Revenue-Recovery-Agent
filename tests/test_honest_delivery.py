"""A file write is not a contact.

`dispatch()` used to return "dispatched" unconditionally: an .eml sitting
beside a failed SMTP send, or an SMS payload no provider ever carries, was
reported upstream as contact — and `send_recovery_notification` recorded the
offer rung as climbed on the strength of it. Measured from the dispatch log at
the time of the fix: 49 of 87 "dispatched" emails were .eml files only.

Every send here is stubbed; nothing touches a real SMTP server or provider.
"""
import json

import pytest

import recovery_agent.notifications as notifications
import recovery_agent.state_store as state_store
from recovery_agent.notifications import NotificationDispatcher
from recovery_agent.state_store import StateStore


# ── the dispatcher tells the truth per channel ──────────────────────────────

def _dispatcher(tmp_path):
    d = NotificationDispatcher(outbox_dir=tmp_path)
    d._smtp_host = ""                   # whatever .env says, this test has no SMTP
    return d


def test_no_smtp_means_not_delivered_even_though_the_eml_exists(tmp_path):
    r = _dispatcher(tmp_path).dispatch(
        payment_id="pay_x", customer_email="c@example.com",
        customer_phone="+919999999999", amount=2499.0,
        recovery_link="https://rzp.io/x")
    assert r["status"] == "not_delivered"
    assert r["channels"] == []
    assert sorted(r["attempted"]) == ["email", "sms"]
    assert "email" in r["undelivered"] and "sms" in r["undelivered"]
    assert list((tmp_path / "emails").glob("*.eml")), \
        "the inspectable artifact is still written"


def test_sms_is_never_reported_as_delivered(tmp_path):
    r = _dispatcher(tmp_path).dispatch(
        payment_id="pay_x", customer_phone="+919999999999", amount=100.0)
    assert r["status"] == "not_delivered"
    sms = r["results"][0]
    assert sms["simulated"] is True and sms["delivered"] is False


def test_a_real_smtp_send_is_delivered(tmp_path, monkeypatch):
    sent = []

    class _FakeSMTP:
        def __init__(self, host, port, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, context=None):
            pass

        def login(self, user, password):
            pass

        def sendmail(self, frm, to, msg):
            sent.append(to)

    monkeypatch.setattr(notifications.smtplib, "SMTP", _FakeSMTP)
    d = NotificationDispatcher(outbox_dir=tmp_path)
    d._smtp_host = "smtp.test.invalid"
    r = d.dispatch(payment_id="pay_x", customer_email="c@example.com",
                   customer_phone="+919999999999", amount=2499.0)
    assert sent == [["c@example.com"]]
    assert r["status"] == "dispatched"
    assert r["channels"] == ["email"], "sms must not ride along as delivered"


def test_a_failed_smtp_send_names_its_error(tmp_path, monkeypatch):
    class _BrokenSMTP:
        def __init__(self, host, port, timeout=None):
            raise ConnectionRefusedError("connection refused by smtp host")

    monkeypatch.setattr(notifications.smtplib, "SMTP", _BrokenSMTP)
    d = NotificationDispatcher(outbox_dir=tmp_path)
    d._smtp_host = "smtp.test.invalid"
    r = d.dispatch(payment_id="pay_x", customer_email="c@example.com", amount=1.0)
    assert r["status"] == "not_delivered"
    assert "refused" in r["undelivered"]["email"]


# ── the tool only climbs the rung on genuine delivery ───────────────────────

@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    state_store.StateStore.reset_instances()
    yield StateStore()
    state_store.StateStore.reset_instances()


class _StubDispatcher:
    result = None

    def __init__(self, *a, **k):
        pass

    def dispatch(self, **kwargs):
        return dict(_StubDispatcher.result)


def _send(payment_id):
    from recovery_agent.agent.tools import send_recovery_notification
    return json.loads(send_recovery_notification.invoke({
        "payment_id": payment_id,
        "customer_email": "c@example.com",
        "customer_phone": "+919999999999",
        "message": "5% off if you finish your order",
        "payment_link": "https://rzp.io/offer",
        "amount": 2374.05,
    }))


def test_an_undelivered_message_does_not_climb_the_rung(isolated_store, monkeypatch):
    isolated_store.save_payment("pay_a", {"payment_id": "pay_a", "amount": 2499.0,
                                          "status": "failed", "trail": []})
    monkeypatch.setattr(notifications, "NotificationDispatcher", _StubDispatcher)
    _StubDispatcher.result = {"status": "not_delivered", "channels": [],
                              "attempted": ["email", "sms"],
                              "undelivered": {"email": "smtp not configured"},
                              "results": []}
    r = _send("pay_a")
    assert r["status"] == "error"
    assert "smtp not configured" in r["message"]
    assert "ladder" not in (isolated_store.get_payment("pay_a") or {}) or \
        "offer" not in (isolated_store.get_payment("pay_a").get("ladder") or {})


def test_a_delivered_message_climbs_the_rung(isolated_store, monkeypatch):
    isolated_store.save_payment("pay_b", {"payment_id": "pay_b", "amount": 2499.0,
                                          "status": "failed", "trail": []})
    monkeypatch.setattr(notifications, "NotificationDispatcher", _StubDispatcher)
    _StubDispatcher.result = {"status": "dispatched", "channels": ["email"],
                              "attempted": ["email", "sms"],
                              "undelivered": {"sms": "no provider"},
                              "results": []}
    r = _send("pay_b")
    assert r["status"] == "ok"
    assert r["channels"] == ["email"]
    assert "offer" in (isolated_store.get_payment("pay_b").get("ladder") or {})
