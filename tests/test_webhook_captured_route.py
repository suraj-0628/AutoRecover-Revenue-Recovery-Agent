"""The real-webhook capture path must feed the learning loop.

`webhook_forward`'s `payment.captured` branch used to write `status` and
`recovered_amount` directly — skipping `_mark_recovered`, the single recorder
every other route uses. That skipped the already-recovered guard, skipped the
order cross-reference, and never called `_notify_agent_of_recovery`: on the
production path the agent was never told it worked, so it never recorded what
actually recovered the money.

These tests drive the real Flask route with `_mark_recovered` recorded (so no
agent run — and therefore no LLM call — is spawned from a test), against an
isolated temp store so the route's `flush()` never touches live data files.
"""
from pathlib import Path

import pytest

import recovery_agent.state_store as state_store
import recovery_agent.frontend as F
from recovery_agent.frontend import _how_it_arrived


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    state_store.StateStore.reset_instances()
    monkeypatch.setattr(F, "store", state_store.StateStore())
    yield
    state_store.StateStore.reset_instances()


def _post_captured(notes):
    return F.app.test_client().post("/api/webhook-forward", json={
        "event": "payment.captured",
        "payload": {"payment": {"entity": {
            "id": "pay_new123",
            "amount": 249900,          # paise on the wire
            "notes": notes,
        }}},
    })


def test_a_webhook_capture_goes_through_the_one_recorder(monkeypatch):
    calls = []
    monkeypatch.setattr(F, "_mark_recovered",
                        lambda pid, amount, rzp_id, seconds, how:
                        calls.append((pid, amount, rzp_id, how)) or True)

    F.store.save_payment("pay_orig1", {"payment_id": "pay_orig1",
                                       "amount": 2499.0, "status": "recovering",
                                       "trail": []})
    resp = _post_captured({"original_payment": "pay_orig1"})
    assert resp.get_json()["status"] == "captured"
    assert calls == [("pay_orig1", 2499.0, "pay_new123",
                      "recovery link paid (webhook payment.captured)")]


def test_the_webhook_how_strings_attribute_correctly():
    """The `how` string becomes the agent's permanent lesson via
    _how_it_arrived — a wrong phrase here is a false memory later."""
    assert _how_it_arrived("recovery link paid (webhook payment.captured)") \
        == "the customer paid the recovery payment link you sent"
    assert _how_it_arrived("original payment captured (webhook payment.captured)") \
        == "the original payment was captured after all"


def test_an_already_recovered_case_is_not_double_counted():
    """No stub here — the real _mark_recovered must refuse the second write."""
    F.store.save_payment("pay_done1", {"payment_id": "pay_done1",
                                       "amount": 2499.0, "status": "recovered",
                                       "recovered_amount": 2499.0, "trail": []})
    resp = _post_captured({"original_payment": "pay_done1"})
    assert resp.get_json()["status"] == "already_recovered"
    assert F.store.get_payment("pay_done1")["recovered_amount"] == 2499.0


def test_no_route_hand_rolls_the_recovered_write():
    """The captured branch may not write status directly any more."""
    text = Path(F.__file__).read_text()
    i = text.index('elif event == "payment.captured"')
    # Slice to the branch's FINAL return — the clean-sale guard returns
    # earlier for first-try sales, and that early return must not hide the
    # recorder from this check.
    body = text[i:text.index('return jsonify({"status": status', i)]
    assert '"status"] = "recovered"' not in body
    assert "_mark_recovered(" in body
