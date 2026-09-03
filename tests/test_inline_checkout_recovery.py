"""The customer pays on the checkout page itself — does anything notice?

pay_hhltoul1s was recovered for real (INR 1,299 captured as pay_TXNanVOxjJXurT)
and the case sat in `awaiting_customer` until it timed out. Three things were
wrong at once, and each of these tests fails without its fix:

  1. Razorpay's inline `handler` callback POSTed nowhere.
  2. Recording a push outcome of "acted" claimed "the recovery watcher has it"
     when no watcher had been started for a link-less push.
  3. The watcher had no strategy that looked at the original checkout order —
     the one object that gets paid on that path.
"""
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "recovery_agent" / "frontend.py"
TEXT = SRC.read_text()


def _handler_body() -> str:
    i = TEXT.index("handler: function(r) {")
    return TEXT[i:TEXT.index("prefill:", i)]


def test_inline_success_is_reported_to_the_server():
    assert "/api/payment-succeeded" in _handler_body(), (
        "Razorpay's inline handler must tell the server the payment landed; "
        "painting the button green is not a record."
    )


def test_reported_success_is_verified_against_razorpay_not_trusted():
    i = TEXT.index("def payment_succeeded()")
    body = TEXT[i:TEXT.index("\n@app.route", i)]
    assert "payment.fetch(" in body, "must fetch the payment from Razorpay"
    assert 'pay.get("status") != "captured"' in body, "must require a captured status"
    assert "order_mismatch" in body, (
        "a real capture belonging to a different case must not be credited here"
    )


def test_acting_on_a_push_starts_a_watcher():
    i = TEXT.index("def _record_push_outcome(")
    body = TEXT[i:TEXT.index("\ndef _link_original_order_to_recovery", i)]
    acted = body[body.index('if action == "acted":'):]
    assert "_watch_for_recovery" in acted.split("if action ==")[0] + acted[:600], (
        'recording "acted" must start the watcher it hands over to'
    )


def test_watcher_checks_the_original_checkout_order():
    i = TEXT.index("def _watch_for_recovery(")
    body = TEXT[i:TEXT.index("\ndef deliver_page_push", i)]
    assert "original_order_id" in body and "order.payments(" in body, (
        "a link-less push sends the customer back to the original order; the "
        "watcher has to look at it"
    )


def test_every_recovery_route_goes_through_one_recorder():
    """Each route learned about the money differently and did a different subset
    of the work — which is how a case could read `recovered` while the agent was
    never told, and how another could be paid while the store never changed."""
    i = TEXT.index("def _watch_for_recovery(")
    body = TEXT[i:TEXT.index("\ndef deliver_page_push", i)]
    assert body.count("_mark_recovered(") == 3, "all three poll strategies"
    assert '"status"] = "recovered"' not in body, (
        "no strategy may hand-roll the recovered write"
    )

    i = TEXT.index("def _mark_recovered(")
    rec = TEXT[i:TEXT.index("\ndef _handoff_to_agent", i)]
    for required in ("_notify_agent_of_recovery(", "push_event(", "store.flush()"):
        assert required in rec, f"_mark_recovered must call {required}"


def test_mark_recovered_will_not_double_count():
    i = TEXT.index("def _mark_recovered(")
    rec = TEXT[i:TEXT.index("\ndef _handoff_to_agent", i)]
    assert 'sp.get("status") == "recovered"' in rec and "return False" in rec, (
        "the browser callback and the poller race on every successful recovery; "
        "the second one to arrive must be a no-op"
    )


def test_a_handoff_run_does_not_erase_the_cases_history():
    """`trail = []` at the top of every run meant the closing run deleted the
    `recovery_confirmed` entry that had triggered it, and the dashboard showed
    only the final rung of a multi-rung recovery."""
    i = TEXT.index("def emit_thought(")
    head = TEXT[TEXT.index("def run_agent_for_payment"):i]
    assert "trail = []" not in head, "the trail must be seeded from the existing case"
    assert 'get("trail")' in head
