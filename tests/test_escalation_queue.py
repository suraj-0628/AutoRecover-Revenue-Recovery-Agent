"""Stage 6 — the batch a human works.

The previous escalation wrote `{ticket_id, payment_id, reason, status}` to a
loose JSON file. Whoever picked one up had no amount, no way to contact the
customer, no record of what had been tried and no link to send — so the queue was
somewhere cases went to be forgotten. These tests pin the properties that make it
actionable instead.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from recovery_agent import escalation_queue as q


@pytest.fixture(autouse=True)
def isolated_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "_QUEUE_PATH", tmp_path / "queue.jsonl")


def _add(payment_id="pay_1", **kw):
    kw.setdefault("reason", "customer did not pay")
    kw.setdefault("amount", 2499.0)
    return q.enqueue(payment_id=payment_id, **kw)


# ── a ticket must be actionable on its own ──────────────────────────────────

def test_a_ticket_carries_what_a_human_needs_to_act():
    t = _add(
        customer={"email": "a@b.com", "name": "Suraj", "contact": "+919999999999"},
        attempts=[{"action": "send_page_push", "result": "dismissed"},
                  {"action": "send_recovery_notification", "result": "ok"}],
        customer_signals=["in-page notification dismissed after 4.2s"],
        offer={"discount_pct": 5, "payable_rupees": 2374.05},
        recovery_link="https://rzp.io/rzp/abc",
        failure_code="card_expired",
    )
    assert t["amount"] == 2499.0
    assert t["customer"]["contact"] == "+919999999999"
    assert len(t["attempts"]) == 2
    assert t["customer_signals"] == ["in-page notification dismissed after 4.2s"]
    assert t["offer"]["payable_rupees"] == 2374.05
    assert t["recovery_link"].startswith("https://")
    assert t["failure_code"] == "card_expired"


def test_empty_customer_fields_are_dropped_not_shown_blank():
    t = _add(customer={"email": "a@b.com", "name": "", "contact": None})
    assert t["customer"] == {"email": "a@b.com"}


# ── ticket ids ──────────────────────────────────────────────────────────────

def test_ticket_id_is_never_malformed_for_a_missing_payment_id():
    """The old scheme produced `ESC--...`; several are sitting in data/escalations."""
    for bad in ("", None, "   "):
        tid = q.make_ticket_id(bad)
        assert tid.startswith("ESC-") and "--" not in tid
        assert len(tid.split("-")[1]) == 8


def test_ticket_ids_are_unique_across_rapid_calls():
    ids = {q.make_ticket_id(f"pay_{i}") for i in range(50)}
    assert len(ids) == 50


# ── the queue must not fill with duplicates ─────────────────────────────────

def test_re_escalating_an_open_case_returns_the_same_ticket():
    first = _add("pay_dup")
    again = q.enqueue(payment_id="pay_dup", reason="tried again")
    assert again["ticket_id"] == first["ticket_id"]
    assert len(q.list_tickets()) == 1


def test_a_case_can_be_escalated_again_after_it_is_resolved():
    first = _add("pay_re")
    q.resolve(first["ticket_id"], outcome="called, no answer")
    second = q.enqueue(payment_id="pay_re", reason="failed again later")
    assert second["ticket_id"] != first["ticket_id"]
    assert len(q.list_tickets(status="open")) == 1


# ── resolution is append-only ───────────────────────────────────────────────

def test_resolving_appends_and_never_rewrites_history():
    t = _add("pay_res")
    q.resolve(t["ticket_id"], outcome="paid by UPI over the phone", by="asha")

    rows = [json.loads(l) for l in q._QUEUE_PATH.read_text().splitlines() if l.strip()]
    assert len(rows) == 2, "the original entry was overwritten"
    assert rows[0]["status"] == "open" and rows[1]["status"] == "resolved"
    assert rows[1]["outcome"] == "paid by UPI over the phone"
    assert rows[1]["resolved_by"] == "asha"
    assert q.list_tickets(status="open") == []


def test_resolving_an_unknown_ticket_returns_none():
    assert q.resolve("ESC-nope-1") is None


# ── the queue survives a damaged file ───────────────────────────────────────

def test_a_torn_line_does_not_hide_the_rest_of_the_queue():
    _add("pay_a")
    with q._QUEUE_PATH.open("a") as f:
        f.write('{"ticket_id": "broken", "amount":\n')      # truncated write
    _add("pay_b")
    ids = {t["payment_id"] for t in q.list_tickets()}
    assert ids == {"pay_a", "pay_b"}


def test_summary_reports_open_value_for_triage():
    _add("pay_x", amount=2499.0)
    _add("pay_y", amount=1500.0)
    s = q.summary()
    assert s["open"] == 2 and s["open_value"] == 3999.0
    q.resolve(q.list_tickets()[0]["ticket_id"])
    assert q.summary()["open"] == 1


def test_missing_queue_file_is_an_empty_queue_not_an_error():
    assert q.list_tickets() == []
    assert q.summary()["open"] == 0
