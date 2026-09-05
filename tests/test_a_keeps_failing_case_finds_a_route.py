"""When the rail switch also fails, give the instrument TIME — don't loop on wait.

Live (pay_7ml7v2pea, INR 7,297): bank decline, customer said "the payment kept
failing", agent switched rails (link + in-page banner + email) — correct. The
rail didn't get paid either. On the wake the ladder said next_rung=alternate_path,
but the agent had no concrete route for it: the offer was refused (instrument,
not price) and voice is off, so it just called wait_for_customer again and again.

The ideal: schedule a timed retry (give the bank issue time to clear), which
advances the ladder AND hands the case to the batch, then wait. Two mechanics
had to change: retry_in_hours must record the rung that actually belongs to the
ladder it is on, and the briefing must name the retry as the alternate route.
"""
from __future__ import annotations

import json

import pytest

import recovery_agent.state_store as state_store
from recovery_agent.agent import ladder
from recovery_agent.agent import tools as T
from recovery_agent.state_store import StateStore


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    from recovery_agent import audit
    monkeypatch.setattr(audit, "record", lambda *a, **k: None)
    StateStore.reset_instances()
    yield
    StateStore.reset_instances()


def _retry(payment_id, hours=24.0):
    fn = getattr(T.retry_in_hours, "func", T.retry_in_hours)
    return json.loads(fn(payment_id=payment_id, hours=hours))


# ── 1. the retry advances the ladder it is actually on ──────────────────

def test_a_retry_on_a_keeps_failing_case_is_the_alternate_path():
    s = StateStore()
    s.save_payment("p", {
        "payment_id": "p", "amount": 7297.0, "status": "recovering",
        "customer": {"email": "a@b.com"},
        "drop_reason": {"code": "payment_kept_failing"},
        "ladder": {"rail_switch": {"at": "2026-09-04T00:00:00+00:00"}}})
    s.flush()
    assert _retry("p")["status"] == "scheduled"
    rec = s.get_payment("p")
    # The method ladder has no silent_retry rung, so the retry IS the
    # different route — recording it there lets the ladder settle instead of
    # sitting at alternate_path forever.
    assert ladder.climbed(rec, "alternate_path")
    assert not ladder.climbed(rec, "silent_retry")


def test_a_retry_on_a_funds_case_is_still_the_silent_first_rung():
    s = StateStore()
    s.save_payment("p", {
        "payment_id": "p", "amount": 999.0, "status": "recovering",
        "customer": {"email": "a@b.com"},
        "failure_reason": "insufficient funds in the account", "ladder": {}})
    s.flush()
    _retry("p", 48.0)
    rec = s.get_payment("p")
    assert ladder.climbed(rec, "silent_retry")
    assert not ladder.climbed(rec, "alternate_path")


def test_the_scheduled_retry_keeps_the_case_waiting_not_exhausted():
    """A retry on the clock means the case is waiting (in the batch), not a
    lost cause — escalation must not fire while it is pending."""
    s = StateStore()
    s.save_payment("p", {
        "payment_id": "p", "amount": 7297.0, "status": "recovering",
        "customer": {"email": "a@b.com"},
        "drop_reason": {"code": "payment_kept_failing"},
        "ladder": {"rail_switch": {"at": "2026-09-04T00:00:00+00:00"}}})
    s.flush()
    _retry("p")
    rec = s.get_payment("p")
    assert ladder.retry_pending(rec)
    assert not ladder.exhausted(rec)   # waiting on the retry, not finished


# ── 2. the briefing names the retry as the route ────────────────────────

def _facts(**over):
    f = {"known": True, "settled": False, "owed": 7297.0, "received": 0.0,
         "outstanding": 7297.0, "failure_kind": "method",
         "next_rung": "alternate_path", "climbed": ["rail_switch"],
         "ladder_rungs": ["rail_switch", "voice_call", "alternate_path"],
         "retry_pending": False}
    f.update(over)
    return f


def test_the_briefing_points_a_stuck_rail_switch_at_a_retry():
    from recovery_agent.agent.perception import as_briefing
    text = as_briefing(_facts())
    assert "retry_in_hours" in text
    assert "KEEPS FAILING" in text
    assert "do NOT mint a new one" in text or "not send one" in text


def test_the_retry_steer_is_gone_once_one_is_scheduled():
    from recovery_agent.agent.perception import as_briefing
    text = as_briefing(_facts(retry_pending=True))
    assert "ALREADY SCHEDULED" in text
    # it should not still be telling the agent to schedule one
    assert "schedule a retry with retry_in_hours" not in text


def test_a_dropoff_at_alternate_path_is_not_pushed_to_a_retry():
    """A price-sensitive drop-off is a different animal — a retry is not its
    alternate route, so the keeps-failing steer must not fire for it."""
    from recovery_agent.agent.perception import as_briefing
    text = as_briefing(_facts(failure_kind="dropoff"))
    assert "KEEPS FAILING" not in text


# ── 3. the checkout banner clears when the money lands elsewhere ─────────

def test_the_checkout_retires_the_banner_on_recovery():
    from recovery_agent import frontend
    src = frontend.PAY_PAGE
    assert 'data.event === "recovery_confirmed"' in src
    assert "agent-offer-banner" in src
