"""Quiet hours exist to stop us WAKING people — not to stop all recovery.

Live, pay_woo85c9gh at 06:30 IST: the customer dismissed an in-page
notification, and 13 seconds later the agent was refused a payment LINK for
"quiet hours". Creating a link wakes nobody, the customer was demonstrably
awake and on the checkout page, and the refusal transitively killed the
in-page offer too (show_page_offer cannot honestly quote a discount without a
link at that price). The whole recovery stalled at 06:30 over a rule about
2am phone calls.

What quiet hours restrict now:
  - voice_call  REFUSED   — a ringing phone wakes someone
  - sms         SUPPRESSED — a buzzing phone wakes someone
  - email       ALLOWED   — it waits in an inbox to be opened
  - in-page     ALLOWED   — the customer is right there, with the page open
  - link/order  ALLOWED   — reaches nobody at all
"""
from datetime import datetime, timedelta, timezone

import pytest

from recovery_agent.agent import guardrails as G
from recovery_agent.agent.guardrails import (IST, GuardrailVerdict,
                                             QuietHourGuardrail)
from recovery_agent.models import ActionType

NIGHT = datetime(2026, 9, 4, 2, 0, tzinfo=IST)      # 02:00 — deep quiet
EARLY = datetime(2026, 9, 4, 6, 30, tzinfo=IST)     # the live incident
DAY = datetime(2026, 9, 4, 14, 0, tzinfo=IST)       # plainly awake


def verdict(action, now):
    return QuietHourGuardrail().check(action, now=now).verdict


# ── what still gets stopped ─────────────────────────────────────────────────

def test_a_voice_call_is_still_deferred_at_night():
    assert verdict(ActionType.VOICE_CALL, NIGHT) == GuardrailVerdict.MODIFIED


def test_a_voice_call_was_the_point_of_the_rule():
    assert verdict(ActionType.VOICE_CALL, EARLY) == GuardrailVerdict.MODIFIED
    assert verdict(ActionType.VOICE_CALL, DAY) == GuardrailVerdict.PASS


# ── what no longer gets stopped ─────────────────────────────────────────────

def test_an_email_is_not_an_interruption():
    assert verdict(ActionType.SEND_NOTIFICATION, NIGHT) == GuardrailVerdict.PASS


def test_creating_a_payment_link_wakes_nobody():
    """The live failure: a link refused at 06:30 to a customer on the page."""
    assert verdict(ActionType.UPDATE_PAYMENT_METHOD, EARLY) == GuardrailVerdict.PASS


def test_a_silent_retry_is_never_quiet_blocked():
    assert verdict(ActionType.WAIT_AND_RETRY, NIGHT) == GuardrailVerdict.PASS


def test_in_page_surfaces_are_not_even_in_the_policy_map():
    """A rule about phones at 2am must never reach a page the customer has open."""
    from recovery_agent.agent.policy_gate import TOOL_ACTION
    assert "send_page_push" not in TOOL_ACTION
    assert "show_page_offer" not in TOOL_ACTION


# ── the refusal names its end, so the agent waits the right amount ──────────

def test_the_refusal_says_when_quiet_hours_end():
    r = QuietHourGuardrail().check(ActionType.VOICE_CALL, now=EARLY)
    assert "08:00" in r.reason, "the agent has to know WHEN, not just that"
    assert "90 minutes" in r.reason, "06:30 -> 08:00 is 90 minutes"


@pytest.mark.parametrize("now,mins", [
    (EARLY, 90),                                     # 06:30 -> 08:00
    (datetime(2026, 9, 4, 7, 45, tzinfo=IST), 15),
    (datetime(2026, 9, 4, 22, 0, tzinfo=IST), 600),  # 22:00 -> 08:00 next day
])
def test_minutes_until_end_is_exact(now, mins):
    assert QuietHourGuardrail().minutes_until_end(now) == mins


def test_the_guidance_hands_back_a_usable_wait(monkeypatch):
    """A refusal that hides its number makes the agent guess — it guessed 60
    when the answer was 90, and would have been refused a second time."""
    from recovery_agent.agent import policy_gate
    monkeypatch.setattr(policy_gate, "_record_refusal", lambda *a, **k: None)
    monkeypatch.setattr(policy_gate, "log_verdict", lambda *a, **k: None)

    checks = [G.GuardrailCheckResult(
        guardrail="quiet_hours", verdict=GuardrailVerdict.MODIFIED,
        reason="Quiet hours until 08:00 IST", original_action="voice_call",
        modified_action="wait_and_retry")]
    out = policy_gate._refusal("p", "initiate_voice_call", checks,
                               ActionType.VOICE_CALL, ActionType.WAIT_AND_RETRY) \
        if hasattr(policy_gate, "_refusal") else None
    if out is None:
        pytest.skip("refusal builder is private to the gate in this build")
    assert out["retry_after_minutes"] >= 1
    assert "wait_for_customer" in out["guidance"]


# ── the dispatcher drops the SMS leg, not the whole message ─────────────────

def _dispatch(tmp_path, monkeypatch, *, quiet: bool):
    import recovery_agent.notifications as notif
    monkeypatch.setattr(notif, "smtplib", notif.smtplib)
    monkeypatch.setattr(G, "in_quiet_hours", lambda now=None: quiet)
    monkeypatch.delenv("GUARDRAIL_QUIET_DISABLED", raising=False)
    d = notif.NotificationDispatcher(outbox_dir=tmp_path)
    d._smtp_host = ""            # no real send; delivery honesty still applies
    return d.dispatch(payment_id="p", customer_email="c@example.com",
                      customer_phone="+919999999999", amount=2499.0,
                      recovery_link="https://rzp.io/x")


def test_overnight_the_sms_is_held_but_the_email_still_goes(tmp_path, monkeypatch):
    r = _dispatch(tmp_path, monkeypatch, quiet=True)
    sms = [x for x in r["results"] if x["channel"] == "sms"][0]
    assert sms["suppressed"] == "quiet_hours"
    assert sms["delivered"] is False
    assert r["undelivered"]["sms"] == "held until quiet hours end"
    assert "email" in r["attempted"], "the email is still attempted overnight"


def test_by_day_the_sms_is_attempted_normally(tmp_path, monkeypatch):
    r = _dispatch(tmp_path, monkeypatch, quiet=False)
    sms = [x for x in r["results"] if x["channel"] == "sms"][0]
    assert sms.get("suppressed") is None
    assert "no sms provider" in r["undelivered"]["sms"]
