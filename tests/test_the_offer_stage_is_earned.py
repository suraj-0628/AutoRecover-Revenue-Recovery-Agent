"""Two live defects from pay_qttkbl2b3 (INR 24,990), 2026-09-04.

Inside ninety seconds the agent quoted "ui_offer" for 5%, emailed a payment
link, called check_payment_status forty seconds later, saw the money had not
arrived -- of course it had not -- then quoted "voice" for 20% and emailed a
SECOND, richer link. No voice call was ever placed.

Two separate controls failed:

1. `stage` is a string the model chooses, and it selects the discount ceiling.
   Nothing checked it against the ladder, so the agent quadrupled its own
   spending authority by typing a different word: INR 1,249.50 -> INR 4,998.

2. The frequency cap counts contacts per day but says nothing about spacing,
   so a whole day's allowance can be spent in a minute.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from recovery_agent.agent import tools as T
from recovery_agent.agent.guardrails import (FrequencyCapGuardrail,
                                             GuardrailVerdict)
from recovery_agent.models import ActionType, CustomerProfile, PaymentRecord


def _quote(stage, rec):
    with patch.object(T, "_live_record", return_value=rec):
        fn = getattr(T.get_recovery_offer, "func", T.get_recovery_offer)
        return json.loads(fn(amount=float(rec["amount"]), stage=stage,
                             payment_id=rec["payment_id"]))


def _dropoff(**over):
    rec = {"payment_id": "pay_qttkbl2b3", "amount": 24990.0,
           "status": "recovering", "failure_code": "customer_cancelled",
           "drop_reason": {"code": "better_price",
                           "label": "I found a better price elsewhere"},
           "ladder": {"offer": True}}
    rec.update(over)
    return rec


# ── 1. the ceiling cannot be raised by the thing that spends it ─────────

def test_asking_for_the_voice_ceiling_without_a_voice_call_gets_the_offer_one():
    """The live escalation. 20% is the NEGOTIATION ceiling — its whole point is
    that a person spoke to the customer first."""
    got = _quote("voice", _dropoff())
    assert got["discount_pct"] == 5.0, (
        f"agent bought a {got['discount_pct']}% ceiling with a word")
    assert got["discount_rupees"] == 1249.5


def test_the_voice_ceiling_is_available_once_the_call_has_happened():
    """Clamping must not break the rung it protects."""
    got = _quote("voice", _dropoff(ladder={"offer": True, "voice_call": True}))
    assert got["discount_pct"] == 20.0


def test_the_ordinary_offer_stage_is_untouched():
    got = _quote("ui_offer", _dropoff())
    assert got["discount_pct"] == 5.0


# ── 2. a day's allowance cannot be spent in a minute ────────────────────

def _profile(seconds_ago: int) -> CustomerProfile:
    now = datetime.now(timezone.utc)
    return CustomerProfile(customer_id="c1", payment_history=[
        PaymentRecord(payment_id="p1", amount=100.0, status="contact",
                      channel_used="email",
                      timestamp=now - timedelta(seconds=seconds_ago))])


def test_a_second_email_forty_seconds_later_is_refused():
    """Exactly the live gap. Nobody decides on a INR 24,990 purchase in forty
    seconds, so the second offer bought nothing the first had not."""
    g = FrequencyCapGuardrail(max_contacts_per_24h=12, min_gap_minutes=5)
    r = g.check(ActionType.SEND_NOTIFICATION, _profile(40),
                datetime.now(timezone.utc))
    assert r.verdict is GuardrailVerdict.BLOCKED
    assert "minute" in r.reason


def test_the_gap_expires_and_does_not_become_a_permanent_block():
    g = FrequencyCapGuardrail(max_contacts_per_24h=12, min_gap_minutes=5)
    r = g.check(ActionType.SEND_NOTIFICATION, _profile(9 * 60),
                datetime.now(timezone.utc))
    assert r.verdict is GuardrailVerdict.PASS


def test_the_gap_can_be_switched_off():
    """An operator who wants it off must be able to turn it off."""
    g = FrequencyCapGuardrail(max_contacts_per_24h=12, min_gap_minutes=0)
    r = g.check(ActionType.SEND_NOTIFICATION, _profile(5),
                datetime.now(timezone.utc))
    assert r.verdict is GuardrailVerdict.PASS
