"""Escalation is rung 6, not the answer to a blocked tool.

`pay_rxfo0unaq`: the approval gate refused a discounted link at rung 2 and the
agent filed a human ticket for a INR 1,19,970 order that had been contacted
exactly once — one silent in-page push. No email, no call, no offer. A human
queue is the last resort; it had become the fallback for any refusal.
"""
import os
from datetime import datetime, timedelta, timezone

import pytest

from recovery_agent.agent import ladder


def case(**over) -> dict:
    rec = {
        "payment_id": "pay_t", "amount": 2499.0, "failure_code": "customer_cancelled",
        "customer": {"email": "a@b.com", "name": "A", "contact": "9000000000"},
        "ladder": {},
    }
    rec.update(over)
    return rec


def climb(rec: dict, *rungs: str) -> dict:
    rec["ladder"] = dict(rec.get("ladder") or {})
    for r in rungs:
        rec["ladder"][r] = {"at": datetime.now(timezone.utc).isoformat(), "detail": ""}
    return rec


# ── order ───────────────────────────────────────────────────────────────

def test_the_first_rung_is_the_silent_push():
    assert ladder.next_rung(case())["rung"] == "page_push"


def test_the_offer_follows_the_push():
    assert ladder.next_rung(climb(case(), "page_push"))["rung"] == "offer"


def test_nothing_is_left_once_every_possible_rung_is_climbed(monkeypatch):
    monkeypatch.setenv("VOICE_CALLS_ENABLED", "1")
    rec = climb(case(), "page_push", "offer", "voice_call", "post_call_email",
                "alternate_path")
    assert ladder.exhausted(rec)
    assert ladder.next_rung(rec) is None


# ── availability, not silent completion ─────────────────────────────────

def test_voice_is_unavailable_rather_than_done_when_calls_are_off(monkeypatch):
    monkeypatch.delenv("VOICE_CALLS_ENABLED", raising=False)
    st = ladder.state(climb(case(), "page_push", "offer"))
    unavailable = {r["rung"] for r in st["unavailable"]}
    assert "voice_call" in unavailable, "a switched-off channel is not a climbed rung"
    assert "voice_call" not in st["climbed"]
    assert any("switched off" in r["why_not"] for r in st["unavailable"])


def test_a_call_that_cannot_be_made_does_not_block_escalation_forever(monkeypatch):
    """Otherwise a deployment with voice off could never escalate anything."""
    monkeypatch.delenv("VOICE_CALLS_ENABLED", raising=False)
    rec = climb(case(), "page_push", "offer", "alternate_path")
    assert ladder.exhausted(rec)


def test_a_customer_with_no_contact_details_cannot_be_emailed():
    rec = case(customer={}, customer_email="", customer_phone="")
    st = ladder.state(rec)
    assert {r["rung"] for r in st["unavailable"]} >= {"offer", "alternate_path"}


# ── the 15-minute gap before a call ─────────────────────────────────────

def test_a_call_is_not_allowed_immediately_after_the_offer():
    rec = climb(case(), "page_push", "offer")
    assert ladder.voice_wait_remaining_minutes(rec) > 14


def test_the_call_opens_up_once_the_offer_has_had_its_window():
    rec = case()
    old = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    rec["ladder"] = {"page_push": {"at": old}, "offer": {"at": old}}
    assert ladder.voice_wait_remaining_minutes(rec) == 0


def test_no_wait_applies_before_an_offer_has_gone_out():
    assert ladder.voice_wait_remaining_minutes(climb(case(), "page_push")) == 0


# ── the one carve-out ───────────────────────────────────────────────────

@pytest.mark.parametrize("code", ["fraud_suspected", "risk_declined",
                                  "dispute_raised", "chargeback"])
def test_do_not_pursue_cases_skip_the_ladder(code):
    assert ladder.pursuit_barred(case(failure_code=code))


def test_a_dead_card_is_not_a_do_not_pursue_case():
    """The recovery for a dead instrument is another rail, not a human ticket."""
    assert not ladder.pursuit_barred(case(failure_code="card_expired"))
    assert not ladder.pursuit_barred(case(failure_code="customer_cancelled"))


def test_an_opted_out_customer_is_not_chased():
    assert ladder.pursuit_barred(case(opted_out=True))


# ── the tool actually enforces it ───────────────────────────────────────

def test_escalation_is_refused_while_rungs_remain(tmp_path, monkeypatch):
    monkeypatch.setenv("RECOVERY_DATA_DIR", str(tmp_path))
    import json
    from recovery_agent.agent.tools import TOOLS_BY_NAME
    out = json.loads(TOOLS_BY_NAME["escalate_to_human"].invoke(
        {"payment_id": "pay_never_seen", "reason": "a tool refused me"}))
    assert out["status"] == "blocked"
    assert out["next_rung"] == "page_push"
    assert "final step" in out["reason"]


def test_the_refusal_names_what_to_do_instead():
    import json
    from recovery_agent.agent.tools import TOOLS_BY_NAME
    out = json.loads(TOOLS_BY_NAME["escalate_to_human"].invoke(
        {"payment_id": "pay_never_seen_2", "reason": "blocked"}))
    assert out["next_step"], "a refusal must name the next move"
    assert "guidance" in out
