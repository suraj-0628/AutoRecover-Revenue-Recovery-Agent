"""What the customer said must change the PLAN, not just the advice.

The stated reason picked the ladder and then had no say in where on it to
start. `offer_ok` could only ever VETO a discount; nothing could reach for
one. So a customer who answered "I found a better price elsewhere" — the one
answer a discount exists for — was met with rung one of the drop-off ladder:
a page push reading "You left this payment incomplete. Can we help?" and a
five-minute wait, while they compared prices in another tab (pay_p9c6jv4x1).

The briefing above that decision said, correctly, "This is the one case a
discount genuinely answers... offer what policy allows, promptly". The advice
was right and the plan ignored it, and the agent follows the plan.
"""
import pytest

from recovery_agent.agent import ladder

CUST = {"email": "s@example.com", "contact": "9363064502"}


def _case(**over):
    rec = {"payment_id": "pay_t1", "amount": 5196.0, "status": "recovering",
           "failure_code": "customer_cancelled", "customer": CUST}
    rec.update(over)
    return rec


def _rungs(rec):
    return [k for k, _ in ladder.rungs_for(rec)]


def test_a_price_objection_opens_on_the_offer():
    rec = _case(drop_reason={"code": "better_price"})
    assert _rungs(rec)[0] == "offer"
    assert "page_push" not in _rungs(rec), \
        "asking 'can we help?' spends the moment they are comparing in"


def test_a_repeated_failure_opens_on_a_different_rail():
    """They have told us the instrument is broken; saying so again is noise."""
    rec = _case(drop_reason={"code": "payment_kept_failing"})
    assert _rungs(rec)[0] == "rail_switch"


def test_an_empty_account_still_opens_on_the_quiet_retry():
    """No money is a timing problem — unchanged, and it was already right."""
    rec = _case(drop_reason={"code": "no_balance"})
    assert _rungs(rec)[0] == "silent_retry"


def test_a_reason_that_names_no_rung_leaves_the_ladder_alone():
    rec = _case(drop_reason={"code": "just_browsing"})
    assert _rungs(rec)[0] == "page_push"


def test_no_reason_at_all_leaves_the_ladder_alone():
    assert _rungs(_case())[0] == "page_push"


def test_skipped_rungs_are_dropped_not_recorded_as_climbed():
    """Marking them climbed would be a lie the rest of the ladder reasons from,
    and would show up in the escalation ticket as contact that never happened."""
    rec = _case(drop_reason={"code": "better_price"})
    assert ladder.state(rec)["climbed"] == []


def test_the_offer_is_what_the_agent_is_pointed_at():
    rec = _case(drop_reason={"code": "better_price"})
    assert (ladder.next_rung(rec) or {}).get("rung") == "offer"


def test_an_unknown_entry_rung_does_not_empty_the_ladder(monkeypatch):
    """A typo in a spec must degrade to the normal ladder, not to no plan."""
    import recovery_agent.drop_reasons as dr
    monkeypatch.setattr(dr, "get",
                        lambda code: {"kind": "dropoff", "entry_rung": "nonsense"})
    rec = _case(drop_reason={"code": "better_price"})
    assert _rungs(rec)[0] == "page_push"


def test_a_reason_cannot_rewind_to_a_rung_already_used():
    """Entering at a rung already climbed must not re-offer it."""
    rec = _case(drop_reason={"code": "better_price"},
                ladder={"offer": {"at": "2026-09-04T10:00:00+00:00"}})
    assert "offer" not in [r["rung"] for r in ladder.state(rec)["remaining"]]
