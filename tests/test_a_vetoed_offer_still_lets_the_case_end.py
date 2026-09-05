"""A customer's "no discount fixes this" answer must not trap the case open.

Live shape (no_balance cases, 2026-09-04): drop reasons with offer_ok False
made get_recovery_offer refuse permanently, while the ladder still listed the
offer rung under "remaining" and rung 5 could only be recorded after a
climbed offer. exhausted() therefore stayed false forever — and both
escalate_to_human and close_case refuse while it is — so the case could not
end by any path except the customer spontaneously paying.
"""
from __future__ import annotations

import pytest

import recovery_agent.state_store as state_store
from recovery_agent.agent import ladder
from recovery_agent.state_store import StateStore


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    # record_rung double-writes to the append-only audit log, which defaults
    # to the live data/ directory — a test must not leave rows there.
    from recovery_agent import audit
    monkeypatch.setattr(audit, "record", lambda *a, **k: None)
    StateStore.reset_instances()
    yield
    StateStore.reset_instances()


def _no_balance(**over):
    rec = {"payment_id": "pay_veto", "amount": 2499.0, "status": "recovering",
           "customer": {"email": "customer@example.com"},
           "drop_reason": {"code": "no_balance",
                           "label": "I didn't have enough balance"}}
    rec.update(over)
    return rec


def test_a_vetoed_offer_is_unavailable_not_remaining():
    st = ladder.state(_no_balance())
    assert "offer" not in [r["rung"] for r in st["remaining"]]
    why = [r for r in st["unavailable"] if r["rung"] == "offer"]
    assert why and "discount" in why[0]["why_not"]


def test_the_ladder_can_exhaust_without_the_vetoed_offer():
    rec = _no_balance(
        ladder={"silent_retry": {"at": "2026-09-04T00:00:00+00:00"},
                "alternate_path": {"at": "2026-09-04T01:00:00+00:00"}})
    # page_push impossible (no live checkout page), offer vetoed — nothing
    # is left that could still be tried.
    assert ladder.exhausted(rec)


def test_rung_five_is_reachable_when_the_offer_never_can_be():
    s = StateStore()
    s.save_payment("pay_veto", _no_balance(
        ladder={"silent_retry": {"at": "2026-09-04T00:00:00+00:00"}}))
    ladder.record_action("pay_veto", "retry:24h")
    ladder.record_action("pay_veto", "notify:email:full_price")
    assert ladder.climbed(s.get_payment("pay_veto"), "alternate_path")


def test_an_offer_that_is_merely_unclimbed_still_gates_rung_five():
    """The normal path is untouched: no veto, the offer is still owed — a
    second distinct action is the ladder being climbed, not rung 5."""
    s = StateStore()
    rec = _no_balance(drop_reason=None,
                      failure_reason="insufficient funds in account")
    s.save_payment("pay_veto", rec)
    ladder.record_action("pay_veto", "retry:24h")
    ladder.record_action("pay_veto", "notify:email:full_price")
    assert not ladder.climbed(s.get_payment("pay_veto"), "alternate_path")


def test_a_ladder_with_no_offer_rung_can_still_reach_rung_five():
    """The transient ladder never had an offer rung, so the old climbed-offer
    gate made rung 5 unreachable for it — the same trap by another door."""
    s = StateStore()
    rec = _no_balance(drop_reason=None,
                      failure_reason="gateway timeout, please retry")
    s.save_payment("pay_veto", rec)
    assert not ladder.has_rung(rec, "offer")
    ladder.record_action("pay_veto", "retry:2h")
    ladder.record_action("pay_veto", "notify:email:retry_done")
    assert ladder.climbed(s.get_payment("pay_veto"), "alternate_path")
