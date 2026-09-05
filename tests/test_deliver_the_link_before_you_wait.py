"""A link the customer never sees is waste — deliver it before waiting or re-minting.

Live (pay_iwvp2sx7h, INR 5,298): the agent minted a wallet link, then a
netbanking link, then called wait_for_customer — delivering NEITHER. It then
sat through the full 5-minute wait window before finally showing a banner and
sending an email on the wake. The customer saw nothing for five minutes and
three units of the 30-per-lifetime link quota were spent for one case.

A second link is only a different ROUTE once the first has reached the
customer; a wait only waits on something once there is something to act on.
Both are now refused while a created link is undelivered.
"""
from __future__ import annotations

import json

import pytest

import recovery_agent.state_store as state_store
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


def _case(**over):
    s = StateStore()
    rec = {"payment_id": "p", "amount": 5298.0, "status": "recovering",
           "customer": {"email": "a@b.com", "contact": "+919999999999"}}
    rec.update(over)
    s.save_payment("p", rec)
    s.flush()
    return s


def _wait():
    fn = getattr(T.wait_for_customer, "func", T.wait_for_customer)
    return json.loads(fn(payment_id="p", waiting_for="x",
                         expected_within_minutes=15))


# ── a fresh undelivered link blocks the next link and the wait ──────────

def test_a_second_link_is_refused_while_the_first_is_undelivered():
    s = _case(recovery_link_id="plink_1", link_awaiting_delivery=True)
    fn = getattr(T.generate_recovery_payment_link, "func",
                 T.generate_recovery_payment_link)
    r = json.loads(fn(payment_id="p", amount=5298.0, customer_email="a@b.com",
                      allowed_rails="netbanking"))
    assert r["status"] == "blocked"
    assert "already created" in r["reason"]
    assert "show_page_offer" in r["guidance"]


def test_waiting_is_refused_while_a_link_is_undelivered():
    _case(recovery_link_id="plink_1", link_awaiting_delivery=True)
    r = _wait()
    assert r["status"] == "blocked"
    assert "have not shown or sent it" in r["reason"]


def test_once_delivered_the_wait_goes_through():
    s = _case(recovery_link_id="plink_1", link_awaiting_delivery=True)
    T._mark_link_delivered("p")
    assert s.get_payment("p")["link_awaiting_delivery"] is False
    r = _wait()
    assert r["status"] == "ok"


def test_a_delivered_link_does_not_block_a_genuinely_new_one():
    """The ladder legitimately mints another link for a later rung — that is
    only blocked while the PREVIOUS one is still undelivered."""
    s = _case(recovery_link_id="plink_1", link_awaiting_delivery=False)
    assert T._undelivered_link("p") == ""


def test_a_case_with_no_link_yet_can_wait():
    """Waiting on a scheduled retry with no link outstanding is fine."""
    _case()
    r = _wait()
    assert r["status"] == "ok"


# ── the checkout no longer shows the "section below" ────────────────────

def test_the_waiting_panel_is_not_shown_on_the_checkout():
    from recovery_agent import frontend
    src = frontend.PAY_PAGE
    i = src.index('data.event === "waiting_for_customer"')
    block = src[i:i + 400]
    assert 'classList.add("visible")' not in block
    assert 'classList.remove("visible")' in block
