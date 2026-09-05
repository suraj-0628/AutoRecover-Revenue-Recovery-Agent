"""A spin-up must not ambush the operator with yesterday's work.

A `wake_agent` job carries no notion of the restart that happened between it
being scheduled and it firing, so the daemon's first poll fired the whole
accumulated backlog at once. Live (2026-09-04): four stale cases woke together
seconds after start.sh, each climbing a rung and emailing a real customer about
a fifteen-minute window that had expired hours earlier.

Wrong twice over: the message is stale, and the live view is for what is
happening NOW — work nobody triggered has no business seizing it.

Parking is not dropping. History is kept, jobs are cancelled rather than lost,
and the case lands in the batch it belongs to.
"""
from __future__ import annotations

import pytest

from recovery_agent.agent.classify import classify


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from recovery_agent import state_store
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    state_store.StateStore.reset_instances()
    yield
    state_store.StateStore.reset_instances()


def _case(pid, status, **over):
    rec = {"payment_id": pid, "amount": 2499.0, "status": status,
           "failure_code": "customer_cancelled",
           "failure_reason": "Payment cancelled by customer",
           "customer": {"email": "c@example.com", "contact": "+919999999999"}}
    rec.update(over)
    return rec


def _store():
    from recovery_agent.state_store import StateStore
    return StateStore()


def test_open_cases_are_parked_and_their_wakes_cancelled():
    """The live failure: cases mid-flight at shutdown woke into the HUD."""
    from recovery_agent import daemon_worker as dw
    s = _store()
    for pid, status in (("pay_a", "recovering"), ("pay_b", "awaiting_customer"),
                        ("pay_c", "scheduled")):
        s.save_payment(pid, _case(pid, status))
    s.flush()

    assert dw.park_open_cases_on_boot() == 3
    s.refresh()
    for pid in ("pay_a", "pay_b", "pay_c"):
        rec = s.get_payment(pid)
        assert rec["status"] == "parked", f"{pid} still wakes into the live view"
        assert rec["parked_from"], "the case must remember what it was doing"


def test_a_parked_case_is_recoverable_from_its_batch():
    """Parking must not be a quiet way of dropping revenue."""
    from recovery_agent import daemon_worker as dw
    s = _store()
    s.save_payment("pay_d", _case("pay_d", "awaiting_customer"))
    s.flush()
    dw.park_open_cases_on_boot()
    s.refresh()
    assert classify(s.get_payment("pay_d")) == "dropoff", (
        "a parked case must land in a batch, or the money is simply lost")


def test_settled_and_closed_cases_are_left_alone():
    """Only work still in flight is parked."""
    from recovery_agent import daemon_worker as dw
    s = _store()
    s.save_payment("pay_won", _case("pay_won", "recovered",
                                    recovered_amount=2499.0))
    s.save_payment("pay_done", _case("pay_done", "awaiting_customer",
                                     closed={"outcome": "escalated"}))
    s.flush()
    assert dw.park_open_cases_on_boot() == 0
    s.refresh()
    assert s.get_payment("pay_won")["status"] == "recovered"
    assert s.get_payment("pay_done")["status"] == "awaiting_customer"


def test_parking_runs_before_the_first_poll():
    """Order is the whole point: a backlogged wake must not beat it."""
    import inspect
    from recovery_agent import daemon_worker as dw
    src = inspect.getsource(dw.daemon_loop)
    assert "park_open_cases_on_boot()" in src
    assert src.index("park_open_cases_on_boot()") < src.index("_process_due_jobs()"), \
        "a due job would fire before the backlog was parked"
