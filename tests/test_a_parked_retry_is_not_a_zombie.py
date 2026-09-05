"""Parking a case must not file it under a retry that will never fire.

park_open_cases_on_boot cancels the case's pending jobs in the jobs table,
but classify() and ladder.retry_pending() read the record's `scheduled_job`
snapshot — which still said "scheduled". The case landed in awaiting_retry
("Nothing to do until it fires", runnable: False) waiting on a retry nothing
would ever fire: not on the HUD, not workable in a batch, forever.
"""
from __future__ import annotations

import pytest

import recovery_agent.state_store as state_store
from recovery_agent.agent import ladder
from recovery_agent.agent.classify import classify
from recovery_agent.state_store import StateStore

TARGET = "2030-01-01T00:00:00+00:00"


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    StateStore.reset_instances()
    yield
    StateStore.reset_instances()


def _case_with_retry(store):
    store.save_payment("p", {
        "payment_id": "p", "amount": 2499.0, "status": "recovering",
        "customer": {"email": "customer@example.com"},
        "failure_reason": "insufficient funds in account",
        "scheduled_job": {"status": "scheduled", "job_id": "job_p_1",
                          "action": "retry_payment",
                          "target_timestamp": TARGET}})
    store.schedule_job(job_id="job_p_1", payment_id="p",
                       target_time=TARGET, action="retry_payment")


def test_cancelling_the_jobs_also_cancels_the_snapshot():
    s = StateStore()
    _case_with_retry(s)
    assert s.cancel_jobs_for("p", reason="parked at restart") == ["job_p_1"]
    snap = s.get_payment("p")["scheduled_job"]
    assert snap["status"] == "cancelled"
    assert snap["cancelled_reason"] == "parked at restart"


def test_a_parked_retry_case_lands_in_a_runnable_batch():
    s = StateStore()
    _case_with_retry(s)
    s.cancel_jobs_for("p", reason="parked at restart")
    s.update_payment("p", status="parked")
    rec = s.get_payment("p")
    assert classify(rec) == "insufficient_funds"
    assert not ladder.retry_pending(rec)


def test_a_snapshot_left_stale_by_an_earlier_sweep_heals_on_the_next_call():
    s = StateStore()
    _case_with_retry(s)
    s._jobs["job_p_1"]["status"] = "cancelled"      # cancelled some other way
    assert s.cancel_jobs_for("p") == []             # nothing pending any more
    assert s.get_payment("p")["scheduled_job"]["status"] == "cancelled"


def test_a_completed_retry_snapshot_is_not_a_pending_one():
    s = StateStore()
    _case_with_retry(s)
    s.update_payment("p", status="parked",
                     scheduled_job={"status": "completed", "job_id": "job_p_1",
                                    "target_timestamp": TARGET})
    assert classify(s.get_payment("p")) == "insufficient_funds"
