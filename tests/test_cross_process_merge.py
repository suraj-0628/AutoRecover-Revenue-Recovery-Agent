"""A flush must not write a stale snapshot over another process's work.

start.sh runs the frontend, the webhook receiver and the daemon as separate
processes; each holds its own in-memory copy of the live_*.json files, and
`flush()` used to rewrite all four wholesale from that copy. Any daemon or
webhook flush therefore reverted every case the frontend had advanced since
that process started — silently, atomically, and file by file.

Two StateStore instances on one directory (forced apart with
reset_instances) are exactly the two-process shape.
"""
import pytest

import recovery_agent.state_store as state_store
from recovery_agent.state_store import StateStore


@pytest.fixture(autouse=True)
def _tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    StateStore.reset_instances()
    yield
    StateStore.reset_instances()


def _second_process() -> StateStore:
    """A fresh instance that loads from disk — another process, in effect."""
    StateStore.reset_instances()
    return StateStore()


def test_a_stale_writer_no_longer_clobbers_another_processs_case():
    a = StateStore()
    a.save_payment("p1", {"payment_id": "p1", "amount": 100.0, "status": "failed"})
    a.flush()

    b = _second_process()
    b.save_payment("p2", {"payment_id": "p2", "amount": 200.0, "status": "failed"})
    b.flush()

    # `a` has never heard of p2. Its flush used to erase it.
    a.update_payment("p1", status="awaiting_customer")
    a.flush()

    c = _second_process()
    assert c.get_payment("p1")["status"] == "awaiting_customer"
    assert c.get_payment("p2") is not None, \
        "the stale writer erased another process's payment"


def test_an_in_place_mutation_is_still_detected_as_ours():
    """Call sites mutate the returned dict directly; the shadow must see it."""
    a = StateStore()
    a.save_payment("p1", {"payment_id": "p1", "attempts": 0, "status": "failed"})
    a.flush()

    b = _second_process()
    b.save_payment("p2", {"payment_id": "p2", "status": "failed"})
    b.flush()

    p = a.get_payment("p1")
    p["attempts"] = 3                    # no update_payment, no save_payment
    a.flush()

    c = _second_process()
    assert c.get_payment("p1")["attempts"] == 3
    assert c.get_payment("p2") is not None


def test_a_deletion_survives_another_processs_flush():
    a = StateStore()
    a.save_payment("p1", {"payment_id": "p1", "status": "failed"})
    a.save_pending("p1", {"action": "send_notification"})
    a.flush()

    b = _second_process()
    b.remove_pending("p1")
    b.flush()

    a.update_payment("p1", status="awaiting_customer")   # unrelated change
    a.flush()

    c = _second_process()
    assert not c.has_pending("p1"), "the stale writer resurrected a pending"


def test_recovered_is_absorbing_across_processes():
    """Money in the bank survives even a stale-dirty write to the SAME record."""
    a = StateStore()
    a.save_payment("p1", {"payment_id": "p1", "amount": 2499.0,
                          "status": "recovering"})
    a.flush()

    b = _second_process()
    b.update_payment("p1", status="recovered", recovered_amount=2499.0,
                     recovered_payment_id="pay_real")
    b.flush()

    # `a` still thinks p1 is live and writes its own (stale) verdict.
    a.update_payment("p1", status="failed", last_error="LLM 404")
    a.flush()

    c = _second_process()
    rec = c.get_payment("p1")
    assert rec["status"] == "recovered"
    assert rec["recovered_amount"] == 2499.0
    assert rec["last_error"] == "LLM 404", "non-status fields still merge"


def test_refresh_makes_another_processs_job_visible():
    a = StateStore()
    a.flush()

    b = _second_process()
    b.schedule_job("job_1", "p1", "2020-01-01T00:00:00+00:00", "retry_payment")
    b.flush()

    assert not a.get_due_jobs(), "the stale snapshot cannot know the job"
    a.refresh()
    assert [j["job_id"] for j in a.get_due_jobs()] == ["job_1"]


def test_a_local_change_survives_refresh_and_reaches_the_next_flush():
    a = StateStore()
    a.save_payment("p1", {"payment_id": "p1", "status": "failed"})
    a.flush()

    b = _second_process()
    b.save_payment("p2", {"payment_id": "p2", "status": "failed"})
    b.flush()

    a.update_payment("p1", status="scheduled")   # dirty, unflushed
    a.refresh()                                  # pulls p2 in
    assert a.get_payment("p2") is not None
    assert a.get_payment("p1")["status"] == "scheduled", \
        "refresh must not roll back an unflushed local change"
    a.flush()

    c = _second_process()
    assert c.get_payment("p1")["status"] == "scheduled"
