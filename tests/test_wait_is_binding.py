"""A wait the agent asks for must actually happen.

`wait_for_customer` used to write its wait into Case metadata that nothing
read, and it failed in BOTH directions:

  - pay_woo85c9gh (2026-09-04): refused a link during quiet hours, the agent
    said "wake me in 60 minutes", and nothing ever did — no timer, no job, no
    watcher. The case sat at `awaiting_customer` permanently.
  - C1 (2026-09-03): the opposite — the run was classified `failed`, the ladder
    hand-off fired ten seconds later, and the agent was marched up rungs it had
    explicitly asked to wait through.

Now it registers a real `wake_agent` job that the daemon fires and the frontend
turns back into a hand-off.
"""
import json
from datetime import datetime, timezone

import pytest

import recovery_agent.state_store as state_store
from recovery_agent.state_store import StateStore


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    StateStore.reset_instances()
    yield
    StateStore.reset_instances()


def _wait(minutes=60, reason="customer recovery outside of quiet hours"):
    from recovery_agent.agent.tools import wait_for_customer
    s = StateStore()
    s.save_payment("p", {"payment_id": "p", "amount": 2499.0,
                         "status": "awaiting_customer", "trail": []})
    s.flush()
    return json.loads(wait_for_customer.invoke({
        "payment_id": "p", "waiting_for": reason,
        "expected_within_minutes": minutes}))


# ── the promise is registered ───────────────────────────────────────────────

def test_waiting_schedules_a_real_wake_up():
    r = _wait(60)
    assert r["status"] == "ok"
    assert r["wake_at"], "the wait must name when it ends"

    jobs = StateStore()._jobs
    wake = [j for j in jobs.values() if j.get("action") == "wake_agent"]
    assert len(wake) == 1, "exactly one wake-up per wait"
    assert wake[0]["payment_id"] == "p"
    assert wake[0]["status"] == "pending"


def test_the_wake_up_is_due_at_the_time_the_agent_asked_for():
    r = _wait(90)
    due = datetime.fromisoformat(r["wake_at"])
    minutes = (due - datetime.now(timezone.utc)).total_seconds() / 60
    assert 88 < minutes < 92, "the agent asked for 90 minutes"


def test_the_reason_is_carried_so_the_agent_resumes_its_own_plan():
    _wait(30, reason="the 5% offer to be seen")
    job = [j for j in StateStore()._jobs.values()
           if j.get("action") == "wake_agent"][0]
    assert "5% offer" in job["metadata"]["reason"]
    assert (StateStore().get_payment("p").get("waiting_for") or {})["reason"]


def test_a_wait_on_an_unknown_case_does_not_explode():
    from recovery_agent.agent.tools import wait_for_customer
    r = json.loads(wait_for_customer.invoke({
        "payment_id": "nope", "waiting_for": "x", "expected_within_minutes": 5}))
    assert r["status"] == "ok", "bookkeeping must never break the turn"


# ── the daemon keeps it ─────────────────────────────────────────────────────

def test_the_daemon_executes_a_wake_up_without_touching_razorpay(monkeypatch):
    from recovery_agent import daemon_worker

    def _boom(*a, **k):
        raise AssertionError("a wake-up must not call the gateway")

    monkeypatch.setattr(daemon_worker, "RazorpayClient", _boom, raising=False)
    out = daemon_worker.execute_retry({
        "job_id": "wake_p_1", "payment_id": "p", "action": "wake_agent",
        "metadata": {"reason": "quiet hours to end"}})
    assert out["status"] == "woken"
    assert "quiet hours" in out["reason"]


def test_a_due_wake_up_is_completed_and_reported(monkeypatch):
    from recovery_agent import daemon_worker
    told = []
    monkeypatch.setattr(daemon_worker, "notify_frontend",
                        lambda job, result: told.append((job, result)) or True)

    s = StateStore()
    s.save_payment("p", {"payment_id": "p", "amount": 2499.0, "trail": []})
    s.schedule_job(job_id="wake_p_1", payment_id="p",
                   target_time="2020-01-01T00:00:00+00:00",
                   action="wake_agent", metadata={"reason": "quiet hours"})
    s.flush()

    assert daemon_worker._process_due_jobs() == 1
    assert told and told[0][1]["status"] == "woken"
    assert StateStore()._jobs["wake_p_1"]["status"] == "completed", \
        "a fired wake-up must not fire again"


# ── the frontend turns it back into a hand-off ──────────────────────────────

def test_the_frontend_hands_the_case_back_when_the_wait_elapses(monkeypatch):
    import recovery_agent.frontend as F
    monkeypatch.setattr(F, "store", StateStore())
    handoffs = []
    monkeypatch.setattr(F, "_handoff_to_agent",
                        lambda pid, obs, scenario: handoffs.append((pid, obs, scenario)))

    F.store.save_payment("p", {"payment_id": "p", "amount": 2499.0, "trail": [],
                               "waiting_for": {"reason": "quiet hours to end"}})
    F.store.flush()

    resp = F.app.test_client().post("/api/daemon-retry-complete", json={
        "job_id": "wake_p_1", "payment_id": "p", "action": "wake_agent",
        "result": {"status": "woken", "reason": "quiet hours to end"}})
    assert resp.status_code == 200
    assert handoffs, "the agent must actually be restarted"
    pid, obs, scenario = handoffs[0]
    assert pid == "p"
    assert "quiet hours to end" in obs, "it resumes its own plan, not a new one"
    assert scenario.startswith("wait_elapsed:"), \
        "keyed per job so two waits cannot collapse into one hand-off"
    assert not (F.store.get_payment("p") or {}).get("waiting_for"), \
        "the wait is cleared once it has been honoured"


# ── an ended case keeps no timers ───────────────────────────────────────────

def test_closing_a_case_cancels_its_pending_wake_up(monkeypatch):
    """pay_4tnzl57fu: the agent asked to be woken in 16 minutes, the customer
    paid 15 seconds later, and the case closed — leaving an alarm set on a
    finished case."""
    from recovery_agent.agent.tools import close_case
    import recovery_agent.agent.perception as P

    _wait(16)
    s = StateStore()
    s.update_payment("p", status="recovered", recovered_amount=2374.05,
                     recovered_payment_id="pay_real", amount=2499.0,
                     ladder={"page_push": {"at": "x"}, "offer": {"at": "x"}})
    s.flush()
    monkeypatch.setattr(P, "ground_truth", lambda pid, verify=False: {
        "known": True, "settled": True, "received": 2374.05, "owed": 2499.0})

    r = json.loads(close_case.invoke({
        "payment_id": "p", "outcome": "recovered",
        "what_happened": "paid the discounted link", "lesson": ""}))
    assert r["status"] == "closed"

    jobs = StateStore()._jobs
    wake = [j for j in jobs.values() if j.get("action") == "wake_agent"]
    assert wake and wake[0]["status"] == "cancelled", \
        "a closed case must not keep an alarm set for itself"
    assert not (StateStore().get_payment("p") or {}).get("waiting_for")


def test_recovery_cancels_the_wake_up_even_without_a_close(monkeypatch):
    """The money arriving is itself the end of the wait."""
    import recovery_agent.frontend as F
    monkeypatch.setattr(F, "store", StateStore())
    monkeypatch.setattr(F, "_notify_agent_of_recovery", lambda *a, **k: None)
    monkeypatch.setattr(F, "_link_original_order_to_recovery", lambda *a, **k: None)

    _wait(16)
    assert F._mark_recovered("p", 2374.05, "pay_real", 15, "link paid")

    wake = [j for j in StateStore()._jobs.values()
            if j.get("action") == "wake_agent"]
    assert wake and wake[0]["status"] == "cancelled"


def test_cancelling_leaves_other_cases_alone():
    _wait(16)
    s = StateStore()
    s.save_payment("other", {"payment_id": "other", "amount": 1.0})
    s.schedule_job(job_id="wake_other_1", payment_id="other",
                   target_time="2030-01-01T00:00:00+00:00", action="wake_agent")
    s.flush()
    assert s.cancel_jobs_for("p") == [j for j in s._jobs
                                      if s._jobs[j]["payment_id"] == "p"]
    assert s._jobs["wake_other_1"]["status"] == "pending"
