"""The desk the exceptions land on — one worker, honest waiting."""
import threading
import time

import pytest

from recovery_agent import audit
from recovery_agent.batch.agent_queue import AgentQueue, Referral


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from recovery_agent import state_store
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path, raising=False)
    state_store.StateStore.reset_instances()
    audit.AuditLog.reset_instances()
    audit._default = None
    yield state_store.StateStore()
    state_store.StateStore.reset_instances()
    audit.AuditLog.reset_instances()
    audit._default = None


def wait_until(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def test_referrals_are_worked_in_arrival_order_by_one_worker(env):
    worked, order_lock = [], threading.Lock()

    def runner(referral):
        with order_lock:
            worked.append(referral.payment_id)
        time.sleep(0.03)

    q = AgentQueue(runner, workers=1)
    for i in range(5):
        assert q.submit(Referral(f"pay_{i}", "needs judgement")) == ""
    assert wait_until(lambda: q.stats()["worked"] == 5), q.stats()
    assert worked == [f"pay_{i}" for i in range(5)], (
        "one worker means fair, arrival-ordered service")


def test_a_case_already_waiting_is_not_queued_twice(env):
    gate = threading.Event()
    q = AgentQueue(lambda r: gate.wait(3), workers=1)
    assert q.submit(Referral("pay_1", "wave 2")) == ""
    assert q.submit(Referral("pay_1", "wave 3")) == "already_queued", (
        "the wave loop re-refers every wave; the queue holds one place")
    gate.set()


def test_a_full_queue_refuses_loudly_instead_of_growing_forever(env):
    gate = threading.Event()
    q = AgentQueue(lambda r: gate.wait(3), workers=1, depth=2)
    verdicts = [q.submit(Referral(f"pay_{i}", "x")) for i in range(5)]
    assert "queue_full" in verdicts
    gate.set()
    refused = [e for e in audit.log().for_payment(
        [f"pay_{i}" for i in range(5)][verdicts.index("queue_full")])
        if "queue full" in e.get("reason", "")]
    assert refused, "a dropped referral leaves a trace, never a silence"


def test_a_referred_case_is_stamped_before_the_session_starts(env):
    store = env
    store.save_payment("pay_x", {"payment_id": "pay_x", "amount": 100.0})
    store.flush()
    seen = {}

    def runner(referral):
        seen["stamp"] = (store.get_payment("pay_x") or {}).get("batch_run_id")

    q = AgentQueue(runner, workers=1)
    q.submit(Referral("pay_x", "over the ceiling", batch_run_id="run_9"))
    assert wait_until(lambda: "stamp" in seen)
    assert seen["stamp"] == "run_9", (
        "whatever the agent recovers lands on the run that referred it")


def test_a_crashing_session_does_not_kill_the_worker(env):
    worked = []

    def runner(referral):
        if referral.payment_id == "pay_bad":
            raise RuntimeError("proxy fell over")
        worked.append(referral.payment_id)

    q = AgentQueue(runner, workers=1)
    q.submit(Referral("pay_bad", "x"))
    q.submit(Referral("pay_good", "x"))
    assert wait_until(lambda: "pay_good" in worked)
    assert q.stats()["failed"] == 1 and q.stats()["worked"] == 1
