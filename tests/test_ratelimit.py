"""The gate that stays a gate under concurrency.

The regression these tests exist for: the old module-attribute gate let N
threads read one timestamp, sleep the same delay, and wake together — a burst
of N against the provider's limit, which is the opposite of pacing.
"""
import threading
import time

from recovery_agent.ratelimit import TokenBucket, llm_gate


def grants(bucket, n, spawn_together=True):
    """n threads race the bucket; returns each grant's monotonic time."""
    times, lock = [], threading.Lock()
    start = threading.Event()

    def one():
        start.wait()
        bucket.acquire()
        with lock:
            times.append(time.monotonic())

    threads = [threading.Thread(target=one) for _ in range(n)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join(timeout=10)
    return sorted(times)


def test_simultaneous_callers_are_spaced_never_bursty():
    """Six threads arriving at once must come out one interval apart — the
    exact scenario where the old gate fired all six together."""
    bucket = TokenBucket(calls_per_minute=600)          # 0.1s interval
    times = grants(bucket, 6)
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert all(g >= bucket.interval * 0.8 for g in gaps), gaps
    assert times[-1] - times[0] >= bucket.interval * 4.5


def test_the_first_caller_does_not_wait():
    bucket = TokenBucket(calls_per_minute=6)
    assert bucket.acquire() == 0.0


def test_a_quiet_gap_is_not_hoarded_into_a_later_burst():
    """After idle time, callers still space out from *now* — unused slots do
    not accumulate into permission to burst."""
    bucket = TokenBucket(calls_per_minute=600)
    bucket.acquire()
    time.sleep(0.35)                     # three intervals of silence
    times = grants(bucket, 3)
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert all(g >= bucket.interval * 0.8 for g in gaps), gaps


def test_the_wait_is_a_number_someone_can_be_shown():
    bucket = TokenBucket(calls_per_minute=600)
    for _ in range(3):
        bucket.acquire()
    stats = bucket.stats()
    assert stats["calls"] == 3
    assert stats["waited_seconds"] > 0, (
        "time spent waiting on the quota is reported, not hidden")


def test_under_pytest_the_gate_is_open():
    """The pause respects a real provider; charging it to a scripted model is
    how a fast suite stops being run."""
    gate = llm_gate()
    started = time.monotonic()
    for _ in range(20):
        gate.acquire()
    assert time.monotonic() - started < 0.5
    assert gate.stats()["calls"] >= 20
