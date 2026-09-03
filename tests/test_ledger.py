"""D1 gate tests for the case ledger.

These are the acceptance criteria from REBUILD-PLAN.md Track A, block D1:

    "SIGKILL mid-case -> intact; double-fire -> one case"

They use real subprocesses and a real SIGKILL. Nothing here is mocked, because
the property under test is crash-safety, and a mocked crash proves nothing.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import threading
from decimal import Decimal

import pytest

from recovery_agent.ledger import (
    CaseNotFound,
    EventKind,
    Ledger,
    LedgerError,
    TERMINAL_STATUSES,
    to_paise,
    to_rupees,
)
from recovery_agent.models import CaseStatus


@pytest.fixture
def ledger(tmp_path):
    return Ledger(db_path=tmp_path / "ledger.db")


def _run_child(db_path, body: str, expect_signal: int | None = None):
    """Run a snippet in a fresh interpreter against the same ledger file."""
    src = textwrap.dedent(f"""
        import os, signal, sys
        os.environ["LEDGER_DB_PATH"] = {str(db_path)!r}
        sys.path.insert(0, {str(os.path.join(os.getcwd(), "src"))!r})
        from recovery_agent.ledger import Ledger, EventKind, to_paise
        from recovery_agent.models import CaseStatus
        led = Ledger(db_path={str(db_path)!r})
    """) + textwrap.dedent(body)
    proc = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
    if expect_signal is not None:
        assert proc.returncode == -expect_signal, (
            f"expected signal {expect_signal}, got rc={proc.returncode}\n{proc.stderr}"
        )
    else:
        assert proc.returncode == 0, proc.stderr
    return proc


# ── Money: integers only (AUDIT-FINDINGS S1-4) ───────────────────────────────

def test_to_paise_is_exact_and_half_up():
    assert to_paise(2999) == 299900
    assert to_paise("2999.99") == 299999
    assert to_paise(0.1 + 0.2) == 30          # float noise must not leak through
    assert to_paise(Decimal("1234.565")) == 123457   # half-up, not banker's
    assert to_rupees(299900) == Decimal("2999.00")


def test_open_case_rejects_float_amounts(ledger):
    """The rupee/paise mix-up that put the escalation threshold 100x off."""
    with pytest.raises(LedgerError, match="int paise"):
        ledger.open_case(payment_id="pay_f", amount_paise=2999.0)
    with pytest.raises(LedgerError, match="int paise"):
        ledger.open_case(payment_id="pay_b", amount_paise=True)


def test_open_case_rejects_negative_amount(ledger):
    with pytest.raises(LedgerError):
        ledger.open_case(payment_id="pay_n", amount_paise=-1)


# ── Idempotency: double-fire -> one case ─────────────────────────────────────

def test_open_case_is_idempotent_in_process(ledger):
    a = ledger.open_case(payment_id="pay_dup", amount_paise=to_paise(2999),
                         customer_id="c1", failure_code="card_expired")
    b = ledger.open_case(payment_id="pay_dup", amount_paise=to_paise(2999),
                         customer_id="c1", failure_code="card_expired")
    assert a.case_id == b.case_id
    assert len(ledger.all_cases()) == 1


def test_open_case_idempotent_after_progress(ledger):
    """A redelivered failure must not reset a case that has already moved on."""
    a = ledger.open_case(payment_id="pay_prog", amount_paise=to_paise(500))
    ledger.record_transition(a.case_id, CaseStatus.ACTING, reason="agent decided")
    b = ledger.open_case(payment_id="pay_prog", amount_paise=to_paise(500))
    assert b.case_id == a.case_id
    assert b.status == CaseStatus.ACTING


def test_open_case_idempotent_across_processes(tmp_path):
    """Eight racing processes, one payment_id, one case."""
    db = tmp_path / "ledger.db"
    Ledger(db_path=db)  # create schema first

    body = """
        led.open_case(payment_id="pay_race", amount_paise=to_paise(1500))
    """
    procs = []
    for _ in range(8):
        src = textwrap.dedent(f"""
            import sys, os
            sys.path.insert(0, {str(os.path.join(os.getcwd(), "src"))!r})
            from recovery_agent.ledger import Ledger, to_paise
            led = Ledger(db_path={str(db)!r})
        """) + textwrap.dedent(body)
        procs.append(subprocess.Popen([sys.executable, "-c", src],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    for p in procs:
        _, err = p.communicate(timeout=60)
        assert p.returncode == 0, err.decode()

    led = Ledger(db_path=db)
    assert len(led.all_cases()) == 1
    case = led.get_case_by_payment("pay_race")
    assert case is not None
    # Exactly one OPENED event — no duplicate genesis.
    opened = [e for e in led.events(case.case_id) if e.kind == EventKind.OPENED]
    assert len(opened) == 1


# ── Append-only, enforced by the storage layer ───────────────────────────────

def test_events_cannot_be_updated(ledger):
    case = ledger.open_case(payment_id="pay_ao", amount_paise=to_paise(100))
    with sqlite3.connect(str(ledger.db_path)) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE events SET reason='tampered' WHERE case_id=?",
                         (case.case_id,))


def test_events_cannot_be_deleted(ledger):
    case = ledger.open_case(payment_id="pay_ao2", amount_paise=to_paise(100))
    with sqlite3.connect(str(ledger.db_path)) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM events WHERE case_id=?", (case.case_id,))


# ── Projection is derived, not authoritative ─────────────────────────────────

def test_projection_matches_replay(ledger):
    case = ledger.open_case(payment_id="pay_rep", amount_paise=to_paise(2999),
                            customer_id="c9", failure_code="card_expired")
    ledger.record_note(case.case_id, reason="diagnosed: card_expired")
    ledger.record_transition(case.case_id, CaseStatus.ACTING, reason="decided")
    ledger.record_attempt(case.case_id, action="create_payment_link",
                          result="ok", receipt={"link_id": "plink_x"})
    ledger.record_transition(case.case_id, CaseStatus.AWAITING_CUSTOMER)
    assert ledger.verify(case.case_id)


def test_projection_is_rebuilt_from_log_when_tampered(ledger):
    """The log wins. A doctored projection is repaired by replay."""
    case = ledger.open_case(payment_id="pay_tamper", amount_paise=to_paise(700))
    ledger.record_transition(case.case_id, CaseStatus.ACTING)

    with sqlite3.connect(str(ledger.db_path)) as conn:
        conn.execute("UPDATE cases SET status=?, attempt_count=99 WHERE case_id=?",
                     (CaseStatus.RECOVERED.value, case.case_id))
        conn.commit()
    assert ledger.get_case(case.case_id).status == CaseStatus.RECOVERED

    rebuilt = ledger.rebuild_projection(case.case_id)
    assert rebuilt.status == CaseStatus.ACTING
    assert rebuilt.attempt_count == 0


# ── Attempts actually count (AUDIT-FINDINGS S0-2) ────────────────────────────

def test_attempt_count_increments(ledger):
    """attempt_count was permanently 0 in the old code; it must move here."""
    case = ledger.open_case(payment_id="pay_att", amount_paise=to_paise(1200))
    assert case.attempt_count == 0
    for i in range(3):
        ledger.record_attempt(case.case_id, action="retry_payment",
                              result="failed", request={"n": i})
    assert ledger.require_case(case.case_id).attempt_count == 3


def test_attempt_stores_receipt(ledger):
    """The receipt is the evidence the action really happened."""
    case = ledger.open_case(payment_id="pay_rcpt", amount_paise=to_paise(1000))
    ledger.record_attempt(
        case.case_id, action="create_payment_link", result="ok",
        request={"amount_paise": 100000},
        receipt={"link_id": "plink_abc", "short_url": "https://rzp.io/i/abc"},
    )
    ev = [e for e in ledger.events(case.case_id) if e.kind == EventKind.ATTEMPT][0]
    assert ev.payload["receipt"]["link_id"] == "plink_abc"
    assert ev.payload["request"]["amount_paise"] == 100000


# ── Recovery linkage: original stays failed, a NEW payment captures ──────────

def test_recovery_links_new_payment(ledger):
    case = ledger.open_case(payment_id="pay_orig", amount_paise=to_paise(2999))
    ledger.record_observation(
        case.case_id, observed="payment_link.paid", recovered=True,
        recovery_payment_id="pay_new_123",
        recovered_amount_paise=to_paise(2999),
    )
    ledger.record_transition(case.case_id, CaseStatus.RECOVERED, reason="link paid")

    got = ledger.require_case(case.case_id)
    assert got.payment_id == "pay_orig"          # original is untouched
    assert got.recovery_payment_id == "pay_new_123"
    assert got.recovered_amount_paise == 299900
    assert got.recovered_amount_rupees == Decimal("2999.00")
    assert got.is_terminal


def test_observation_without_recovery_does_not_set_amount(ledger):
    case = ledger.open_case(payment_id="pay_obs", amount_paise=to_paise(400))
    ledger.record_observation(case.case_id, observed="payment_link.created")
    got = ledger.require_case(case.case_id)
    assert got.recovery_payment_id == ""
    assert got.recovered_amount_paise == 0


# ── The work queue ───────────────────────────────────────────────────────────

def test_open_cases_excludes_terminal(ledger):
    live = ledger.open_case(payment_id="pay_live", amount_paise=to_paise(10))
    for pid, status in [("pay_r", CaseStatus.RECOVERED),
                        ("pay_s", CaseStatus.STOPPED),
                        ("pay_e", CaseStatus.ESCALATED)]:
        c = ledger.open_case(payment_id=pid, amount_paise=to_paise(10))
        if status is CaseStatus.RECOVERED:
            # D2: recovery must be observed before it can be declared.
            ledger.record_observation(c.case_id, observed="payment.captured",
                                      recovered=True, recovery_payment_id="pay_x",
                                      recovered_amount_paise=to_paise(10))
        ledger.record_transition(c.case_id, status)

    open_ids = {c.case_id for c in ledger.open_cases()}
    assert open_ids == {live.case_id}
    assert all(s in TERMINAL_STATUSES for s in
               (CaseStatus.RECOVERED, CaseStatus.STOPPED, CaseStatus.ESCALATED))


def test_append_to_unknown_case_raises(ledger):
    with pytest.raises(CaseNotFound):
        ledger.record_note("case_nope", reason="x")


# ── Concurrency: seq stays unique and monotonic ──────────────────────────────

def test_concurrent_appends_have_unique_seq(ledger):
    case = ledger.open_case(payment_id="pay_conc", amount_paise=to_paise(10))
    errors: list[Exception] = []

    def worker(n: int):
        try:
            for i in range(10):
                ledger.record_note(case.case_id, reason=f"t{n}-{i}")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    events = ledger.events(case.case_id)
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs)) == 61      # 1 opened + 60 notes
    assert ledger.verify(case.case_id)


# ── THE GATE: SIGKILL mid-case leaves the ledger intact ──────────────────────

def test_sigkill_mid_case_leaves_ledger_readable(tmp_path):
    """Kill -9 a process holding the ledger; state up to the last commit survives."""
    db = tmp_path / "ledger.db"
    Ledger(db_path=db)

    _run_child(db, """
        c = led.open_case(payment_id="pay_kill", amount_paise=to_paise(2999),
                          customer_id="c1", failure_code="card_expired")
        led.record_transition(c.case_id, CaseStatus.ACTING, reason="decided")
        led.record_attempt(c.case_id, action="create_payment_link", result="ok",
                           receipt={"link_id": "plink_kill"})
        os.kill(os.getpid(), signal.SIGKILL)
    """, expect_signal=9)

    led = Ledger(db_path=db)
    case = led.get_case_by_payment("pay_kill")
    assert case is not None, "case did not survive SIGKILL"
    assert case.status == CaseStatus.ACTING
    assert case.attempt_count == 1
    assert case.amount_paise == 299900
    assert led.verify(case.case_id), "projection diverged from log after crash"


def test_sigkill_then_resume_continues_same_case(tmp_path):
    """After the crash, work resumes on the same case — no duplicate, no reset."""
    db = tmp_path / "ledger.db"
    Ledger(db_path=db)

    _run_child(db, """
        c = led.open_case(payment_id="pay_resume", amount_paise=to_paise(1500))
        led.record_transition(c.case_id, CaseStatus.ACTING)
        os.kill(os.getpid(), signal.SIGKILL)
    """, expect_signal=9)

    led = Ledger(db_path=db)
    before = led.get_case_by_payment("pay_resume")
    assert before.status == CaseStatus.ACTING

    # Redelivery after restart must not create a second case or rewind status.
    again = led.open_case(payment_id="pay_resume", amount_paise=to_paise(1500))
    assert again.case_id == before.case_id
    assert again.status == CaseStatus.ACTING

    led.record_observation(again.case_id, observed="payment_link.paid",
                           recovered=True, recovery_payment_id="pay_new_9",
                           recovered_amount_paise=to_paise(1500))
    led.record_transition(again.case_id, CaseStatus.RECOVERED)

    final = led.require_case(again.case_id)
    assert final.is_terminal
    assert final.recovery_payment_id == "pay_new_9"
    assert len(led.all_cases()) == 1
    assert led.verify(final.case_id)


def test_sigkill_during_burst_never_leaves_partial_event(tmp_path):
    """A kill at an arbitrary moment must not produce a half-written event."""
    db = tmp_path / "ledger.db"
    Ledger(db_path=db)

    _run_child(db, """
        import time
        c = led.open_case(payment_id="pay_burst", amount_paise=to_paise(100))
        import threading
        def killer():
            time.sleep(0.15)
            os.kill(os.getpid(), signal.SIGKILL)
        threading.Thread(target=killer, daemon=True).start()
        for i in range(100000):
            led.record_note(c.case_id, reason="n%d" % i)
    """, expect_signal=9)

    led = Ledger(db_path=db)
    case = led.get_case_by_payment("pay_burst")
    assert case is not None
    events = led.events(case.case_id)
    assert [e.seq for e in events] == list(range(1, len(events) + 1)), "gap in seq"
    assert case.seq == len(events), "projection seq ahead of log"
    assert led.verify(case.case_id)


# ═══════════════════════════════════════════════════════════════════════════
# D2 — transition legality and evidence
# ═══════════════════════════════════════════════════════════════════════════

from datetime import datetime, timedelta, timezone     # noqa: E402

from recovery_agent.ledger import to_paise as _tp      # noqa: E402
from recovery_agent.statemachine import (              # noqa: E402
    IllegalTransition,
    MissingEvidence,
)


def _recover(ledger, case_id, amount_paise=1000, pid="pay_new"):
    ledger.record_observation(case_id, observed="payment_link.paid", recovered=True,
                              recovery_payment_id=pid,
                              recovered_amount_paise=amount_paise)
    return ledger.record_transition(case_id, CaseStatus.RECOVERED, reason="observed")


# ── Terminal is terminal ─────────────────────────────────────────────────────

@pytest.mark.parametrize("terminal", [CaseStatus.RECOVERED, CaseStatus.STOPPED,
                                      CaseStatus.ESCALATED])
def test_cannot_leave_a_terminal_case(ledger, terminal):
    case = ledger.open_case(payment_id=f"pay_term_{terminal.value}",
                            amount_paise=to_paise(10))
    if terminal is CaseStatus.RECOVERED:
        _recover(ledger, case.case_id, to_paise(10))
    else:
        ledger.record_transition(case.case_id, terminal)

    for target in (CaseStatus.ACTING, CaseStatus.OPEN, CaseStatus.SCHEDULED,
                   CaseStatus.RECOVERED):
        with pytest.raises(IllegalTransition, match="terminal"):
            ledger.record_transition(case.case_id, target)


def test_illegal_transition_writes_no_event(ledger):
    """A rejected transition must leave the log and seq untouched."""
    case = ledger.open_case(payment_id="pay_atomic", amount_paise=to_paise(10))
    ledger.record_transition(case.case_id, CaseStatus.STOPPED)
    before = ledger.require_case(case.case_id)
    n_before = len(ledger.events(case.case_id))

    with pytest.raises(IllegalTransition):
        ledger.record_transition(case.case_id, CaseStatus.ACTING)

    after = ledger.require_case(case.case_id)
    assert len(ledger.events(case.case_id)) == n_before
    assert after.seq == before.seq
    assert after.status == CaseStatus.STOPPED
    assert ledger.verify(case.case_id)


def test_scheduled_cannot_jump_to_awaiting_customer(ledger):
    """Off-graph moves are rejected even when evidence would exist."""
    case = ledger.open_case(payment_id="pay_jump", amount_paise=to_paise(10))
    ledger.record_transition(case.case_id, CaseStatus.ACTING)
    ledger.record_attempt(case.case_id, action="retry_payment", result="ok")
    ledger.record_transition(case.case_id, CaseStatus.SCHEDULED,
                             wake_at=datetime.now(timezone.utc))
    with pytest.raises(IllegalTransition, match="legal targets"):
        ledger.record_transition(case.case_id, CaseStatus.AWAITING_CUSTOMER)


# ── Evidence rule 1: recovery is observed, never declared ────────────────────

def test_cannot_declare_recovered_without_observation(ledger):
    """The rule the old system most needed (AUDIT-FINDINGS S2-1)."""
    case = ledger.open_case(payment_id="pay_fake", amount_paise=to_paise(2999))
    ledger.record_transition(case.case_id, CaseStatus.ACTING)
    ledger.record_attempt(case.case_id, action="create_payment_link", result="ok")
    ledger.record_transition(case.case_id, CaseStatus.AWAITING_CUSTOMER)

    with pytest.raises(MissingEvidence, match="never self-declared"):
        ledger.record_transition(case.case_id, CaseStatus.RECOVERED)
    assert ledger.require_case(case.case_id).status == CaseStatus.AWAITING_CUSTOMER


def test_a_note_claiming_success_is_not_evidence(ledger):
    """An agent asserting 'I recovered it' in prose proves nothing."""
    case = ledger.open_case(payment_id="pay_prose", amount_paise=to_paise(500))
    ledger.record_note(case.case_id, reason="Payment recovered successfully!",
                       payload={"recovered": True})
    with pytest.raises(MissingEvidence):
        ledger.record_transition(case.case_id, CaseStatus.RECOVERED)


def test_a_non_recovery_observation_is_not_evidence(ledger):
    case = ledger.open_case(payment_id="pay_obs2", amount_paise=to_paise(500))
    ledger.record_observation(case.case_id, observed="payment_link.created")
    with pytest.raises(MissingEvidence):
        ledger.record_transition(case.case_id, CaseStatus.RECOVERED)


def test_recovery_succeeds_once_observed(ledger):
    case = ledger.open_case(payment_id="pay_real_rec", amount_paise=to_paise(2999))
    ledger.record_transition(case.case_id, CaseStatus.ACTING)
    ledger.record_attempt(case.case_id, action="create_payment_link", result="ok",
                          receipt={"link_id": "plink_1"})
    ledger.record_transition(case.case_id, CaseStatus.AWAITING_CUSTOMER)
    got = _recover(ledger, case.case_id, to_paise(2999), pid="pay_new_1")

    assert got.status == CaseStatus.RECOVERED
    assert got.is_terminal
    assert got.recovered_amount_paise == 299900
    assert ledger.verify(case.case_id)


# ── Attribution: recovered != caused by the agent ────────────────────────────

def test_self_recovery_is_not_attributed_to_the_agent(ledger):
    """Customer pays unprompted. Real recovery, but not the agent's win."""
    case = ledger.open_case(payment_id="pay_self", amount_paise=to_paise(1000))
    got = _recover(ledger, case.case_id, to_paise(1000))
    assert got.status == CaseStatus.RECOVERED
    assert got.self_recovered
    assert not got.attributed_to_agent


def test_recovery_after_an_attempt_is_attributed(ledger):
    case = ledger.open_case(payment_id="pay_attr", amount_paise=to_paise(1000))
    ledger.record_transition(case.case_id, CaseStatus.ACTING)
    ledger.record_attempt(case.case_id, action="create_payment_link", result="ok")
    got = _recover(ledger, case.case_id, to_paise(1000))
    assert got.attributed_to_agent
    assert not got.self_recovered


# ── Evidence rule 2: only wait on a customer you reached ─────────────────────

def test_cannot_await_customer_without_an_attempt(ledger):
    case = ledger.open_case(payment_id="pay_noattempt", amount_paise=to_paise(10))
    ledger.record_transition(case.case_id, CaseStatus.ACTING)
    with pytest.raises(MissingEvidence, match="nothing was sent"):
        ledger.record_transition(case.case_id, CaseStatus.AWAITING_CUSTOMER)


def test_failed_effector_does_not_count_as_reaching_the_customer(ledger):
    case = ledger.open_case(payment_id="pay_err", amount_paise=to_paise(10))
    ledger.record_transition(case.case_id, CaseStatus.ACTING)
    ledger.record_attempt(case.case_id, action="send_email", result="error",
                          receipt={"error": "smtp refused"})
    with pytest.raises(MissingEvidence):
        ledger.record_transition(case.case_id, CaseStatus.AWAITING_CUSTOMER)


# ── Evidence rule 3: SCHEDULED needs a wake time ─────────────────────────────

def test_scheduled_requires_wake_at(ledger):
    case = ledger.open_case(payment_id="pay_nowake", amount_paise=to_paise(10))
    with pytest.raises(MissingEvidence, match="wake_at"):
        ledger.record_transition(case.case_id, CaseStatus.SCHEDULED)


def test_wake_at_is_stored_and_normalised_to_utc(ledger):
    case = ledger.open_case(payment_id="pay_wake", amount_paise=to_paise(10))
    when = datetime.now(timezone.utc) + timedelta(days=5)
    got = ledger.record_transition(case.case_id, CaseStatus.SCHEDULED,
                                   reason="retry on payday", wake_at=when)
    assert got.wake_at is not None
    assert abs((got.wake_at - when).total_seconds()) < 1
    assert got.wake_at.tzinfo is not None


def test_leaving_scheduled_clears_wake_at(ledger):
    """A stale wake time would make the scheduler pick up a case twice."""
    case = ledger.open_case(payment_id="pay_clear", amount_paise=to_paise(10))
    ledger.record_transition(case.case_id, CaseStatus.SCHEDULED,
                             wake_at=datetime.now(timezone.utc))
    assert ledger.require_case(case.case_id).wake_at is not None
    got = ledger.record_transition(case.case_id, CaseStatus.ACTING)
    assert got.wake_at is None
    assert ledger.verify(case.case_id)


# ── The two work queues ──────────────────────────────────────────────────────

def test_due_cases_returns_only_past_due_scheduled(ledger):
    now = datetime.now(timezone.utc)
    due = ledger.open_case(payment_id="pay_due", amount_paise=to_paise(10))
    ledger.record_transition(due.case_id, CaseStatus.SCHEDULED,
                             wake_at=now - timedelta(minutes=5))
    later = ledger.open_case(payment_id="pay_later", amount_paise=to_paise(10))
    ledger.record_transition(later.case_id, CaseStatus.SCHEDULED,
                             wake_at=now + timedelta(days=5))

    ids = {c.case_id for c in ledger.due_cases(now=now)}
    assert ids == {due.case_id}


def test_awaiting_customer_queue_is_what_the_sensor_polls(ledger):
    waiting = ledger.open_case(payment_id="pay_wait", amount_paise=to_paise(10))
    ledger.record_transition(waiting.case_id, CaseStatus.ACTING)
    ledger.record_attempt(waiting.case_id, action="create_payment_link", result="ok")
    ledger.record_transition(waiting.case_id, CaseStatus.AWAITING_CUSTOMER)

    idle = ledger.open_case(payment_id="pay_idle", amount_paise=to_paise(10))
    ledger.record_transition(idle.case_id, CaseStatus.SCHEDULED,
                             wake_at=datetime.now(timezone.utc))

    assert {c.case_id for c in ledger.awaiting_customer_cases()} == {waiting.case_id}


# ── Migration from a pre-D2 ledger ───────────────────────────────────────────

def test_pre_d2_database_is_migrated(tmp_path):
    """A genuine pre-D2 ledger (no wake_at column, no wake index) must open."""
    db = tmp_path / "old.db"
    with sqlite3.connect(str(db)) as conn:
        conn.executescript("""
            CREATE TABLE cases (
                case_id TEXT PRIMARY KEY, payment_id TEXT NOT NULL UNIQUE,
                customer_id TEXT NOT NULL DEFAULT '', amount_paise INTEGER NOT NULL DEFAULT 0,
                currency TEXT NOT NULL DEFAULT 'INR', failure_code TEXT NOT NULL DEFAULT '',
                failure_reason TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0,
                recovery_payment_id TEXT NOT NULL DEFAULT '',
                recovered_amount_paise INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                seq INTEGER NOT NULL DEFAULT 0, metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL,
                seq INTEGER NOT NULL, kind TEXT NOT NULL, from_status TEXT, to_status TEXT,
                action TEXT NOT NULL DEFAULT '', result TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '', payload TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL, UNIQUE(case_id, seq)
            );
            INSERT INTO cases (case_id, payment_id, status, amount_paise,
                               created_at, updated_at, seq)
            VALUES ('case_legacy', 'pay_old', 'open', 1000,
                    '2026-08-01T00:00:00+00:00', '2026-08-01T00:00:00+00:00', 1);
            INSERT INTO events (case_id, seq, kind, to_status, payload, created_at)
            VALUES ('case_legacy', 1, 'opened', 'open',
                    '{"payment_id":"pay_old","amount_paise":1000}',
                    '2026-08-01T00:00:00+00:00');
        """)
        conn.commit()

    led = Ledger(db_path=db)                      # must migrate, not crash
    got = led.get_case_by_payment("pay_old")
    assert got is not None and got.wake_at is None
    assert got.amount_paise == 1000

    led.record_transition("case_legacy", CaseStatus.SCHEDULED,
                          wake_at=datetime.now(timezone.utc))
    assert led.require_case("case_legacy").wake_at is not None
    assert led.verify("case_legacy")


def test_migration_is_idempotent(tmp_path):
    """Opening the same ledger repeatedly must not re-run or duplicate anything."""
    db = tmp_path / "repeat.db"
    first = Ledger(db_path=db)
    case = first.open_case(payment_id="pay_repeat", amount_paise=to_paise(10))
    for _ in range(3):
        led = Ledger(db_path=db)
        assert led.get_case_by_payment("pay_repeat").case_id == case.case_id
    assert len(Ledger(db_path=db).all_cases()) == 1


# ═══════════════════════════════════════════════════════════════════════════
# API review fixes — atomicity, idempotency, optimistic concurrency
# ═══════════════════════════════════════════════════════════════════════════

from recovery_agent.ledger import ConcurrentModification    # noqa: E402


def test_act_records_attempt_and_transition_atomically(ledger):
    case = ledger.open_case(payment_id="pay_act", amount_paise=to_paise(2999))
    ledger.record_transition(case.case_id, CaseStatus.ACTING)

    got = ledger.act(
        case.case_id, action="create_payment_link",
        to_status=CaseStatus.AWAITING_CUSTOMER, result="ok",
        request={"amount_paise": 299900},
        receipt={"link_id": "plink_1", "short_url": "https://rzp.io/i/1"},
    )
    assert got.status == CaseStatus.AWAITING_CUSTOMER
    assert got.attempt_count == 1
    kinds = [e.kind for e in ledger.events(case.case_id)]
    assert kinds[-2:] == [EventKind.ATTEMPT, EventKind.TRANSITION]
    assert ledger.verify(case.case_id)


def test_act_rolls_back_both_events_when_the_transition_is_illegal(ledger):
    """No half-written act: an illegal target must undo the attempt too."""
    case = ledger.open_case(payment_id="pay_act_bad", amount_paise=to_paise(10))
    ledger.record_transition(case.case_id, CaseStatus.STOPPED)
    n_before = len(ledger.events(case.case_id))

    with pytest.raises(IllegalTransition):
        ledger.act(case.case_id, action="create_payment_link",
                   to_status=CaseStatus.AWAITING_CUSTOMER)

    after = ledger.require_case(case.case_id)
    assert len(ledger.events(case.case_id)) == n_before
    assert after.attempt_count == 0
    assert after.status == CaseStatus.STOPPED
    assert ledger.verify(case.case_id)


def test_act_is_idempotent_under_a_key(ledger):
    """Replaying a request after the effector succeeded must not charge twice."""
    case = ledger.open_case(payment_id="pay_idem", amount_paise=to_paise(2999))
    ledger.record_transition(case.case_id, CaseStatus.ACTING)

    for _ in range(4):
        got = ledger.act(
            case.case_id, action="retry_payment",
            to_status=CaseStatus.AWAITING_CUSTOMER,
            idempotency_key="retry-1", receipt={"rzp_id": "pay_x"},
        )
    assert got.attempt_count == 1
    attempts = [e for e in ledger.events(case.case_id) if e.kind == EventKind.ATTEMPT]
    assert len(attempts) == 1
    assert ledger.verify(case.case_id)


def test_different_idempotency_keys_are_separate_attempts(ledger):
    case = ledger.open_case(payment_id="pay_idem2", amount_paise=to_paise(10))
    ledger.record_transition(case.case_id, CaseStatus.ACTING)
    ledger.act(case.case_id, action="retry_payment", to_status=CaseStatus.ACTING,
               idempotency_key="k1")
    got = ledger.act(case.case_id, action="retry_payment", to_status=CaseStatus.ACTING,
                     idempotency_key="k2")
    assert got.attempt_count == 2


def test_expected_seq_rejects_a_stale_writer(ledger):
    """Two workers read the same case; only the first write wins."""
    case = ledger.open_case(payment_id="pay_race2", amount_paise=to_paise(10))
    stale_seq = case.seq

    ledger.record_transition(case.case_id, CaseStatus.ACTING)   # worker A wins

    with pytest.raises(ConcurrentModification, match="moved on"):
        ledger.record_transition(case.case_id, CaseStatus.STOPPED,
                                 expected_seq=stale_seq)        # worker B is stale

    assert ledger.require_case(case.case_id).status == CaseStatus.ACTING
    assert ledger.verify(case.case_id)


def test_expected_seq_allows_the_current_writer(ledger):
    case = ledger.open_case(payment_id="pay_race3", amount_paise=to_paise(10))
    got = ledger.record_transition(case.case_id, CaseStatus.ACTING,
                                   expected_seq=case.seq)
    assert got.status == CaseStatus.ACTING


def test_to_status_is_rejected_on_non_transition_events(ledger):
    """Silently ignoring it would let a caller think it changed state."""
    case = ledger.open_case(payment_id="pay_bad_kind", amount_paise=to_paise(10))
    with pytest.raises(LedgerError, match="only meaningful on a TRANSITION"):
        ledger.append(case.case_id, EventKind.NOTE, to_status=CaseStatus.RECOVERED)
    assert ledger.require_case(case.case_id).status == CaseStatus.OPEN


def test_verify_does_not_mutate(ledger):
    """verify() is a read-only check — it must not repair as a side effect."""
    case = ledger.open_case(payment_id="pay_pure", amount_paise=to_paise(10))
    ledger.record_transition(case.case_id, CaseStatus.ACTING)

    with sqlite3.connect(str(ledger.db_path)) as conn:
        conn.execute("UPDATE cases SET attempt_count=42 WHERE case_id=?",
                     (case.case_id,))
        conn.commit()

    assert ledger.verify(case.case_id) is False
    assert ledger.require_case(case.case_id).attempt_count == 42   # untouched
    assert ledger.replay(case.case_id).attempt_count == 0          # log says 0

    ledger.rebuild_projection(case.case_id)                        # explicit repair
    assert ledger.require_case(case.case_id).attempt_count == 0
    assert ledger.verify(case.case_id) is True
