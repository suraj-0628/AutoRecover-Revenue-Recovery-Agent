"""The audit trail the track asks for.

What existed was three fragments that did not join, and the one built for the
job was dead — `AuditLogger`'s only caller, `RecoveryAgent.run()`, is invoked
nowhere, leaving 1,302 files from a removed code path. None of them carried a
batch run, so "what did this run do, and on whose authority" was unanswerable.
"""
import pathlib
import sqlite3

import pytest

from recovery_agent import audit


@pytest.fixture()
def log(tmp_path):
    audit.AuditLog.reset_instances()
    yield audit.AuditLog(tmp_path / "audit.db")
    audit.AuditLog.reset_instances()


# ── immutability is the whole point ──────────────────────────────────────

def test_an_event_cannot_be_updated(log, tmp_path):
    """Enforced by SQLite, not by convention. An auditor's guarantee must not
    rest on every future writer remembering to be careful."""
    log.record(audit.MONEY_RECOVERED, payment_id="p", amount_rupees=100)
    with pytest.raises(sqlite3.Error, match="append-only"):
        with sqlite3.connect(str(tmp_path / "audit.db")) as c:
            c.execute("UPDATE events SET amount_paise = 999999")


def test_an_event_cannot_be_deleted(log, tmp_path):
    log.record(audit.MONEY_RECOVERED, payment_id="p", amount_rupees=100)
    with pytest.raises(sqlite3.Error, match="append-only"):
        with sqlite3.connect(str(tmp_path / "audit.db")) as c:
            c.execute("DELETE FROM events")


def test_tampering_leaves_the_money_intact(log, tmp_path):
    log.record(audit.MONEY_RECOVERED, payment_id="p", batch_run_id="r",
               amount_rupees=2374.05)
    for sql in ("UPDATE events SET amount_paise = 1", "DELETE FROM events"):
        with pytest.raises(sqlite3.Error):
            with sqlite3.connect(str(tmp_path / "audit.db")) as c:
                c.execute(sql)
    assert log.recovered_paise("r") == 237405


def test_seq_is_per_subject_and_gapless(log):
    for _ in range(3):
        log.record(audit.ACTION_ATTEMPTED, payment_id="pay_a")
    log.record(audit.ACTION_ATTEMPTED, payment_id="pay_b")
    assert [e["seq"] for e in log.for_payment("pay_a")] == [1, 2, 3]
    assert [e["seq"] for e in log.for_payment("pay_b")] == [1]


def test_a_duplicate_seq_is_rejected(log, tmp_path):
    log.record(audit.ACTION_ATTEMPTED, payment_id="p")
    with pytest.raises(sqlite3.IntegrityError):
        with sqlite3.connect(str(tmp_path / "audit.db")) as c:
            c.execute(
                "INSERT INTO events (subject_type, subject_id, seq, kind, created_at) "
                "VALUES ('case', 'p', 1, 'forged', '2026-01-01')")


# ── money is integer paise ───────────────────────────────────────────────

@pytest.mark.parametrize("rupees,paise", [
    (2374.05, 237405), ("2374.05", 237405), (0, 0), (None, 0), (99975, 9997500),
])
def test_money_converts_without_float_drift(rupees, paise):
    assert audit.to_paise(rupees) == paise


def test_a_long_run_of_discounted_amounts_does_not_drift(log):
    """The reason for integer paise: at these sizes float error is invisible,
    which is what makes it dangerous in a number presented as measured."""
    for _ in range(1000):
        log.record(audit.MONEY_RECOVERED, payment_id="p", batch_run_id="r",
                   amount_rupees="2374.05")
    assert log.recovered_paise("r") == 237405 * 1000


# ── the join: one table, two views ───────────────────────────────────────

def test_a_case_and_a_run_are_the_same_table_read_two_ways(log):
    log.record(audit.BATCH_OPENED, subject_type=audit.BATCH_RUN,
               subject_id="r1", batch_run_id="r1")
    log.record(audit.CASE_SELECTED, payment_id="p1", batch_run_id="r1")
    log.record(audit.MONEY_RECOVERED, payment_id="p1", batch_run_id="r1",
               amount_rupees=500)
    log.record(audit.CASE_SELECTED, payment_id="p2")          # not in the run

    assert [e["kind"] for e in log.for_payment("p1")] == [
        audit.CASE_SELECTED, audit.MONEY_RECOVERED]
    assert len(log.for_run("r1")) == 3
    assert log.for_payment("p2")[0]["batch_run_id"] == ""


def test_recovered_paise_counts_only_that_run(log):
    """Attribution is a join, not a timestamp heuristic."""
    log.record(audit.MONEY_RECOVERED, payment_id="a", batch_run_id="r1",
               amount_rupees=100)
    log.record(audit.MONEY_RECOVERED, payment_id="b", batch_run_id="r2",
               amount_rupees=250)
    log.record(audit.MONEY_RECOVERED, payment_id="c", amount_rupees=999)  # live path
    assert log.recovered_paise("r1") == 10000
    assert log.recovered_paise("r2") == 25000


def test_money_recovered_after_a_run_finishes_still_counts(log):
    """A batch finishes sending in seconds; customers pay over minutes. The
    total is resolved on read, so it climbs afterwards — which is correct."""
    log.record(audit.BATCH_FINISHED, subject_type=audit.BATCH_RUN,
               subject_id="r1", batch_run_id="r1")
    assert log.recovered_paise("r1") == 0
    log.record(audit.MONEY_RECOVERED, payment_id="late", batch_run_id="r1",
               amount_rupees=1234.05)
    assert log.recovered_paise("r1") == 123405


# ── it must never break a recovery ───────────────────────────────────────

def test_a_failed_write_returns_rather_than_raising(log, monkeypatch):
    """A missing row is a gap in the record. A raised exception here would be a
    customer who did not get their link because the logging broke."""
    def boom(*a, **k):
        raise sqlite3.OperationalError("disk is full")
    monkeypatch.setattr(log, "_conn", boom)
    assert log.record(audit.ACTION_ATTEMPTED, payment_id="p") == 0


def test_an_event_with_no_subject_is_dropped_not_raised(log):
    assert log.record(audit.ACTION_ATTEMPTED) == 0


# ── shape ────────────────────────────────────────────────────────────────

def test_payload_round_trips_as_json(log):
    log.record(audit.ACTION_RESULT, payment_id="p", link_id="plink_1",
               rails=["upi", "netbanking"])
    payload = log.for_payment("p")[0]["payload"]
    assert payload["link_id"] == "plink_1"
    assert payload["rails"] == ["upi", "netbanking"]


def test_summary_counts_by_kind(log):
    log.record(audit.CASE_SKIPPED, payment_id="a", reason="already_paid")
    log.record(audit.CASE_SKIPPED, payment_id="b", reason="no_contact")
    log.record(audit.MONEY_RECOVERED, payment_id="c", amount_rupees=1)
    s = log.summary()
    assert s["events"] == 3
    assert s["by_kind"][audit.CASE_SKIPPED] == 2


def test_the_vocabulary_is_constants_not_loose_strings():
    """A typo should be an ImportError, not an event that silently never
    matches a query."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
           / "audit.py").read_text()
    for name in ("BATCH_OPENED", "CASE_SKIPPED", "MONEY_RECOVERED",
                 "LADDER_RUNG", "ESCALATION_RAISED", "BUDGET_EXHAUSTED"):
        assert f"{name} = " in src


def test_one_instance_per_database_file(tmp_path):
    audit.AuditLog.reset_instances()
    try:
        assert audit.AuditLog(tmp_path / "a.db") is audit.AuditLog(tmp_path / "a.db")
        assert audit.AuditLog(tmp_path / "a.db") is not audit.AuditLog(tmp_path / "b.db")
    finally:
        audit.AuditLog.reset_instances()
