"""An append-only record of everything that spent money or reached a customer.

The track's bar asks for an audit trail. What existed was three fragments that
did not join, and the one built for the job was dead:

  - `logging/AuditLogger` writes `data/audit_logs/case_<uuid>.jsonl`. Its only
    caller is `RecoveryAgent.run()`, which is invoked nowhere. 1,302 files, all
    from a code path that was removed.
  - the case `trail` is a mutable list inside the payment record, rewritten
    wholesale on every flush. A thing you can silently rewrite is not a trail.
  - `escalation_queue`'s JSONL is genuinely append-only, but only for tickets.

None of them carries a batch run, so "what did this run do, and on whose
authority" has never been answerable.

This is one table, two subjects. **Immutability is enforced by SQLite, not by
convention** — triggers ABORT any UPDATE or DELETE, so an auditor's guarantee
does not rest on every future writer remembering to be careful. Money is
integer paise; a float total that drifts is not evidence.

The storage layer is lifted from the `ledger.py` deleted in commit 3046b72
(WAL, `synchronous=FULL`, `BEGIN IMMEDIATE` for seq allocation, Decimal money).
What is deliberately left behind is that module's `cases` projection: it modelled
its own case lifecycle, which would be a *second* source of truth alongside
`StateStore`, `record["closed"]` and `ladder`. This codebase's comment history is
largely the story of two copies of one fact drifting apart, and a hackathon is
not the moment to add another.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Iterator

CASE = "case"
BATCH_RUN = "batch_run"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_type  TEXT NOT NULL,              -- 'case' | 'batch_run'
    subject_id    TEXT NOT NULL,
    seq           INTEGER NOT NULL,
    kind          TEXT NOT NULL,
    -- denormalised on purpose: "what happened to this payment" and "what did
    -- this run do" are the same table read two ways, with no subquery.
    payment_id    TEXT NOT NULL DEFAULT '',
    batch_run_id  TEXT NOT NULL DEFAULT '',
    actor         TEXT NOT NULL DEFAULT '',
    action        TEXT NOT NULL DEFAULT '',
    result        TEXT NOT NULL DEFAULT '',
    reason        TEXT NOT NULL DEFAULT '',
    amount_paise  INTEGER NOT NULL DEFAULT 0,
    payload       TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    UNIQUE(subject_type, subject_id, seq)
);
"""

#: Applied after the table, because a trigger on a missing table is an error and
#: an index on a missing column makes an older database unopenable.
_SCHEMA_GUARDS = """
CREATE INDEX IF NOT EXISTS idx_events_payment ON events(payment_id, event_id);
CREATE INDEX IF NOT EXISTS idx_events_run     ON events(batch_run_id, event_id);
CREATE INDEX IF NOT EXISTS idx_events_kind    ON events(kind, event_id);

-- The whole point. Immutability at the storage layer, not by convention:
-- no future writer can rewrite history by forgetting to be careful.
CREATE TRIGGER IF NOT EXISTS events_no_update
BEFORE UPDATE ON events
BEGIN
    SELECT RAISE(ABORT, 'events is append-only: UPDATE forbidden');
END;

CREATE TRIGGER IF NOT EXISTS events_no_delete
BEFORE DELETE ON events
BEGIN
    SELECT RAISE(ABORT, 'events is append-only: DELETE forbidden');
END;
"""


# ── money ────────────────────────────────────────────────────────────────
#
# Integer paise everywhere. `/api/payments` sums floats today; at these amounts
# the drift is invisible, which is exactly what makes it dangerous in a number
# presented as measured.

def to_paise(rupees: Any) -> int:
    if rupees in (None, ""):
        return 0
    return int((Decimal(str(rupees)) * 100).quantize(Decimal("1"),
                                                     rounding=ROUND_HALF_UP))


def to_rupees(paise: int) -> Decimal:
    return (Decimal(int(paise or 0)) / 100).quantize(Decimal("0.01"),
                                                     rounding=ROUND_HALF_UP)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path() -> Path:
    return Path(os.getenv("STATE_DIR", "data")) / "audit.db"


class AuditLog:
    """Append-only event log. One instance per database file, per process."""

    _INSTANCES: dict[Path, "AuditLog"] = {}
    _INSTANCES_LOCK = threading.Lock()

    def __new__(cls, db_path: str | Path | None = None) -> "AuditLog":
        key = Path(db_path or _db_path()).resolve()
        with cls._INSTANCES_LOCK:
            inst = cls._INSTANCES.get(key)
            if inst is None:
                inst = super().__new__(cls)
                inst._initialised = False
                cls._INSTANCES[key] = inst
            return inst

    def __init__(self, db_path: str | Path | None = None, timeout: float = 30.0):
        if getattr(self, "_initialised", False):
            return
        self._path = Path(db_path or _db_path())
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout
        self._lock = threading.Lock()
        self._ensure_schema()
        self._initialised = True

    @classmethod
    def reset_instances(cls) -> None:
        """For tests."""
        with cls._INSTANCES_LOCK:
            cls._INSTANCES.clear()

    # ── connection ───────────────────────────────────────────────────────

    @contextmanager
    def _conn(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._path), timeout=self._timeout,
                               isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute(f"PRAGMA busy_timeout={int(self._timeout * 1000)}")
            if write:
                # seq allocation is read-then-write; IMMEDIATE takes the write
                # lock up front so two appenders cannot pick the same number.
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            if write:
                conn.execute("COMMIT")
        except BaseException:
            if write:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        # Not via `_conn(write=True)`: `executescript` commits any open
        # transaction itself, so an explicit COMMIT afterwards has nothing to
        # close and raises.
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            conn.executescript(_SCHEMA_GUARDS)

    # ── write ────────────────────────────────────────────────────────────

    def record(
        self,
        kind: str,
        *,
        subject_type: str = CASE,
        subject_id: str = "",
        payment_id: str = "",
        batch_run_id: str = "",
        actor: str = "agent",
        action: str = "",
        result: str = "",
        reason: str = "",
        amount_paise: int = 0,
        amount_rupees: Any = None,
        **payload: Any,
    ) -> int:
        """Append one fact. Returns its seq.

        Never raises: an audit write must not be able to fail a recovery. A
        missing row is a gap in the record; a raised exception here would be a
        customer who did not get their payment link because the logging broke.
        """
        subject_id = subject_id or payment_id or batch_run_id
        if not subject_id:
            return 0
        if amount_rupees is not None and not amount_paise:
            amount_paise = to_paise(amount_rupees)
        try:
            with self._lock, self._conn(write=True) as conn:
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) AS s FROM events "
                    "WHERE subject_type = ? AND subject_id = ?",
                    (subject_type, subject_id),
                ).fetchone()
                seq = int(row["s"]) + 1
                conn.execute(
                    "INSERT INTO events (subject_type, subject_id, seq, kind, "
                    "payment_id, batch_run_id, actor, action, result, reason, "
                    "amount_paise, payload, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (subject_type, subject_id, seq, kind, payment_id,
                     batch_run_id, actor, action, result, reason,
                     int(amount_paise or 0), json.dumps(payload, default=str),
                     _now()),
                )
                return seq
        except Exception as exc:      # pragma: no cover - defensive
            print(f"[audit] could not record {kind}: {exc}", flush=True)
            return 0

    # ── read ─────────────────────────────────────────────────────────────

    @staticmethod
    def _row(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        try:
            d["payload"] = json.loads(d.get("payload") or "{}")
        except (ValueError, TypeError):
            d["payload"] = {}
        return d

    def for_payment(self, payment_id: str, limit: int = 500) -> list[dict]:
        """Everything that happened to one case, in order — including the events
        a batch run caused, because both carry `payment_id`."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE payment_id = ? "
                "ORDER BY event_id LIMIT ?", (payment_id, limit)).fetchall()
        return [self._row(r) for r in rows]

    def for_run(self, batch_run_id: str, limit: int = 5000) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE batch_run_id = ? "
                "ORDER BY event_id LIMIT ?", (batch_run_id, limit)).fetchall()
        return [self._row(r) for r in rows]

    def batch_activity(self, since_event_id: int = 0,
                       limit: int = 200) -> list[dict]:
        """Everything batch-shaped, oldest first. The batch window's feed.

        It is a straight read of this table — not a narration layer — which is
        what lets the UI say "every line here is the append-only log" and mean
        it.

        Two modes, so a hard refresh restores the log exactly as it was:
          - `since_event_id > 0` (incremental poll): the events AFTER the
            cursor, in order — one indexed range scan every couple of seconds.
          - `since_event_id == 0` (a fresh page load): the most RECENT `limit`
            events, still returned oldest-first. Returning the oldest `limit`
            instead meant that once a demo wrote more than `limit` batch events,
            a reload showed the very first ones ever logged and had to crawl
            forward a page at a time to reach what was actually on screen. The
            tail is what "keep the log as it is" means.
        """
        with self._conn() as conn:
            if int(since_event_id or 0) > 0:
                rows = conn.execute(
                    "SELECT * FROM events WHERE event_id > ? AND "
                    "(batch_run_id != '' OR subject_type = ? OR kind = ?) "
                    "ORDER BY event_id LIMIT ?",
                    (int(since_event_id), BATCH_RUN, CASE_LABELED,
                     int(limit))).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM (SELECT * FROM events WHERE "
                    "(batch_run_id != '' OR subject_type = ? OR kind = ?) "
                    "ORDER BY event_id DESC LIMIT ?) ORDER BY event_id",
                    (BATCH_RUN, CASE_LABELED, int(limit))).fetchall()
        return [self._row(r) for r in rows]

    def of_kind(self, kind: str, *, batch_run_id: str | None = None,
                limit: int = 5000) -> list[dict]:
        sql = "SELECT * FROM events WHERE kind = ?"
        args: list[Any] = [kind]
        if batch_run_id is not None:
            sql += " AND batch_run_id = ?"
            args.append(batch_run_id)
        sql += " ORDER BY event_id LIMIT ?"
        args.append(limit)
        with self._conn() as conn:
            return [self._row(r) for r in conn.execute(sql, args).fetchall()]

    def recovered_paise(self, batch_run_id: str) -> int:
        """Money attributed to one run.

        A join, not a heuristic: the executor stamps `batch_run_id` on the
        record, `_mark_recovered` reads it back onto the event, and this sums
        those events. No timestamp windows, no guessing which run "probably"
        caused a payment.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(amount_paise), 0) AS t FROM events "
                "WHERE kind = ? AND batch_run_id = ?",
                (MONEY_RECOVERED, batch_run_id)).fetchone()
        return int(row["t"])

    def summary(self) -> dict[str, Any]:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
            kinds = conn.execute(
                "SELECT kind, COUNT(*) AS c FROM events GROUP BY kind "
                "ORDER BY c DESC").fetchall()
        return {"events": int(total), "by_kind": {r["kind"]: int(r["c"]) for r in kinds}}


# ── the vocabulary ───────────────────────────────────────────────────────
#
# Named constants rather than loose strings, so a typo is an ImportError instead
# of an event that silently never matches a query.

BATCH_OPENED = "batch_run.opened"
BATCH_PLANNED = "batch_run.planned"
BATCH_PLAN_REJECTED = "batch_run.plan_rejected"
BATCH_FINISHED = "batch_run.finished"
BATCH_ABORTED = "batch_run.aborted"

CASE_SELECTED = "case.selected"
CASE_SKIPPED = "case.skipped"
CASE_EXCEPTION = "case.exception"
CASE_CLOSED = "case.closed"
CASE_LABELED = "case.labeled"
CASE_REASONING = "case.reasoning"

ACTION_ATTEMPTED = "action.attempted"
ACTION_RESULT = "action.result"

CYCLE_OPENED = "batch_cycle.opened"
CYCLE_WAVE = "batch_cycle.wave"
CYCLE_FINISHED = "batch_cycle.finished"
LADDER_RUNG = "ladder.rung_climbed"
ESCALATION_RAISED = "escalation.raised"
MONEY_RECOVERED = "money.recovered"
BUDGET_EXHAUSTED = "budget.exhausted"


_default: AuditLog | None = None


def log() -> AuditLog:
    """The process-wide log. Cheap to call; the instance is cached per path."""
    global _default
    if _default is None or Path(_default._path).resolve() != _db_path().resolve():
        _default = AuditLog()
    return _default


def record(kind: str, **kw: Any) -> int:
    """Module-level shorthand, so a choke point is one line and one import."""
    return log().record(kind, **kw)
