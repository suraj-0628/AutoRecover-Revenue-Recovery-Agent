"""Case ledger — the durable source of truth for every recovery case.

Block D1 of REBUILD-PLAN.md.

Design contract
---------------
1. APPEND-ONLY.   `events` is never updated or deleted. It is the truth.
2. PROJECTION.    `cases` is a materialised view of `events`, rebuildable at any
                  time via `rebuild_projection()`. If the two ever disagree, the
                  event log wins. Live writes and replay share one `_apply()`
                  function so they cannot drift.
3. IDEMPOTENT.    `open_case()` is keyed on `payment_id`. Calling it twice for the
                  same payment returns the same case; it never creates two.
4. CRASH-SAFE.    SQLite in WAL mode with synchronous=FULL. A SIGKILL between any
                  two calls leaves a consistent, readable ledger.
5. INTEGER MONEY. All amounts are integer paise. Floats never touch money.
                  (AUDIT-FINDINGS S1-4: a rupee/paise mix-up put the escalation
                  threshold 100x too high. Fixed at the foundation.)
6. RECOVERY LINK. A recovered case records the *new* payment id that actually
                  captured the money. The original payment stays `failed` in
                  Razorpay forever, so without this the recovery rate can never
                  reconcile against the dashboard.

7. EVIDENCE.     A case cannot be declared RECOVERED without an observation
                  event proving money actually moved. Success is never
                  self-declared. (D2)

Not in this block: effectors (D3), sensing (D4).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

from recovery_agent.models import CaseStatus
from recovery_agent.statemachine import (
    IllegalTransition,
    MissingEvidence,
    TERMINAL,
    assert_transition,
    is_terminal,
    is_waiting,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB = _REPO_ROOT / "data" / "ledger.db"

#: Statuses from which no further work is scheduled. Owned by `statemachine`.
TERMINAL_STATUSES: frozenset[CaseStatus] = TERMINAL


# ── Money ────────────────────────────────────────────────────────────────────

def to_paise(rupees: float | int | str | Decimal) -> int:
    """Convert a rupee amount to integer paise. Half-up, never float rounding.

    Raises `ValueError` on anything unparseable, so callers handle one exception
    type rather than leaking `decimal.InvalidOperation` up the stack.
    """
    try:
        return int(
            (Decimal(str(rupees)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        )
    except (ArithmeticError, TypeError) as exc:
        raise ValueError(f"not a valid amount: {rupees!r}") from exc


def to_rupees(paise: int) -> Decimal:
    """Convert integer paise back to a exact rupee Decimal (for display only)."""
    return (Decimal(int(paise)) / 100).quantize(Decimal("0.01"))


# ── Records ──────────────────────────────────────────────────────────────────

class EventKind(str, Enum):
    OPENED = "opened"            # case created
    TRANSITION = "transition"    # status changed
    ATTEMPT = "attempt"          # an effector was invoked (D3)
    OBSERVATION = "observation"  # the sensor observed reality (D4)
    DECISION = "decision"        # the agent chose an action, with its reason (D6)
    NOTE = "note"                # agent reasoning / diagnosis, no state change


class LedgerEvent(BaseModel):
    """One immutable fact about a case."""
    event_id: int = 0
    case_id: str
    seq: int
    kind: EventKind
    created_at: datetime
    from_status: CaseStatus | None = None
    to_status: CaseStatus | None = None
    action: str = ""
    result: str = ""
    reason: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class CaseRecord(BaseModel):
    """Projection of a case, derived entirely from its events."""
    case_id: str
    payment_id: str
    customer_id: str = ""
    amount_paise: int = 0
    currency: str = "INR"
    failure_code: str = ""
    failure_reason: str = ""
    source: str = ""
    status: CaseStatus = CaseStatus.OPEN
    attempt_count: int = 0
    recovery_payment_id: str = ""
    recovered_amount_paise: int = 0
    wake_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    seq: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return is_terminal(self.status)

    @property
    def is_waiting(self) -> bool:
        """Parked on the outside world: a person (sensor) or a clock (scheduler)."""
        return is_waiting(self.status)

    @property
    def attributed_to_agent(self) -> bool:
        """Recovered *after* the agent did something.

        A customer can pay of their own accord before any effector runs. That is
        still a recovered case, but crediting it to the agent inflates the
        recovery rate — so attribution is derived, never assumed.
        """
        return self.status == CaseStatus.RECOVERED and self.attempt_count > 0

    @property
    def self_recovered(self) -> bool:
        """Recovered with no agent action taken."""
        return self.status == CaseStatus.RECOVERED and self.attempt_count == 0

    @property
    def amount_rupees(self) -> Decimal:
        return to_rupees(self.amount_paise)

    @property
    def recovered_amount_rupees(self) -> Decimal:
        return to_rupees(self.recovered_amount_paise)


class LedgerError(RuntimeError):
    pass


class CaseNotFound(LedgerError):
    pass


class ConcurrentModification(LedgerError):
    """Another worker changed the case since it was read (optimistic locking)."""
    pass


# ── Reducer — the single place state is derived from an event ────────────────

def _apply(state: dict[str, Any], ev: LedgerEvent) -> dict[str, Any]:
    """Fold one event into case state.

    Used by BOTH live writes and `rebuild_projection`, so the stored projection
    and a replay of the log cannot diverge.
    """
    s = dict(state)
    p = ev.payload or {}

    if ev.kind == EventKind.OPENED:
        s.update(
            payment_id=p.get("payment_id", s.get("payment_id", "")),
            customer_id=p.get("customer_id", ""),
            amount_paise=int(p.get("amount_paise", 0)),
            currency=p.get("currency", "INR"),
            failure_code=p.get("failure_code", ""),
            failure_reason=p.get("failure_reason", ""),
            source=p.get("source", ""),
            metadata=p.get("metadata", {}) or {},
            status=CaseStatus.OPEN.value,
            created_at=ev.created_at,
        )
    elif ev.kind == EventKind.TRANSITION:
        if ev.to_status is not None:
            s["status"] = ev.to_status.value
            # A wake time belongs to SCHEDULED and to nothing else; leaving the
            # state clears it so the scheduler can never pick up a stale case.
            s["wake_at"] = _norm_wake(p.get("wake_at")) \
                if ev.to_status == CaseStatus.SCHEDULED else ""
    elif ev.kind == EventKind.ATTEMPT:
        s["attempt_count"] = int(s.get("attempt_count", 0)) + 1
    elif ev.kind == EventKind.OBSERVATION:
        if p.get("recovered"):
            s["recovery_payment_id"] = p.get("recovery_payment_id", "")
            s["recovered_amount_paise"] = int(p.get("recovered_amount_paise", 0))
    # DECISION and NOTE never change derived state — deciding is not doing.

    s["seq"] = ev.seq
    s["updated_at"] = ev.created_at
    return s


# ── Ledger ───────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id                 TEXT PRIMARY KEY,
    payment_id              TEXT NOT NULL UNIQUE,
    customer_id             TEXT NOT NULL DEFAULT '',
    amount_paise            INTEGER NOT NULL DEFAULT 0,
    currency                TEXT NOT NULL DEFAULT 'INR',
    failure_code            TEXT NOT NULL DEFAULT '',
    failure_reason          TEXT NOT NULL DEFAULT '',
    source                  TEXT NOT NULL DEFAULT '',
    status                  TEXT NOT NULL,
    attempt_count           INTEGER NOT NULL DEFAULT 0,
    recovery_payment_id     TEXT NOT NULL DEFAULT '',
    recovered_amount_paise  INTEGER NOT NULL DEFAULT 0,
    wake_at                 TEXT NOT NULL DEFAULT '',
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    seq                     INTEGER NOT NULL DEFAULT 0,
    metadata                TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS events (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id      TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    from_status  TEXT,
    to_status    TEXT,
    action       TEXT NOT NULL DEFAULT '',
    result       TEXT NOT NULL DEFAULT '',
    reason       TEXT NOT NULL DEFAULT '',
    payload      TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    UNIQUE(case_id, seq)
);

"""

# Indexes and triggers are applied *after* column migrations, because some of
# them reference columns added by a later block. Creating them first makes an
# older ledger unopenable.
_SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_events_case ON events(case_id, seq);
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
CREATE INDEX IF NOT EXISTS idx_cases_wake ON cases(status, wake_at);

-- Append-only enforcement: the log is immutable at the storage layer, not by
-- convention. Any UPDATE or DELETE on `events` aborts the transaction.
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


class Ledger:
    """Durable, append-only case ledger.

    Safe across threads and processes: every operation opens its own connection
    in WAL mode, and writes run inside BEGIN IMMEDIATE so `seq` allocation is
    serialised.
    """

    def __init__(self, db_path: str | Path | None = None, timeout: float = 30.0):
        self.db_path = Path(
            db_path or os.getenv("LEDGER_DB_PATH") or _DEFAULT_DB
        ).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout
        self._init_lock = threading.Lock()
        self._ensure_schema()

    # ── connection plumbing ──

    @contextmanager
    def _conn(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=self._timeout,
                               isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(f"PRAGMA busy_timeout={int(self._timeout * 1000)}")
            if write:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    yield conn
                    conn.execute("COMMIT")
                except Exception:
                    try:
                        conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    raise
            else:
                yield conn
        finally:
            conn.close()

    #: Columns added after the first release, applied to older ledgers in order.
    _MIGRATIONS: tuple[tuple[str, str, str], ...] = (
        ("cases", "wake_at", "TEXT NOT NULL DEFAULT ''"),   # D2
    )

    def _ensure_schema(self) -> None:
        with self._init_lock, self._conn() as conn:
            conn.executescript(_SCHEMA)          # tables only
            self._migrate(conn)                  # add any missing columns
            conn.executescript(_SCHEMA_INDEXES)  # indexes/triggers may use them

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Additively add columns introduced by later blocks. Never destructive."""
        for table, column, decl in self._MIGRATIONS:
            cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    # ── writes ──

    def open_case(
        self,
        payment_id: str,
        amount_paise: int,
        customer_id: str = "",
        currency: str = "INR",
        failure_code: str = "",
        failure_reason: str = "",
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CaseRecord:
        """Create a case for a failed payment. Idempotent on `payment_id`.

        Two deliveries of the same failure — a double checkout callback, a
        redelivered webhook, a batch overlapping a live case — produce ONE case.
        """
        if not payment_id:
            raise LedgerError("payment_id is required")
        if not isinstance(amount_paise, int) or isinstance(amount_paise, bool):
            raise LedgerError(
                f"amount_paise must be int paise, got {type(amount_paise).__name__}. "
                "Use ledger.to_paise(rupees)."
            )
        if amount_paise < 0:
            raise LedgerError("amount_paise must be >= 0")

        existing = self.get_case_by_payment(payment_id)
        if existing is not None:
            return existing

        case_id = f"case_{uuid.uuid4().hex[:12]}"
        now = _utcnow()
        payload = {
            "payment_id": payment_id,
            "customer_id": customer_id,
            "amount_paise": amount_paise,
            "currency": currency,
            "failure_code": failure_code,
            "failure_reason": failure_reason,
            "source": source,
            "metadata": metadata or {},
        }

        try:
            with self._conn(write=True) as conn:
                # Re-check inside the transaction: another process may have won.
                row = conn.execute(
                    "SELECT case_id FROM cases WHERE payment_id = ?", (payment_id,)
                ).fetchone()
                if row is not None:
                    winner = row["case_id"]
                else:
                    winner = None
                    conn.execute(
                        "INSERT INTO cases (case_id, payment_id, customer_id, "
                        "amount_paise, currency, failure_code, failure_reason, "
                        "source, status, created_at, updated_at, seq, metadata) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (case_id, payment_id, customer_id, amount_paise, currency,
                         failure_code, failure_reason, source,
                         CaseStatus.OPEN.value, now.isoformat(), now.isoformat(),
                         1, json.dumps(metadata or {})),
                    )
                    conn.execute(
                        "INSERT INTO events (case_id, seq, kind, to_status, reason, "
                        "payload, created_at) VALUES (?,?,?,?,?,?,?)",
                        (case_id, 1, EventKind.OPENED.value, CaseStatus.OPEN.value,
                         "case opened", json.dumps(payload), now.isoformat()),
                    )
        except sqlite3.IntegrityError:
            winner = None  # lost the race; fall through to the re-read below

        found = self.get_case_by_payment(payment_id)
        if found is None:
            raise LedgerError(f"failed to open case for {payment_id}")
        return found

    def append(
        self,
        case_id: str,
        kind: EventKind,
        *,
        to_status: CaseStatus | None = None,
        action: str = "",
        result: str = "",
        reason: str = "",
        payload: dict[str, Any] | None = None,
        expected_seq: int | None = None,
    ) -> LedgerEvent:
        """Append one event and fold it into the projection, atomically.

        `expected_seq` gives optimistic concurrency: pass the `seq` you read and
        the write is rejected if another worker moved the case first. Two pollers
        picking up the same case then cannot both act on it.
        """
        with self._conn(write=True) as conn:
            state = self._load_state(conn, case_id, expected_seq)
            state, ev = self._append_in_txn(
                conn, state, kind, to_status=to_status, action=action,
                result=result, reason=reason, payload=payload or {},
            )
            _write_projection(conn, case_id, state)
        return ev

    def act(
        self,
        case_id: str,
        action: str,
        *,
        to_status: CaseStatus,
        result: str = "ok",
        request: dict[str, Any] | None = None,
        receipt: dict[str, Any] | None = None,
        reason: str = "",
        idempotency_key: str = "",
        wake_at: datetime | str | None = None,
        expected_seq: int | None = None,
    ) -> CaseRecord:
        """Record an effector call **and** the state change it caused, atomically.

        Effectors move money and send messages. Recording the attempt and the
        resulting transition as two separate writes leaves a window where a crash
        strands the case in ACTING with an attempt but no transition — every
        caller would then need its own reconciliation. One transaction removes
        the window.

        `idempotency_key` makes the whole operation replay-safe: if an attempt
        with that key already exists on this case, nothing is written and the
        current case is returned. Retrying a request after the effector
        succeeded but the ledger write failed cannot double-charge.
        """
        with self._conn(write=True) as conn:
            state = self._load_state(conn, case_id, expected_seq)

            if idempotency_key and _attempt_exists(conn, case_id, idempotency_key):
                return _row_state_to_case(state)

            attempt_payload: dict[str, Any] = {
                "request": request or {}, "receipt": receipt or {},
            }
            if idempotency_key:
                attempt_payload["idempotency_key"] = idempotency_key

            state, _ = self._append_in_txn(
                conn, state, EventKind.ATTEMPT, action=action, result=result,
                reason=reason, payload=attempt_payload,
            )
            transition_payload: dict[str, Any] = {}
            if wake_at is not None:
                transition_payload["wake_at"] = _norm_wake(wake_at)
            state, _ = self._append_in_txn(
                conn, state, EventKind.TRANSITION, to_status=to_status,
                reason=reason or action, payload=transition_payload,
            )
            _write_projection(conn, case_id, state)

        return self.require_case(case_id)

    # ── in-transaction primitives ──

    def _load_state(
        self, conn: sqlite3.Connection, case_id: str, expected_seq: int | None
    ) -> dict[str, Any]:
        row = conn.execute(
            "SELECT * FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        if row is None:
            raise CaseNotFound(case_id)
        state = _row_to_state(row)
        if expected_seq is not None and int(state["seq"]) != int(expected_seq):
            raise ConcurrentModification(
                f"{case_id} moved on: expected seq {expected_seq}, found "
                f"{state['seq']}. Re-read the case and decide again."
            )
        return state

    def _append_in_txn(
        self,
        conn: sqlite3.Connection,
        state: dict[str, Any],
        kind: EventKind,
        *,
        to_status: CaseStatus | None = None,
        action: str = "",
        result: str = "",
        reason: str = "",
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], LedgerEvent]:
        case_id = state["case_id"]
        from_status = CaseStatus(state["status"])

        if kind == EventKind.TRANSITION and to_status is not None:
            _validate_transition(conn, case_id, from_status, to_status, payload)
        elif to_status is not None:
            raise LedgerError(
                f"to_status is only meaningful on a TRANSITION event, not {kind.value}"
            )

        now = _utcnow()
        next_seq = int(state.get("seq", 0)) + 1
        ev = LedgerEvent(
            case_id=case_id, seq=next_seq, kind=kind, created_at=now,
            from_status=from_status, to_status=to_status, action=action,
            result=result, reason=reason, payload=payload,
        )
        conn.execute(
            "INSERT INTO events (case_id, seq, kind, from_status, to_status, "
            "action, result, reason, payload, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (case_id, next_seq, kind.value, from_status.value,
             to_status.value if to_status else None, action, result, reason,
             json.dumps(payload), now.isoformat()),
        )
        return _apply(state, ev), ev

    def record_transition(
        self,
        case_id: str,
        to_status: CaseStatus,
        reason: str = "",
        actor: str = "",
        wake_at: datetime | str | None = None,
        expected_seq: int | None = None,
    ) -> CaseRecord:
        """Move a case to a new status, or raise.

        Legality comes from `statemachine.LEGAL_TRANSITIONS`; the evidence rules
        live in `_validate_transition`. Both run inside the write transaction, so
        a concurrent writer cannot slip a case past a guard.

        There is deliberately **no force/override flag**. A ledger you can
        override is a ledger you cannot trust; to correct a case, append a
        further event rather than rewriting history.
        """
        payload: dict[str, Any] = {}
        if actor:
            payload["actor"] = actor
        if wake_at is not None:
            payload["wake_at"] = _norm_wake(wake_at)
        self.append(
            case_id, EventKind.TRANSITION, to_status=to_status, reason=reason,
            payload=payload, expected_seq=expected_seq,
        )
        return self.require_case(case_id)

    def record_attempt(
        self,
        case_id: str,
        action: str,
        *,
        result: str = "pending",
        request: dict[str, Any] | None = None,
        receipt: dict[str, Any] | None = None,
        reason: str = "",
        idempotency_key: str = "",
    ) -> LedgerEvent:
        """Record that an effector was invoked, with its request and receipt.

        `receipt` is whatever the outside world handed back (a Razorpay link id,
        an outbox path). It is the evidence the action really happened — D3
        depends on this being stored, not inferred.
        """
        payload: dict[str, Any] = {"request": request or {}, "receipt": receipt or {}}
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        return self.append(
            case_id, EventKind.ATTEMPT, action=action, result=result, reason=reason,
            payload=payload,
        )

    def record_observation(
        self,
        case_id: str,
        *,
        observed: str,
        recovered: bool = False,
        recovery_payment_id: str = "",
        recovered_amount_paise: int = 0,
        payload: dict[str, Any] | None = None,
    ) -> LedgerEvent:
        """Record something the sensor saw in the real world (D4)."""
        if recovered and not isinstance(recovered_amount_paise, int):
            raise LedgerError("recovered_amount_paise must be int paise")
        body = dict(payload or {})
        body.update(
            observed=observed,
            recovered=recovered,
            recovery_payment_id=recovery_payment_id,
            recovered_amount_paise=int(recovered_amount_paise),
        )
        return self.append(
            case_id, EventKind.OBSERVATION, result=observed, payload=body
        )

    def record_decision(
        self,
        case_id: str,
        action: str,
        reason: str,
        *,
        source: str = "",
        payload: dict[str, Any] | None = None,
        expected_seq: int | None = None,
    ) -> LedgerEvent:
        """Record what the agent chose and why, before anything is done about it.

        Kept separate from NOTE so decisions are directly queryable: evaluating an
        agent means reading its decisions, and filtering free-text notes for them
        is how eval data goes stale.
        """
        body = dict(payload or {})
        body["source"] = source
        return self.append(
            case_id, EventKind.DECISION, action=action, result=source,
            reason=reason, payload=body, expected_seq=expected_seq,
        )

    def decisions(self, case_id: str) -> list[LedgerEvent]:
        """Every decision the agent made on this case, in order."""
        return [e for e in self.events(case_id) if e.kind is EventKind.DECISION]

    def record_note(
        self, case_id: str, reason: str, payload: dict[str, Any] | None = None
    ) -> LedgerEvent:
        """Record agent reasoning or a diagnosis. Never changes derived state."""
        return self.append(case_id, EventKind.NOTE, reason=reason, payload=payload or {})

    # ── reads ──

    def get_case(self, case_id: str) -> CaseRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        return _row_to_case(row) if row else None

    def require_case(self, case_id: str) -> CaseRecord:
        case = self.get_case(case_id)
        if case is None:
            raise CaseNotFound(case_id)
        return case

    def get_case_by_payment(self, payment_id: str) -> CaseRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM cases WHERE payment_id = ?", (payment_id,)
            ).fetchone()
        return _row_to_case(row) if row else None

    def has_attempt(self, case_id: str, idempotency_key: str) -> bool:
        """True if an attempt with this key was already recorded.

        Lets a caller detect a replay *before* writing anything, so a repeated
        request is a genuine no-op rather than a partial one.
        """
        if not idempotency_key:
            return False
        with self._conn() as conn:
            return _attempt_exists(conn, case_id, idempotency_key)

    def events(self, case_id: str) -> list[LedgerEvent]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE case_id = ? ORDER BY seq", (case_id,)
            ).fetchall()
        return [_row_to_event(r) for r in rows]

    def open_cases(self, limit: int = 500) -> list[CaseRecord]:
        """Cases still needing work — the queue D4 and Track B poll."""
        placeholders = ",".join("?" * len(TERMINAL_STATUSES))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM cases WHERE status NOT IN ({placeholders}) "
                "ORDER BY created_at LIMIT ?",
                (*[s.value for s in TERMINAL_STATUSES], limit),
            ).fetchall()
        return [_row_to_case(r) for r in rows]

    def due_cases(self, now: datetime | None = None, limit: int = 500) -> list[CaseRecord]:
        """SCHEDULED cases whose wake time has passed — the scheduler's queue (D6/B2)."""
        cutoff = (now or _utcnow()).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM cases WHERE status = ? AND wake_at != '' "
                "AND wake_at <= ? ORDER BY wake_at LIMIT ?",
                (CaseStatus.SCHEDULED.value, cutoff, limit),
            ).fetchall()
        return [_row_to_case(r) for r in rows]

    def awaiting_customer_cases(self, limit: int = 500) -> list[CaseRecord]:
        """Cases with something outstanding in the world — the sensor's queue (D4)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM cases WHERE status = ? ORDER BY updated_at LIMIT ?",
                (CaseStatus.AWAITING_CUSTOMER.value, limit),
            ).fetchall()
        return [_row_to_case(r) for r in rows]

    def all_cases(self, limit: int = 1000) -> list[CaseRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM cases ORDER BY created_at LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_case(r) for r in rows]

    # ── integrity ──

    def replay(self, case_id: str) -> CaseRecord:
        """Recompute a case purely from its event log. Read-only.

        Use this to check the stored projection without touching the database —
        `verify()` is built on it.
        """
        return _row_state_to_case(self._replay_state(case_id))

    def _replay_state(self, case_id: str) -> dict[str, Any]:
        events = self.events(case_id)
        if not events:
            raise CaseNotFound(case_id)

        state: dict[str, Any] = {
            "case_id": case_id, "payment_id": "", "customer_id": "",
            "amount_paise": 0, "currency": "INR", "failure_code": "",
            "failure_reason": "", "source": "", "status": CaseStatus.OPEN.value,
            "attempt_count": 0, "recovery_payment_id": "",
            "recovered_amount_paise": 0, "wake_at": "", "seq": 0, "metadata": {},
            "created_at": events[0].created_at, "updated_at": events[0].created_at,
        }
        for ev in events:
            state = _apply(state, ev)
        return state

    def rebuild_projection(self, case_id: str) -> CaseRecord:
        """Replay the log and OVERWRITE the stored projection from it.

        The repair operation: if the projection has drifted, the log wins.
        Mutates. For a read-only check use `verify()` or `replay()`.
        """
        state = self._replay_state(case_id)
        with self._conn(write=True) as conn:
            _write_projection(conn, case_id, state)
        return self.require_case(case_id)

    def verify(self, case_id: str) -> bool:
        """True if the stored projection matches a replay of the log.

        Read-only — safe to call in assertions and health checks.
        """
        return self.require_case(case_id).model_dump() == self.replay(case_id).model_dump()


# ── transition validation ────────────────────────────────────────────────────

def _validate_transition(
    conn: sqlite3.Connection,
    case_id: str,
    frm: CaseStatus,
    to: CaseStatus,
    payload: dict[str, Any],
) -> None:
    """Enforce the state machine plus the evidence rules. Raises on violation.

    Runs inside the caller's write transaction, so the evidence it reads cannot
    change underneath it.
    """
    assert_transition(frm, to)

    # ── Evidence rule 1: recovery must be observed, never asserted ──
    #
    # This is the rule the old system most needed. Nothing stopped it from
    # declaring success on its own say-so, which is how the Chaos Gym could
    # report "100% recovery" while never contacting anyone (AUDIT-FINDINGS S2-1).
    # Money moved, or the case is not recovered.
    if to == CaseStatus.RECOVERED:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE case_id = ? AND kind = ? "
            "AND json_extract(payload, '$.recovered') = 1",
            (case_id, EventKind.OBSERVATION.value),
        ).fetchone()
        if not row or row["n"] == 0:
            raise MissingEvidence(
                frm, to,
                "no observation proves the money moved. Call record_observation("
                "recovered=True, recovery_payment_id=...) first — recovery is "
                "never self-declared",
            )

    # ── Evidence rule 2: you can only await a customer you actually reached ──
    if to == CaseStatus.AWAITING_CUSTOMER:
        # 'pending' counts — an async send in flight has still reached out.
        # 'error' and 'blocked' do not: nothing left the building.
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE case_id = ? AND kind = ? "
            "AND result NOT IN ('error', 'blocked')",
            (case_id, EventKind.ATTEMPT.value),
        ).fetchone()
        if not row or row["n"] == 0:
            raise MissingEvidence(
                frm, to,
                "nothing was sent. Call record_attempt() for a successful "
                "effector before waiting on the customer",
            )

    # ── Evidence rule 3: a scheduled case must say when to wake ──
    if to == CaseStatus.SCHEDULED and not _norm_wake(payload.get("wake_at")):
        raise MissingEvidence(
            frm, to, "SCHEDULED requires wake_at; a case with no wake time is "
                     "invisible to the scheduler and never runs again",
        )


def _attempt_exists(conn: sqlite3.Connection, case_id: str, key: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM events WHERE case_id = ? AND kind = ? "
        "AND json_extract(payload, '$.idempotency_key') = ? LIMIT 1",
        (case_id, EventKind.ATTEMPT.value, key),
    ).fetchone()
    return row is not None


def _norm_wake(value: Any) -> str:
    """Normalise a wake time to a UTC ISO string, or '' if absent."""
    if value in (None, ""):
        return ""
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


# ── row helpers ──────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _row_to_state(row: sqlite3.Row) -> dict[str, Any]:
    state = {k: row[k] for k in row.keys()}
    state["metadata"] = json.loads(state.get("metadata") or "{}")
    state["created_at"] = _parse_dt(state["created_at"])
    state["updated_at"] = _parse_dt(state["updated_at"])
    state["wake_at"] = state.get("wake_at") or ""
    return state


def _row_state_to_case(state: dict[str, Any]) -> CaseRecord:
    s = dict(state)
    wake = s.get("wake_at") or ""
    s["wake_at"] = _parse_dt(wake) if isinstance(wake, str) and wake else (
        wake if isinstance(wake, datetime) else None
    )
    return CaseRecord(**s)


def _row_to_case(row: sqlite3.Row) -> CaseRecord:
    return _row_state_to_case(_row_to_state(row))


def _row_to_event(row: sqlite3.Row) -> LedgerEvent:
    return LedgerEvent(
        event_id=row["event_id"],
        case_id=row["case_id"],
        seq=row["seq"],
        kind=EventKind(row["kind"]),
        created_at=_parse_dt(row["created_at"]),
        from_status=CaseStatus(row["from_status"]) if row["from_status"] else None,
        to_status=CaseStatus(row["to_status"]) if row["to_status"] else None,
        action=row["action"] or "",
        result=row["result"] or "",
        reason=row["reason"] or "",
        payload=json.loads(row["payload"] or "{}"),
    )


def _write_projection(conn: sqlite3.Connection, case_id: str, state: dict[str, Any]) -> None:
    conn.execute(
        "UPDATE cases SET payment_id=?, customer_id=?, amount_paise=?, currency=?, "
        "failure_code=?, failure_reason=?, source=?, status=?, attempt_count=?, "
        "recovery_payment_id=?, recovered_amount_paise=?, wake_at=?, updated_at=?, seq=?, "
        "metadata=? WHERE case_id=?",
        (
            state["payment_id"], state["customer_id"], int(state["amount_paise"]),
            state["currency"], state["failure_code"], state["failure_reason"],
            state["source"], state["status"], int(state["attempt_count"]),
            state["recovery_payment_id"], int(state["recovered_amount_paise"]),
            _norm_wake(state.get("wake_at")),
            _parse_dt(state["updated_at"]).isoformat(), int(state["seq"]),
            json.dumps(state.get("metadata") or {}), case_id,
        ),
    )


_default_ledger: Ledger | None = None
_default_lock = threading.Lock()


def get_ledger() -> Ledger:
    """Process-wide default ledger."""
    global _default_ledger
    with _default_lock:
        if _default_ledger is None:
            _default_ledger = Ledger()
    return _default_ledger
