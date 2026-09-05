"""One batch run: what it was allowed to do, what it did, and what came back.

Two decisions shape this file.

**The run is its events.** Nothing here keeps a running total in a field and
hopes it stays true. `projection()` folds the append-only log back into a report,
so the number on screen and the number in the audit trail cannot disagree —
there is only one of them. It also means the report is honest about time: a batch
finishes *sending* in seconds and customers pay over the following minutes, so
recovered money is resolved on read and climbs after the run has closed. A total
frozen at `finished_at` would be wrong in the direction that flatters us.

**Stopping is checked in four places, because there are four ways to overrun.**
A budget bounds what may be spent, a deadline bounds how long it may take, a
consecutive-failure count catches a systemic fault (an expired key failing every
case one at a time), and an abort flag lets a human stop it. Each writes why it
stopped. A run that halts without saying which of the four fired is a run nobody
can trust the totals of.

What is parallel here is I/O, not judgement. The decision was made once, by the
plan; the workers below only carry it out, which is why they can run at four at a
time without needing to agree about anything.
"""
from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from recovery_agent import audit
from recovery_agent.batch import executor as ex
from recovery_agent.batch.plan import BatchBudget, BatchPlan
from recovery_agent.batch.tiers import amount_tier

#: Concurrency is over network calls, not reasoning. Kept low deliberately: the
#: workers share one Razorpay account and one state file, and going wider buys
#: nothing once those are the constraint.
WORKERS = max(1, int(os.getenv("BATCH_WORKERS", "4")))

OPEN, DONE, ABORTED = "running", "finished", "aborted"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── the live run ─────────────────────────────────────────────────────────

@dataclass
class BatchRun:
    """A single execution of one plan set over one batch."""
    batch_key: str
    plans: dict[str, BatchPlan]              # tier -> plan
    budget: BatchBudget = field(default_factory=BatchBudget)
    dry_run: bool = False
    started_by: str = "dashboard"
    #: Set when this run is one wave of a cycle. Purely identity — the run
    #: behaves identically either way; the cycle reads these back off the
    #: audit trail to tell its waves apart.
    wave: int = 0
    cycle_id: str = ""
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")

    spend: ex.Spend = field(default_factory=ex.Spend)
    status: str = OPEN
    stop_reason: str = ""
    decisions: list[ex.Decision] = field(default_factory=list)

    _abort: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _consecutive_failures: int = 0
    _deadline: float = 0.0

    # -- lifecycle ---------------------------------------------------------

    def abort(self, why: str = "aborted by a human") -> None:
        """Stop before the next case, and before any side effect in flight.

        An action already begun is allowed to finish: a payment link created and
        never mentioned to the customer is worse than one extra email.
        """
        self._abort.set()
        with self._lock:
            if not self.stop_reason:
                self.stop_reason = why

    @property
    def aborted(self) -> bool:
        return self._abort.is_set()

    def _stop_now(self) -> str:
        """Which stopping rule has fired, or '' to keep going."""
        if self._abort.is_set():
            return self.stop_reason or "aborted"
        if self._deadline and _now().timestamp() > self._deadline:
            return "wallclock"
        if (self.budget.abort_after_consecutive_failures
                and self._consecutive_failures
                >= self.budget.abort_after_consecutive_failures):
            return "consecutive_failures"
        return ""

    # -- execution ---------------------------------------------------------

    def execute(self, records: list[dict], *,
                on_decision: Callable[[ex.Decision], None] | None = None
                ) -> dict[str, Any]:
        """Work the batch and return the report.

        Cases are grouped by amount band because that is the unit a plan was
        made for: `merchant_dunning_rules.md` prescribes a different window,
        channel set and incentive for a INR 400 order than a INR 60,000 one, so
        one plan across both would apply one band's policy to the other band's
        customers.
        """
        self._deadline = _now().timestamp() + self.budget.max_wallclock_seconds
        candidates = list(records)
        at_risk = sum(audit.to_paise(r.get("amount")) for r in candidates)

        audit.record(audit.BATCH_OPENED, subject_type=audit.BATCH_RUN,
                     subject_id=self.run_id, batch_run_id=self.run_id,
                     actor=self.started_by, batch_key=self.batch_key,
                     dry_run=self.dry_run, candidates=len(candidates),
                     amount_paise=at_risk, budget=self.budget.as_dict(),
                     workers=WORKERS, wave=self.wave, cycle_id=self.cycle_id)
        for tier, plan in self.plans.items():
            audit.record(audit.BATCH_PLANNED, subject_type=audit.BATCH_RUN,
                         subject_id=self.run_id, batch_run_id=self.run_id,
                         tier=tier, plan=plan.digest(),
                         # Nested, not spread: the plan carries its own
                         # `tier` and `batch_key` and would collide.
                         detail=plan.as_dict())

        for tier, group in _by_tier(candidates).items():
            plan = self.plans.get(tier)
            if plan is None:
                # Not silently dropped. A band nobody planned for is a band
                # whose cases still need somebody, so they go to the agent.
                for rec in group:
                    self._settle(self._unhandled(
                        rec, f"no plan for tier {tier}"), on_decision)
                continue
            self._work_group(group, plan, on_decision)

        with self._lock:
            if self.status == OPEN:
                self.status = ABORTED if self.aborted else DONE
        audit.record(
            audit.BATCH_ABORTED if self.status == ABORTED else audit.BATCH_FINISHED,
            subject_type=audit.BATCH_RUN, subject_id=self.run_id,
            batch_run_id=self.run_id, reason=self.stop_reason,
            **self.spend.as_dict())
        return self.report()

    def _unhandled(self, rec: dict, reason: str) -> ex.Decision:
        """A case the run declined to work, recorded rather than dropped."""
        pid = str(rec.get("payment_id") or "")
        paise = audit.to_paise(rec.get("amount"))
        audit.record(audit.CASE_EXCEPTION, payment_id=pid,
                     batch_run_id=self.run_id, actor="batch_run",
                     reason=reason, amount_paise=paise, dry_run=self.dry_run)
        return ex.Decision(payment_id=pid, outcome=ex.EXCEPTION,
                           reason=reason, amount_paise=paise)

    def _work_group(self, group: list[dict], plan: BatchPlan,
                    on_decision) -> None:
        """One band, worked concurrently. The plan is frozen and shared; the
        only mutable thing the workers touch is `spend`, which is locked."""
        def one(rec: dict) -> ex.Decision:
            stop = self._stop_now()
            if stop:
                return ex.Decision(
                    payment_id=str(rec.get("payment_id") or ""),
                    outcome=ex.BUDGET, reason=f"stopped: {stop}",
                    amount_paise=audit.to_paise(rec.get("amount")))
            try:
                decision = ex.work_case(rec, plan, run_id=self.run_id,
                                        budget=self.budget, spend=self.spend,
                                        dry_run=self.dry_run)
            except Exception as exc:          # pragma: no cover - defensive
                decision = ex.Decision(
                    payment_id=str(rec.get("payment_id") or ""),
                    outcome=ex.EXCEPTION,
                    reason=f"{type(exc).__name__}: {exc}",
                    amount_paise=audit.to_paise(rec.get("amount")))
            self._count(decision)
            return decision

        workers = 1 if self.dry_run else min(WORKERS, max(1, len(group)))
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="batch") as pool:
            for decision in pool.map(one, group):
                self._settle(decision, on_decision)

    def _count(self, decision: ex.Decision) -> None:
        """Update the failure streak from inside the worker.

        A pool submits every item eagerly, so a counter that only the consumer
        updates is read after the later workers have already started — and the
        systemic fault this exists to catch (an expired key failing every case
        identically) runs through the whole batch anyway.
        """
        with self._lock:
            if decision.outcome == ex.ACTED:
                self._consecutive_failures = 0
            elif decision.outcome == ex.EXCEPTION:
                self._consecutive_failures += 1

    def _settle(self, decision: ex.Decision, on_decision) -> None:
        with self._lock:
            self.decisions.append(decision)
            if decision.outcome == ex.BUDGET and not self.stop_reason:
                self.stop_reason = decision.reason
        if on_decision:
            try:
                on_decision(decision)
            except Exception:                 # pragma: no cover - a listener
                pass                          # must not stop a run

    # -- reporting ---------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """The live view. `projection()` is the same numbers from the log."""
        base = projection(self.run_id)
        base.update({"status": self.status, "stop_reason": self.stop_reason,
                     "dry_run": self.dry_run, "wave": self.wave,
                     "cycle_id": self.cycle_id,
                     "budget": self.budget.as_dict(),
                     "spend": self.spend.as_dict(),
                     "decisions": [d.as_dict() for d in self.decisions]})
        if self.dry_run:
            base.update(self._projected())
        return base

    def _projected(self) -> dict[str, Any]:
        """What a dry run would have done, counted from its decisions.

        `projection()` counts actions by their `action.result` events, and a dry
        run correctly writes none — so read from the log it looks like a run
        that decided to do nothing, which is the opposite of what it found. The
        live path is untouched: this only fills in the column a run without side
        effects cannot have.
        """
        acted = [d for d in self.decisions if d.outcome == ex.ACTED]
        return {
            "acted": len(acted),
            "selected": len(acted),
            "acted_paise": sum(d.charged_paise for d in acted),
            "discount_paise": sum(d.discount_paise for d in acted),
            "acted_rupees": float(audit.to_rupees(
                sum(d.charged_paise for d in acted))),
            "links_created": sum(1 for d in acted
                                 if d.action == "link_and_notify"),
            "emails_sent": sum(1 for d in acted
                               if d.action in ("link_and_notify", "notify_only")),
        }


def _by_tier(records: Iterable[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for rec in records:
        out.setdefault(amount_tier(rec.get("amount")).key, []).append(rec)
    return out


# ── the report, folded out of the events ─────────────────────────────────

def projection(run_id: str) -> dict[str, Any]:
    """Rebuild a run's report from the audit log alone.

    This is the number that gets shown. It is a fold over immutable events
    rather than a counter someone remembered to increment, so it survives a
    restart, cannot drift from the trail it is meant to summarise, and keeps
    climbing as customers pay after the run has closed.
    """
    events = audit.log().for_run(run_id)
    out: dict[str, Any] = {
        "batch_run_id": run_id, "batch_key": "", "started_by": "",
        "started_at": "", "finished_at": "", "status": OPEN, "dry_run": False,
        "candidates": 0, "selected": 0, "acted": 0, "skipped": 0,
        "exceptions": 0, "deferred": 0, "escalations": 0,
        "at_risk_paise": 0, "acted_paise": 0, "discount_paise": 0,
        "recovered_paise": 0, "links_created": 0, "emails_sent": 0,
        "skipped_by_reason": {}, "exception_by_reason": {},
        "stop_reason": "", "events": len(events),
    }
    recovered_by_case: dict[str, int] = {}

    for e in events:
        kind, payload = e["kind"], e.get("payload") or {}
        if kind == audit.BATCH_OPENED:
            out.update(batch_key=payload.get("batch_key", ""),
                       started_by=e.get("actor", ""),
                       started_at=e["created_at"],
                       wave=int(payload.get("wave") or 0),
                       cycle_id=str(payload.get("cycle_id") or ""),
                       dry_run=bool(payload.get("dry_run")),
                       candidates=int(payload.get("candidates") or 0),
                       at_risk_paise=int(e.get("amount_paise") or 0))
        elif kind in (audit.BATCH_FINISHED, audit.BATCH_ABORTED):
            out["finished_at"] = e["created_at"]
            out["status"] = ABORTED if kind == audit.BATCH_ABORTED else DONE
            out["stop_reason"] = e.get("reason") or out["stop_reason"]
        elif kind == audit.ACTION_RESULT:
            out["acted"] += 1
            out["acted_paise"] += int(e.get("amount_paise") or 0)
            out["discount_paise"] += int(payload.get("discount_paise") or 0)
            if payload.get("link_url"):
                out["links_created"] += 1
            if payload.get("channels"):
                out["emails_sent"] += 1
        elif kind == audit.CASE_SKIPPED:
            bucket = "deferred" if _is_deferral(e) else "skipped"
            out[bucket] += 1
            _tally(out["skipped_by_reason"], e.get("reason"))
        elif kind == audit.CASE_EXCEPTION:
            out["exceptions"] += 1
            _tally(out["exception_by_reason"], e.get("reason"))
        elif kind == audit.ESCALATION_RAISED:
            out["escalations"] += 1
        elif kind == audit.BUDGET_EXHAUSTED:
            out["stop_reason"] = out["stop_reason"] or (e.get("reason") or "")
        elif kind == audit.MONEY_RECOVERED:
            # Keyed by case, so a payment reported twice — a webhook and a poll
            # seeing the same settlement — is counted once.
            recovered_by_case[e.get("payment_id") or e["subject_id"]] = \
                int(e.get("amount_paise") or 0)

    out["recovered_paise"] = sum(recovered_by_case.values())
    out["selected"] = out["acted"]
    out["net_paise"] = out["recovered_paise"] - out["discount_paise"]
    out["recovery_rate"] = (round(len(recovered_by_case) / out["acted"], 4)
                            if out["acted"] else 0.0)
    for key in ("at_risk", "acted", "discount", "recovered", "net"):
        out[f"{key}_rupees"] = float(audit.to_rupees(out[f"{key}_paise"]))
    return out


def _is_deferral(event: dict) -> bool:
    """Quiet hours and the frequency cap are recorded as skips, but they mean
    'not now' rather than 'not this case', and reporting them together would
    make a compliant pause look like a case we gave up on."""
    reason = str(event.get("reason") or "")
    return reason.startswith(("quiet_hours", "frequency_cap", "opt_out"))


def _tally(bucket: dict[str, int], reason: Any) -> None:
    key = str(reason or "unknown").split(":")[0].strip() or "unknown"
    bucket[key] = bucket.get(key, 0) + 1


# ── the live registry, so a run can be aborted from a request thread ─────

_RUNS: dict[str, BatchRun] = {}
_RUNS_LOCK = threading.Lock()


def register(run: BatchRun) -> BatchRun:
    with _RUNS_LOCK:
        _RUNS[run.run_id] = run
    return run


def get(run_id: str) -> BatchRun | None:
    with _RUNS_LOCK:
        return _RUNS.get(run_id)


def live() -> list[BatchRun]:
    with _RUNS_LOCK:
        return [r for r in _RUNS.values() if r.status == OPEN]


def forget(run_id: str) -> None:
    with _RUNS_LOCK:
        _RUNS.pop(run_id, None)
