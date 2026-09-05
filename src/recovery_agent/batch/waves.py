"""The wave cycle: bin, decide once, apply, watch, re-bin, repeat.

This is the batch agent — deliberately not the live agent. It never reasons
about an individual case; it decides *when the live agent's reasoning is
needed*, and applies it everywhere else:

    wave N:  re-read every open case fresh
             bin them: cause x amount band          (deterministic)
             per bin, get a plan:
               - a cached plan that still covers the bin's rung, or
               - the CHAMPION: one case worked by the real agent, its
                 decisions distilled into the plan  (the intelligence)
               - or the policy default               (the safe floor)
             run the bin through the executor        (zero LLM)
             settle: customers pay on their own clock
    wave N+1: the recovered are gone; the rest have re-binned themselves —
             a dropoff whose link attempt bounced on funds is now a funds
             case, and the funds plan is what it gets

Waves end for a stated reason, never by exhaustion of the loop counter alone:
everyone recovered; a wave changed nothing (no new money, no case moved bins,
no rung climbed — sending the same batch again would send the same batch
again); the wave allowance is spent; or a human aborted.

Second touches concentrate, they do not spread. The ladder forbids repeating a
rung, and each bin's policy deliberately routes post-offer rungs to the agent —
so wave 2 is structurally incapable of re-spamming wave 1's customers. What it
can do is retry quietly, work the re-binned, and put the genuinely stuck on a
person's desk with their full history attached.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from recovery_agent import audit
from recovery_agent.agent import ladder
from recovery_agent.batch import distill, planner
from recovery_agent.batch import run as batch_run
from recovery_agent.batch.executor import _stamp
from recovery_agent.batch.plan import BatchBudget, BatchPlan, PlanRejected
from recovery_agent.batch.tiers import amount_tier

RUNNING, DONE, ABORTED = "running", "finished", "aborted"

#: A champion session is a real agent run over a gated LLM; it is allowed to be
#: slow, but not to hold a whole cycle hostage.
CHAMPION_TIMEOUT_SECONDS = 300.0


@dataclass
class WaveConfig:
    max_waves: int = 3
    #: How long a wave waits for money before concluding. Customers pay over
    #: minutes; the link poller checks every 15s — shorter than ~30s and a
    #: wave would out-run its own instruments.
    settle_seconds: float = 60.0
    #: "live" runs a champion through the real agent when a bin needs a
    #: decision; "off" uses the policy defaults — the mode for CI and for any
    #: room where the LLM proxy cannot be trusted to show up.
    champion_mode: str = "off"
    budget: BatchBudget = field(default_factory=BatchBudget)

    def as_dict(self) -> dict[str, Any]:
        return {"max_waves": self.max_waves,
                "settle_seconds": self.settle_seconds,
                "champion_mode": self.champion_mode,
                "budget": self.budget.as_dict()}


@dataclass
class WaveCycle:
    """One cycle of waves over one set of cases."""
    payment_ids: list[str]
    config: WaveConfig = field(default_factory=WaveConfig)
    #: Blocking callable (record, context) -> None that runs one full live
    #: agent session. Injected by the web layer; None means champions cannot
    #: run and the defaults carry every bin. The cycle must not import the
    #: web layer to find it — that inversion is how circular imports start.
    champion_runner: Callable[[dict, str], None] | None = None
    on_decision: Callable | None = None
    started_by: str = "dashboard"
    cycle_id: str = field(
        default_factory=lambda: f"cyc_{uuid.uuid4().hex[:12]}")

    status: str = RUNNING
    stop_reason: str = ""
    waves: list[dict[str, Any]] = field(default_factory=list)
    champions: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    exceptions: list[dict[str, Any]] = field(default_factory=list)

    _abort: threading.Event = field(default_factory=threading.Event,
                                    repr=False)
    _plans: dict[tuple[str, str], BatchPlan] = field(default_factory=dict,
                                                     repr=False)

    # -- control -----------------------------------------------------------

    def abort(self, why: str = "aborted by a human") -> None:
        self._abort.set()
        if not self.stop_reason:
            self.stop_reason = why
        for rid in self.run_ids:
            live = batch_run.get(rid)
            if live is not None and live.status == batch_run.OPEN:
                live.abort(why)

    @property
    def aborted(self) -> bool:
        return self._abort.is_set()

    # -- the loop ----------------------------------------------------------

    def execute(self) -> dict[str, Any]:
        from recovery_agent.state_store import StateStore
        store = StateStore()

        audit.record(audit.CYCLE_OPENED, subject_type=audit.BATCH_RUN,
                     subject_id=self.cycle_id, actor=self.started_by,
                     cases=len(self.payment_ids), **self.config.as_dict())

        for wave in range(1, self.config.max_waves + 1):
            if self.aborted:
                break

            records = self._open_records(store)
            if not records:
                self.stop_reason = "all_settled"
                break

            shape = {r["payment_id"]: self._shape(r) for r in records}
            summary = self._run_wave(wave, records, store)
            self.waves.append(summary)
            audit.record(audit.CYCLE_WAVE, subject_type=audit.BATCH_RUN,
                         subject_id=self.cycle_id, actor=self.started_by,
                         reason=f"wave {wave}", **summary)

            if self.aborted:
                break
            self._settle()

            after = self._open_records(store)
            recovered_now = len(records) - len(after)
            moved = any(self._shape(r) != shape.get(r["payment_id"])
                        for r in after)
            if not after:
                self.stop_reason = "all_settled"
                break
            if recovered_now == 0 and not moved and not summary["acted"]:
                # Nothing changed and nothing was even attempted: running the
                # same wave again would run the same wave again.
                self.stop_reason = "dry"
                break

        if not self.stop_reason and not self.aborted:
            self.stop_reason = "max_waves"
        self.status = ABORTED if self.aborted else DONE
        audit.record(audit.CYCLE_FINISHED, subject_type=audit.BATCH_RUN,
                     subject_id=self.cycle_id, actor=self.started_by,
                     result=self.status, reason=self.stop_reason,
                     waves=len(self.waves))
        return self.report()

    def _run_wave(self, wave: int, records: list[dict], store) -> dict:
        from recovery_agent.agent.classify import BATCH_BY_KEY, classify

        by_key: dict[str, list[dict]] = {}
        held: list[dict] = []
        for rec in records:
            # A human's label can say "this one needs judgement, not a plan" —
            # a stated price objection, a dead contact. Those never enter a
            # bin: the whole point of the label is that the next move is a
            # decision about ONE customer.
            label = (rec.get("operator_label") or {}).get("code") or ""
            if label:
                try:
                    from recovery_agent.labels import get as _label
                    spec = _label(label)
                except Exception:
                    spec = None
                if spec and spec.get("to_agent"):
                    self._note_exception(
                        rec, f"labeled by a person: {spec['label']}")
                    continue
            key = classify(rec)
            meta = BATCH_BY_KEY.get(key or "")
            if key and meta and meta.get("runnable"):
                by_key.setdefault(key, []).append(rec)
            else:
                # Un-runnable is not un-tracked: risk, unclassified and their
                # kin are cases with no shared cause to plan on, so they are
                # the agent's one at a time.
                held.append({"payment_id": rec.get("payment_id", ""),
                             "batch": key or "unclassified",
                             "why": "no shared cause to plan on"})

        summary = {"wave": wave, "cases": len(records), "acted": 0,
                   "skipped": 0, "deferred": 0, "exceptions": 0,
                   "links_created": 0, "champions": [], "runs": [],
                   "held": len(held)}

        for key, group in sorted(by_key.items()):
            if self.aborted:
                break
            plans, excluded = self._plans_for(key, group, wave, store)
            if not plans:
                for rec in group:
                    self._note_exception(rec, "no plan could be made")
                summary["exceptions"] += len(group)
                continue

            run = batch_run.BatchRun(
                batch_key=key, plans=plans, budget=self.config.budget,
                started_by=self.started_by, wave=wave,
                cycle_id=self.cycle_id)
            batch_run.register(run)
            self.run_ids.append(run.run_id)
            for pid in excluded:
                # The champion was just worked by its own session — being
                # worked IS its treatment this wave. Stamped so its money
                # still lands on this run.
                _stamp(pid, run.run_id)
            workable = [r for r in group
                        if r.get("payment_id") not in excluded]
            report = run.execute(workable, on_decision=self.on_decision)

            summary["acted"] += report["acted"]
            summary["skipped"] += report["skipped"]
            summary["deferred"] += report["deferred"]
            summary["exceptions"] += report["exceptions"]
            summary["links_created"] += report["links_created"]
            summary["champions"].extend(excluded)
            summary["runs"].append(run.run_id)
            for decision in run.decisions:
                if decision.outcome == "exception":
                    # The run id rides along so that when the agent later works
                    # this case off the queue, whatever it recovers still lands
                    # on the run that referred it — without this, an
                    # agent-rescued exception is money the cycle earned and
                    # cannot count.
                    self._note_exception(
                        {"payment_id": decision.payment_id},
                        decision.reason, run_id=run.run_id)

        for rec in held:
            self._note_exception(rec, rec.get("why", ""))
        return summary

    # -- planning ----------------------------------------------------------

    def _plans_for(self, key: str, group: list[dict], wave: int,
                   store) -> tuple[dict[str, BatchPlan], set[str]]:
        """One plan per band present — champion, cache, or default."""
        excluded: set[str] = set()
        plans: dict[str, BatchPlan] = {}
        by_tier: dict[str, list[dict]] = {}
        for rec in group:
            by_tier.setdefault(amount_tier(rec.get("amount")).key,
                               []).append(rec)

        for tier, members in sorted(by_tier.items()):
            cached = self._plans.get((key, tier))
            modal = self._modal_rung(members)
            if cached is not None and self._covers(cached, modal):
                plans[tier] = cached
                continue

            plan: BatchPlan | None = None
            if (self.config.champion_mode == "live"
                    and self.champion_runner is not None):
                champ = self._run_champion(key, tier, members, modal, store)
                if champ is not None:
                    plan, pid = champ
                    excluded.add(pid)
                    self.champions.append(pid)

            if plan is None:
                try:
                    plan = planner.plan_for(key, tier)
                except PlanRejected as exc:
                    audit.record(audit.BATCH_PLAN_REJECTED,
                                 subject_type=audit.BATCH_RUN,
                                 subject_id=self.cycle_id,
                                 reason=str(exc), batch_key=key, tier=tier)
                    continue
            self._plans[(key, tier)] = plan
            plans[tier] = plan
        return plans, excluded

    def _run_champion(self, key: str, tier: str, members: list[dict],
                      modal: str | None, store) -> tuple[BatchPlan, str] | None:
        """One real case, worked by the real agent, becomes the bin's plan."""
        candidates = [r for r in members
                      if not ladder.pursuit_barred(r)
                      and not r.get("opted_out")]
        if modal:
            on_modal = [r for r in candidates
                        if (ladder.next_rung(r) or {}).get("rung") == modal]
            candidates = on_modal or candidates
        if not candidates:
            return None
        candidates.sort(key=lambda r: float(r.get("amount") or 0))
        champ = candidates[len(candidates) // 2]     # the median: ordinary
        pid = str(champ.get("payment_id") or "")

        before = distill.snapshot(champ)
        context = (
            f"CHAMPION CASE: this payment is the representative of a batch of "
            f"{len(members)} similar failures ({key}, {tier} band). Work it on "
            f"its own merits — what you decide here will inform how the rest "
            f"of the batch is treated, so decide it the way this cause "
            f"deserves, not the way this one customer's mood suggests.")

        audit.record(audit.CASE_SELECTED, payment_id=pid,
                     actor="wave_cycle", reason="champion",
                     batch_key=key, tier=tier, cycle_id=self.cycle_id)
        done = threading.Event()
        error: list[str] = []

        def _session() -> None:
            try:
                self.champion_runner(dict(champ), context)
            except Exception as exc:          # pragma: no cover - defensive
                error.append(f"{type(exc).__name__}: {exc}")
            finally:
                done.set()

        worker = threading.Thread(target=_session, daemon=True,
                                  name=f"champion-{pid}")
        worker.start()
        done.wait(CHAMPION_TIMEOUT_SECONDS)
        if not done.is_set() or error:
            audit.record(audit.CASE_EXCEPTION, payment_id=pid,
                         actor="wave_cycle", cycle_id=self.cycle_id,
                         reason=("champion session timed out" if not error
                                 else f"champion failed: {error[0]}"))
            return None

        after = store.get_payment(pid) or dict(champ)
        try:
            plan = distill.distill(after, before, batch_key=key, tier=tier)
        except distill.DistillationFailed as exc:
            audit.record(audit.CASE_EXCEPTION, payment_id=pid,
                         actor="wave_cycle", cycle_id=self.cycle_id,
                         reason=f"distillation: {exc}")
            # The session still happened and still counts as this case's
            # treatment; only the generalisation failed.
            return None
        return plan, pid

    # -- bookkeeping -------------------------------------------------------

    @staticmethod
    def _shape(record: dict) -> str:
        """What would have to change for a new wave to act differently."""
        from recovery_agent.agent.classify import classify
        nxt = ladder.next_rung(record)
        return f"{classify(record)}|{(nxt or {}).get('rung')}"

    def _open_records(self, store) -> list[dict]:
        out = []
        for pid in self.payment_ids:
            rec = store.get_payment(pid)
            if rec is None:
                continue
            if rec.get("status") in ("recovered", "escalated"):
                continue
            if rec.get("closed"):
                continue
            out.append(dict(rec))
        return out

    def _settle(self) -> None:
        """Give the customers their turn. Abort still lands within a second."""
        deadline = time.monotonic() + max(0.0, self.config.settle_seconds)
        while time.monotonic() < deadline and not self.aborted:
            time.sleep(min(1.0, deadline - time.monotonic()) or 0.01)

    def _note_exception(self, rec: dict, why: str, run_id: str = "") -> None:
        pid = rec.get("payment_id", "")
        for entry in self.exceptions:
            if entry["payment_id"] == pid:
                entry["why"] = why            # the latest wave's reason wins
                entry["batch_run_id"] = entry.get("batch_run_id") or run_id
                return
        self.exceptions.append({"payment_id": pid, "why": why,
                                "batch_run_id": run_id})

    @staticmethod
    def _modal_rung(members: list[dict]) -> str | None:
        counts: dict[str, int] = {}
        for rec in members:
            nxt = ladder.next_rung(rec)
            if nxt:
                counts[nxt["rung"]] = counts.get(nxt["rung"], 0) + 1
        return max(counts, key=counts.get) if counts else None

    @staticmethod
    def _covers(plan: BatchPlan, rung: str | None) -> bool:
        return rung is not None and plan.step_for(rung) is not None

    # -- reporting ---------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """The cycle's story: waves, money, and why it stopped.

        Money comes from the runs' projections — the same audit-log fold the
        rest of the system trusts — summed across the cycle's runs, so this
        number also keeps climbing after the cycle closes.
        """
        recovered = discount = acted_paise = 0
        links = emails = 0
        for rid in self.run_ids:
            proj = batch_run.projection(rid)
            recovered += proj["recovered_paise"]
            discount += proj["discount_paise"]
            acted_paise += proj["acted_paise"]
            links += proj["links_created"]
            emails += proj["emails_sent"]
        return {
            "cycle_id": self.cycle_id, "status": self.status,
            "stop_reason": self.stop_reason, "config": self.config.as_dict(),
            "cases": len(self.payment_ids), "waves": self.waves,
            "runs": self.run_ids, "champions": self.champions,
            "exceptions": self.exceptions,
            "recovered_paise": recovered, "discount_paise": discount,
            "net_paise": recovered - discount, "acted_paise": acted_paise,
            "links_created": links, "emails_sent": emails,
            "recovered_rupees": float(audit.to_rupees(recovered)),
            "net_rupees": float(audit.to_rupees(recovered - discount)),
        }


# ── the report, rebuilt from the log ─────────────────────────────────────

def cycle_projection(cycle_id: str) -> dict[str, Any] | None:
    """A cycle's report folded out of the audit log alone.

    The live `WaveCycle` object dies with its process; the money it moved must
    not. Every fact the report needs was written as an event when it happened —
    opened (config, case count), one event per wave (the summary), finished
    (status, stop reason) — and the runs are found by the cycle id each one
    carries. So a restart loses the *object* and keeps the *report*, which is
    the only one of the two an auditor ever asks for.

    Returns None for a cycle the log has never heard of.
    """
    log = audit.log()
    mine = lambda events: [e for e in events                 # noqa: E731
                           if e["subject_id"] == cycle_id]
    opened = mine(log.of_kind(audit.CYCLE_OPENED))
    if not opened:
        return None
    head = opened[0]

    wave_events = mine(log.of_kind(audit.CYCLE_WAVE))
    finished = mine(log.of_kind(audit.CYCLE_FINISHED))

    run_ids = [e["batch_run_id"] for e in log.of_kind(audit.BATCH_OPENED)
               if (e.get("payload") or {}).get("cycle_id") == cycle_id]

    recovered = discount = acted_paise = links = emails = 0
    exceptions: dict[str, dict[str, Any]] = {}
    for rid in run_ids:
        proj = batch_run.projection(rid)
        recovered += proj["recovered_paise"]
        discount += proj["discount_paise"]
        acted_paise += proj["acted_paise"]
        links += proj["links_created"]
        emails += proj["emails_sent"]
        for e in log.for_run(rid):
            if e["kind"] == audit.CASE_EXCEPTION and e["payment_id"]:
                # Last wave's reason wins, matching the live object.
                exceptions[e["payment_id"]] = {
                    "payment_id": e["payment_id"],
                    "why": e.get("reason", ""), "batch_run_id": rid}

    champions = [e["payment_id"] for e in log.of_kind(audit.CASE_SELECTED)
                 if (e.get("payload") or {}).get("cycle_id") == cycle_id
                 and e.get("reason") == "champion"]

    head_payload = head.get("payload") or {}
    end = finished[0] if finished else None
    return {
        "cycle_id": cycle_id,
        "status": ((ABORTED if end.get("result") == ABORTED else DONE)
                   if end else RUNNING),
        "stop_reason": (end or {}).get("reason", ""),
        "config": {k: head_payload.get(k) for k in
                   ("max_waves", "settle_seconds", "champion_mode", "budget")},
        "cases": int(head_payload.get("cases") or 0),
        "waves": [e.get("payload") or {} for e in wave_events],
        "runs": run_ids, "champions": champions,
        "exceptions": list(exceptions.values()),
        "recovered_paise": recovered, "discount_paise": discount,
        "net_paise": recovered - discount, "acted_paise": acted_paise,
        "links_created": links, "emails_sent": emails,
        "recovered_rupees": float(audit.to_rupees(recovered)),
        "net_rupees": float(audit.to_rupees(recovered - discount)),
        "rebuilt_from_audit": True,
    }


def known_cycle_ids(limit: int = 100) -> list[str]:
    """Every cycle the log has seen, newest first."""
    events = audit.log().of_kind(audit.CYCLE_OPENED, limit=limit)
    return [e["subject_id"] for e in reversed(events)]


# ── the registry the routes need ─────────────────────────────────────────

_CYCLES: dict[str, WaveCycle] = {}
_CYCLES_LOCK = threading.Lock()


def register(cycle: WaveCycle) -> WaveCycle:
    with _CYCLES_LOCK:
        _CYCLES[cycle.cycle_id] = cycle
    return cycle


def get(cycle_id: str) -> WaveCycle | None:
    with _CYCLES_LOCK:
        return _CYCLES.get(cycle_id)


def all_cycles() -> list[WaveCycle]:
    with _CYCLES_LOCK:
        return list(_CYCLES.values())
