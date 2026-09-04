"""The opposite batch: a population of customers who behave like people.

The batch system's own driver proves the machinery; this proves the *strategy*.
Every archetype here acts only through surfaces a real customer touches — the
inbox (the rig's outbox sink), a payment link (the fake gateway registry), a
retried order — and never by reaching into the recovery system's internals.
Where an event would arrive from the gateway in production (a link attempt
bouncing), the sim writes what the webhook would have written, through the same
`StateStore` the servers use, whose file lock exists for exactly this kind of
cross-process writer.

The sim also owns the clock: a plan's 18-hour retry is honoured as "the retry
window passed" once the scheduled job has existed for a few seconds. And it
knows its own ground truth — which archetypes would ever pay, and how much —
so a run against it ends in a score, not an anecdote:

    recovered X of Y recoverable, in N waves, with 0 compliance violations

Voice never happens here: SuperU credits are real money reserved for the
judges, `VOICE_CALLS_ENABLED` stays false, and the ladder routes around the
missing rung exactly as it would for a merchant who never bought calling.
"""
from __future__ import annotations

import json
import random
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Which archetypes pay, if the system treats them the way their situation
#: calls for. `drop_then_try` pays only when an agent-judged second touch
#: reaches it, so in champion-off runs its PASS is "landed on the agent's
#: desk with full history" rather than "paid".
PAYS_AUTOMATICALLY = ("pays_first_nudge", "pays_on_retry", "rail_switcher",
                      "fails_differently")
PAYS_FOR_THE_AGENT = ("drop_then_try",)
NEVER_RECOVERABLE = ("never_pays", "opted_out")

#: How each archetype enters the store: (decline_strategy, extra fields).
#: `drop_then_try` carries the aftermath of its live phase — it was on the
#: page, got the push, dismissed it — which is exactly what a hand-off records.
_SEEDS: dict[str, tuple[str, dict[str, Any]]] = {
    "pays_first_nudge": ("user_dropoff", {}),
    "pays_on_retry": ("insufficient_funds", {}),
    "rail_switcher": ("card_expired", {}),
    "drop_then_try": ("user_dropoff", {
        "ladder": {"page_push": {"at": "2026-09-04T08:00:00+00:00",
                                 "detail": "dismissed after 4s"}},
        "push_outcome": {"action": "dismissed", "seconds_shown": 4},
    }),
    "fails_differently": ("user_dropoff", {}),
    "never_pays": ("card_expired", {}),
    "opted_out": ("user_dropoff", {"opted_out": True}),
}


@dataclass
class SimCustomer:
    payment_id: str
    archetype: str
    amount: float
    email: str
    name: str
    order_id: str
    state: str = "waiting"
    notes: list[str] = field(default_factory=list)
    _acts_at: float = 0.0

    def note(self, what: str) -> None:
        self.notes.append(what)


def build_population(per_archetype: int = 1, *, amounts=(1499.0, 7999.0),
                     seed: int = 7) -> list[SimCustomer]:
    """A population spread across archetypes and amount bands, so every
    (cause x band) plan is exercised, not just the medium one."""
    rng = random.Random(seed)
    out: list[SimCustomer] = []
    n = 0
    for archetype in _SEEDS:
        for i in range(per_archetype):
            n += 1
            amount = amounts[(n + i) % len(amounts)] + rng.randint(0, 9)
            pid = f"pay_sim_{archetype[:12]}_{i}"
            out.append(SimCustomer(
                payment_id=pid, archetype=archetype, amount=float(amount),
                email=f"{archetype}.{i}@sim.fake", name=f"Sim {archetype} {i}",
                order_id=f"order_sim_{archetype[:12]}_{i}"))
    return out


def seed_store(customers: list[SimCustomer], state_dir: Path) -> None:
    """Write the population as post-hoc failed payments — the batch's natural
    input: failures from hours ago whose customers are long gone."""
    payments = {}
    for c in customers:
        strategy, extra = _SEEDS[c.archetype]
        payments[c.payment_id] = {
            "payment_id": c.payment_id, "amount": c.amount, "currency": "INR",
            "status": "failed", "failure_code": "",
            "decline_strategy": strategy,
            "failure_reason": f"seeded {strategy}",
            # Both, as the live create-order flow writes both: the link
            # watcher later overwrites `order_id` with a link id, and a
            # retried-order payment must still match the case through
            # `original_order_id` or the money is verified and then dropped.
            "order_id": c.order_id,
            "original_order_id": c.order_id,
            "customer": {"email": c.email, "name": c.name,
                         "contact": "+919000000099"},
            "created_at": "2026-09-04T06:00:00+00:00",
            **extra,
        }
    state_dir.mkdir(exist_ok=True)
    (state_dir / "live_payments.json").write_text(json.dumps(payments, indent=2))
    for name in ("live_jobs.json", "live_pending.json"):
        (state_dir / name).write_text("{}")
    (state_dir / "fake_gateway.json").write_text(json.dumps(
        {"captured": {}, "paid_links": {}, "paid_orders": {}, "payments": []}))
    (state_dir / "audit.db").unlink(missing_ok=True)
    import shutil
    shutil.rmtree(state_dir / "outbox", ignore_errors=True)


class Simulator(threading.Thread):
    """Runs the population against a live rig until told to stop."""

    def __init__(self, customers: list[SimCustomer], state_dir: Path,
                 base_url: str, tick_seconds: float = 1.5):
        super().__init__(daemon=True, name="customer-sim")
        self.customers = customers
        self.state = state_dir
        self.base = base_url
        self.tick_seconds = tick_seconds
        self.stop_flag = threading.Event()
        self._store = None
        self._rng = random.Random(11)

    # -- senses ------------------------------------------------------------

    def _registry(self) -> dict:
        try:
            return json.loads((self.state / "fake_gateway.json").read_text())
        except Exception:
            return {}

    def _emails_delivered(self) -> dict[str, int]:
        """payment_id -> delivered email count, from the dispatch log."""
        out: dict[str, int] = {}
        log = self.state / "outbox" / "dispatch_log.jsonl"
        if not log.exists():
            return out
        for line in log.read_text().splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            pid = entry.get("payment_id", "")
            delivered = any(r.get("channel") == "email" and r.get("delivered")
                            for r in entry.get("results", []))
            if pid and delivered:
                out[pid] = out.get(pid, 0) + 1
        return out

    def _link_for(self, pid: str) -> dict | None:
        links = [l for l in self._registry().get("links_created", [])
                 if (l.get("notes") or {}).get("original_payment") == pid]
        return links[-1] if links else None

    def _job_for(self, pid: str) -> dict | None:
        try:
            jobs = json.loads((self.state / "live_jobs.json").read_text())
        except Exception:
            return None
        for job in jobs.values():
            if job.get("payment_id") == pid and job.get("status") not in (
                    "completed", "cancelled"):
                return job
        return None

    # -- hands -------------------------------------------------------------

    def _pay_link(self, c: SimCustomer, link: dict) -> None:
        reg = self._registry()
        reg.setdefault("paid_links", {})[link["link_id"]] = {
            "amount": float(link["amount"]),
            "payment_id": f"cap_sim_{c.payment_id}"}
        (self.state / "fake_gateway.json").write_text(json.dumps(reg))
        c.state = "paid"
        c.note(f"paid link {link['link_id']} for {link['amount']}")

    def _pay_order(self, c: SimCustomer) -> None:
        """The retried order goes through — reported the way a completed
        checkout reports itself, then verified against the (fake) gateway."""
        cap = f"cap_sim_{c.payment_id}"
        reg = self._registry()
        reg.setdefault("paid_orders", {})[c.order_id] = {
            "amount": c.amount, "payment_id": cap}
        reg.setdefault("captured", {})[cap] = {
            "amount": int(round(c.amount * 100)), "order_id": c.order_id}
        (self.state / "fake_gateway.json").write_text(json.dumps(reg))
        try:
            req = urllib.request.Request(
                self.base + "/api/payment-succeeded",
                data=json.dumps({"payment_id": c.payment_id,
                                 "razorpay_payment_id": cap,
                                 "razorpay_order_id": c.order_id}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=15).read()
            c.state = "paid"
            c.note("retried order captured and reported")
        except Exception as exc:
            c.note(f"payment-succeeded report failed: {exc}")

    def _bounce_link(self, c: SimCustomer) -> None:
        """The link attempt fails for a NEW reason — written the way gateway
        telemetry writes it, straight onto the record."""
        store = self._get_store()
        store.refresh()
        store.update_payment(
            c.payment_id, decline_strategy="insufficient_funds",
            failure_code="",
            failure_reason="recovery link attempt declined: insufficient funds")
        store.flush()
        c.state = "bounced_once"
        c.note("link attempt bounced: now a funds case")

    def _get_store(self):
        if self._store is None:
            from recovery_agent.state_store import StateStore
            self._store = StateStore()
        return self._store

    # -- the population's behaviour ----------------------------------------

    def run(self) -> None:
        while not self.stop_flag.is_set():
            try:
                self._tick()
            except Exception:                    # pragma: no cover
                pass
            self.stop_flag.wait(self.tick_seconds)

    def _tick(self) -> None:
        emails = self._emails_delivered()
        now = time.monotonic()

        for c in self.customers:
            if c.state == "paid":
                continue
            if c.archetype in ("never_pays", "opted_out"):
                continue                          # they do nothing, ever

            if c.archetype == "pays_first_nudge":
                link = self._link_for(c.payment_id)
                if emails.get(c.payment_id) and link:
                    self._after_jitter(c, now,
                                       lambda: self._pay_link(c, link))

            elif c.archetype == "rail_switcher":
                link = self._link_for(c.payment_id)
                if emails.get(c.payment_id) and link:
                    rails = str((link.get("notes") or {})
                                .get("allowed_rails") or "")
                    if "upi" in rails.lower():
                        self._after_jitter(c, now,
                                           lambda: self._pay_link(c, link))
                    elif c.state != "refused_rails":
                        c.state = "refused_rails"
                        c.note(f"link offered rails {rails!r}; wanted upi")

            elif c.archetype == "pays_on_retry":
                job = self._job_for(c.payment_id)
                if job:
                    # The scheduled window "passes" a few seconds after the
                    # job exists — the sim owns the clock.
                    self._after_jitter(c, now, lambda: self._pay_order(c),
                                       delay=5.0)

            elif c.archetype == "fails_differently":
                if c.state == "waiting":
                    link = self._link_for(c.payment_id)
                    if emails.get(c.payment_id) and link:
                        self._after_jitter(c, now,
                                           lambda: self._bounce_link(c))
                elif c.state == "bounced_once":
                    job = self._job_for(c.payment_id)
                    if job:
                        self._after_jitter(c, now,
                                           lambda: self._pay_order(c),
                                           delay=5.0)

            elif c.archetype == "drop_then_try":
                # Ignores the first contact entirely. Pays the second, if a
                # second ever reaches it — which takes an agent's judgement.
                contacts = emails.get(c.payment_id, 0)
                link = self._link_for(c.payment_id)
                if contacts >= 2 and link:
                    self._after_jitter(c, now, lambda: self._pay_link(c, link))
                elif contacts == 1 and c.state == "waiting":
                    c.state = "ignored_wave1"
                    c.note("saw wave 1's email; not interested yet")

    def _after_jitter(self, c: SimCustomer, now: float, act, *,
                      delay: float | None = None) -> None:
        """People do not click the second an email lands."""
        if c._acts_at == 0.0:
            c._acts_at = now + (delay if delay is not None
                                else self._rng.uniform(2.0, 6.0))
            return
        if now >= c._acts_at:
            c._acts_at = 0.0
            act()


# ── the scorecard ────────────────────────────────────────────────────────

def score(customers: list[SimCustomer], state_dir: Path) -> dict[str, Any]:
    """Ground truth vs what happened — the number the run is judged by."""
    payments = json.loads((state_dir / "live_payments.json").read_text())
    registry = json.loads((state_dir / "fake_gateway.json").read_text())
    links_by_pid: dict[str, int] = {}
    for link in registry.get("links_created", []):
        pid = (link.get("notes") or {}).get("original_payment", "")
        links_by_pid[pid] = links_by_pid.get(pid, 0) + 1

    emails: dict[str, int] = {}
    log = state_dir / "outbox" / "dispatch_log.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                pid = json.loads(line).get("payment_id", "")
            except ValueError:
                continue
            emails[pid] = emails.get(pid, 0) + 1

    rows = []
    recovered_paise = recoverable_paise = 0
    violations: list[str] = []
    for c in customers:
        rec = payments.get(c.payment_id) or {}
        got = float(rec.get("recovered_amount") or 0)
        recovered = rec.get("status") == "recovered" and got > 0
        if c.archetype in PAYS_AUTOMATICALLY + PAYS_FOR_THE_AGENT:
            recoverable_paise += int(round(c.amount * 100))
        if recovered:
            recovered_paise += int(round(got * 100))
        if c.archetype == "opted_out" and (
                emails.get(c.payment_id) or links_by_pid.get(c.payment_id)):
            violations.append(f"{c.payment_id}: opted out but was contacted")
        rows.append({
            "payment_id": c.payment_id, "archetype": c.archetype,
            "amount": c.amount, "recovered": recovered,
            "recovered_amount": got,
            "emails": emails.get(c.payment_id, 0),
            "links": links_by_pid.get(c.payment_id, 0),
            "final_status": rec.get("status", "?"),
            "sim_notes": c.notes,
        })

    return {
        "rows": rows,
        "recovered_paise": recovered_paise,
        "recoverable_paise": recoverable_paise,
        "capture_rate": (round(recovered_paise / recoverable_paise, 4)
                         if recoverable_paise else 0.0),
        "violations": violations,
    }
