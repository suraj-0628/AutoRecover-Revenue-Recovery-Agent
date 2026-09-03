"""Recovery sensor — observes whether the money actually came back.

Block D4 of REBUILD-PLAN.md. **The keystone.**

Everything before this block could be built and tested without ever knowing if a
recovery worked. That is exactly the hole the old system had: no component ever
reconciled an action against reality, so "recovered" was whatever the code chose
to assert, and the Chaos Gym could report 100% recovery without contacting anyone
(AUDIT-FINDINGS S2-1, Gap #1). D2 made recovery unassertable; D4 is the only
thing that can prove it.

How it works
------------
Poll `ledger.awaiting_customer_cases()` — every case with an offer outstanding —
ask Razorpay what happened to that offer, and write the answer to the ledger.

    outstanding offer ──probe──▶ paid in full   ──▶ observation + RECOVERED
                                 paid in part   ──▶ observation, keep waiting
                                 expired/void   ──▶ observation + back to the queue
                                 still pending  ──▶ **nothing written**

Three rules
-----------
1. **Write only when reality changed.** A sensor polling every 15 seconds that
   logged "still pending" each time would bury the real events. Silence is the
   normal case.
2. **The sensor never decides strategy.** A dead offer goes back on the work
   queue (`SCHEDULED`, wake now) for the agent to reconsider. Choosing what to do
   next is D6's job, not the sensor's.
3. **Partial payment is not recovery.** Anything less than the debt keeps the
   case open, with the amount recorded.

Polling, not webhooks — deliberately. Razorpay cannot reach localhost, the
webhook secret is unset, and a tunnel is one more thing to fail mid-demo. B4
adds webhook ingress for scale; polling remains the fallback.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from recovery_agent.ledger import CaseRecord, EventKind, Ledger, ConcurrentModification
from recovery_agent.models import CaseStatus

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = 15.0

# Outcomes a probe can report.
PAID = "paid"
PARTIALLY_PAID = "partially_paid"
PENDING = "pending"
EXPIRED = "expired"
CANCELLED = "cancelled"
ERROR = "error"

#: Outcomes meaning the offer is dead and the agent must choose again.
DEAD_OUTCOMES = frozenset({EXPIRED, CANCELLED})


@dataclass(frozen=True)
class Observation:
    """What the outside world says happened to an offer."""
    outcome: str
    recovered: bool = False
    recovery_payment_id: str = ""
    amount_paid_paise: int = 0
    detail: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def is_error(self) -> bool:
        return self.outcome == ERROR


@dataclass
class SensorRun:
    """Summary of one polling pass."""
    checked: int = 0
    recovered: int = 0
    changed: int = 0
    unchanged: int = 0
    dead: int = 0
    errors: int = 0
    skipped: int = 0

    def __str__(self) -> str:
        return (f"checked={self.checked} recovered={self.recovered} "
                f"changed={self.changed} unchanged={self.unchanged} "
                f"dead={self.dead} errors={self.errors} skipped={self.skipped}")


class Probe(Protocol):
    """Knows how to ask the world about one kind of receipt."""
    def supports(self, receipt: dict[str, Any]) -> bool: ...
    def check(self, receipt: dict[str, Any]) -> Observation: ...


# ── Probes ───────────────────────────────────────────────────────────────────

class OrderProbe:
    """Checks a recovery Order: is it paid, and by which payment?"""

    def __init__(self, api: Any = None, client: Any = None):
        self._api = api
        self._client = client

    @property
    def api(self) -> Any:
        if self._api is None:
            client = self._client
            if client is None:
                from recovery_agent.razorpay_client import RazorpayClient
                client = RazorpayClient()
            self._api = client.client.order
        return self._api

    def supports(self, receipt: dict[str, Any]) -> bool:
        return bool(receipt.get("order_id"))

    def check(self, receipt: dict[str, Any]) -> Observation:
        order_id = receipt["order_id"]
        try:
            order = self.api.fetch(order_id)
        except Exception as exc:
            return Observation(outcome=ERROR, error=f"{type(exc).__name__}: {exc}")

        status = order.get("status", "")
        amount = int(order.get("amount") or 0)
        paid = int(order.get("amount_paid") or 0)
        detail = {"order_id": order_id, "order_status": status,
                  "amount_paise": amount, "amount_paid_paise": paid}

        if status != "paid" and paid <= 0:
            # Razorpay has no "expired" order status; an order simply stays
            # `created` until something pays it. Expiry is the *offer's* concern.
            return Observation(outcome=PENDING, detail=detail)

        payment_id = self._captured_payment_id(order_id)
        detail["recovery_payment_id"] = payment_id

        if paid >= amount and amount > 0:
            return Observation(outcome=PAID, recovered=True,
                               recovery_payment_id=payment_id,
                               amount_paid_paise=paid, detail=detail)
        return Observation(outcome=PARTIALLY_PAID, recovered=False,
                           recovery_payment_id=payment_id,
                           amount_paid_paise=paid, detail=detail)

    def _captured_payment_id(self, order_id: str) -> str:
        """The payment that actually captured the money — the recovery linkage."""
        try:
            page = self.api.payments(order_id) or {}
        except Exception as exc:
            logger.warning("[sensor] order.payments(%s) failed: %s", order_id, exc)
            return ""
        for p in page.get("items") or []:
            if p.get("status") == "captured":
                return p.get("id", "")
        return ""


class PaymentLinkProbe:
    """Checks a Razorpay payment link."""

    def __init__(self, api: Any = None, client: Any = None):
        self._api = api
        self._client = client

    @property
    def api(self) -> Any:
        if self._api is None:
            client = self._client
            if client is None:
                from recovery_agent.razorpay_client import RazorpayClient
                client = RazorpayClient()
            self._api = client.client.payment_link
        return self._api

    def supports(self, receipt: dict[str, Any]) -> bool:
        return bool(receipt.get("link_id"))

    def check(self, receipt: dict[str, Any]) -> Observation:
        link_id = receipt["link_id"]
        try:
            link = self.api.fetch(link_id)
        except Exception as exc:
            return Observation(outcome=ERROR, error=f"{type(exc).__name__}: {exc}")

        status = link.get("status", "")
        amount = int(link.get("amount") or 0)
        paid = int(link.get("amount_paid") or 0)
        detail = {"link_id": link_id, "link_status": status,
                  "amount_paise": amount, "amount_paid_paise": paid}

        if status == "cancelled":
            return Observation(outcome=CANCELLED, detail=detail)
        if status == "expired":
            return Observation(outcome=EXPIRED, detail=detail)

        if status == "paid" or paid > 0:
            payment_id = ""
            for p in link.get("payments") or []:
                if p.get("status") == "captured":
                    payment_id = p.get("payment_id") or p.get("id", "")
                    break
            detail["recovery_payment_id"] = payment_id
            if paid >= amount and amount > 0:
                return Observation(outcome=PAID, recovered=True,
                                   recovery_payment_id=payment_id,
                                   amount_paid_paise=paid, detail=detail)
            return Observation(outcome=PARTIALLY_PAID,
                               recovery_payment_id=payment_id,
                               amount_paid_paise=paid, detail=detail)

        return Observation(outcome=PENDING, detail=detail)


# ── Sensor ───────────────────────────────────────────────────────────────────

class RecoverySensor:
    """Reconciles outstanding offers against Razorpay and records the truth."""

    def __init__(
        self,
        ledger: Ledger | None = None,
        probes: list[Probe] | None = None,
        client: Any = None,
    ):
        self.ledger = ledger or Ledger()
        self.probes = probes if probes is not None else [
            OrderProbe(client=client), PaymentLinkProbe(client=client),
        ]

    # ── observing ──

    def observe(self, case: CaseRecord) -> Observation:
        """Ask the world about this case's outstanding offer."""
        receipt = _last_successful_receipt(self.ledger, case.case_id)
        if not receipt:
            return Observation(outcome=ERROR, error="no receipt to check")
        for probe in self.probes:
            if probe.supports(receipt):
                return probe.check(receipt)
        return Observation(
            outcome=ERROR,
            error=f"no probe handles receipt keys {sorted(receipt)}",
        )

    # ── recording ──

    def poll_case(self, case: CaseRecord, run: SensorRun | None = None) -> CaseRecord:
        """Observe one case and record any change. Returns the current case."""
        run = run or SensorRun()
        run.checked += 1

        obs = self.observe(case)

        if obs.is_error:
            # Transient failures must not touch state, or a Razorpay blip would
            # look like a recovery outcome.
            run.errors += 1
            logger.warning("[sensor] %s: %s", case.case_id, obs.error)
            return case

        if obs.outcome == PENDING or not _is_new(self.ledger, case.case_id, obs):
            run.unchanged += 1
            return case

        try:
            return self._record(case, obs, run)
        except ConcurrentModification:
            # Another worker got there first. Its write is as good as ours.
            run.skipped += 1
            logger.info("[sensor] %s changed under us; skipping", case.case_id)
            return self.ledger.require_case(case.case_id)

    def _record(self, case: CaseRecord, obs: Observation, run: SensorRun) -> CaseRecord:
        seq = case.seq
        self.ledger.record_observation(
            case.case_id,
            observed=obs.outcome,
            recovered=obs.recovered,
            recovery_payment_id=obs.recovery_payment_id,
            recovered_amount_paise=obs.amount_paid_paise if obs.recovered else 0,
            payload={"amount_paid_paise": obs.amount_paid_paise, **obs.detail},
        )
        run.changed += 1

        if obs.recovered:
            got = self.ledger.record_transition(
                case.case_id, CaseStatus.RECOVERED,
                reason=f"observed {obs.outcome} via {obs.recovery_payment_id or 'razorpay'}",
                actor="sensor", expected_seq=seq + 1,
            )
            run.recovered += 1
            logger.info("[sensor] RECOVERED %s -> %s (%s paise)", case.case_id,
                        obs.recovery_payment_id, obs.amount_paid_paise)
            return got

        if obs.outcome in DEAD_OUTCOMES:
            # Back on the work queue. The sensor does not choose what to do next.
            got = self.ledger.record_transition(
                case.case_id, CaseStatus.SCHEDULED,
                reason=f"offer {obs.outcome}; needs a new decision",
                actor="sensor", wake_at=datetime.now(timezone.utc),
                expected_seq=seq + 1,
            )
            run.dead += 1
            return got

        # Partial payment: recorded, still owed, keep waiting.
        return self.ledger.require_case(case.case_id)

    # ── loops ──

    def poll_once(self, limit: int = 200) -> SensorRun:
        run = SensorRun()
        for case in self.ledger.awaiting_customer_cases(limit=limit):
            self.poll_case(case, run)
        return run

    def run_forever(
        self, interval: float = DEFAULT_POLL_INTERVAL, iterations: int | None = None
    ) -> SensorRun:
        """Poll until stopped. `iterations` bounds it for tests."""
        total = SensorRun()
        n = 0
        while iterations is None or n < iterations:
            run = self.poll_once()
            for f in ("checked", "recovered", "changed", "unchanged", "dead",
                      "errors", "skipped"):
                setattr(total, f, getattr(total, f) + getattr(run, f))
            if run.changed or run.errors:
                logger.info("[sensor] %s", run)
            n += 1
            if iterations is None or n < iterations:
                time.sleep(interval)
        return total


# ── helpers ──────────────────────────────────────────────────────────────────

def _last_successful_receipt(ledger: Ledger, case_id: str) -> dict[str, Any]:
    for ev in reversed(ledger.events(case_id)):
        if ev.kind is EventKind.ATTEMPT and ev.result == "ok":
            receipt = ev.payload.get("receipt") or {}
            if receipt:
                return receipt
    return {}


def _is_new(ledger: Ledger, case_id: str, obs: Observation) -> bool:
    """True unless the last observation already said exactly this.

    Without it, a partially-paid case would append an identical observation on
    every poll — hundreds of rows saying nothing new.
    """
    for ev in reversed(ledger.events(case_id)):
        if ev.kind is EventKind.OBSERVATION:
            same_outcome = ev.result == obs.outcome
            same_amount = int(ev.payload.get("amount_paid_paise") or 0) == obs.amount_paid_paise
            return not (same_outcome and same_amount)
    return True


def main() -> None:  # pragma: no cover - operational entry point
    import os
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    interval = float(os.getenv("SENSOR_POLL_INTERVAL", DEFAULT_POLL_INTERVAL))
    print(f"[sensor] polling every {interval}s — Ctrl-C to stop")
    RecoverySensor().run_forever(interval=interval)


if __name__ == "__main__":  # pragma: no cover
    main()
