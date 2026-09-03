"""Effectors — the only code allowed to change the outside world.

Block D3 of REBUILD-PLAN.md.

An effector does three things and nothing else:
  1. performs one real side effect,
  2. returns a **receipt** — evidence from outside that it happened,
  3. fails loudly when it cannot.

Two rules, both learned from the audit
--------------------------------------
**No simulation mode.** `RazorpayClient` silently fabricated plausible responses
when unconfigured — including a real-looking but dead `https://rzp.io/i/...` URL
and a hardcoded `error_code: gateway_timeout` (AUDIT-FINDINGS S3-4). Nothing
downstream could tell invention from fact. Here an unconfigured client raises
`NotConfigured`. Tests inject a fake API object; production never pretends.

**Paise in, paise out.** The old chain converted rupees to paise in `tools.py`
and *again* inside `RazorpayClient.create_payment_link`, so a Rs 2,999 recovery
link was created for Rs 29,99,000 — a 10,000x error. Effectors take
`amount_paise` straight from the ledger and never multiply.

Idempotency
-----------
The dangerous window is: the link is created at Razorpay, then the process dies
before the ledger records it. A retry would create a second link for the same
debt. Two defences:

  * `reference_id` is derived deterministically from (case_id, attempt). Razorpay
    enforces it as unique, so the duplicate is rejected server-side and we
    recover the original link instead of making another.
  * `EffectorResult.idempotency_key` is handed to `ledger.act()`, which refuses
    to record the same attempt twice.

The first covers a crash before the write; the second covers a replayed request.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from recovery_agent.ledger import CaseRecord

logger = logging.getLogger(__name__)

DEFAULT_LINK_TTL = timedelta(hours=24)

#: Where a recovery order is completed. The customer lands back on the merchant's
#: own checkout, which renders Razorpay Checkout for this order.
DEFAULT_CHECKOUT_URL = "http://localhost:5002/pay"


def checkout_base_url() -> str:
    """Read at call time, not import time — the web app sets this once it knows
    which port it actually bound to. A constant frozen at import produces links
    pointing at a port nothing is listening on."""
    return os.getenv("CHECKOUT_BASE_URL") or DEFAULT_CHECKOUT_URL
#: Razorpay rejects an expiry closer than 15 minutes out.
_MIN_TTL = timedelta(minutes=16)


class EffectorError(RuntimeError):
    """The side effect did not happen."""


class NotConfigured(EffectorError):
    """Credentials are missing. Never substituted with a fake success."""


@dataclass(frozen=True)
class EffectorResult:
    """What an effector did, and the proof."""
    action: str
    ok: bool
    request: dict[str, Any] = field(default_factory=dict)
    receipt: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    idempotency_key: str = ""
    reused: bool = False          # an existing side effect was recovered, not remade

    def raise_for_status(self) -> "EffectorResult":
        if not self.ok:
            raise EffectorError(f"{self.action} failed: {self.error}")
        return self


class PaymentLinkAPI(Protocol):
    """The slice of the Razorpay SDK this effector needs. Fakes implement it."""
    def create(self, params: dict[str, Any]) -> dict[str, Any]: ...
    def fetch(self, link_id: str) -> dict[str, Any]: ...
    def all(self, params: dict[str, Any] | None = None) -> dict[str, Any]: ...


def reference_id_for(case: CaseRecord, intent: str = "") -> str:
    """The server-side idempotency anchor. Stable for a case by default.

    Deriving this from `attempt_count` is a trap: the attempt being recorded
    increments the counter, so a retry computes a *different* reference and
    Razorpay happily creates a second link for the same debt. The default is
    therefore stable per case — **one outstanding recovery link per case** —
    which is also the right product behaviour: a customer should never be holding
    two live links for one payment.

    Pass `intent` to deliberately mint another link (say, after the first
    expired). The caller owns that decision; a replay must never make it by
    accident. In D6 the agent's decision id becomes the intent.
    """
    return f"rec-{case.case_id}-{intent}" if intent else f"rec-{case.case_id}"


class PaymentLinkEffector:
    """Creates a real Razorpay payment link for a failed payment."""

    action = "create_payment_link"

    def idempotency_key(self, case: CaseRecord, intent: str = "") -> str:
        return reference_id_for(case, intent)

    def __init__(self, api: PaymentLinkAPI | None = None, client: Any = None):
        self._api = api
        self._client = client

    @property
    def api(self) -> PaymentLinkAPI:
        if self._api is not None:
            return self._api
        client = self._client
        if client is None:
            from recovery_agent.razorpay_client import RazorpayClient
            client = RazorpayClient()
        if not getattr(client, "is_configured", False):
            raise NotConfigured(
                "Razorpay credentials are not configured. Set RAZORPAY_KEY_ID and "
                "RAZORPAY_KEY_SECRET. This effector does not simulate."
            )
        self._api = client.client.payment_link
        return self._api

    def run(
        self,
        case: CaseRecord,
        *,
        customer_name: str = "",
        description: str = "",
        allowed_rails: str = "",
        intent: str = "",
        ttl: timedelta = DEFAULT_LINK_TTL,
        now: datetime | None = None,
    ) -> EffectorResult:
        """Create the link. Returns a result; never raises for API failure.

        `intent` distinguishes a deliberate second link from a replay of the
        first — see `reference_id_for`.
        """
        now = now or datetime.now(timezone.utc)
        ref = reference_id_for(case, intent)
        expire_by = int((now + max(ttl, _MIN_TTL)).timestamp())

        notes = {
            # D4 correlates the paid link back to this case using these.
            "case_id": case.case_id,
            "original_payment_id": case.payment_id,
        }
        if allowed_rails:
            notes["allowed_rails"] = allowed_rails

        params: dict[str, Any] = {
            "amount": int(case.amount_paise),      # already paise. Do not multiply.
            "currency": case.currency,
            "accept_partial": False,
            "reference_id": ref,
            "description": description or f"Recovery for {case.payment_id}",
            "expire_by": expire_by,
            "notify": {"sms": False, "email": False},   # dispatch is D7's job
            "reminder_enable": False,
            "notes": notes,
        }
        contact = _customer_params(case, customer_name)
        if contact:
            params["customer"] = contact

        request = {"reference_id": ref, "amount_paise": params["amount"],
                   "currency": params["currency"], "expire_by": expire_by}

        try:
            link = self.api.create(params)
        except NotConfigured:
            raise
        except Exception as exc:                     # SDK raises many shapes
            if _is_duplicate_reference(exc):
                existing = self._find_by_reference(ref)
                if existing:
                    logger.info("[effector] reusing existing link %s for %s",
                                existing.get("id"), case.case_id)
                    return EffectorResult(
                        action=self.action, ok=True, request=request,
                        receipt=_receipt(existing, ref), idempotency_key=ref,
                        reused=True,
                    )
            return EffectorResult(
                action=self.action, ok=False, request=request,
                error=f"{type(exc).__name__}: {exc}", idempotency_key=ref,
            )

        receipt = _receipt(link, ref)
        if not receipt.get("link_url"):
            return EffectorResult(
                action=self.action, ok=False, request=request, receipt=receipt,
                error="Razorpay returned no short_url", idempotency_key=ref,
            )
        return EffectorResult(action=self.action, ok=True, request=request,
                              receipt=receipt, idempotency_key=ref)

    def _find_by_reference(self, ref: str) -> dict[str, Any] | None:
        try:
            page = self.api.all({"reference_id": ref}) or {}
        except Exception as exc:
            logger.warning("[effector] lookup by reference_id failed: %s", exc)
            return None
        items = page.get("items") or []
        for item in items:
            if item.get("reference_id") == ref:
                return item
        return items[0] if len(items) == 1 else None


# ── helpers ──────────────────────────────────────────────────────────────────

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _customer_params(case: CaseRecord, name: str) -> dict[str, str]:
    """Razorpay rejects a malformed contact block, so only send what is valid."""
    out: dict[str, str] = {}
    email = (case.metadata.get("customer_email") or case.customer_id or "").strip()
    if _EMAIL.match(email):
        out["email"] = email
    phone = str(case.metadata.get("customer_phone") or "").strip()
    if phone:
        out["contact"] = phone
    display = name or str(case.metadata.get("customer_name") or "").strip()
    if display:
        out["name"] = display
    return out


def _receipt(link: dict[str, Any], ref: str) -> dict[str, Any]:
    """The evidence we store. Amounts stay in paise."""
    return {
        "link_id": link.get("id", ""),
        "link_url": link.get("short_url", ""),
        "reference_id": link.get("reference_id", ref),
        "status": link.get("status", ""),
        "amount_paise": link.get("amount"),
        "currency": link.get("currency", ""),
        "expire_by": link.get("expire_by"),
    }


def _is_duplicate_reference(exc: Exception) -> bool:
    text = str(exc).lower()
    return "reference_id" in text and (
        "exist" in text or "duplicate" in text or "already" in text
    )


# ── wiring: side effect first, then the ledger ───────────────────────────────

def send_recovery_offer(
    ledger: Any,
    case: CaseRecord,
    *,
    effector: Any = None,
    gate: Any = None,
    reason: str = "",
    now: Any = None,
    **kwargs: Any,
) -> CaseRecord:
    """Create a recovery link for `case` and record what happened.

    Order matters: the side effect runs **first**, then the ledger records it.
    Writing an intent first would let the ledger claim an action that never
    happened; this way the ledger can only ever lag reality, never lead it — and
    `reference_id` closes that lag (see the module docstring).

    On success the case moves to `AWAITING_CUSTOMER` in one transaction with the
    attempt. On failure the error is recorded as evidence and the case stays in
    `ACTING` for the agent to decide again — a failed send is deliberately not
    enough to satisfy the AWAITING_CUSTOMER evidence rule.
    """
    from recovery_agent.models import CaseStatus

    eff = effector or default_effector()

    # Replay guard comes FIRST. Moving the case to ACTING and only then noticing
    # the attempt already exists would leave it parked in ACTING — a replay that
    # changes state is not idempotent, however harmless each individual write
    # looks. Nothing is written unless something new is going to happen.
    key = eff.idempotency_key(case, kwargs.get("intent", ""))
    if ledger.has_attempt(case.case_id, key):
        return ledger.require_case(case.case_id)

    # D5: the policy gate runs here — before the side effect and before any
    # state change. Not by filtering what the model may propose, which is
    # advisory, but in the one path every effect must pass through.
    from recovery_agent.policy import PolicyGate
    decision = (gate if gate is not None else PolicyGate()).check(
        case, eff.action, ledger=ledger, amount_paise=case.amount_paise, now=now,
    )
    if not decision.allowed:
        # Recorded as evidence, with the reason. `result="blocked"` deliberately
        # does not satisfy D2's AWAITING_CUSTOMER rule — nothing left the building.
        ledger.record_attempt(
            case.case_id, action=eff.action, result="blocked",
            request={"amount_paise": case.amount_paise},
            receipt=decision.as_receipt(), reason=decision.reason,
        )
        return ledger.require_case(case.case_id)

    # "I am about to act" is a real state change, so record it rather than
    # letting the effector fire from OPEN.
    if case.status != CaseStatus.ACTING:
        case = ledger.record_transition(
            case.case_id, CaseStatus.ACTING,
            reason=reason or "creating recovery offer",
        )

    result = eff.run(case, **kwargs)

    if not result.ok:
        ledger.record_attempt(
            case.case_id, action=result.action, result="error",
            request=result.request, receipt={"error": result.error},
            reason=result.error,
        )
        return ledger.require_case(case.case_id)

    return ledger.act(
        case.case_id,
        action=result.action,
        to_status=CaseStatus.AWAITING_CUSTOMER,
        result="ok",
        request=result.request,
        receipt=result.receipt,
        reason=reason or ("reused existing offer" if result.reused else "offer sent"),
        idempotency_key=result.idempotency_key,
    )


#: Back-compat alias — the wiring is the same whichever effector is used.
send_recovery_link = send_recovery_offer


def last_receipt(ledger: Any, case_id: str, action: str = "") -> dict[str, Any]:
    """Most recent successful receipt for a case — how D4 finds what to poll."""
    from recovery_agent.ledger import EventKind

    for ev in reversed(ledger.events(case_id)):
        if ev.kind is EventKind.ATTEMPT and ev.result == "ok":
            if not action or ev.action == action:
                return ev.payload.get("receipt", {})
    return {}


# ── Recovery order ───────────────────────────────────────────────────────────
#
# Razorpay caps test-mode payment links at **30 per account, for the lifetime of
# the account** — cancelling them does not free the quota (verified: 30 cancelled,
# the next create still failed). Orders have no such cap, so on a capped account
# the recovery offer is a fresh Order plus a checkout URL rather than a link.
#
# This is also the better demo: the customer returns to the merchant's own
# checkout, pays with a real test card through real Razorpay Checkout, and the
# result is a genuine captured payment for the sensor (D4) to observe.


class OrderAPI(Protocol):
    def create(self, params: dict[str, Any]) -> dict[str, Any]: ...
    def fetch(self, order_id: str) -> dict[str, Any]: ...
    def all(self, params: dict[str, Any] | None = None) -> dict[str, Any]: ...


class RecoveryOrderEffector:
    """Creates a real Razorpay Order the customer can pay to recover the debt."""

    action = "create_recovery_order"

    def idempotency_key(self, case: CaseRecord, intent: str = "") -> str:
        return reference_id_for(case, intent)

    def __init__(
        self,
        api: OrderAPI | None = None,
        client: Any = None,
        checkout_base_url: str = "",
    ):
        self._api = api
        self._client = client
        self._checkout_base_url = checkout_base_url

    @property
    def api(self) -> OrderAPI:
        if self._api is not None:
            return self._api
        client = self._client
        if client is None:
            from recovery_agent.razorpay_client import RazorpayClient
            client = RazorpayClient()
        if not getattr(client, "is_configured", False):
            raise NotConfigured(
                "Razorpay credentials are not configured. Set RAZORPAY_KEY_ID and "
                "RAZORPAY_KEY_SECRET. This effector does not simulate."
            )
        self._api = client.client.order
        return self._api

    @property
    def checkout_base_url(self) -> str:
        return self._checkout_base_url or checkout_base_url()

    def run(self, case: CaseRecord, *, intent: str = "", **_: Any) -> EffectorResult:
        ref = reference_id_for(case, intent)
        params = {
            "amount": int(case.amount_paise),     # already paise. Do not multiply.
            "currency": case.currency,
            "receipt": ref[:40],                  # Razorpay caps receipt length
            "notes": {
                "case_id": case.case_id,
                "original_payment_id": case.payment_id,
                "recovery": "true",
            },
        }
        request = {"reference_id": ref, "amount_paise": params["amount"],
                   "currency": params["currency"]}

        # Recover a prior create that succeeded but whose ledger write did not.
        existing = self._find_by_receipt(params["receipt"])
        if existing:
            return EffectorResult(
                action=self.action, ok=True, request=request,
                receipt=self._receipt(existing, ref), idempotency_key=ref, reused=True,
            )

        try:
            order = self.api.create(params)
        except NotConfigured:
            raise
        except Exception as exc:
            return EffectorResult(
                action=self.action, ok=False, request=request,
                error=f"{type(exc).__name__}: {exc}", idempotency_key=ref,
            )

        receipt = self._receipt(order, ref)
        if not receipt.get("order_id"):
            return EffectorResult(
                action=self.action, ok=False, request=request, receipt=receipt,
                error="Razorpay returned no order id", idempotency_key=ref,
            )
        return EffectorResult(action=self.action, ok=True, request=request,
                              receipt=receipt, idempotency_key=ref)

    def _receipt(self, order: dict[str, Any], ref: str) -> dict[str, Any]:
        order_id = order.get("id", "")
        sep = "&" if "?" in self.checkout_base_url else "?"
        return {
            "order_id": order_id,
            "link_url": f"{self.checkout_base_url}{sep}order_id={order_id}" if order_id else "",
            "reference_id": order.get("receipt", ref),
            "status": order.get("status", ""),
            "amount_paise": order.get("amount"),
            "currency": order.get("currency", ""),
        }

    def _find_by_receipt(self, receipt_id: str) -> dict[str, Any] | None:
        try:
            page = self.api.all({"receipt": receipt_id}) or {}
        except Exception as exc:
            logger.warning("[effector] order lookup by receipt failed: %s", exc)
            return None
        for item in page.get("items") or []:
            if item.get("receipt") == receipt_id:
                return item
        return None


def default_effector() -> Any:
    """The recovery offer this deployment can actually create.

    Orders by default: Razorpay's test-mode payment-link quota is 30 for the life
    of the account and cancelling does not reclaim it, so a link-based demo stops
    working permanently. Set RECOVERY_EFFECTOR=payment_link on an account with
    quota to spare.
    """
    if os.getenv("RECOVERY_EFFECTOR", "order").lower() in ("payment_link", "link"):
        return PaymentLinkEffector()
    return RecoveryOrderEffector()
