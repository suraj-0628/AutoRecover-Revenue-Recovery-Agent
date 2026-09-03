"""Policy gate — deterministic limits enforced at the effector boundary.

Block D5 of REBUILD-PLAN.md.

Where this runs is the whole point
----------------------------------
The old system tried to enforce safety by filtering which tools got bound to the
LLM (`graph.py` + `governance.get_allowed_tools`). That is **advisory**: it fails
if the model is wrong, jailbroken, or — as actually happened — the policy names
drift out of sync with the tool registry, leaving eight phantom names and the one
real retry tool ungrantable (AUDIT-FINDINGS S1-1). Worse, the real guardrails
were never called at all from the runtime path (S0-3).

Here the check runs **inside the execution path, immediately before the side
effect**. It does not matter what the model decided, what tools it could see, or
whether it was tricked: nothing reaches Razorpay or a customer without passing
this gate.

Two design choices worth stating
--------------------------------
**Actions are classified, not lumped together.** Creating a recovery order at
3 a.m. is fine; *texting the customer* at 3 a.m. is not. The old engine treated
every action alike, so it could only be either too strict or useless.

**Escalation can never be blocked.** A gate able to refuse `escalate_to_human`
can trap a case with no way out. Escalation is always allowed.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from recovery_agent.ledger import CaseRecord, EventKind, Ledger
from recovery_agent.models import HARD_DECLINES
from recovery_agent.statemachine import is_terminal

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


#: Largest recovery a policy will allow without a human. Rs 50,000.
MAX_RECOVERY_PAISE = _env_int("POLICY_MAX_RECOVERY_PAISE", 50_00_000)
QUIET_START_HOUR = _env_int("POLICY_QUIET_START_HOUR", 21)
QUIET_END_HOUR = _env_int("POLICY_QUIET_END_HOUR", 8)
MAX_CONTACTS_PER_DAY = _env_int("POLICY_MAX_CONTACTS_24H", 2)

#: Failure codes that are permanent. Retrying burns network fees and never works.
HARD_DECLINE_REASONS = frozenset(HARD_DECLINES) | {
    "card_expired", "expiry_date_invalid", "invalid_card", "lost_card",
    "stolen_card", "closed_account", "card_not_permitted",
}


# ── Actions ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ActionSpec:
    """What an action does, so policies can judge it accurately."""
    name: str
    moves_money: bool = False        # creates a payable obligation or charges
    contacts_customer: bool = False  # reaches a human
    always_allowed: bool = False     # never blockable (escalation)


ACTIONS: dict[str, ActionSpec] = {
    "create_recovery_order": ActionSpec("create_recovery_order", moves_money=True),
    "create_payment_link": ActionSpec("create_payment_link", moves_money=True),
    "retry_payment": ActionSpec("retry_payment", moves_money=True),
    "send_recovery_notification": ActionSpec(
        "send_recovery_notification", contacts_customer=True),
    "initiate_voice_call": ActionSpec("initiate_voice_call", contacts_customer=True),
    "escalate_to_human": ActionSpec("escalate_to_human", always_allowed=True),
}


def spec_for(action: str) -> ActionSpec:
    """Unknown actions are treated as the most dangerous kind, not the safest."""
    return ACTIONS.get(action, ActionSpec(action, moves_money=True,
                                          contacts_customer=True))


# ── Results ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CheckResult:
    policy: str
    allowed: bool
    reason: str = ""
    applicable: bool = True


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    allowed: bool
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def blocked_by(self) -> str:
        return next((c.policy for c in self.checks if not c.allowed), "")

    @property
    def reason(self) -> str:
        return next((c.reason for c in self.checks if not c.allowed), "")

    def as_receipt(self) -> dict[str, Any]:
        return {
            "blocked_by": self.blocked_by,
            "reason": self.reason,
            "checks": [
                {"policy": c.policy, "allowed": c.allowed, "reason": c.reason}
                for c in self.checks if c.applicable
            ],
        }


@dataclass
class PolicyContext:
    case: CaseRecord
    action: ActionSpec
    ledger: Ledger
    now: datetime
    amount_paise: int


class Policy(Protocol):
    name: str
    def check(self, ctx: PolicyContext) -> CheckResult: ...


# ── Policies ─────────────────────────────────────────────────────────────────

class TerminalCaseLock:
    """A closed case is closed. No effect may touch it."""
    name = "terminal_case"

    def check(self, ctx: PolicyContext) -> CheckResult:
        if is_terminal(ctx.case.status):
            return CheckResult(self.name, False,
                               f"case is {ctx.case.status.value} and cannot be acted on")
        return CheckResult(self.name, True)


class DoubleDebitLock:
    """Never leave a customer holding two live demands for one debt.

    The most expensive mistake this system could make is charging someone twice.
    D3's `reference_id` prevents an accidental duplicate; this prevents a
    *deliberate* one — an agent that decides to "try again" while an offer is
    still outstanding.
    """
    name = "double_debit"

    def check(self, ctx: PolicyContext) -> CheckResult:
        if not ctx.action.moves_money:
            return CheckResult(self.name, True, applicable=False)
        from recovery_agent.models import CaseStatus
        if ctx.case.status == CaseStatus.AWAITING_CUSTOMER:
            return CheckResult(
                self.name, False,
                "an offer is already outstanding with the customer; "
                "cancel or expire it before creating another",
            )
        return CheckResult(self.name, True)


class HardDeclineBlock:
    """Never re-charge an instrument that is permanently dead.

    Note the scope: this blocks *retrying the same instrument*. Asking the
    customer to pay another way is exactly the right response to an expired
    card, so offers are unaffected.
    """
    name = "hard_decline"

    def check(self, ctx: PolicyContext) -> CheckResult:
        if ctx.action.name != "retry_payment":
            return CheckResult(self.name, True, applicable=False)
        code = (ctx.case.failure_code or "").strip().lower()
        if code in HARD_DECLINE_REASONS:
            return CheckResult(
                self.name, False,
                f"{code} is a hard decline; retrying incurs network fees and "
                "cannot succeed. Offer another instrument or escalate.",
            )
        return CheckResult(self.name, True)


class AmountIntegrity:
    """The customer is asked for the debt, and nothing larger.

    Defence in depth against the class of bug that actually shipped: a double
    rupee-to-paise conversion billed Rs 1,299 as Rs 1,29,900 (S1-4b). D3 fixed
    the conversion; this makes any future recurrence unable to reach a customer.
    """
    name = "amount_integrity"

    def check(self, ctx: PolicyContext) -> CheckResult:
        if not ctx.action.moves_money:
            return CheckResult(self.name, True, applicable=False)
        if ctx.amount_paise <= 0:
            return CheckResult(self.name, False,
                               f"refusing to charge {ctx.amount_paise} paise")
        if ctx.amount_paise != ctx.case.amount_paise:
            return CheckResult(
                self.name, False,
                f"amount {ctx.amount_paise} paise does not match the debt "
                f"{ctx.case.amount_paise} paise",
            )
        if ctx.amount_paise > MAX_RECOVERY_PAISE:
            return CheckResult(
                self.name, False,
                f"{ctx.amount_paise} paise exceeds the automatic limit of "
                f"{MAX_RECOVERY_PAISE}; needs a human",
            )
        return CheckResult(self.name, True)


class OptOutRespect:
    """A customer who opted out is never contacted again."""
    name = "opt_out"

    def check(self, ctx: PolicyContext) -> CheckResult:
        if not ctx.action.contacts_customer:
            return CheckResult(self.name, True, applicable=False)
        if _truthy(ctx.case.metadata.get("opted_out")):
            return CheckResult(self.name, False,
                               "customer has opted out of recovery contact")
        return CheckResult(self.name, True)


class QuietHours:
    """No contact overnight. Creating an offer is fine; waking someone is not."""
    name = "quiet_hours"

    def __init__(self, start: int = QUIET_START_HOUR, end: int = QUIET_END_HOUR):
        self.start, self.end = start, end

    def check(self, ctx: PolicyContext) -> CheckResult:
        if not ctx.action.contacts_customer:
            return CheckResult(self.name, True, applicable=False)
        hour = _local_hour(ctx.case, ctx.now)
        if self.start <= hour or hour < self.end:
            return CheckResult(
                self.name, False,
                f"local time {hour:02d}:00 is inside quiet hours "
                f"({self.start:02d}:00-{self.end:02d}:00)",
            )
        return CheckResult(self.name, True)


class FrequencyCap:
    """A bounded number of contacts per day. Every extra message is a churn risk."""
    name = "frequency_cap"

    def __init__(self, max_per_day: int = MAX_CONTACTS_PER_DAY):
        self.max_per_day = max_per_day

    def check(self, ctx: PolicyContext) -> CheckResult:
        if not ctx.action.contacts_customer:
            return CheckResult(self.name, True, applicable=False)
        cutoff = ctx.now - timedelta(hours=24)
        n = 0
        for ev in ctx.ledger.events(ctx.case.case_id):
            if (ev.kind is EventKind.ATTEMPT and ev.result == "ok"
                    and spec_for(ev.action).contacts_customer
                    and ev.created_at >= cutoff):
                n += 1
        if n >= self.max_per_day:
            return CheckResult(
                self.name, False,
                f"{n} contact(s) already sent in the last 24h "
                f"(limit {self.max_per_day})",
            )
        return CheckResult(self.name, True)


DEFAULT_POLICIES: tuple[Policy, ...] = (
    TerminalCaseLock(), DoubleDebitLock(), HardDeclineBlock(),
    AmountIntegrity(), OptOutRespect(), QuietHours(), FrequencyCap(),
)


# ── The gate ─────────────────────────────────────────────────────────────────

class PolicyGate:
    """Runs every policy against a proposed action. Deterministic, no LLM."""

    def __init__(self, policies: tuple[Policy, ...] | list[Policy] | None = None):
        self.policies = tuple(policies if policies is not None else DEFAULT_POLICIES)

    def check(
        self,
        case: CaseRecord,
        action: str,
        *,
        ledger: Ledger,
        amount_paise: int | None = None,
        now: datetime | None = None,
    ) -> PolicyDecision:
        spec = spec_for(action)
        if spec.always_allowed:
            return PolicyDecision(action, True, [
                CheckResult("always_allowed", True,
                            "escalation is never blocked — a gate that can refuse "
                            "it can trap a case")
            ])

        ctx = PolicyContext(
            case=case, action=spec, ledger=ledger,
            now=now or datetime.now(timezone.utc),
            amount_paise=case.amount_paise if amount_paise is None else amount_paise,
        )
        checks = [p.check(ctx) for p in self.policies]
        allowed = all(c.allowed for c in checks)
        if not allowed:
            first = next(c for c in checks if not c.allowed)
            logger.info("[policy] BLOCKED %s on %s: %s (%s)",
                        action, case.case_id, first.reason, first.policy)
        return PolicyDecision(action, allowed, checks)


# ── helpers ──────────────────────────────────────────────────────────────────

def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _local_hour(case: CaseRecord, now: datetime) -> int:
    """Customer-local hour. Defaults to IST — this is a Razorpay-first system."""
    offset_minutes = case.metadata.get("utc_offset_minutes")
    try:
        offset = timedelta(minutes=int(offset_minutes))
    except (TypeError, ValueError):
        offset = timedelta(hours=5, minutes=30)
    return (now.astimezone(timezone.utc) + offset).hour
