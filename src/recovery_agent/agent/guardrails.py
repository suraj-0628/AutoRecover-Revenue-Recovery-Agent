"""Guardrail Engine — pre-execution safety interceptor.

Validates every action against 6 enterprise safety policies before execution.
If a policy is violated, the action is vetoed or modified to a compliant fallback.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from recovery_agent.models import ActionType, Case, CustomerProfile, HARD_DECLINES

IST = timezone(timedelta(hours=5, minutes=30))

# A voice call is the most intrusive contact there is — it was missing from
# every communication set, so quiet hours, the frequency cap and opt-out all
# waved the phone through while fussing over an email.
_COMMUNICATION_ACTIONS = frozenset({
    ActionType.SEND_NOTIFICATION,
    ActionType.UPDATE_PAYMENT_METHOD,
    ActionType.VOICE_CALL,
})

#: What quiet hours actually restrict: contact that INTERRUPTS. A phone
#: ringing wakes someone; an email does not. `send_recovery_notification`
#: therefore stays allowed overnight and drops to email-only (its SMS leg is
#: suppressed by `sms_allowed_now()`), while a voice call is refused outright.
#: In-page surfaces are deliberately absent — the customer is right there.
_QUIET_INTERRUPTS = frozenset({
    ActionType.VOICE_CALL,
})


def in_quiet_hours(now: datetime | None = None) -> bool:
    """Is it currently the customer's quiet window?"""
    g = QuietHourGuardrail()
    current = now or datetime.now(IST)
    return current.hour >= g.quiet_start or current.hour < g.quiet_end


def sms_allowed_now(now: datetime | None = None) -> bool:
    """SMS buzzes a phone, so it keeps the quiet-hours restriction email does not."""
    if os.getenv("GUARDRAIL_QUIET_DISABLED", "").strip().lower() in ("1", "true", "yes"):
        return True
    return not in_quiet_hours(now)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


class GuardrailVerdict(str, Enum):
    PASS = "pass"
    MODIFIED = "modified"
    BLOCKED = "blocked"


class GuardrailCheckResult(BaseModel):
    """Result of a single guardrail check."""
    guardrail: str
    verdict: GuardrailVerdict
    reason: str
    original_action: str
    modified_action: str = ""


# --- Individual Guardrail Evaluators ---

class QuietHourGuardrail:
    """Restricts customer communications during quiet hours (9 PM – 8 AM)."""

    def __init__(self, quiet_start: int = 21, quiet_end: int = 8):
        self.quiet_start = quiet_start
        self.quiet_end = quiet_end

    def minutes_until_end(self, now: datetime | None = None) -> int:
        """How long the quiet window still has to run, in whole minutes.

        The agent has to wait exactly this long — waiting less means being
        refused again, waiting more spends recovery window for nothing.
        """
        current = now or datetime.now(IST)
        end = current.replace(hour=self.quiet_end % 24, minute=0,
                              second=0, microsecond=0)
        if current.hour >= self.quiet_end:      # evening side: ends tomorrow
            end += timedelta(days=1)
        return max(1, int((end - current).total_seconds() // 60))

    def check(
        self,
        action: ActionType,
        profile: CustomerProfile | None = None,
        now: datetime | None = None,
    ) -> GuardrailCheckResult:
        current = now or datetime.now(IST)
        hour = current.hour

        is_quiet = hour >= self.quiet_start or hour < self.quiet_end

        if not is_quiet:
            return GuardrailCheckResult(
                guardrail="quiet_hours",
                verdict=GuardrailVerdict.PASS,
                reason="Outside quiet hours",
                original_action=action.value,
            )

        # Quiet hours exist to stop us WAKING people, so they gate what
        # interrupts: a voice call rings and an SMS buzzes a phone at 2am. An
        # email does neither — it waits in an inbox until the customer chooses
        # to open it — so it stays permitted, and `send_recovery_notification`
        # suppresses only its SMS leg overnight (see _QUIET_INTERRUPTS).
        #
        # In-page surfaces are not here at all and never should be: the
        # customer is on the checkout with their hand on the mouse. Live
        # (pay_woo85c9gh), someone dismissed a notification at 06:30 IST — 13
        # seconds before the agent was refused for "quiet hours" — which is
        # about as awake as a customer gets.
        if action in _QUIET_INTERRUPTS:
            return GuardrailCheckResult(
                guardrail="quiet_hours",
                verdict=GuardrailVerdict.MODIFIED,
                # Name the END, not just the fact.
                #
                # This said only "Quiet hours active (6:00)". Live
                # (pay_woo85c9gh), the agent had no way to know how long that
                # lasted, guessed 60 minutes, and would have woken at 07:00 to
                # be refused again — quiet hours ran to 08:00. Every other
                # refusal in this system names the number it is enforcing
                # (`_approval_refusal` gives the exact chargeable minimum);
                # this one made the agent guess.
                reason=(f"Quiet hours until {self.quiet_end:02d}:00 IST "
                        f"({self.minutes_until_end(current)} minutes from now; "
                        f"it is {current.strftime('%H:%M')} IST). "
                        f"Communication deferred."),
                original_action=action.value,
                modified_action=ActionType.WAIT_AND_RETRY.value,
            )

        return GuardrailCheckResult(
            guardrail="quiet_hours",
            verdict=GuardrailVerdict.PASS,
            reason="Action is not a communication, permitted during quiet hours",
            original_action=action.value,
        )


class FrequencyCapGuardrail:
    """Maximum 2 communications per customer per 24-hour window."""

    def __init__(self, max_contacts_per_24h: int = 2):
        self.max_contacts = max_contacts_per_24h

    def check(
        self,
        action: ActionType,
        profile: CustomerProfile | None = None,
        now: datetime | None = None,
    ) -> GuardrailCheckResult:
        communication_actions = _COMMUNICATION_ACTIONS

        if action not in communication_actions:
            return GuardrailCheckResult(
                guardrail="frequency_cap",
                verdict=GuardrailVerdict.PASS,
                reason="Action is not a communication, no frequency cap applies",
                original_action=action.value,
            )

        if not profile:
            return GuardrailCheckResult(
                guardrail="frequency_cap",
                verdict=GuardrailVerdict.PASS,
                reason="No profile data, allowing action",
                original_action=action.value,
            )

        # Count recent contacts (last 24h). "contact" is a delivery that was
        # neither a payment success nor a failure — a notification that went
        # out. It counts against the cap without polluting channel win-rates.
        current = now or datetime.now(timezone.utc)
        recent_contacts = 0
        for record in profile.payment_history:
            if record.status in ("success", "failed", "contact") and record.channel_used:
                time_diff = (current - record.timestamp).total_seconds()
                if time_diff < 86400:  # 24 hours
                    recent_contacts += 1

        if recent_contacts >= self.max_contacts:
            return GuardrailCheckResult(
                guardrail="frequency_cap",
                verdict=GuardrailVerdict.BLOCKED,
                reason=f"Frequency cap reached: {recent_contacts}/{self.max_contacts} contacts in 24h",
                original_action=action.value,
            )

        return GuardrailCheckResult(
            guardrail="frequency_cap",
            verdict=GuardrailVerdict.PASS,
            reason=f"Frequency OK: {recent_contacts}/{self.max_contacts} contacts in 24h",
            original_action=action.value,
        )


class DoubleDebitLockGuardrail:
    """Prevents duplicate debit retries on pending/captured payments."""

    def check(
        self,
        action: ActionType,
        profile: CustomerProfile | None = None,
        case: Case | None = None,
        **kwargs: Any,
    ) -> GuardrailCheckResult:
        if action != ActionType.RETRY_PAYMENT:
            return GuardrailCheckResult(
                guardrail="double_debit_lock",
                verdict=GuardrailVerdict.PASS,
                reason="Action is not a payment retry, no lock applies",
                original_action=action.value,
            )

        if not case:
            return GuardrailCheckResult(
                guardrail="double_debit_lock",
                verdict=GuardrailVerdict.PASS,
                reason="No case data, allowing action",
                original_action=action.value,
            )

        # Check if there's already a successful or pending payment in recent attempts
        for attempt in case.attempts:
            if attempt.result == "success":
                return GuardrailCheckResult(
                    guardrail="double_debit_lock",
                    verdict=GuardrailVerdict.BLOCKED,
                    reason=f"Double-debit blocked: payment already succeeded in attempt {attempt.id}",
                    original_action=action.value,
                )

        # Check for concurrent pending retries
        pending_count = sum(
            1 for a in case.attempts
            if a.action_type == ActionType.RETRY_PAYMENT and a.result == "pending"
        )
        if pending_count > 0:
            return GuardrailCheckResult(
                guardrail="double_debit_lock",
                verdict=GuardrailVerdict.BLOCKED,
                reason=f"Double-debit blocked: {pending_count} payment(s) already pending",
                original_action=action.value,
            )

        return GuardrailCheckResult(
            guardrail="double_debit_lock",
            verdict=GuardrailVerdict.PASS,
            reason="No duplicate payment detected",
            original_action=action.value,
        )


class OptOutGuardrail:
    """Blocks all customer messages if customer has opted out."""

    def check(
        self,
        action: ActionType,
        profile: CustomerProfile | None = None,
        **kwargs: Any,
    ) -> GuardrailCheckResult:
        if not profile or not profile.opt_out:
            return GuardrailCheckResult(
                guardrail="opt_out",
                verdict=GuardrailVerdict.PASS,
                reason="Customer has not opted out",
                original_action=action.value,
            )

        communication_actions = _COMMUNICATION_ACTIONS

        if action in communication_actions:
            return GuardrailCheckResult(
                guardrail="opt_out",
                verdict=GuardrailVerdict.BLOCKED,
                reason="Customer opted out of automated communications",
                original_action=action.value,
            )

        # Non-communication actions are allowed, but escalate if needed
        return GuardrailCheckResult(
            guardrail="opt_out",
            verdict=GuardrailVerdict.PASS,
            reason="Action is not a communication, permitted for opted-out customer",
            original_action=action.value,
        )


class MonetaryCapGuardrail:
    """Bounded monetary cap per single retry action."""

    def __init__(self, max_single_retry: float = 500_000.0):
        self.max_single_retry = max_single_retry

    def check(
        self,
        action: ActionType,
        profile: CustomerProfile | None = None,
        case: Case | None = None,
        **kwargs: Any,
    ) -> GuardrailCheckResult:
        if action != ActionType.RETRY_PAYMENT:
            return GuardrailCheckResult(
                guardrail="monetary_cap",
                verdict=GuardrailVerdict.PASS,
                reason="Action is not a payment retry, no monetary cap applies",
                original_action=action.value,
            )

        if not case:
            return GuardrailCheckResult(
                guardrail="monetary_cap",
                verdict=GuardrailVerdict.PASS,
                reason="No case data, allowing action",
                original_action=action.value,
            )

        if case.payment.amount > self.max_single_retry:
            return GuardrailCheckResult(
                guardrail="monetary_cap",
                verdict=GuardrailVerdict.BLOCKED,
                reason=f"Amount INR {case.payment.amount:,.2f} exceeds single retry cap of INR {self.max_single_retry:,.2f}",
                original_action=action.value,
            )

        return GuardrailCheckResult(
            guardrail="monetary_cap",
            verdict=GuardrailVerdict.PASS,
            reason=f"Amount INR {case.payment.amount:,.2f} within cap",
            original_action=action.value,
        )


class HardDeclineGuardrail:
    """Intercepts retry attempts on hard decline codes.

    Prevents $0.10/attempt Visa/Mastercard network penalties.
    Hard decline codes: 41, 43, 54, 14, 04, 46, 57, 93

    Source: Redux hard decline handling, Stripe Schedule+Skip
    """

    def check(
        self,
        action: ActionType,
        profile: CustomerProfile | None = None,
        case: Case | None = None,
        **kwargs: Any,
    ) -> GuardrailCheckResult:
        if action not in (ActionType.RETRY_PAYMENT, ActionType.WAIT_AND_RETRY):
            return GuardrailCheckResult(
                guardrail="hard_decline",
                verdict=GuardrailVerdict.PASS,
                reason="Action is not a retry, no hard decline check applies",
                original_action=action.value,
            )

        if not case:
            return GuardrailCheckResult(
                guardrail="hard_decline",
                verdict=GuardrailVerdict.PASS,
                reason="No case data, allowing action",
                original_action=action.value,
            )

        failure_code = case.payment.failure_code

        if failure_code in HARD_DECLINES:
            case.penalties_prevented += 1

            return GuardrailCheckResult(
                guardrail="hard_decline",
                verdict=GuardrailVerdict.BLOCKED,
                reason=(
                    f"Hard decline code {failure_code} detected. "
                    f"Retrying would incur $0.10/attempt Visa/MC network penalty. "
                    f"Penalties prevented: {case.penalties_prevented}. "
                    f"Blocking retry and escalating to human."
                ),
                original_action=action.value,
                modified_action=ActionType.ESCALATE_TO_HUMAN.value,
            )

        return GuardrailCheckResult(
            guardrail="hard_decline",
            verdict=GuardrailVerdict.PASS,
            reason=f"Decline code {failure_code} is not a hard decline",
            original_action=action.value,
        )





# --- Guardrail Engine ---

class GuardrailEngine:
    """Pre-execution interceptor that validates actions against all guardrails.

    Runs 7 guardrails in sequence:
    1. Hard Decline (highest priority) — prevents network penalties
    2. Quiet Hours — communication timing
    3. Frequency Cap — communication rate limiting
    4. Double-Debit Lock — duplicate payment prevention
    5. Opt-Out — customer preference compliance
    6. Monetary Cap — high-value transaction safety
    7. Semantic — LLM-based safety evaluation for complex cases
    """

    def __init__(self) -> None:
        # Env-tunable so the live deployment and the test rig can move the
        # boundaries without a code change; the class defaults are the policy.
        #
        # The engine's contact cap defaults to 3, not the class's 2: the
        # recovery ladder itself can legitimately reach a customer three times
        # in one evening (offer email → voice call → the agreed link by email),
        # and a cap that blocks the ladder's own last rung is a cap on the
        # policy, not on spam. GUARDRAIL_QUIET_DISABLED=1 empties the quiet
        # window entirely — for the integration rig, which runs at any hour.
        # Operator settings first (the dashboard), then env, then the class
        # defaults — see guardrail_config. A merchant mid-recovery should be
        # able to raise a contact cap without shell access and a restart; live
        # on pay_ls1dep23k a real bank-decline recovery was refused at "4/3
        # contacts in 24h" with no way to move the ceiling.
        try:
            from recovery_agent import guardrail_config as gc
            cfg = gc.all_values()
        except Exception:
            cfg = {}

        def _cfg(key, fallback):
            return cfg.get(key, fallback)

        quiet_off = os.getenv("GUARDRAIL_QUIET_DISABLED", "").strip().lower() in (
            "1", "true", "yes") or not _cfg("quiet_enabled", True)
        if quiet_off:
            quiet_start, quiet_end = 24, 0        # no hour satisfies either bound
        else:
            quiet_start = int(_cfg("quiet_start", _env_int("GUARDRAIL_QUIET_START", 21)))
            quiet_end = int(_cfg("quiet_end", _env_int("GUARDRAIL_QUIET_END", 8)))
        self.quiet_hours = QuietHourGuardrail(quiet_start=quiet_start,
                                              quiet_end=quiet_end)
        self.frequency_cap = FrequencyCapGuardrail(
            max_contacts_per_24h=int(_cfg(
                "max_contacts_24h", _env_int("GUARDRAIL_MAX_CONTACTS_24H", 5))))
        self.double_debit = DoubleDebitLockGuardrail()
        self.opt_out = OptOutGuardrail()
        self.monetary_cap = MonetaryCapGuardrail(
            max_single_retry=float(_cfg(
                "max_single_retry",
                _env_float("GUARDRAIL_MAX_SINGLE_RETRY", 500_000.0))))
        self.hard_decline = HardDeclineGuardrail()
        # Two safety policies can be switched off deliberately; opt-out cannot
        # (guardrail_config refuses the change). A disabled check is skipped in
        # validate_action rather than silently passing, so the verdict log
        # still shows it was not consulted.
        self._double_debit_on = bool(_cfg("double_debit_enabled", True))
        self._hard_decline_on = bool(_cfg("hard_decline_enabled", True))

    def validate_action(
        self,
        case: Case,
        action: ActionType,
        profile: CustomerProfile | None = None,
        now: datetime | None = None,
    ) -> tuple[ActionType, list[GuardrailCheckResult]]:
        """Intercept proposed action and run all guardrails.

        Returns (approved_or_modified_action, list_of_check_results).
        """
        checks: list[GuardrailCheckResult] = []
        current_action = action

        # Run each guardrail — they may modify or block the action
        # Hard decline check runs FIRST (highest priority)
        guardrail_checks = [
            ("hard_decline", lambda: self.hard_decline.check(current_action, profile, case=case)),
            ("quiet_hours", lambda: self.quiet_hours.check(current_action, profile, now)),
            ("frequency_cap", lambda: self.frequency_cap.check(current_action, profile, now)),
            ("double_debit_lock", lambda: self.double_debit.check(current_action, profile, case=case)),
            ("opt_out", lambda: self.opt_out.check(current_action, profile)),
            ("monetary_cap", lambda: self.monetary_cap.check(current_action, profile, case=case)),
        ]

        # A policy an operator switched off is SKIPPED, not quietly passed —
        # the verdict log should show it was never consulted rather than imply
        # it looked and approved.
        _off = set()
        if not getattr(self, "_double_debit_on", True):
            _off.add("double_debit_lock")
        if not getattr(self, "_hard_decline_on", True):
            _off.add("hard_decline")
        guardrail_checks = [c for c in guardrail_checks if c[0] not in _off]

        for name, check_fn in guardrail_checks:
            result = check_fn()
            checks.append(result)

            # If guardrail modifies the action, use the modified version for subsequent checks
            if result.verdict == GuardrailVerdict.MODIFIED and result.modified_action:
                try:
                    current_action = ActionType(result.modified_action)
                except ValueError:
                    pass

            # If guardrail blocks, use a safe compliant fallback instead of the blocked action
            if result.verdict == GuardrailVerdict.BLOCKED:
                if current_action in (ActionType.SEND_NOTIFICATION, ActionType.UPDATE_PAYMENT_METHOD):
                    fallback = ActionType.WAIT_AND_RETRY
                elif current_action == ActionType.RETRY_PAYMENT:
                    fallback = ActionType.ESCALATE_TO_HUMAN
                elif current_action == ActionType.WAIT_AND_RETRY:
                    fallback = ActionType.ESCALATE_TO_HUMAN
                else:
                    fallback = ActionType.WAIT_AND_RETRY

                current_action = fallback
                checks.append(GuardrailCheckResult(
                    guardrail="interceptor",
                    verdict=GuardrailVerdict.BLOCKED,
                    reason=f"Action blocked by {name}: {result.reason}. Fallback: {fallback.value}.",
                    original_action=action.value,
                    modified_action=fallback.value,
                ))
                break

        # Store results in case metadata
        case.payment.metadata["guardrail_checks"] = [c.model_dump() for c in checks]
        case.payment.metadata["guardrail_final_action"] = current_action.value

        return current_action, checks
