"""Guardrail Engine — NVIDIA NAT-style pre-execution safety interceptor.

Validates every action against 5 enterprise safety policies before execution.
If a policy is violated, the action is vetoed or modified to a compliant fallback.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from recovery_agent.models import ActionType, Case, CustomerProfile, HARD_DECLINES


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

    def check(
        self,
        action: ActionType,
        profile: CustomerProfile | None = None,
        now: datetime | None = None,
    ) -> GuardrailCheckResult:
        current = now or datetime.now(timezone.utc)
        hour = current.hour

        is_quiet = hour >= self.quiet_start or hour < self.quiet_end

        if not is_quiet:
            return GuardrailCheckResult(
                guardrail="quiet_hours",
                verdict=GuardrailVerdict.PASS,
                reason="Outside quiet hours",
                original_action=action.value,
            )

        # During quiet hours, communication actions are deferred
        communication_actions = {
            ActionType.SEND_NOTIFICATION,
            ActionType.UPDATE_PAYMENT_METHOD,
        }

        if action in communication_actions:
            return GuardrailCheckResult(
                guardrail="quiet_hours",
                verdict=GuardrailVerdict.MODIFIED,
                reason=f"Quiet hours active ({hour}:00). Communication deferred.",
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
        communication_actions = {
            ActionType.SEND_NOTIFICATION,
            ActionType.UPDATE_PAYMENT_METHOD,
        }

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

        # Count recent contacts (last 24h)
        current = now or datetime.now(timezone.utc)
        recent_contacts = 0
        for record in profile.payment_history:
            if record.status in ("success", "failed") and record.channel_used:
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

        communication_actions = {
            ActionType.SEND_NOTIFICATION,
            ActionType.UPDATE_PAYMENT_METHOD,
        }

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


class SemanticGuardrail:
    """LLM-based semantic evaluation for complex safety policy decisions.

    Uses NeMo Colang-style reasoning to evaluate if an action violates
    broad safety policies that cannot be captured by static IF statements.

    Falls back to deterministic rules if LLM is unavailable.
    """

    SEMANTIC_EVAL_PROMPT = """You are a payment recovery safety guardrail evaluating whether an action is safe.

ACTION TO EVALUATE:
  Type: {action_type}
  Customer opted out: {opt_out}
  Communication count (24h): {comm_count_24h}
  Failure code: {failure_code}
  Attempt count: {attempt_count}
  Amount: INR {amount}

POLICY RULES:
1. Do NOT send communications to customers who have opted out
2. Do NOT exceed 3 communications per 24 hours
3. Do NOT retry a payment that has already been captured
4. Do NOT execute high-value retries (>INR 1,00,000) without human review
5. Do NOT send notifications during quiet hours (9 PM - 8 AM IST)
6. If the customer has complained or the case is disputed, escalate to human

Determine if this action is SAFE, UNSAFE, or REQUIRES_REVIEW.
Output JSON:
{{
  "verdict": "safe" | "unsafe" | "requires_review",
  "reason": "Brief explanation",
  "suggested_action": "If unsafe, suggest the correct action"
}}"""

    def __init__(self):
        self._use_llm = True

    def check(
        self,
        action: ActionType,
        profile: CustomerProfile | None = None,
        case: Case | None = None,
        **kwargs: Any,
    ) -> GuardrailCheckResult:
        """Semantic safety evaluation using LLM (with deterministic fallback)."""
        # Fast path: deterministic checks first
        if action == ActionType.ESCALATE_TO_HUMAN:
            return GuardrailCheckResult(
                guardrail="semantic",
                verdict=GuardrailVerdict.PASS,
                reason="Escalation to human is always safe",
                original_action=action.value,
            )

        if not self._use_llm:
            return self._deterministic_fallback(action, profile, case)

        # LLM-based semantic evaluation
        try:
            from recovery_agent.agent.llm_client import invoke_llm_json

            comm_count = 0
            if profile:
                from recovery_agent.agent.memory import CustomerMemoryStore
                store = CustomerMemoryStore()
                comm_count = store.get_communication_count_24h(profile.customer_id)

            prompt = self.SEMANTIC_EVAL_PROMPT.format(
                action_type=action.value,
                opt_out=profile.opt_out if profile else False,
                comm_count_24h=comm_count,
                failure_code=case.payment.failure_code if case else "unknown",
                attempt_count=case.attempt_count if case else 0,
                amount=case.payment.amount if case else 0,
            )

            result = invoke_llm_json(
                prompt=prompt,
                system="You are a payment recovery safety guardrail. Output only JSON.",
                temperature=0,
                max_tokens=256,
            )

            if result is None:
                return self._deterministic_fallback(action, profile, case)

            verdict = result.get("verdict", "safe")
            reason = result.get("reason", "LLM semantic evaluation")
            suggested = result.get("suggested_action", "")

            if verdict == "unsafe":
                try:
                    modified = ActionType(suggested) if suggested else ActionType.WAIT_AND_RETRY
                except ValueError:
                    modified = ActionType.WAIT_AND_RETRY
                return GuardrailCheckResult(
                    guardrail="semantic",
                    verdict=GuardrailVerdict.BLOCKED,
                    reason=f"Semantic evaluation: {reason}",
                    original_action=action.value,
                    modified_action=modified.value,
                )
            elif verdict == "requires_review":
                return GuardrailCheckResult(
                    guardrail="semantic",
                    verdict=GuardrailVerdict.MODIFIED,
                    reason=f"Semantic evaluation: {reason}",
                    original_action=action.value,
                    modified_action=ActionType.ESCALATE_TO_HUMAN.value,
                )
            else:
                return GuardrailCheckResult(
                    guardrail="semantic",
                    verdict=GuardrailVerdict.PASS,
                    reason=f"Semantic evaluation: {reason}",
                    original_action=action.value,
                )

        except Exception as e:
            return self._deterministic_fallback(action, profile, case)

    def _deterministic_fallback(
        self,
        action: ActionType,
        profile: CustomerProfile | None,
        case: Case | None,
    ) -> GuardrailCheckResult:
        """Deterministic fallback when LLM is unavailable."""
        if profile and profile.opt_out:
            if action in (ActionType.SEND_NOTIFICATION, ActionType.UPDATE_PAYMENT_METHOD):
                return GuardrailCheckResult(
                    guardrail="semantic",
                    verdict=GuardrailVerdict.BLOCKED,
                    reason="Customer opted out (deterministic fallback)",
                    original_action=action.value,
                    modified_action=ActionType.WAIT_AND_RETRY.value,
                )

        if case and case.payment.amount > 100_000:
            if action == ActionType.RETRY_PAYMENT:
                return GuardrailCheckResult(
                    guardrail="semantic",
                    verdict=GuardrailVerdict.MODIFIED,
                    reason=f"High value INR {case.payment.amount:,.2f} requires human review (deterministic fallback)",
                    original_action=action.value,
                    modified_action=ActionType.ESCALATE_TO_HUMAN.value,
                )

        return GuardrailCheckResult(
            guardrail="semantic",
            verdict=GuardrailVerdict.PASS,
            reason="Deterministic fallback: action appears safe",
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
        self.quiet_hours = QuietHourGuardrail()
        self.frequency_cap = FrequencyCapGuardrail()
        self.double_debit = DoubleDebitLockGuardrail()
        self.opt_out = OptOutGuardrail()
        self.monetary_cap = MonetaryCapGuardrail()
        self.hard_decline = HardDeclineGuardrail()
        self.semantic = SemanticGuardrail()

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
            ("semantic", lambda: self.semantic.check(current_action, profile, case=case)),
        ]

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
                # Communication blocks → WAIT_AND_RETRY (defer to later)
                # Payment blocks → ESCALATE_TO_HUMAN (let human handle)
                # Other blocks → WAIT_AND_RETRY
                if current_action in (ActionType.SEND_NOTIFICATION, ActionType.UPDATE_PAYMENT_METHOD):
                    fallback = ActionType.WAIT_AND_RETRY
                elif current_action == ActionType.RETRY_PAYMENT:
                    fallback = ActionType.ESCALATE_TO_HUMAN
                else:
                    fallback = ActionType.WAIT_AND_RETRY

                current_action = fallback
                # Record the block
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
