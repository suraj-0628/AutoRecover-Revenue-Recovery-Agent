"""Stopping rules — enforces explicit boundaries on the agent.

Two-tier stopping logic:
- Tier 1 (Silent): Separate max attempts for background retries
- Tier 2 (Active): Separate max attempts for customer-facing actions
- tier transition: Silent exhausted → escalate to Active

Source: Guardrails concept from Multi AI Agent Systems with crewAI
        Governance — risk management pillar from Governing AI Agents
"""
from __future__ import annotations

from recovery_agent.models import (
    ACTIVE_ACTIONS,
    ActionType,
    Case,
    CaseStatus,
    RecoveryTier,
)


def check_stopping_rules(case: Case) -> tuple[bool, str]:
    """Evaluate whether the agent should stop.

    Returns (should_stop, reason).

    Two-tier stopping rules:
    1. Recovery succeeded -> stop
    2. Silent tier failed on last attempt -> transition to Active (not stop)
    3. Silent tier exhausted -> transition to Active (not stop)
    4. Active tier exhausted -> stop (escalate to human)
    5. Escalated to human -> stop (human takes over)
    6. Abandoned -> stop
    7. Hard decline detected -> stop (escalate immediately)
    """
    # Rule 1: Already recovered
    if case.recovered:
        return True, "Recovery succeeded"

    # Rule 2: Hard decline detected → stop immediately
    if case.payment.metadata.get("hard_decline_blocked"):
        return True, f"Hard decline code {case.payment.failure_code} — cannot retry"

    # Rule 3: Last attempt was escalation (human takes over)
    if case.attempts:
        last_attempt = case.attempts[-1]
        if last_attempt.action_type == ActionType.ESCALATE_TO_HUMAN:
            return True, "Escalated to human agent"

    # Rule 4: Last attempt was abandon
    if case.attempts:
        last_attempt = case.attempts[-1]
        if last_attempt.action_type == ActionType.ABANDON:
            return True, "Case abandoned — no viable recovery path"

    # Rule 5: Silent tier — if last silent retry FAILED, escalate to Active immediately
    if case.recovery_tier == RecoveryTier.SILENT and case.attempts:
        last_attempt = case.attempts[-1]
        if last_attempt.action_type in (ActionType.RETRY_PAYMENT, ActionType.WAIT_AND_RETRY):
            if last_attempt.result == "failed":
                return False, "SILENT_RETRY_FAILED"

    # Rule 6: Silent tier exhausted → transition to Active
    if case.recovery_tier == RecoveryTier.SILENT:
        if case.silent_attempts >= case.max_silent_attempts:
            return False, "SILENT_TIER_EXHAUSTED"

    # Rule 7: Max attempts reached (any tier) → stop
    if case.attempt_count >= case.max_attempts:
        return True, f"Max attempts ({case.max_attempts}) reached"

    # Rule 8: A customer-facing action succeeded → the ball is in their court.
    # Without this a case that had done everything right still read as ACTING,
    # so the dashboard could not distinguish "agent still working" from "link
    # sent, waiting for the customer to pay".
    if case.attempts:
        last = case.attempts[-1]
        if last.result == "success" and last.action_type in ACTIVE_ACTIONS:
            return False, "AWAITING_CUSTOMER"

    return False, ""


def transition_to_active_tier(case: Case, reason: str = "") -> Case:
    """Transition from Silent to Active tier."""
    case.recovery_tier = RecoveryTier.ACTIVE
    case.payment.metadata["tier_transition"] = "silent_to_active"
    if reason == "SILENT_RETRY_FAILED":
        case.payment.metadata["tier_transition_reason"] = (
            f"Silent tier retry FAILED on attempt #{case.attempt_count}. "
            f"Automatic escalation to Active tier — customer must switch instruments."
        )
    else:
        case.payment.metadata["tier_transition_reason"] = (
            f"Silent tier exhausted after {case.silent_attempts} attempts. "
            f"Transitioning to Active tier for customer-facing recovery."
        )
    return case


def run_stopping_check(case: Case) -> Case:
    """Check stopping rules and update case status."""
    should_stop, reason = check_stopping_rules(case)

    # Handle silent tier failure (transition, not stop)
    if reason == "SILENT_RETRY_FAILED":
        case = transition_to_active_tier(case, reason=reason)
        return case

    # Handle silent tier exhaustion (transition, not stop)
    if reason == "SILENT_TIER_EXHAUSTED":
        case = transition_to_active_tier(case, reason=reason)
        return case

    if should_stop:
        if case.recovered:
            case.status = CaseStatus.RECOVERED
        elif any(a.action_type == ActionType.ESCALATE_TO_HUMAN for a in case.attempts):
            case.status = CaseStatus.ESCALATED
        else:
            case.status = CaseStatus.STOPPED

    return case
