"""Stopping rules — enforces explicit boundaries on the agent.

Source: Guardrails concept from Multi AI Agent Systems with crewAI
        Governance — risk management pillar from Governing AI Agents
"""
from __future__ import annotations

from recovery_agent.models import (
    ActionType,
    Case,
    CaseStatus,
)


def check_stopping_rules(case: Case) -> tuple[bool, str]:
    """Evaluate whether the agent should stop.

    Returns (should_stop, reason).

    Stopping rules:
    1. Recovery succeeded -> stop
    2. Max attempts reached -> stop
    3. Escalated to human -> stop (human takes over)
    4. Abandoned -> stop
    5. Max loops reached -> stop (safety valve)

    Source: Guardrails — break infinite loops, handle errors gracefully
    https://www.deeplearning.ai/courses/multi-ai-agent-systems-with-crewai
    """
    # Rule 1: Already recovered
    if case.recovered:
        return True, "Recovery succeeded"

    # Rule 2: Max attempts reached
    if case.attempt_count >= case.max_attempts:
        return True, f"Max attempts ({case.max_attempts}) reached"

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

    # Rule 5: Safety valve — max loops in the graph
    # (checked externally in the agent loop)

    return False, ""


def run_stopping_check(case: Case) -> Case:
    """Check stopping rules and update case status.

    Source: Governance — observability pillar, track decisions
    https://www.deeplearning.ai/courses/governing-ai-agents
    """
    should_stop, reason = check_stopping_rules(case)

    if should_stop:
        if case.recovered:
            case.status = CaseStatus.RECOVERED
        elif any(a.action_type == ActionType.ESCALATE_TO_HUMAN for a in case.attempts):
            case.status = CaseStatus.ESCALATED
        else:
            case.status = CaseStatus.STOPPED

    return case
