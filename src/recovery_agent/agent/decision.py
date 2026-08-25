"""Decision layer — maps root cause to intervention strategy.

Source: Planning pattern from Agentic AI (Andrew Ng), Module 5
        Tool selection from Evaluating AI Agents
"""
from __future__ import annotations

from recovery_agent.models import (
    ActionType,
    Case,
    CaseStatus,
    FailureType,
)


def decide_intervention(case: Case) -> ActionType:
    """Choose the appropriate intervention based on diagnosis and attempt history.

    Maps: root_cause + attempt_count -> action type

    Source: Planning pattern — agent creates plan then executes
    https://www.deeplearning.ai/courses/agentic-ai (Module 5)
    """
    if not case.diagnosis:
        return ActionType.ABANDON

    cause = case.diagnosis.root_cause
    attempts = case.attempt_count

    # Decision matrix: cause x attempt count -> action
    decision_tree: dict[FailureType, dict[int, ActionType]] = {
        FailureType.CARD_EXPIRED: {
            0: ActionType.SEND_NOTIFICATION,   # Ask to update card
            1: ActionType.UPDATE_PAYMENT_METHOD,
            2: ActionType.ESCALATE_TO_HUMAN,
        },
        FailureType.INSUFFICIENT_FUNDS: {
            0: ActionType.WAIT_AND_RETRY,      # Wait, funds may arrive
            1: ActionType.RETRY_PAYMENT,
            2: ActionType.SEND_NOTIFICATION,
            3: ActionType.ESCALATE_TO_HUMAN,
        },
        FailureType.BANK_DECLINED: {
            0: ActionType.RETRY_PAYMENT,       # Transient, retry once
            1: ActionType.SEND_NOTIFICATION,
            2: ActionType.ESCALATE_TO_HUMAN,
        },
        FailureType.NETWORK_TIMEOUT: {
            0: ActionType.RETRY_PAYMENT,       # Transient, retry immediately
            1: ActionType.WAIT_AND_RETRY,
            2: ActionType.RETRY_PAYMENT,
            3: ActionType.ESCALATE_TO_HUMAN,
        },
        FailureType.RISK_BLOCK: {
            0: ActionType.ESCALATE_TO_HUMAN,   # Always escalate risk blocks
        },
        FailureType.MANDATE_REVOKED: {
            0: ActionType.SEND_NOTIFICATION,   # Inform customer
            1: ActionType.ESCALATE_TO_HUMAN,
        },
        FailureType.UNKNOWN: {
            0: ActionType.RETRY_PAYMENT,       # Try generic retry
            1: ActionType.SEND_NOTIFICATION,
            2: ActionType.ESCALATE_TO_HUMAN,
        },
    }

    cause_tree = decision_tree.get(cause, decision_tree[FailureType.UNKNOWN])

    # Find the right action for this attempt count
    # Use the highest attempt key that's <= current attempt
    action = ActionType.ABANDON
    for threshold in sorted(cause_tree.keys(), reverse=True):
        if attempts >= threshold:
            action = cause_tree[threshold]
            break

    return action


def run_decision(case: Case) -> Case:
    """Run decision layer on a case and update its state.

    Transitions case from DIAGNOSING → DIAGNOSED after selecting an intervention.

    Source: Planning with code execution
    https://www.deeplearning.ai/courses/agentic-ai (Module 5)
    """
    case.status = CaseStatus.DIAGNOSED

    action = decide_intervention(case)

    # Store the decided action in the case metadata for the act step
    case.payment.metadata["decided_action"] = action.value

    return case
