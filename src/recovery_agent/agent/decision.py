"""Decision layer — maps root cause to intervention strategy.

Source: Planning pattern from Agentic AI (Andrew Ng), Module 5
        Tool selection from Evaluating AI Agents
"""
from __future__ import annotations

from datetime import datetime

from recovery_agent.agent.memory import CustomerMemoryStore
from recovery_agent.models import (
    ActionType,
    Case,
    CaseStatus,
    CustomerProfile,
    FailureType,
)


def decide_intervention(
    case: Case,
    profile: CustomerProfile | None = None,
    memory: CustomerMemoryStore | None = None,
) -> ActionType:
    """Choose the appropriate intervention based on diagnosis, attempt history, and memory.

    Memory-aware rules:
    - INSUFFICIENT_FUNDS + salary_dependent + not in salary window → WAIT_AND_RETRY
    - Best channel from profile overrides default
    - Promise-to-pay tracked if customer commits

    Source: Planning pattern — agent creates plan then executes
    https://www.deeplearning.ai/courses/agentic-ai (Module 5)
    """
    if not case.diagnosis:
        return ActionType.ABANDON

    cause = case.diagnosis.root_cause
    attempts = case.attempt_count

    # Memory-aware: If insufficient funds and customer is salary-dependent,
    # keep waiting until salary window is active
    if (
        cause == FailureType.INSUFFICIENT_FUNDS
        and profile
        and memory
        and profile.salary_window.typical_pay_day > 0
    ):
        current_day = datetime.now().day
        in_window = memory.check_salary_liquidity(profile.customer_id, current_day)
        if not in_window and attempts < 3:
            # Don't burn attempts outside salary window — keep waiting
            return ActionType.WAIT_AND_RETRY

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


def run_decision(
    case: Case,
    profile: CustomerProfile | None = None,
    memory: CustomerMemoryStore | None = None,
) -> Case:
    """Run decision layer on a case and update its state.

    Transitions case from DIAGNOSING → DIAGNOSED after selecting an intervention.

    Source: Planning with code execution
    https://www.deeplearning.ai/courses/agentic-ai (Module 5)
    """
    case.status = CaseStatus.DIAGNOSED

    action = decide_intervention(case, profile=profile, memory=memory)

    # Store the decided action in the case metadata for the act step
    case.payment.metadata["decided_action"] = action.value

    # Store optimal channel from memory for execution
    if memory and profile:
        case.payment.metadata["optimal_channel"] = memory.get_optimal_channel(
            profile.customer_id
        )

    return case
