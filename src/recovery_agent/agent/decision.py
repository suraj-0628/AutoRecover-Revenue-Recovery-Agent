"""Decision layer — maps root cause to intervention strategy.

Source: Planning pattern from Agentic AI (Andrew Ng), Module 5
        Tool selection from Evaluating AI Agents
"""
from __future__ import annotations

from datetime import datetime

from recovery_agent.agent.kg_router import RazorpayKnowledgeGraph
from recovery_agent.agent.memory import CustomerMemoryStore
from recovery_agent.models import (
    ActionType,
    Case,
    CaseStatus,
    CustomerProfile,
    FailureType,
)


# Shared KG router instance (built once, reused)
_kg_router: RazorpayKnowledgeGraph | None = None


def _get_kg_router() -> RazorpayKnowledgeGraph:
    """Lazy-init the shared KG router."""
    global _kg_router
    if _kg_router is None:
        _kg_router = RazorpayKnowledgeGraph()
    return _kg_router


def decide_intervention(
    case: Case,
    profile: CustomerProfile | None = None,
    memory: CustomerMemoryStore | None = None,
    kg_router: RazorpayKnowledgeGraph | None = None,
) -> ActionType:
    """Choose the appropriate intervention based on diagnosis, attempt history, memory, and KG.

    Memory-aware rules:
    - INSUFFICIENT_FUNDS + salary_dependent + not in salary window → WAIT_AND_RETRY
    - Best channel from profile overrides default

    KG-aware rules:
    - On card/bank failures, discover optimal recovery rail path
    - Attach rail recommendation to case metadata

    Source: Planning pattern — agent creates plan then executes
    https://www.deeplearning.ai/courses/agentic-ai (Module 5)
    """
    if not case.diagnosis:
        return ActionType.ABANDON

    cause = case.diagnosis.root_cause
    attempts = case.attempt_count
    router = kg_router or _get_kg_router()

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
            return ActionType.WAIT_AND_RETRY

    # KG-aware: Discover recovery rails for card/bank failures
    failure_code = cause.value
    preferred_channel = ""
    if profile and memory:
        preferred_channel = memory.get_optimal_channel(profile.customer_id)

    if cause in (
        FailureType.CARD_EXPIRED,
        FailureType.BANK_DECLINED,
        FailureType.NETWORK_TIMEOUT,
        FailureType.MANDATE_REVOKED,
        FailureType.RISK_BLOCK,
    ):
        path = router.discover_recovery_path(
            failure_code=failure_code,
            customer_id=case.payment.customer_id,
            preferred_channel=preferred_channel,
        )
        recommended_rail = router.recommend_optimal_rail(
            failure_code=failure_code,
            preferred_channel=preferred_channel,
        )
        # Store KG results in metadata for execution layer
        case.payment.metadata["discovered_rail_path"] = path
        case.payment.metadata["recommended_api_rail"] = recommended_rail

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
    kg_router: RazorpayKnowledgeGraph | None = None,
) -> Case:
    """Run decision layer on a case and update its state.

    Transitions case from DIAGNOSING → DIAGNOSED after selecting an intervention.
    Queries KG router for optimal recovery rails on payment failures.

    Source: Planning with code execution
    https://www.deeplearning.ai/courses/agentic-ai (Module 5)
    """
    case.status = CaseStatus.DIAGNOSED

    action = decide_intervention(case, profile=profile, memory=memory, kg_router=kg_router)

    # Store the decided action in the case metadata for the act step
    case.payment.metadata["decided_action"] = action.value

    # Store optimal channel from memory for execution
    if memory and profile:
        case.payment.metadata["optimal_channel"] = memory.get_optimal_channel(
            profile.customer_id
        )

    return case
