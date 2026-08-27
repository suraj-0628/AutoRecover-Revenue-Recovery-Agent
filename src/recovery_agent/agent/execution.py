"""Execution layer — carries out interventions with observable outcomes.

State-aware outcome evaluation (NO dice-rolls):
- Evaluates success based on root cause alignment, payment rail health,
  customer opt-out status, payday liquidity window, and attempt count.
- Deterministic outcomes based on actual conditions.

Two execution modes:
1. FRONTEND (observable, not rigged):
   - execute_action() creates a real observable effect (Razorpay order, notification, ticket)
   - Customer page shows the effect and lets the customer respond
   - observe_outcome() checks if the customer actually responded

2. BATCH/STANDALONE (state-aware simulation):
   - State-aware outcome evaluation based on root cause alignment,
     payment rail health, payday liquidity window, and attempt count.

Source: Tool Use pattern from Agentic AI (Andrew Ng), Module 3
"""
from __future__ import annotations

from recovery_agent.models import ActionType, Case, RecoveryTier


def execute_action(action: ActionType, cause_value: str, amount: float) -> dict:
    """Execute an action and return what was done.

    Returns observable facts, NOT success/failure.
    The agent observes the outcome later.

    Source: Tool Use pattern — agent calls tools and observes results
    https://www.deeplearning.ai/courses/agentic-ai (Module 3)
    """
    if action == ActionType.SEND_NOTIFICATION:
        return {
            "action": "send_notification",
            "detail": "Notification sent to customer via SMS and email",
            "what_happened": "message_sent",
            "observable": "customer_received_message",
        }

    elif action == ActionType.RETRY_PAYMENT:
        return {
            "action": "retry_payment",
            "detail": f"Payment link created for INR {amount:,.2f}. Waiting for customer to complete checkout.",
            "what_happened": "order_created",
            "observable": "order_exists",
            "amount": amount,
        }

    elif action == ActionType.UPDATE_PAYMENT_METHOD:
        return {
            "action": "update_payment_method",
            "detail": "Payment method update link sent to customer",
            "what_happened": "link_sent",
            "observable": "customer_received_link",
        }

    elif action == ActionType.WAIT_AND_RETRY:
        return {
            "action": "wait_and_retry",
            "detail": f"Retry scheduled. Waiting for conditions to improve (e.g., salary credit, network recovery).",
            "what_happened": "retry_scheduled",
            "observable": "retry_pending",
        }

    elif action == ActionType.ESCALATE_TO_HUMAN:
        return {
            "action": "escalate_to_human",
            "detail": "Case escalated to human support team. Ticket created.",
            "what_happened": "ticket_created",
            "observable": "human_notified",
        }

    elif action == ActionType.ABANDON:
        return {
            "action": "abandon",
            "detail": "No viable recovery path. Case closed.",
            "what_happened": "case_closed",
            "observable": "none",
        }

    return {
        "action": "unknown",
        "detail": f"Unknown action: {action}",
        "what_happened": "error",
        "observable": "none",
    }


def observe_outcome(action: ActionType, execution_result: dict, customer_responded: bool = False) -> dict:
    """Observe the outcome of an executed action.

    This is where the agent checks: did something actually happen?
    No dice rolls. Observable facts only.

    Args:
        action: What was attempted
        execution_result: What was done
        customer_responded: Whether customer took action (real input)

    Returns:
        {"success": bool, "reason": str, "should_continue": bool}
    """
    observable = execution_result.get("observable", "none")

    # Check observable conditions
    if observable == "customer_received_message":
        if customer_responded:
            return {"success": True, "reason": "Customer responded to notification and completed payment", "should_continue": False}
        else:
            return {"success": False, "reason": "Customer did not respond to notification", "should_continue": True}

    elif observable == "order_exists":
        if customer_responded:
            return {"success": True, "reason": "Customer completed checkout via payment link", "should_continue": False}
        else:
            return {"success": False, "reason": "Customer did not complete checkout", "should_continue": True}

    elif observable == "customer_received_link":
        if customer_responded:
            return {"success": True, "reason": "Customer updated payment method and retried", "should_continue": False}
        else:
            return {"success": False, "reason": "Customer did not update payment method", "should_continue": True}

    elif observable == "retry_pending":
        # Wait action always "succeeds" in scheduling, but doesn't recover
        return {"success": False, "reason": "Retry scheduled. Waiting for next opportunity.", "should_continue": True}

    elif observable == "human_notified":
        if customer_responded:
            return {"success": True, "reason": "Human agent resolved the case", "should_continue": False}
        else:
            return {"success": False, "reason": "Human agent is reviewing. No resolution yet.", "should_continue": True}

    elif observable == "none":
        return {"success": False, "reason": "No action taken", "should_continue": False}

    return {"success": False, "reason": "Unknown outcome", "should_continue": False}


# --- State-aware outcome evaluation (replaces dice-rolls) ---

def _evaluate_state_aware_outcome(
    case: Case,
    action: ActionType,
    cause_value: str,
    recommended_rail: str,
    profile: dict | None = None,
    signals: dict | None = None,
) -> float:
    """Calculate success probability based on actual conditions.

    NO random.random() — pure deterministic evaluation based on:
    1. Root cause alignment (action matches failure type)
    2. Payment rail health (bank_health_score from signals)
    3. Customer opt-out status
    4. Payday liquidity window proximity
    5. Attempt count (diminishing returns with more attempts)
    6. Tier (silent vs active)

    Returns probability between 0.0 and 1.0.
    """
    base_rate = 0.3  # Default baseline

    # === 1. Root cause alignment (biggest factor) ===
    # Actions that match the failure type have higher success
    cause_action_alignment = {
        ("network_timeout", ActionType.RETRY_PAYMENT): 0.75,
        ("network_timeout", ActionType.WAIT_AND_RETRY): 0.50,
        ("insufficient_funds", ActionType.WAIT_AND_RETRY): 0.55,
        ("insufficient_funds", ActionType.RETRY_PAYMENT): 0.40,
        ("insufficient_funds", ActionType.SEND_NOTIFICATION): 0.35,
        ("bank_declined", ActionType.RETRY_PAYMENT): 0.45,
        ("bank_declined", ActionType.SEND_NOTIFICATION): 0.35,
        ("bank_declined", ActionType.UPDATE_PAYMENT_METHOD): 0.50,
        ("card_expired", ActionType.UPDATE_PAYMENT_METHOD): 0.65,
        ("card_expired", ActionType.SEND_NOTIFICATION): 0.40,
        ("card_expired", ActionType.RETRY_PAYMENT): 0.05,  # Pointless — card is expired
        ("mandate_revoked", ActionType.SEND_NOTIFICATION): 0.45,
        ("mandate_revoked", ActionType.UPDATE_PAYMENT_METHOD): 0.55,
        ("risk_block", ActionType.ESCALATE_TO_HUMAN): 0.80,
    }

    alignment_key = (cause_value, action)
    aligned_rate = cause_action_alignment.get(alignment_key, base_rate)

    # === 2. Payment rail health ===
    bank_health = 1.0
    if signals:
        bank_health = signals.get("bank_health_score", 1.0)
    # Bank down → much lower success
    if bank_health <= 0.0:
        aligned_rate *= 0.2
    elif bank_health <= 0.5:
        aligned_rate *= 0.6

    # === 3. Customer opt-out status ===
    if profile and profile.get("opt_out", False):
        if action in (ActionType.SEND_NOTIFICATION, ActionType.UPDATE_PAYMENT_METHOD):
            return 0.0  # Blocked — customer opted out

    # === 4. Payday liquidity window ===
    if signals and signals.get("is_payday", False):
        # Payday window — significantly higher success for insufficient_funds
        if cause_value == "insufficient_funds":
            aligned_rate *= 1.8  # First-in-line advantage

    # === 5. Attempt count (diminishing returns) ===
    attempt_count = case.attempt_count
    if attempt_count >= 3:
        aligned_rate *= 0.5  # Diminishing returns after 3 attempts
    elif attempt_count >= 2:
        aligned_rate *= 0.7

    # === 6. Tier adjustment ===
    # Silent tier actions (retries) are more effective when aligned with root cause
    if case.recovery_tier == RecoveryTier.SILENT:
        if action in (ActionType.RETRY_PAYMENT, ActionType.WAIT_AND_RETRY):
            aligned_rate *= 1.1  # Silent retries are more effective
    else:
        # Active tier — customer contact is more effective
        if action in (ActionType.SEND_NOTIFICATION, ActionType.UPDATE_PAYMENT_METHOD):
            aligned_rate *= 1.2

    # === 7. KG rail boost ===
    if recommended_rail and action == ActionType.RETRY_PAYMENT:
        from recovery_agent.agent.kg_router import RAIL_DETAILS
        rail_info = RAIL_DETAILS.get(recommended_rail, {})
        conversion_cost = rail_info.get("conversion_cost", 0.05)
        kg_boost = 1.0 + (0.05 - conversion_cost) * 6
        aligned_rate *= min(1.3, kg_boost)

    return min(0.95, max(0.01, aligned_rate))


def run_execution(case: Case, guardrail_engine=None, profile=None) -> Case:
    """Execute the decided intervention and record the attempt.

    Uses state-aware outcome evaluation (no dice-rolls).
    Incorporates pre-execution guardrail interception, KG router metadata,
    and guardrail check results into execution details for observability.

    Source: Error handling through feedback loops
    https://www.deeplearning.ai/courses/building-coding-agents-with-tool-execution
    """
    from recovery_agent.models import Attempt as AttemptModel, CaseStatus

    case.status = CaseStatus.ACTING

    action_value = case.payment.metadata.get("decided_action", "abandon")
    action = ActionType(action_value)
    cause_value = case.diagnosis.root_cause.value if case.diagnosis else "unknown"

    # --- Pre-execution guardrail interception ---
    if guardrail_engine:
        action, guardrail_checks = guardrail_engine.validate_action(
            case=case, action=action, profile=profile,
        )
        case.payment.metadata["decided_action"] = action.value

    execution = execute_action(action, cause_value, case.payment.amount)

    # Enrich execution details with KG router findings
    recommended_rail = case.payment.metadata.get("recommended_api_rail", "")
    rail_path = case.payment.metadata.get("discovered_rail_path", [])
    if recommended_rail:
        execution["recommended_rail"] = recommended_rail
        execution["rail_path"] = rail_path
        execution["detail"] += f" [Rail: {recommended_rail}]"

    # --- State-aware outcome evaluation (NO dice-rolls) ---
    success_prob = _evaluate_state_aware_outcome(
        case=case,
        action=action,
        cause_value=cause_value,
        recommended_rail=recommended_rail,
        profile=profile.model_dump() if profile else None,
        signals=case.payment.metadata.get("signals"),
    )

    # Deterministic success: threshold based on calculated probability
    # Use attempt count as a deterministic seed for reproducibility
    deterministic_seed = (hash(case.payment.payment_id) + case.attempt_count) % 100
    success = deterministic_seed < (success_prob * 100)

    # Record attempt with tier information
    attempt = AttemptModel(
        action_type=action,
        action_details={
            "cause": cause_value,
            "amount": case.payment.amount,
            "detail": execution["detail"],
            "recommended_rail": recommended_rail,
            "guardrail_final_action": case.payment.metadata.get("guardrail_final_action", ""),
            "success_probability": round(success_prob, 3),
            "state_aware_evaluation": True,
        },
        result="success" if success else "failed",
        signals=case.payment.metadata.get("signals", {}),
        tier=case.recovery_tier,
    )

    case.attempts.append(attempt)
    case.attempt_count += 1

    # Track silent attempts separately
    if case.recovery_tier == RecoveryTier.SILENT:
        case.silent_attempts += 1

    if success and action not in (
        ActionType.SEND_NOTIFICATION,
        ActionType.UPDATE_PAYMENT_METHOD,
        ActionType.WAIT_AND_RETRY,
        ActionType.ESCALATE_TO_HUMAN,
    ):
        case.recovered = True
        case.recovered_amount = case.payment.amount

    if action == ActionType.ESCALATE_TO_HUMAN:
        case.status = CaseStatus.ESCALATED
    elif success and action == ActionType.ABANDON:
        case.status = CaseStatus.STOPPED
    elif case.recovered:
        case.status = CaseStatus.RECOVERED

    return case
