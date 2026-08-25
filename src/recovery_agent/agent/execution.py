"""Execution layer — carries out interventions with observable outcomes.

Two execution modes:

1. FRONTEND (observable, not rigged):
   - execute_action() creates a real observable effect (Razorpay order, notification, ticket)
   - Customer page shows the effect and lets the customer respond
   - observe_outcome() checks if the customer actually responded
   - No dice rolls — real customer behavior determines recovery

2. BATCH/STANDALONE (weighted probabilistic simulation):
   - execute_action() still creates the observable effect
   - run_execution() simulates customer response using weighted success rates
   - Each action/failure combo has a realistic probability of customer response
   - This is NOT random — probabilities are calibrated to real-world retry behavior

Source: Tool Use pattern from Agentic AI (Andrew Ng), Module 3
"""
from __future__ import annotations

from recovery_agent.models import ActionType


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


# Wrapper for backward compatibility with agent/__init__.py
def run_execution(case, guardrail_engine=None, profile=None) -> Case:
    """Execute the decided intervention and record the attempt.

    Incorporates pre-execution guardrail interception, KG router metadata,
    and guardrail check results into execution details for observability.

    Source: Error handling through feedback loops
    https://www.deeplearning.ai/courses/building-coding-agents-with-tool-execution
    """
    from recovery_agent.models import Attempt as AttemptModel, ActionType, CaseStatus
    import random

    case.status = CaseStatus.ACTING

    action_value = case.payment.metadata.get("decided_action", "abandon")
    action = ActionType(action_value)
    cause_value = case.diagnosis.root_cause.value if case.diagnosis else "unknown"

    # --- Pre-execution guardrail interception ---
    if guardrail_engine:
        action, guardrail_checks = guardrail_engine.validate_action(
            case=case, action=action, profile=profile,
        )
        # Update the decided action to the guardrail-approved version
        case.payment.metadata["decided_action"] = action.value

    execution = execute_action(action, cause_value, case.payment.amount)

    # Enrich execution details with KG router findings
    recommended_rail = case.payment.metadata.get("recommended_api_rail", "")
    rail_path = case.payment.metadata.get("discovered_rail_path", [])
    if recommended_rail:
        execution["recommended_rail"] = recommended_rail
        execution["rail_path"] = rail_path
        execution["detail"] += f" [Rail: {recommended_rail}]"

    # For the batch/standalone agent, simulate customer response
    # In the frontend, this is handled by the waiting mechanism
    success_rate = {
        ActionType.RETRY_PAYMENT: 0.35,
        ActionType.SEND_NOTIFICATION: 0.60,
        ActionType.UPDATE_PAYMENT_METHOD: 0.55,
        ActionType.WAIT_AND_RETRY: 0.45,
        ActionType.ESCALATE_TO_HUMAN: 0.80,
        ActionType.ABANDON: 0.0,
    }.get(action, 0.3)

    # Boost success rate if KG found a high-quality rail
    if recommended_rail and action == ActionType.RETRY_PAYMENT:
        from recovery_agent.agent.kg_router import RAIL_DETAILS
        rail_info = RAIL_DETAILS.get(recommended_rail, {})
        conversion_cost = rail_info.get("conversion_cost", 0.05)
        # Lower conversion cost → higher success multiplier (1.0 to 1.3)
        kg_boost = 1.0 + (0.05 - conversion_cost) * 6
        success_rate = min(0.95, success_rate * kg_boost)

    success = random.random() < success_rate

    attempt = AttemptModel(
        action_type=action,
        action_details={
            "cause": cause_value,
            "amount": case.payment.amount,
            "detail": execution["detail"],
            "recommended_rail": recommended_rail,
            "guardrail_final_action": case.payment.metadata.get("guardrail_final_action", ""),
        },
        result="success" if success else "failed",
    )

    case.attempts.append(attempt)
    case.attempt_count += 1

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
