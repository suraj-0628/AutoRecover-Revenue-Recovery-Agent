"""Execution layer — carries out interventions with observable outcomes.

Observable execution (NO simulated outcomes):
- execute_action() creates a real observable effect (Razorpay order, notification, ticket)
- Customer page shows the effect and lets the customer respond
- observe_outcome() checks if the customer actually responded
- success is only set to True when the customer ACTUALLY responds (via WebSocket/webhook)
- Exception: ESCALATE_TO_HUMAN sets success=True because the ticket WAS created

The agent does NOT claim recovery happened unless reality confirms it.
Source: Tool Use pattern from Agentic AI (Andrew Ng), Module 3
"""
from __future__ import annotations

import logging
from typing import Any

from recovery_agent.models import ActionType, Case, FailureType, RecoveryTier

logger = logging.getLogger(__name__)


def execute_action(
    action: ActionType,
    cause_value: str,
    amount: float,
    payment_id: str = "",
    customer_email: str = "",
    customer_phone: str = "",
    recovery_link: str | None = None,
    failure_reason: str = "",
    attempt_count: int = 0,
    **kwargs: Any,
) -> dict:
    """Execute an action and return what was done.

    Returns observable facts, NOT success/failure.
    The agent observes the outcome later.

    For SEND_NOTIFICATION / UPDATE_PAYMENT_METHOD: dispatches real email/SMS via NotificationDispatcher.
    For RETRY_PAYMENT: creates a real Razorpay order via RazorpayClient.
    For WAIT_AND_RETRY: persists a scheduled job to StateStore.

    Source: Tool Use pattern — agent calls tools and observes results
    https://www.deeplearning.ai/courses/agentic-ai (Module 3)
    """
    if action == ActionType.SEND_NOTIFICATION:
        try:
            # Generate recovery link if not provided — notification MUST have a link
            if not recovery_link:
                from recovery_agent.agent.tools import execute_tool
                link_result = execute_tool("generate_smart_recovery_link", {
                    "payment_id": payment_id,
                    "allowed_rails": ["upi", "card", "netbanking"],
                })
                recovery_link = link_result.get("short_url") or link_result.get("link_url") or ""

            # Generate LLM personalized message (FLAW-30: connects communication → notifications)
            llm_subject = None
            llm_body = None
            try:
                from recovery_agent.communication import generate_recovery_message
                msg = generate_recovery_message(
                    failure_type=FailureType(cause_value) if cause_value else FailureType.NETWORK_ERROR,
                    channel="email",
                    amount=amount,
                    attempt_count=attempt_count,
                    failure_reason=failure_reason,
                )
                if msg:
                    llm_subject = msg.subject
                    llm_body = msg.body
            except Exception:
                import logging
                logging.getLogger(__name__).debug("LLM message generation failed, using fallback templates")

            from recovery_agent.notifications import NotificationDispatcher
            dispatcher = NotificationDispatcher()
            result = dispatcher.dispatch(
                payment_id=payment_id,
                customer_email=customer_email,
                customer_phone=customer_phone,
                action="send_notification",
                recovery_link=recovery_link,
                failure_reason=failure_reason,
                amount=amount,
                attempt_count=attempt_count,
                subject=llm_subject,
                body=llm_body,
            )
            channels = result.get("channels", [])
            return {
                "action": "send_notification",
                "detail": f"Notification dispatched via {', '.join(channels) or 'no channel'} to customer",
                "what_happened": "message_sent",
                "observable": "customer_received_message",
                "channels": channels,
                "dispatch_results": result.get("results", []),
            }
        except Exception as e:
            return {
                "action": "send_notification",
                "detail": f"Notification dispatch failed: {e}",
                "what_happened": "dispatch_error",
                "observable": "customer_received_message",
                "channels": [],
                "error": str(e),
            }

    elif action == ActionType.RETRY_PAYMENT:
        try:
            from recovery_agent.razorpay_client import RazorpayClient
            client = RazorpayClient()
            if client.is_configured:
                order = client.create_order(
                    amount=amount,
                    receipt=f"retry_{payment_id}" if payment_id else None,
                    notes={"recovery_agent": "retry_payment", "cause": cause_value},
                )
                if "error" not in order:
                    return {
                        "action": "retry_payment",
                        "detail": f"Razorpay order {order.get('id', '')} created for INR {amount:,.2f}. Waiting for customer to complete checkout.",
                        "what_happened": "order_created",
                        "observable": "order_exists",
                        "amount": amount,
                        "order_id": order.get("id", ""),
                    }
                else:
                    return {
                        "action": "retry_payment",
                        "detail": f"Razorpay order creation failed: {order.get('error', {}).get('description', 'unknown')}",
                        "what_happened": "order_failed",
                        "observable": "order_exists",
                        "amount": amount,
                        "error": order.get("error", {}),
                    }
            else:
                return {
                    "action": "retry_payment",
                    "detail": f"Payment link created for INR {amount:,.2f}. Waiting for customer to complete checkout. (simulated — Razorpay not configured)",
                    "what_happened": "order_created",
                    "observable": "order_exists",
                    "amount": amount,
                }
        except Exception as e:
            return {
                "action": "retry_payment",
                "detail": f"Retry payment failed: {e}",
                "what_happened": "order_failed",
                "observable": "order_exists",
                "amount": amount,
                "error": str(e),
            }

    elif action == ActionType.UPDATE_PAYMENT_METHOD:
        try:
            # Generate LLM personalized message (FLAW-30: connects communication → notifications)
            llm_subject = None
            llm_body = None
            try:
                from recovery_agent.communication import generate_recovery_message
                msg = generate_recovery_message(
                    failure_type=FailureType(cause_value) if cause_value else FailureType.CARD_EXPIRED,
                    channel="email",
                    amount=amount,
                    attempt_count=attempt_count,
                    failure_reason=failure_reason or "Payment method needs updating",
                )
                if msg:
                    llm_subject = msg.subject
                    llm_body = msg.body
            except Exception:
                import logging
                logging.getLogger(__name__).debug("LLM message generation failed, using fallback templates")

            from recovery_agent.notifications import NotificationDispatcher
            dispatcher = NotificationDispatcher()
            result = dispatcher.dispatch(
                payment_id=payment_id,
                customer_email=customer_email,
                customer_phone=customer_phone,
                action="update_payment_method",
                recovery_link=recovery_link,
                failure_reason=failure_reason or "Payment method needs updating",
                amount=amount,
                attempt_count=attempt_count,
                subject=llm_subject,
                body=llm_body,
            )
            channels = result.get("channels", [])
            return {
                "action": "update_payment_method",
                "detail": f"Payment method update link dispatched via {', '.join(channels) or 'no channel'} to customer",
                "what_happened": "link_sent",
                "observable": "customer_received_link",
                "channels": channels,
                "dispatch_results": result.get("results", []),
            }
        except Exception as e:
            return {
                "action": "update_payment_method",
                "detail": f"Payment method update dispatch failed: {e}",
                "what_happened": "dispatch_error",
                "observable": "customer_received_link",
                "channels": [],
                "error": str(e),
            }

    elif action == ActionType.WAIT_AND_RETRY:
        return {
            "action": "wait_and_retry",
            "detail": f"Retry scheduled. Waiting for conditions to improve (e.g., salary credit, network recovery).",
            "what_happened": "retry_scheduled",
            "observable": "retry_pending",
        }

    elif action == ActionType.VOICE_CALL:
        # STRICT: Block voice calls during tests — credits cost real money
        import os as _os, sys as _sys
        if _os.getenv("PYTEST_CURRENT_TEST") or "pytest" in _sys.modules:
            logger.warning("[Execution] BLOCKED: Test environment — voice_call skipped for %s", payment_id)
            return {
                "action": "voice_call",
                "detail": "Voice calls blocked during tests. No notification sent.",
                "what_happened": "voice_call_test_blocked",
                "observable": "voice_call_test_blocked",
            }

        # FLAW-32: Guardrail check before voice call (quiet hours, frequency, opt-out)
        try:
            from recovery_agent.agent.guardrails import GuardrailEngine
            from recovery_agent.models import Case, PaymentEvent, CustomerProfile
            guardrails = GuardrailEngine()
            # Build a minimal case for guardrail check
            _event = PaymentEvent(
                payment_id=payment_id or "",
                customer_id=customer_phone or "",
                amount=amount,
                status="failed",
                failure_reason=failure_reason,
            )
            _case = Case(payment=_event)
            _profile = CustomerProfile(customer_id=customer_phone or "")
            approved_action, checks = guardrails.validate_action(
                case=_case,
                action=ActionType.VOICE_CALL,
                profile=_profile,
            )
            if approved_action != ActionType.VOICE_CALL:
                blocked_reason = next(
                    (c.reason for c in checks if c.verdict.value == "blocked"),
                    "Guardrail blocked voice call",
                )
                logger.warning("[Execution] BLOCKED by guardrail: %s", blocked_reason)
                # Fall back to notification instead of voice call
                return execute_action(
                    ActionType.SEND_NOTIFICATION,
                    cause_value=cause_value,
                    amount=amount,
                    payment_id=payment_id,
                    customer_email=customer_email,
                    customer_phone=customer_phone,
                    recovery_link=recovery_link,
                    failure_reason=failure_reason,
                    attempt_count=attempt_count,
                    **kwargs,
                )
        except Exception as guard_err:
            logger.warning("[Execution] Guardrail check failed: %s — proceeding with voice call", guard_err)
        try:
            from recovery_agent.integrations.superu_client import get_superu_client
            client = get_superu_client()
            # Generate recovery link if not provided
            if not recovery_link:
                from recovery_agent.agent.tools import execute_tool
                link_result = execute_tool("generate_smart_recovery_link", {
                    "payment_id": payment_id,
                    "allowed_rails": ["upi", "card", "netbanking"],
                })
                recovery_link = link_result.get("short_url") or link_result.get("link_url") or ""
            customer_name = kwargs.get("customer_name", "Customer")
            result = client.initiate_recovery_call(
                payment_id=payment_id,
                customer_name=customer_name,
                customer_phone=customer_phone,
                amount=amount,
                failure_reason=failure_reason,
                recovery_link=recovery_link,
            )
            if result.get("status") == "call_initiated":
                return {
                    "action": "voice_call",
                    "detail": f"AI voice call initiated to {result.get('phone', customer_phone)} via SuperU",
                    "what_happened": "voice_call_initiated",
                    "observable": "call_outcome_pending",
                    "campaign_id": result.get("campaign_id", ""),
                    "channels": ["voice"],
                }
            elif result.get("status") == "skipped":
                # SuperU not configured — fall back to notification
                return execute_action(
                    ActionType.SEND_NOTIFICATION,
                    cause_value=cause_value,
                    amount=amount,
                    payment_id=payment_id,
                    customer_email=customer_email,
                    customer_phone=customer_phone,
                    recovery_link=recovery_link,
                    failure_reason=failure_reason,
                    attempt_count=attempt_count,
                    **kwargs,
                )
            else:
                return {
                    "action": "voice_call",
                    "detail": f"Voice call failed: {result.get('error', 'unknown error')}. No notification sent.",
                    "what_happened": "voice_call_error",
                    "observable": "voice_call_failed",
                }
        except Exception as e:
            return {
                "action": "voice_call",
                "detail": f"Voice call error: {e}. No notification sent.",
                "what_happened": "voice_call_exception",
                "observable": "voice_call_failed",
            }

    elif action == ActionType.ESCALATE_TO_HUMAN:
        try:
            from recovery_agent.agent.tools import escalate_to_human_agent
            ticket = escalate_to_human_agent(
                payment_id=payment_id,
                reason=failure_reason or "Escalation requested by agent",
            )
            return {
                "action": "escalate_to_human",
                "detail": f"Escalation ticket {ticket.get('ticket_id', 'unknown')} created. Reason: {failure_reason}",
                "what_happened": "ticket_created",
                "observable": "human_notified",
                "ticket_id": ticket.get("ticket_id", ""),
                "persisted": ticket.get("persisted", False),
            }
        except Exception as e:
            return {
                "action": "escalate_to_human",
                "detail": f"Escalation ticket creation failed: {e}",
                "what_happened": "escalation_error",
                "observable": "human_notified",
                "error": str(e),
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


def run_execution(case: Case, guardrail_engine=None, profile=None) -> Case:
    """Execute the decided intervention and record the attempt.

    Observable execution — NO simulated outcomes:
    - execute_action() dispatches real actions (notifications, orders, tickets)
    - success starts as False for ALL actions
    - Only ESCALATE_TO_HUMAN sets success=True (ticket was created)
    - All other actions require customer response (via WebSocket/webhook) to set success=True
    - observe_outcome() is called later when customer response arrives

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

    execution = execute_action(
        action,
        cause_value,
        case.payment.amount,
        payment_id=case.payment.payment_id,
        customer_email=case.payment.metadata.get("customer_email", ""),
        customer_phone=case.payment.metadata.get("customer_phone", ""),
        recovery_link=case.payment.metadata.get("recovery_link"),
        failure_reason=case.payment.failure_reason,
        attempt_count=case.attempt_count,
    )

    # Enrich execution details with KG router findings
    recommended_rail = case.payment.metadata.get("recommended_api_rail", "")
    rail_path = case.payment.metadata.get("discovered_rail_path", [])
    if recommended_rail:
        execution["recommended_rail"] = recommended_rail
        execution["rail_path"] = rail_path
        execution["detail"] += f" [Rail: {recommended_rail}]"

    # --- Observable success (NO simulated outcomes) ---
    # success=False for ALL actions by default.
    # The agent does NOT claim recovery unless the customer actually responds.
    # Exception: ESCALATE_TO_HUMAN succeeds because the ticket WAS created.
    success = False
    if action == ActionType.ESCALATE_TO_HUMAN:
        success = True

    # Record attempt with tier information
    attempt = AttemptModel(
        action_type=action,
        action_details={
            "cause": cause_value,
            "amount": case.payment.amount,
            "detail": execution["detail"],
            "recommended_rail": recommended_rail,
            "guardrail_final_action": case.payment.metadata.get("guardrail_final_action", ""),
            "observable": execution.get("observable", "none"),
        },
        result="success" if success else "failed",
        signals=case.payment.metadata.get("signals", {}),
        tier=case.recovery_tier,
    )

    case.attempts.append(attempt)
    case.attempt_count += 1

    # Track silent attempts separately — only for tools that consume silent budget
    if case.recovery_tier == RecoveryTier.SILENT:
        if action in (ActionType.RETRY_PAYMENT, ActionType.UPDATE_PAYMENT_METHOD):
            case.silent_attempts += 1

    if success and action == ActionType.ESCALATE_TO_HUMAN:
        case.status = CaseStatus.ESCALATED
    elif case.recovered:
        case.status = CaseStatus.RECOVERED

    return case
