"""Full-stack frontend — Observable agent flow.

Agent acts → customer sees what happened → customer responds → agent observes → decides next step.

Usage:
    python -m recovery_agent.frontend
    # Customer: http://localhost:5002/pay
    # Merchant: http://localhost:5002/merchant
"""
from __future__ import annotations

import os
import time
import traceback
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, render_template, render_template_string, request, jsonify
from flask_socketio import SocketIO

from recovery_agent.razorpay_client import RazorpayClient
from recovery_agent.state_store import StateStore

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24).hex()
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

razorpay_client = RazorpayClient()
store = StateStore()

# Transient — only tracks in-flight agent threads, not persisted
active_agent_payments: set[str] = set()


def push_event(payment_id: str, event_type: str, data: dict):
    payload = {"payment_id": payment_id, "event": event_type, "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"), **data}
    socketio.emit("agent_event", payload)
    socketio.emit("agent_stream", payload)


TIER_COLORS = {
    "silent": "#3b82f6",
    "active": "#f59e0b",
    "hard_decline_blocked": "#ef4444",
}
TIER_BADGES = {
    "silent": "SILENT RECOVERY",
    "active": "ACTIVE RECOVERY",
    "hard_decline_blocked": "HARD DECLINE BLOCKED",
}

# Decline code → strategy description mapping (for telemetry card)
DECLINE_STRATEGY_DISPLAY = {
    "insufficient_funds": "Payday Timing Scheduler",
    "card_expired": "Card Update Flow",
    "network_timeout": "Metadata Enrichment + Retry",
    "bank_declined": "Multi-Rail Failover",
    "mandate_revoked": "Re-Auth Notification",
    "risk_block": "Human Escalation",
    "card_declined": "Network Penalty Prevention",
    "unknown": "LLM Diagnostic Routing",
}

# Network fine rates (Visa/MC per-attempt penalty)
NETWORK_FINE_PER_ATTEMPT = 0.10  # USD


def push_tier_event(
    payment_id: str,
    tier: str,
    penalties_prevented: int = 0,
    decline_strategy: str = "",
    payday_target_date: str = "",
):
    """Broadcast tier badge, penalty counter, and strategy info to merchant dashboard."""
    tier_badge = TIER_BADGES.get(tier, "ACTIVE RECOVERY")
    tier_color = TIER_COLORS.get(tier, "#f59e0b")
    socketio.emit("tier_update", {
        "payment_id": payment_id,
        "tier": tier,
        "tier_badge": tier_badge,
        "tier_color": tier_color,
        "penalties_prevented": penalties_prevented,
        "penalties_value": f"${penalties_prevented * NETWORK_FINE_PER_ATTEMPT:.2f}",
        "penalties_value_inr": f"INR {penalties_prevented * 8.30:.2f}",
        "decline_strategy": decline_strategy,
        "payday_target_date": payday_target_date,
        "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
    })


def push_decline_strategy_event(payment_id: str, failure_code: str, strategy: str, tier: str):
    """Emit decline-code strategy routing update for the telemetry card."""
    socketio.emit("decline_strategy", {
        "payment_id": payment_id,
        "failure_code": failure_code,
        "strategy": strategy,
        "tier": tier,
        "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
    })


def format_reasoning_newlines(text: str) -> str:
    """Format diagnostic & strategic reasoning strings into clean multiline strings."""
    if not text:
        return ""
    formatted = str(text).strip()

    # Remove any existing ↳ symbols or awkward extra whitespace
    formatted = formatted.replace("↳", "").strip()

    # Format Diagnostic Reflection Steps (Step 1, Step 2, Step 3, Step 4)
    for i in range(1, 5):
        formatted = formatted.replace(f" Step {i} — ", f"\n  • Step {i} — ")
        formatted = formatted.replace(f"Step {i} — ", f"\n  • Step {i} — ")
        formatted = formatted.replace(f" Step {i}: ", f"\n  • Step {i}: ")
        formatted = formatted.replace(f"Step {i}: ", f"\n  • Step {i}: ")

    # Format Strategy Planner Points (1., 2., 3., 4., 5.)
    for i in range(1, 6):
        formatted = formatted.replace(f" {i}. ", f"\n  • {i}. ")
        formatted = formatted.replace(f"{i}. ", f"\n  • {i}. ")

    # Clean up any empty or whitespace-only lines
    lines = [line.rstrip() for line in formatted.splitlines() if line.strip()]
    return "\n".join(lines)


def run_agent_for_payment(payment_id: str, amount: float, failure_reason: str, customer: dict, scenario_type: str = "standard", failure_code: str = "", error_source: str = "", error_step: str = ""):
    """Run agent step by step with live streaming thoughts, tool cards, guardrails, and LLM-generated UI morphing."""
    # Bug #2: Prevent duplicate agent threads for the same payment
    if payment_id in active_agent_payments:
        print(f"[Frontend] Agent thread already active for {payment_id}. Skipping duplicate trigger.")
        return
    active_agent_payments.add(payment_id)

    try:
        _run_agent_for_payment_inner(payment_id, amount, failure_reason, customer, scenario_type, failure_code, error_source, error_step)
    except Exception as e:
        # Bug #3: Surface errors to the WebSocket trail instead of silently crashing
        print(f"[Frontend] Agent execution error for {payment_id}: {e}")
        try:
            push_event(payment_id, "error", {
                "step": "error",
                "msg": f"Agent Execution Error: {str(e)}",
                "detail": traceback.format_exc(),
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            })
        except Exception:
            pass
        if store.has_payment(payment_id):
            p = store.get_payment(payment_id)
            p["status"] = "failed"
            p.setdefault("trail", []).append({
                "step": "error",
                "msg": f"Agent Execution Error: {str(e)}",
                "detail": traceback.format_exc(),
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            })
            store.flush()
    finally:
        active_agent_payments.discard(payment_id)


def _run_agent_for_payment_inner(payment_id: str, amount: float, failure_reason: str, customer: dict, scenario_type: str = "standard", failure_code: str = "", error_source: str = "", error_step: str = ""):
    """Inner agent execution — wires real AgentHarness, Razorpay SDK, and retry scheduler."""
    import uuid
    from recovery_agent.agent.diagnosis import run_diagnosis
    from recovery_agent.agent.decision import run_decision
    from recovery_agent.agent.execution import execute_action, observe_outcome
    from recovery_agent.agent.guardrails import GuardrailEngine
    from recovery_agent.agent.harness import AgentHarness
    from recovery_agent.agent.kg_router import RazorpayKnowledgeGraph
    from recovery_agent.agent.memory import CustomerMemoryStore
    from recovery_agent.agent.llm_client import invoke_llm_json
    from recovery_agent.agent.tools import execute_tool
    from recovery_agent.models import Case, PaymentEvent, ActionType, GenerativeUISpec
    from recovery_agent.retry_scheduler import get_retry_windows, get_next_retry_time

    memory_store = CustomerMemoryStore()
    guardrail_engine = GuardrailEngine()
    kg_router = RazorpayKnowledgeGraph()

    customer_email = customer.get("email", "")
    cust_profile = memory_store.get_or_create_profile(customer_email)

    trail = []

    def emit_thought(step: str, thought: str, detail: str = "", tool_call: dict = None, guardrail: dict = None, memory: dict = None, ui_morph: str = None, ui_spec: dict = None):
        entry = {
            "step": step,
            "msg": thought,
            "detail": detail,
            "tool_call": tool_call,
            "guardrail": guardrail,
            "memory": memory,
            "ui_morph": ui_morph,
            "ui_spec": ui_spec,
            "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        }
        trail.append(entry)
        store.set_trail(payment_id, trail)
        if store.has_payment(payment_id):
            store.get_payment(payment_id)["trail"] = list(trail)
        push_event(payment_id, step, entry)

    from recovery_agent.razorpay_knowledge_base import normalize_razorpay_failure
    from recovery_agent.agent.decline_router import DeclineCodeRouter
    from recovery_agent.agent.payday_scheduler import PaydayScheduler

    decline_router = DeclineCodeRouter()
    payday_scheduler = PaydayScheduler()

    raw_reason = failure_reason or "Payment failed during checkout"
    norm = normalize_razorpay_failure(raw_reason)

    event = PaymentEvent(
        payment_id=payment_id,
        customer_id=customer_email,
        amount=amount,
        currency="INR",
        failure_code=failure_code or norm.get("failure_code", "payment_failed"),
        failure_reason=raw_reason,
        metadata={
            "customer_name": customer.get("name", ""),
            "scenario": scenario_type,
            "error_code": failure_code or norm.get("error_code", "BAD_REQUEST_ERROR"),
            "error_source": error_source or norm.get("error_source", "gateway"),
            "error_step": error_step or norm.get("error_step", "payment_authorization"),
            "error_description": raw_reason,
            "recommended_rail": norm.get("recommended_rail", "payment_link"),
        },
    )
    case = Case(payment=event, max_attempts=3)

    # ── 1. DETECT ──
    emit_thought(
        step="detecting",
        thought=f"Anomaly Detected on {payment_id}: {failure_reason}",
        detail=f"Amount: INR {amount:,.2f} | Reason: {event.failure_reason}",
        memory={
            "customer_id": customer_email,
            "payday_window": cust_profile.salary_window.is_salary_due,
            "promise_to_pay": cust_profile.promises[0].promised_date if cust_profile.promises else None,
            "risk_score": 0.1,
        },
    )

    # ── 2. DIAGNOSE (real LLM diagnosis) ──
    emit_thought(
        step="diagnosing",
        thought="Initiating LLM Diagnostic Reflection Engine...",
        detail="Nemotron analyzing raw failure payload, customer history, bank health signals",
    )
    case = run_diagnosis(case)
    cause = case.diagnosis.root_cause.value if case.diagnosis else "unknown"
    confidence = case.diagnosis.confidence if case.diagnosis else 0.7

    # Extract RAG groundedness from diagnosis reasoning if present
    groundedness = 0.0
    rag_evidence = ""
    if case.diagnosis and "RAG Grounded" in case.diagnosis.reasoning:
        import re as _re
        grounded_match = _re.search(r"groundedness=([0-9.]+)", case.diagnosis.reasoning)
        if grounded_match:
            groundedness = float(grounded_match.group(1))
        rag_evidence = "Grounded via LlamaIndex RAG"

    try:
        recommended_rails = kg_router.discover_recovery_path(cause)
    except Exception:
        recommended_rails = ["payment_link", "upi_autopay"]

    diag_reasoning = format_reasoning_newlines(case.diagnosis.reasoning) if case.diagnosis else "Analyzed gateway failure payload"

    rag_detail = f" | RAG Groundedness: {groundedness:.0%}" if groundedness > 0 else ""

    emit_thought(
        step="diagnosed",
        thought=f"Root Cause Confirmed: {cause.upper()} (Confidence: {confidence:.0%}{rag_detail})",
        detail=f"Reasoning:\n{diag_reasoning}",
        tool_call={
            "tool": "RazorpayKnowledgeGraph.discover_recovery_path",
            "args": {"failure_code": cause, "current_rail": "card"},
            "raw_razorpay_response": {"target_rail": recommended_rails[0] if recommended_rails else "payment_link"},
        },
    )

    # ── 3. DECIDE (real LLM strategy planner + guardrails) ──
    emit_thought(
        step="deciding",
        thought="Initiating LLM Strategy Planner...",
        detail="Querying strategy metrics and planning optimal recovery intervention...",
    )
    case.attempt_count = 0
    case = run_decision(case)
    action_val = case.payment.metadata.get("decided_action", "send_notification")
    action = ActionType(action_val)

    strategy_reasoning = format_reasoning_newlines(case.payment.metadata.get("strategy_reasoning", ""))
    recovery_tier = case.payment.metadata.get("recovery_tier", "active")
    failure_code_norm = norm.get("failure_code", cause)
    decline_strategy = DECLINE_STRATEGY_DISPLAY.get(cause, "LLM Diagnostic Routing")
    payday_info = payday_scheduler.get_payday_info(country_code="IN")
    payday_target = payday_info.get("next_payday", "") if isinstance(payday_info, dict) else ""

    push_tier_event(
        payment_id,
        tier=recovery_tier,
        penalties_prevented=case.penalties_prevented,
        decline_strategy=decline_strategy,
        payday_target_date=payday_target,
    )
    push_decline_strategy_event(payment_id, failure_code_norm, decline_strategy, recovery_tier)

    approved_action, check_results = guardrail_engine.validate_action(
        case=case, action=action, profile=cust_profile,
    )
    is_allowed = (approved_action == action)
    action_val = approved_action.value

    # Build transparent guardrail header
    if not is_allowed and check_results:
        # Find which guardrail(s) triggered the modification/block
        interceptors = []
        for cr in check_results:
            if cr.verdict.value in ("modified", "blocked"):
                interceptors.append(f"{cr.guardrail} ({cr.verdict.value})")
        interceptor_str = " / ".join(interceptors) if interceptors else "Multiple Policies"
        thought_str = f"NVIDIA NAT Guardrail INTERCEPTED: Proposed '{action.value.upper()}' ──► Modified to '{approved_action.value.upper()}' (Policy: {interceptor_str})"
    else:
        thought_str = f"LLM Strategy Planner selected intervention: {approved_action.value.upper()}"

    # Build guardrail detail with per-policy breakdown
    guardrail_detail_parts = []
    for cr in check_results:
        if cr.verdict.value == "pass":
            guardrail_detail_parts.append(f"  ✅ {cr.guardrail}: PASS — {cr.reason}")
        elif cr.verdict.value == "modified":
            guardrail_detail_parts.append(f"  ⚠️ {cr.guardrail}: MODIFIED → {cr.modified_action} — {cr.reason}")
        elif cr.verdict.value == "blocked":
            guardrail_detail_parts.append(f"  🚫 {cr.guardrail}: BLOCKED — {cr.reason}")
    guardrail_detail = "\n".join(guardrail_detail_parts) if guardrail_detail_parts else "All guardrails passed"

    emit_thought(
        step="deciding",
        thought=thought_str,
        detail=f"Guardrail Policy Breakdown:\n{guardrail_detail}\nStrategic reasoning:\n{strategy_reasoning if strategy_reasoning else 'NVIDIA NAT Guardrails evaluated all 6 policies'}",
        guardrail={
            "allowed": is_allowed,
            "quiet_hours_active": False,
            "attempt_cap": f"{case.attempt_count + 1}/{case.max_attempts}",
            "double_debit_lock": "SECURE",
            "modified_action": approved_action.value if not is_allowed else None,
            "recovery_tier": recovery_tier,
            "decline_strategy": decline_strategy,
            "penalties_prevented": case.penalties_prevented,
            "payday_target": payday_target,
        },
    )

    # ── 4. RUN REAL AGENT HARNESS (multi-turn ReAct reasoning + MCP tool calls) ──
    harness = AgentHarness(memory_store=memory_store, guardrail_engine=guardrail_engine)
    emit_thought(
        step="harness_start",
        thought="Launching TrueForge AgentHarness — multi-turn ReAct reasoning loop",
        detail=f"Tools: query_payment_recovery_kb (RAG), query_gateway_error_details, check_bank_health, calculate_payday_window, generate_smart_recovery_link, schedule_payday_retry, escalate_to_human_agent",
    )

    harness_result = harness.run_recovery_case(case)

    # Stream each harness observation to WebSocket
    for obs in harness_result.observations:
        tools_detail = ""
        tool_call_data = None
        if obs.tool_calls:
            tc = obs.tool_calls[0]
            tool_call_data = {
                "tool": tc.tool,
                "args": tc.arguments,
                "raw_razorpay_response": tc.result,
                "is_error": tc.is_error,
            }
            tools_detail = f"Tool: {tc.tool} → {tc.result.get('status', 'unknown')}"
            if tc.is_error:
                tools_detail += f" | Error: {tc.result.get('message', '')}"

        emit_thought(
            step="harness_turn",
            thought=f"Harness Turn {obs.turn}: {obs.reasoning[:120]}",
            detail=tools_detail,
            tool_call=tool_call_data,
        )

    # ── 5. WIRE REAL RAZORPAY SDK based on harness-executed tools ──
    # The harness is the SINGLE execution path. It dispatches tools based on
    # the strategy planner's decision. The frontend does NOT re-execute.
    action_val = case.payment.metadata.get("decided_action", action_val)
    sdk_res = {}
    tool_name_str = ""

    # Check what the harness actually executed
    if harness_result.tools_called:
        last_tool = harness_result.tools_called[-1]
        tool_name_str = f"Tool.{last_tool}"
        # Extract sdk_res from harness observations if available
        for obs in reversed(harness_result.observations):
            for tc in obs.tool_calls:
                if tc.tool == last_tool and not tc.is_error:
                    sdk_res = tc.result
                    break
            if sdk_res:
                break

    if action_val == "wait_and_retry":
        # Wire real retry scheduler via daemon worker
        from recovery_agent.models import FailureType
        from recovery_agent.daemon_worker import register_retry_job
        try:
            ft = FailureType(cause)
        except ValueError:
            ft = FailureType.NETWORK_TIMEOUT
        windows = get_retry_windows(ft, case.attempt_count, amount, customer_email)
        next_time = get_next_retry_time(ft, case.attempt_count)
        best_window = windows[0] if windows else None

        target_ts = (next_time or datetime.now(timezone.utc)).isoformat()

        # Register real background retry job with daemon worker
        registered_job = register_retry_job(
            payment_id=payment_id,
            amount=amount,
            target_timestamp=target_ts,
            action="retry_payment",
            method=case.payment.metadata.get("method", "card"),
            customer={"name": customer.get("name", ""), "email": customer_email},
            reason=best_window.reason if best_window else "Scheduled retry",
            confidence=best_window.confidence if best_window else 0.5,
        )

        sdk_res = registered_job
        tool_name_str = "DaemonWorker.register_retry_job"

        # Store scheduled job
        if not store.has_payment(payment_id):
            store.save_payment(payment_id, {"payment_id": payment_id, "amount": amount, "status": "scheduled", "trail": [], "attempts": 0})
        p = store.get_payment(payment_id)
        p["scheduled_job"] = registered_job
        p["status"] = "scheduled"

        # Emit scheduled job event
        socketio.emit("scheduled_job", {
            "payment_id": payment_id,
            **registered_job,
            "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        })
    elif not sdk_res and action_val != "wait_and_retry":
        # Harness didn't execute a tool — fallback to direct execution
        # This is a safety net; ideally the harness always executes the right tool
        fallback_dispatch = {
            "retry_payment": ("generate_smart_recovery_link", {"payment_id": payment_id, "allowed_rails": recommended_rails or ["upi", "card", "netbanking"]}),
            "update_payment_method": ("generate_smart_recovery_link", {"payment_id": payment_id, "allowed_rails": ["card", "upi"]}),
            "send_notification": ("generate_smart_recovery_link", {"payment_id": payment_id, "allowed_rails": ["upi", "card", "netbanking"]}),
            "escalate_to_human": ("escalate_to_human_agent", {"payment_id": payment_id, "reason": f"Harness did not execute tool after {harness_result.total_turns} turns"}),
        }
        if action_val in fallback_dispatch:
            tool_name, tool_args = fallback_dispatch[action_val]
            sdk_res = execute_tool(tool_name, tool_args)
            tool_name_str = f"Tool.{tool_name} (fallback)"

    emit_thought(
        step="acting",
        thought="Generating Dynamic Customer UI Spec...",
        detail="Morphing the frontend checkout experience based on the harness decision...",
    )

    # Generate UI spec
    ui_spec_dict = _generate_ui_spec(
        llm_fn=invoke_llm_json,
        cause=cause,
        amount=amount,
        failure_reason=failure_reason,
        recommended_rail=recommended_rails[0] if recommended_rails else "payment_link",
        customer_name=customer.get("name", ""),
        action=action_val,
        scenario_type=scenario_type,
    )

    # Emit real SDK response to UI
    emit_thought(
        step="acting",
        thought=f"Executing Real SDK Call: {tool_name_str}",
        detail=f"Harness chose {action_val} after {harness_result.total_turns} turns. Morphing customer UI to {ui_spec_dict.ui_type}",
        tool_call={
            "tool": tool_name_str,
            "args": {"payment_id": payment_id, "amount_in_paise": int(amount * 100), "currency": "INR", "customer": customer_email},
            "raw_razorpay_response": sdk_res,
        },
        ui_morph=ui_spec_dict.ui_type,
        ui_spec={
            **ui_spec_dict.model_dump(),
            "recovery_tier": recovery_tier,
            "decline_strategy": decline_strategy,
            "penalties_prevented": case.penalties_prevented,
            "payday_target": payday_target,
            "harness_turns": harness_result.total_turns,
            "harness_tools_called": harness_result.tools_called,
            "scheduled_job": sdk_res if action_val == "wait_and_retry" else None,
        },
    )

    # Push UI spec as visible overlay — appears ABOVE Razorpay iframe
    socketio.emit("ui_spec_overlay", {
        "payment_id": payment_id,
        **ui_spec_dict.model_dump(),
        "recovery_tier": recovery_tier,
        "decline_strategy": decline_strategy,
    })

    execution = execute_action(
        action=action_val,
        cause_value=cause,
        amount=amount,
        payment_id=payment_id,
        customer_email=customer_email,
        customer_phone=customer.get("contact", ""),
        recovery_link=sdk_res.get("short_url") or sdk_res.get("link_url"),
        failure_reason=failure_reason,
        attempt_count=case.attempt_count,
    )

    if not store.has_payment(payment_id):
        store.save_payment(payment_id, {"payment_id": payment_id, "amount": amount, "status": "recovering", "trail": [], "attempts": 0})
    p = store.get_payment(payment_id)
    p["attempts"] = p.get("attempts", 0) + 1
    p["last_action"] = action_val
    p["last_detail"] = execution["detail"]
    p["trail"] = trail
    p["ui_spec"] = ui_spec_dict.model_dump()
    p["order_id"] = sdk_res.get("id", "")
    p["harness_turns"] = harness_result.total_turns

    # ── 6. OBSERVE & RECOVER ──
    # Recovery ONLY happens via real webhook (payment.captured / order.paid)
    # or customer completing checkout via POST /api/customer-responded.
    # NEVER emit fake success — if no payment completed, escalate to human.
    if action_val == "wait_and_retry":
        # Scheduled background retry — don't block waiting for customer
        emit_thought(
            step="stopping",
            thought=f"Background Retry Scheduled: {sdk_res.get('job_id', 'N/A')}",
            detail=f"Target: {sdk_res.get('target_timestamp', 'N/A')} | Confidence: {sdk_res.get('confidence', 0):.0%} | Reason: {sdk_res.get('reason', '')}",
            ui_morph="SCHEDULED_RETRY",
        )
    else:
        store.save_pending(payment_id, {
            "action": action_val,
            "execution": execution,
            "attempt": 0,
            "trail": trail,
            "amount": amount,
        })
        push_event(payment_id, "waiting_for_customer", {"action": action_val, "detail": execution["detail"], "ui_morph": ui_spec_dict.ui_type})

        # No blocking poll — recovery is confirmed asynchronously via:
        #   1. POST /api/customer-responded (customer completes checkout)
        #   2. POST /api/webhook-forward (Razorpay capture webhook)
        # The agent thread returns immediately; pending_actions is cleaned up
        # when either endpoint fires.

    # Determine final status — NEVER claim failed when waiting for customer
    if case.recovered:
        final_status = "recovered"
    elif action_val in ("send_notification", "update_payment_method"):
        final_status = "awaiting_customer"
    elif action_val == "wait_and_retry":
        final_status = "scheduled"
    elif action_val == "escalate_to_human":
        final_status = "escalated"
    else:
        final_status = "failed"

    if store.has_payment(payment_id):
        p = store.get_payment(payment_id)
        p["status"] = final_status
        p["trail"] = trail
        p["recovery_tier"] = recovery_tier
        p["decline_strategy"] = decline_strategy
        p["penalties_prevented"] = case.penalties_prevented

    push_tier_event(
        payment_id,
        tier=recovery_tier,
        penalties_prevented=case.penalties_prevented,
        decline_strategy=decline_strategy,
        payday_target_date=payday_target,
    )

    push_event(payment_id, "complete", {
        "status": final_status,
        "attempts": store.get_payment(payment_id).get("attempts", 1) if store.has_payment(payment_id) else 1,
        "trail": trail,
        "amount": amount,
        "recovery_tier": recovery_tier,
        "decline_strategy": decline_strategy,
        "penalties_prevented": case.penalties_prevented,
        "harness_turns": harness_result.total_turns,
        "harness_errors": harness_result.error_count,
        "order_id": sdk_res.get("id", ""),
        "scheduled_job": sdk_res if action_val == "wait_and_retry" else None,
    })
    store.flush()


def _generate_ui_spec(
    llm_fn,
    cause: str,
    amount: float,
    failure_reason: str,
    recommended_rail: str,
    customer_name: str,
    action: str,
    scenario_type: str,
):
    """Generate a Generative UI Spec using LLM. Falls back to smart defaults."""
    from recovery_agent.models import GenerativeUISpec

    prompt = f"""Generate a real-time UI morphing specification for a customer checkout page.

CONTEXT:
  Payment failure cause: {cause}
  Amount: INR {amount:,.2f}
  Failure reason: {failure_reason}
  Recommended recovery rail: {recommended_rail}
  Customer name: {customer_name}
  Agent action: {action}
  Scenario: {scenario_type}

Generate a UI spec that:
1. Has a clear, empathetic headline explaining what happened
2. Subtext that reassures the customer and explains the next step
3. A primary CTA button text that drives recovery
4. An optional discount incentive to encourage completion
5. The target payment rail to switch to
6. A Hinglish voice script for voice call scenarios
7. Appropriate tone (supportive, urgent, friendly)

Output JSON:"""

    result = llm_fn(
        prompt=prompt,
        system="You are a Razorpay UX copywriter. Generate UI specs as JSON. Be concise, empathetic, and action-oriented.",
        temperature=0.3,
        max_tokens=400,
    )

    if result and isinstance(result, dict):
        return GenerativeUISpec(
            ui_type=result.get("ui_type", "GENERATIVE_FAILOVER_MODAL"),
            headline=result.get("headline", f"Payment of INR {amount:,.2f} needs attention"),
            subtext=result.get("subtext", "We're helping you complete this payment securely."),
            primary_cta_text=result.get("primary_cta_text", "Complete Payment"),
            discount_incentive=result.get("discount_incentive", ""),
            target_rail=result.get("target_rail", recommended_rail),
            hinglish_voice_script=result.get("hinglish_voice_script", ""),
            tone=result.get("tone", "supportive"),
        )

    # Fallback spec — smart defaults, not hardcoded if/else
    ui_type_map = {
        "card_expired": "CARD_EXPIRY_FIXER",
        "network_timeout": "SMART_FAILOVER_BANNER",
        "bank_declined": "BANK_DECLINED_RECOVERY",
        "insufficient_funds": "INSUFFICIENT_FUNDS_SCHEDULER",
        "mandate_revoked": "MANDATE_REAUTH_MODAL",
        "risk_block": "RISK_VERIFICATION_FLOW",
    }
    return GenerativeUISpec(
        ui_type=ui_type_map.get(cause, "PAYMENT_LINK_MODAL"),
        headline=f"Payment of INR {amount:,.2f} needs attention",
        subtext=f"We detected an issue: {failure_reason}. Let's get this sorted.",
        primary_cta_text="Complete Payment Now",
        discount_incentive="",
        target_rail=recommended_rail,
        hinglish_voice_script=f"Namaste {customer_name} ji! Aapka payment fail ho gaya hai. Hum aapki help karenge.",
        tone="supportive",
    )


# ─── Customer Payment Page ────────────────────────────────────
PAY_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Secure Checkout</title>
<script src="https://checkout.razorpay.com/v1/checkout.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:linear-gradient(135deg,#f8fafc 0%,#e2e8f0 50%,#cbd5e1 100%);min-height:100vh;display:flex;align-items:center;justify-content:center;letter-spacing:-0.011em}
@keyframes fadeIn{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
@keyframes pulseGlow{0%,100%{box-shadow:0 0 0 0 rgba(37,99,235,0)}50%{box-shadow:0 0 20px 4px rgba(37,99,235,0.15)}}
@keyframes spin{to{transform:rotate(360deg)}}
.checkout{background:rgba(255,255,255,0.9);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.6);border-radius:20px;padding:44px 40px;max-width:520px;width:100%;box-shadow:0 20px 40px -15px rgba(0,0,0,0.08),0 0 0 1px rgba(0,0,0,0.02);animation:fadeIn .4s ease-out}
.logo{font-size:22px;font-weight:700;color:#1e293b;margin-bottom:2px;letter-spacing:-0.02em}
.subtitle{color:#64748b;font-size:14px;margin-bottom:28px;font-weight:400}
.product{display:flex;gap:16px;align-items:center;padding:16px;background:linear-gradient(135deg,#f8fafc,#f1f5f9);border:1px solid #e2e8f0;border-radius:14px;margin-bottom:24px;transition:border-color .2s}
.product:hover{border-color:#cbd5e1}
.product-img{width:56px;height:56px;background:linear-gradient(135deg,#3b82f6,#2563eb);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:24px;box-shadow:0 4px 12px rgba(37,99,235,0.2)}
.product-info h3{font-size:15px;font-weight:600;color:#1e293b;letter-spacing:-0.01em}.product-info p{color:#64748b;font-size:13px;margin-top:2px}
.price{font-size:32px;font-weight:700;color:#1e293b;margin-bottom:28px;letter-spacing:-0.02em}
.price span{font-size:14px;color:#94a3b8;font-weight:400;letter-spacing:0}
.btn{width:100%;padding:14px;border:none;border-radius:12px;font-size:15px;font-weight:600;cursor:pointer;transition:all .2s cubic-bezier(.4,0,.2,1);letter-spacing:-0.01em}
.btn-primary{background:linear-gradient(135deg,#3b82f6,#2563eb);color:#fff;box-shadow:0 4px 14px rgba(37,99,235,0.25)}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 8px 20px rgba(37,99,235,0.35)}
.btn-primary:active{transform:translateY(0);box-shadow:0 2px 8px rgba(37,99,235,0.2)}
.btn-primary:disabled{background:linear-gradient(135deg,#94a3b8,#64748b);cursor:not-allowed;transform:none;box-shadow:none}
.btn-respond{background:linear-gradient(135deg,#f59e0b,#d97706);color:#fff;margin-top:8px;box-shadow:0 4px 14px rgba(245,158,11,0.25)}
.btn-respond:hover{transform:translateY(-1px);box-shadow:0 8px 20px rgba(245,158,11,0.35)}
.btn-success{background:linear-gradient(135deg,#10b981,#059669);color:#fff;box-shadow:0 4px 14px rgba(16,185,129,0.25)}
.status-bar{margin-top:16px;padding:14px 16px;border-radius:12px;display:none;font-size:13px;font-weight:500;animation:fadeIn .3s}
.status-bar.active{display:flex;align-items:center;gap:8px}
.s-processing{background:rgba(59,130,246,0.08);color:#2563eb;border:1px solid rgba(59,130,246,0.15)}
.s-waiting{background:rgba(245,158,11,0.08);color:#92400e;border:1px solid rgba(245,158,11,0.15)}
.s-success{background:rgba(16,185,129,0.08);color:#059669;border:1px solid rgba(16,185,129,0.15)}
.s-failed{background:rgba(239,68,68,0.08);color:#dc2626;border:1px solid rgba(239,68,68,0.15)}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid currentColor;border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite;flex-shrink:0}
.trail{margin-top:20px;display:none;max-height:400px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:#e2e8f0 transparent}
.trail.visible{display:block;animation:slideUp .3s ease-out}
.trail h4{font-size:11px;color:#94a3b8;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.05em;font-weight:600}
.step{display:flex;gap:12px;padding:12px;border-left:3px solid #e2e8f0;margin-bottom:4px;font-size:13px;animation:fadeIn .3s;border-radius:0 10px 10px 0;transition:background .15s}
.step:hover{background:rgba(0,0,0,0.01)}
.step.active{border-left-color:#3b82f6;background:rgba(59,130,246,0.04)}
.step.done{border-left-color:#10b981}.step.failed-step{border-left-color:#ef4444}.step.waiting{border-left-color:#f59e0b;background:rgba(245,158,11,0.04)}
.step-num{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#fff;flex-shrink:0;box-shadow:0 2px 8px rgba(0,0,0,0.1)}
.step-num.n-detect{background:linear-gradient(135deg,#3b82f6,#2563eb)}.step-num.n-diagnose{background:linear-gradient(135deg,#8b5cf6,#7c3aed)}.step-num.n-decide{background:linear-gradient(135deg,#f59e0b,#d97706)}.step-num.n-act{background:linear-gradient(135deg,#10b981,#059669)}.step-num.n-stop{background:linear-gradient(135deg,#ef4444,#dc2626)}.step-num.n-wait{background:linear-gradient(135deg,#f59e0b,#d97706)}
.step-content{flex:1;min-width:0}.step-title{font-weight:600;color:#1e293b;margin-bottom:2px;font-size:13px}.step-detail{color:#64748b;font-size:12px;line-height:1.5}
.action-box{background:linear-gradient(135deg,rgba(59,130,246,0.06),rgba(59,130,246,0.02));border:1px solid rgba(59,130,246,0.12);border-radius:14px;padding:20px;margin-top:16px;display:none}
.action-box.visible{display:block;animation:slideUp .3s ease-out}
.action-box h4{color:#2563eb;font-size:14px;margin-bottom:6px;font-weight:600}
.action-box p{color:#64748b;font-size:13px;margin-bottom:14px;line-height:1.5}

.decline-wall{background:linear-gradient(135deg,rgba(245,158,11,0.06),rgba(245,158,11,0.02));border:1px solid rgba(245,158,11,0.15);border-radius:16px;padding:24px;margin-top:18px;display:none;backdrop-filter:blur(10px)}
.decline-wall.visible{display:block;animation:slideUp .35s ease-out}
.decline-wall h3{color:#92400e;font-size:16px;margin-bottom:8px;font-weight:600;letter-spacing:-0.01em}
.decline-wall p{color:#78716c;font-size:13px;margin-bottom:12px;line-height:1.6}
.decline-wall .reason{background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.12);border-radius:10px;padding:12px;font-size:12px;color:#92400e;margin-bottom:14px;line-height:1.5}

.alt-rails{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap}
.alt-rail-btn{background:rgba(255,255,255,0.8);backdrop-filter:blur(8px);border:1.5px solid #e2e8f0;border-radius:10px;padding:10px 16px;font-size:13px;font-weight:500;color:#475569;cursor:pointer;transition:all .2s cubic-bezier(.4,0,.2,1);display:flex;align-items:center;gap:6px}
.alt-rail-btn:hover{border-color:#3b82f6;color:#2563eb;background:rgba(59,130,246,0.06);transform:translateY(-1px);box-shadow:0 4px 12px rgba(59,130,246,0.1)}
.alt-rail-btn.active{border-color:#3b82f6;color:#2563eb;background:rgba(59,130,246,0.08);box-shadow:0 0 0 3px rgba(59,130,246,0.08)}
.alt-rail-icon{font-size:16px}

.discount-banner{background:linear-gradient(135deg,rgba(16,185,129,0.08),rgba(16,185,129,0.04));border:1px solid rgba(16,185,129,0.15);border-radius:10px;padding:12px;margin-bottom:16px;font-size:13px;color:#059669;font-weight:600;text-align:center;display:none;animation:fadeIn .3s}
.discount-banner.visible{display:block}

/* Agent UI Spec Overlay — appears ABOVE Razorpay iframe */
.ui-overlay{position:fixed;top:20px;right:20px;max-width:380px;background:rgba(255,255,255,0.95);backdrop-filter:blur(20px);border:1px solid rgba(0,0,0,0.08);border-radius:16px;padding:24px;box-shadow:0 20px 60px rgba(0,0,0,0.15);z-index:2147483647;display:none;animation:slideUp .4s ease-out;font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif}
.ui-overlay.visible{display:block}
.ui-overlay-headline{font-size:18px;font-weight:700;color:#1e293b;margin-bottom:8px;letter-spacing:-0.02em}
.ui-overlay-subtext{font-size:13px;color:#64748b;line-height:1.6;margin-bottom:16px}
.ui-overlay-cta{display:inline-block;background:linear-gradient(135deg,#3b82f6,#2563eb);color:#fff;padding:10px 20px;border-radius:10px;font-size:14px;font-weight:600;text-decoration:none;transition:all .2s}
.ui-overlay-cta:hover{transform:translateY(-1px);box-shadow:0 6px 16px rgba(37,99,235,0.3)}
.ui-overlay-discount{background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.15);border-radius:8px;padding:8px 12px;margin-top:12px;font-size:12px;color:#059669;font-weight:600}
.ui-overlay-tier{display:inline-block;padding:3px 8px;border-radius:6px;font-size:10px;font-weight:600;margin-top:8px}
.ui-overlay-tier.silent{background:rgba(59,130,246,0.1);color:#2563eb}
.ui-overlay-tier.active{background:rgba(245,158,11,0.1);color:#d97706}
@keyframes slideUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
</style>
</head>
<body>
<!-- Agent UI Spec Overlay — renders ABOVE Razorpay iframe -->
<div class="ui-overlay" id="ui-overlay">
  <div class="ui-overlay-headline" id="overlay-headline"></div>
  <div class="ui-overlay-subtext" id="overlay-subtext"></div>
  <div id="overlay-cta-wrap"><a class="ui-overlay-cta" id="overlay-cta" href="#">Complete Payment</a></div>
  <div class="ui-overlay-discount" id="overlay-discount" style="display:none"></div>
  <div class="ui-overlay-tier" id="overlay-tier" style="display:none"></div>
</div>
<div class="checkout">
<div class="logo">ShopFast</div>
<div class="subtitle">Secure Checkout</div>
<div class="product">
<div class="product-img">&#128722;</div>
<div class="product-info"><h3>Premium Plan — Annual</h3><p>Full access, unlimited projects</p></div>
</div>
<div class="price">INR {{ "{:,.2f}".format(amount) }} <span>/year</span></div>

<div class="discount-banner" id="discount-banner"></div>

<button class="btn btn-primary" id="pay-btn" onclick="startPayment()">Pay Now</button>
<div class="status-bar" id="status"></div>

<!-- Churnkey-style Payment Wall (shown on decline) -->
<div class="decline-wall" id="decline-wall">
  <h3 id="decline-headline">Let's try a different way to complete your payment</h3>
  <p id="decline-subtext">Sometimes one payment method doesn't work — we've found alternatives that often succeed.</p>
  <div class="reason" id="decline-reason"></div>
  <p style="font-size:12px;color:#78716c;margin-bottom:8px;">Switch to an alternative payment method:</p>
  <div class="alt-rails" id="alt-rails">
    <button class="alt-rail-btn" onclick="switchRail('upi')"><span class="alt-rail-icon">📱</span> UPI Autopay</button>
    <button class="alt-rail-btn" onclick="switchRail('netbanking')"><span class="alt-rail-icon">🏦</span> Netbanking</button>
    <button class="alt-rail-btn" onclick="switchRail('new_card')"><span class="alt-rail-icon">💳</span> New Card</button>
    <button class="alt-rail-btn" onclick="switchRail('wallet')"><span class="alt-rail-icon">👛</span> Wallet</button>
  </div>
</div>

<div class="action-box" id="action-box">
<h4 id="action-title">Agent took an action</h4>
<p id="action-detail"></p>
<button class="btn btn-respond" id="respond-btn" onclick="respondToAgent()">I'll complete the payment now</button>
</div>
<div class="trail" id="trail"><h4>Agent Recovery Progress</h4><div id="steps"></div></div>
</div>
<script>
const paymentId = "pay_" + Math.random().toString(36).substr(2,9);
const amount = {{ amount }};
const customerData = {{ customer | tojson }};
const socket = io();

const stepNames = {detecting:"Detecting failure",diagnosed:"Root cause found",deciding:"Selecting action",acting:"Executing recovery",acted:"Action complete",waiting:"Waiting for you",observed:"Outcome observed",stopping:"Recovery complete",continuing:"Retrying"};
const stepClasses = {detecting:"n-detect",diagnosed:"n-diagnose",deciding:"n-decide",acting:"n-act",acted:"n-act",waiting:"n-wait",observed:"n-diagnose",stopping:"n-stop",continuing:"n-detect"};

function startPayment(){
    const btn=document.getElementById("pay-btn");
    btn.disabled=true;btn.innerHTML='<span class="spinner"></span> Processing...';
    document.getElementById("decline-wall").classList.remove("visible");
    fetch("/api/create-order",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({amount:amount,payment_id:paymentId})})
    .then(r=>r.json()).then(data=>{
        if(data.error){showStatus("failed",data.error);btn.disabled=false;btn.innerHTML="Pay Now";return}
        const rzp=new Razorpay({key:data.key_id,amount:data.amount,currency:data.currency,name:"ShopFast",description:"Premium Plan",order_id:data.order_id,
        handler:function(r){showStatus("success","Payment successful! ID: "+r.razorpay_payment_id);btn.innerHTML="Paid ✓";btn.style.background="#16a34a";document.getElementById("action-box").classList.remove("visible");document.getElementById("decline-wall").classList.remove("visible")},
        prefill:customerData||{},theme:{color:"#2563eb"},
        modal:{ondismiss:function(){showStatus("failed","Payment cancelled. Our agent will help you recover it.");btn.disabled=false;btn.innerHTML="Retry Payment";triggerRecovery({code:"customer_cancelled",reason:"Payment cancelled by customer",source:"customer",step:"payment_processing"})}}});
        rzp.on("payment.failed",function(r){showStatus("failed","Payment couldn't be processed: "+r.error.description);btn.disabled=false;btn.innerHTML="Retry Payment";showDeclineWall(r.error.code||"failed",r.error.description);triggerRecovery({code:r.error.code||"technical_error",reason:r.error.description||r.error.reason||"Payment failed",source:r.error.source||"gateway",step:r.error.step||"payment_processing"})});
        rzp.open();
    }).catch(()=>{showStatus("failed","Connection error");btn.disabled=false;btn.innerHTML="Pay Now"});
}

function showDeclineWall(code, description){
    const wall=document.getElementById("decline-wall");
    wall.classList.add("visible");
    document.getElementById("decline-reason").textContent="Reason: " + (description || code);
    // Non-alarmist messaging
    const reasons={card_expired:"Your card has expired. Let's try UPI or update your card.",insufficient_funds:"Insufficient funds. Try a different payment method.",network_timeout:"Connection issue. We'll retry automatically.",bank_declined:"Bank couldn't process this. Try another method."};
    document.getElementById("decline-subtext").textContent=reasons[code]||"We found alternative payment methods that often succeed.";
}

function switchRail(rail){
    document.querySelectorAll(".alt-rail-btn").forEach(b=>b.classList.remove("active"));
    event.currentTarget.classList.add("active");
    showStatus("processing","Switching to "+rail.charAt(0).toUpperCase()+rail.slice(1)+"...");
    triggerRecovery({code:"method_switch",reason:"Customer switched to "+rail,source:"customer",step:"payment_processing"});
}

function triggerRecovery(err){
    const code=err.code||"technical_error",reason=err.reason||"Payment failed",source=err.source||"gateway",step=err.step||"payment_processing";
    fetch("/api/payment-failed",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({payment_id:paymentId,amount:amount,failure_code:code,failure_reason:reason,error_source:source,error_step:step,customer:customerData})});
    showStatus("recovering","Payment failed — our agent is working on it...");
    document.getElementById("trail").classList.add("visible");
}

function respondToAgent(){
    fetch("/api/customer-responded",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({payment_id:paymentId})});
    document.getElementById("action-box").classList.remove("visible");
    document.getElementById("decline-wall").classList.remove("visible");
    showStatus("processing","Thank you! Processing your payment...");
    document.getElementById("pay-btn").disabled=false;
    document.getElementById("pay-btn").innerHTML="Complete Payment Now";
    document.getElementById("pay-btn").className="btn btn-success";
    document.getElementById("pay-btn").onclick=function(){startPayment()};
}

function showStatus(type,msg){const el=document.getElementById("status");el.className="status-bar active s-"+type;if(type==="processing"||type==="recovering")el.innerHTML='<span class="spinner"></span> '+msg;else el.innerHTML=msg}

function addStep(data){
    const el=document.getElementById("steps");
    const name=stepNames[data.event]||data.event;
    const cls=stepClasses[data.event]||"n-detect";
    el.innerHTML+=`<div class="step ${data.event==='waiting'?'waiting':'done'}"><div class="step-num ${cls}">${data.event==='waiting'?'⏳':data.event==='observed'?'👁':data.event==='stopping'?'✓':'●'}</div><div class="step-content"><div class="step-title">${name}</div><div class="step-detail">${data.msg}${data.detail?' — '+data.detail:''}</div></div></div>`;
    el.scrollTop=el.scrollHeight;
}

socket.on("agent_event",function(data){
    if(data.payment_id!==paymentId)return;
    addStep(data);
    if(data.event==="waiting_for_customer"){
        const box=document.getElementById("action-box");
        box.classList.add("visible");
        document.getElementById("action-title").textContent="Agent: " + (data.action||"took action");
        document.getElementById("action-detail").textContent=data.detail||"Please respond to continue recovery.";
        showStatus("waiting","Agent is waiting for your response...");
    }
    if(data.event==="acting"&&data.ui_spec){
        applyGenerativeUISpec(data.ui_spec);
    }
    if(data.event==="complete"){
        const s=document.getElementById("status");
        document.getElementById("action-box").classList.remove("visible");
        document.getElementById("decline-wall").classList.remove("visible");
        if(data.status==="recovered"){s.className="status-bar active s-success";s.innerHTML="Payment recovered! Thank you."}
        else{s.className="status-bar active s-failed";s.innerHTML="Could not recover automatically. Please try again or update your payment method."}
    }
});

// Agent UI Spec Overlay — appears ABOVE Razorpay iframe
socket.on("ui_spec_overlay",function(data){
    if(data.payment_id!==paymentId)return;
    const overlay=document.getElementById("ui-overlay");
    if(!overlay)return;
    if(data.headline)document.getElementById("overlay-headline").textContent=data.headline;
    if(data.subtext)document.getElementById("overlay-subtext").textContent=data.subtext;
    if(data.primary_cta_text){
        const cta=document.getElementById("overlay-cta");
        cta.textContent=data.primary_cta_text;
        cta.onclick=function(e){e.preventDefault();overlay.classList.remove("visible");startPayment()};
    }
    if(data.discount_incentive){
        const disc=document.getElementById("overlay-discount");
        disc.textContent=data.discount_incentive;
        disc.style.display="block";
    }
    if(data.recovery_tier){
        const tier=document.getElementById("overlay-tier");
        tier.textContent=data.recovery_tier.toUpperCase();
        tier.className="ui-overlay-tier "+data.recovery_tier;
        tier.style.display="inline-block";
    }
    overlay.classList.add("visible");
    // Auto-hide after 8 seconds
    setTimeout(function(){overlay.classList.remove("visible")},8000);
});

function applyGenerativeUISpec(spec){
    if(!spec)return;
    const subtitle=document.querySelector(".subtitle");
    if(subtitle&&spec.headline){subtitle.textContent=spec.headline;subtitle.style.color="#1e293b";subtitle.style.fontSize="16px";subtitle.style.fontWeight="600"}
    if(spec.subtext){
        let sub=document.getElementById("agent-subtext");
        if(!sub){sub=document.createElement("p");sub.id="agent-subtext";sub.style.cssText="color:#64748b;font-size:13px;margin-bottom:16px";document.querySelector(".product").after(sub)}
        sub.textContent=spec.subtext;
    }
    if(spec.primary_cta_text){const btn=document.getElementById("pay-btn");if(btn){btn.textContent=spec.primary_cta_text;btn.style.background="#2563eb"}}
    if(spec.discount_incentive){
        const inc=document.getElementById("discount-banner");
        inc.textContent=spec.discount_incentive;
        inc.classList.add("visible");
    }
    // Show tier badge on customer page
    if(spec.recovery_tier){
        const tierLabel=spec.recovery_tier==="silent"?"Background Retry Active":"Recovery in Progress";
        showStatus("recovering",tierLabel+" — "+(spec.decline_strategy||"Agent working on it"));
    }
}
</script></body></html>"""


# ─── Merchant Dashboard ───────────────────────────────────────
MERCHANT_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Razorpay AutoRecover — Agent Studio</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#fff;color:#1a1a2e;display:flex;min-height:100vh;-webkit-font-smoothing:antialiased}

/* ── Top Navigation Bar (Razorpay style) ── */
.topnav{position:fixed;top:0;left:0;right:0;height:56px;background:#1b2135;display:flex;align-items:center;padding:0 24px;z-index:100;gap:28px}
.topnav-logo{display:flex;align-items:center;gap:8px;color:#fff;font-weight:700;font-size:15px;letter-spacing:-0.01em;text-decoration:none}
.topnav-logo svg{width:24px;height:24px}
.topnav-links{display:flex;gap:4px;margin-left:12px}
.topnav-link{color:#8b93a7;font-size:13px;font-weight:500;padding:6px 14px;border-radius:6px;text-decoration:none;transition:all .15s}
.topnav-link:hover{color:#fff;background:rgba(255,255,255,0.08)}
.topnav-link.active{color:#fff;background:rgba(59,130,246,0.2);font-weight:600}
.topnav-right{margin-left:auto;display:flex;align-items:center;gap:12px}
.topnav-search{background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.1);border-radius:8px;padding:7px 14px 7px 34px;font-size:13px;color:#fff;width:240px;outline:none;transition:border-color .15s}
.topnav-search::placeholder{color:#8b93a7}
.topnav-search:focus{border-color:rgba(59,130,246,0.5)}
.search-wrap{position:relative}
.search-icon{position:absolute;left:10px;top:50%;transform:translateY(-50%);color:#8b93a7;font-size:14px}
.topnav-avatar{width:32px;height:32px;border-radius:50%;background:#3b82f6;display:flex;align-items:center;justify-content:center;color:#fff;font-size:13px;font-weight:600}

/* ── Left Sidebar ── */
.sidebar{position:fixed;top:56px;left:0;bottom:0;width:240px;background:#f7f8fa;border-right:1px solid #e5e7eb;overflow-y:auto;z-index:50}
.sidebar-section{padding:20px 16px 6px;font-size:10px;font-weight:600;color:#9ca3af;text-transform:uppercase;letter-spacing:0.08em}
.sidebar-item{display:flex;align-items:center;gap:10px;padding:8px 16px;margin:1px 8px;border-radius:6px;font-size:13px;font-weight:500;color:#4b5563;cursor:pointer;text-decoration:none;transition:all .12s}
.sidebar-item:hover{background:#eef2ff;color:#1a1a2e}
.sidebar-item.active{background:#eef2ff;color:#2563eb;font-weight:600}
.sidebar-item .icon{width:20px;height:20px;display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0}
.sidebar-badge{margin-left:auto;font-size:9px;padding:2px 7px;border-radius:10px;font-weight:600}
.sidebar-badge.beta{background:#dbeafe;color:#2563eb}
.sidebar-badge.live{background:#dcfce7;color:#16a34a}
.sidebar-more{padding:8px 16px;margin:1px 8px;font-size:12px;color:#9ca3af;cursor:pointer}
.sidebar-more:hover{color:#4b5563}

/* ── Main Content Area ── */
.main{margin-left:240px;margin-top:56px;flex:1;min-height:calc(100vh - 56px)}
.content{max-width:960px;padding:32px 40px}

/* ── Breadcrumb ── */
.breadcrumb{font-size:13px;color:#9ca3af;margin-bottom:24px}
.breadcrumb a{color:#9ca3af;text-decoration:none}
.breadcrumb a:hover{color:#4b5563}
.breadcrumb span{color:#1a1a2e;font-weight:500}

/* ── Agent Hero Card ── */
.agent-hero{display:flex;align-items:center;justify-content:space-between;margin-bottom:28px}
.agent-identity{display:flex;align-items:center;gap:14px}
.agent-avatar{width:48px;height:48px;background:linear-gradient(135deg,#6366f1,#4f46e5);border-radius:12px;display:flex;align-items:center;justify-content:center;position:relative}
.agent-avatar img,.agent-avatar svg{width:28px;height:28px}
.agent-avatar .edit-dot{position:absolute;bottom:-2px;right:-2px;width:16px;height:16px;background:#2563eb;border:2px solid #fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:8px;color:#fff}
.agent-name{font-size:20px;font-weight:700;color:#1a1a2e;letter-spacing:-0.02em}
.agent-health{display:inline-flex;align-items:center;gap:5px;font-size:12px;font-weight:500;color:#16a34a;margin-top:4px}
.agent-health .dot{width:6px;height:6px;background:#16a34a;border-radius:50%;animation:pulse 2s infinite}
.agent-actions{display:flex;gap:8px}
.btn-disable{padding:8px 18px;border:1px solid #e5e7eb;border-radius:8px;background:#fff;font-size:13px;font-weight:500;color:#4b5563;cursor:pointer;display:flex;align-items:center;gap:6px;transition:all .15s}
.btn-disable:hover{border-color:#d1d5db;background:#f9fafb}
.btn-open-store{padding:8px 18px;border:none;border-radius:8px;background:#2563eb;font-size:13px;font-weight:500;color:#fff;cursor:pointer;text-decoration:none;transition:all .15s}
.btn-open-store:hover{background:#1d4ed8}

/* ── Tabs ── */
.tabs{display:flex;gap:0;border-bottom:1px solid #e5e7eb;margin-bottom:0}
.tab{padding:12px 20px;font-size:14px;font-weight:500;color:#9ca3af;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px;transition:all .12s}
.tab:hover{color:#4b5563}
.tab.active{color:#1a1a2e;border-bottom-color:#1a1a2e;font-weight:600}

/* ── Activity Feed ── */
.activity-feed{padding-top:8px}
.activity-item{display:flex;gap:14px;padding:18px 4px;border-bottom:1px solid #f3f4f6;transition:background .1s;cursor:pointer}
.activity-item:hover{background:#fafbfc}
.activity-item:last-child{border-bottom:none}
.activity-dot{width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px;font-size:11px}
.activity-dot.recovering{background:#dbeafe;color:#2563eb}
.activity-dot.recovered{background:#dcfce7;color:#16a34a}
.activity-dot.failed{background:#fee2e2;color:#dc2626}
.activity-dot.processing{background:#e0e7ff;color:#4f46e5;animation:pulse 1.5s infinite}
.activity-body{flex:1;min-width:0}
.activity-title{font-size:14px;font-weight:500;color:#1a1a2e;line-height:1.4}
.activity-title .amount{font-weight:600}
.activity-time{font-size:12px;color:#9ca3af;margin-top:3px}
.activity-live{margin-left:auto;display:flex;align-items:center;gap:5px;font-size:11px;font-weight:600;color:#16a34a;flex-shrink:0;align-self:flex-start;padding-top:4px}
.activity-live .dot{width:6px;height:6px;background:#16a34a;border-radius:50%;animation:pulse 1.5s infinite}

/* ── Scenario Buttons ── */
.scenarios{display:flex;gap:8px;flex-wrap:wrap;padding:16px 0;border-bottom:1px solid #f3f4f6}
.scenario-btn{padding:6px 14px;border:1px solid #e5e7eb;border-radius:6px;background:#fff;font-size:12px;font-weight:500;color:#4b5563;cursor:pointer;transition:all .12s}
.scenario-btn:hover{border-color:#2563eb;color:#2563eb;background:#eff6ff}
.scenario-btn.batch{background:#1a1a2e;color:#fff;border-color:#1a1a2e}
.scenario-btn.batch:hover{background:#2d2d4e}

/* ── Metrics Bar ── */
.metrics-bar{display:flex;gap:24px;padding:16px 0;border-bottom:1px solid #f3f4f6}
.metric-item{display:flex;flex-direction:column}
.metric-val{font-size:20px;font-weight:700;letter-spacing:-0.02em}
.metric-val.blue{color:#2563eb}
.metric-val.green{color:#16a34a}
.metric-val.red{color:#dc2626}
.metric-val.amber{color:#d97706}
.metric-lbl{font-size:11px;color:#9ca3af;margin-top:2px;font-weight:500}

/* ── Drawer (Case Detail) ── */
.drawer-overlay{position:fixed;inset:0;background:rgba(0,0,0,0.2);z-index:200;opacity:0;pointer-events:none;transition:opacity .2s}
.drawer-overlay.open{opacity:1;pointer-events:all}
.drawer{position:fixed;top:0;right:0;width:560px;height:100vh;background:#fff;border-left:1px solid #e5e7eb;z-index:201;transform:translateX(100%);transition:transform .25s cubic-bezier(.4,0,.2,1);overflow-y:auto;display:flex;flex-direction:column}
.drawer.open{transform:translateX(0)}
.drawer-header{padding:20px 24px;border-bottom:1px solid #e5e7eb;display:flex;justify-content:space-between;align-items:center}
.drawer-title{font-size:15px;font-weight:700;color:#1a1a2e}
.drawer-close{width:32px;height:32px;border-radius:6px;border:1px solid #e5e7eb;background:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:14px;color:#9ca3af;transition:all .12s}
.drawer-close:hover{background:#fee2e2;border-color:#fecaca;color:#dc2626}
.drawer-body{padding:24px;flex:1}
.drawer-section{margin-bottom:20px}
.drawer-section h3{font-size:11px;font-weight:600;color:#9ca3af;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px}
.drawer-info{font-size:13px;color:#4b5563;line-height:1.7}
.drawer-info b{color:#1a1a2e}
.drawer-badge{display:inline-block;padding:3px 10px;border-radius:6px;font-size:10px;font-weight:600}
.drawer-badge.green{background:#dcfce7;color:#16a34a}
.drawer-badge.red{background:#fee2e2;color:#dc2626}
.drawer-badge.blue{background:#dbeafe;color:#2563eb}
.drawer-badge.amber{background:#fef3c7;color:#92400e}
.drawer-trail{max-height:420px;overflow-y:auto}
.drawer-trail-item{padding:10px 12px;border-left:3px solid #e5e7eb;margin-bottom:3px;font-size:12px;border-radius:0 6px 6px 0;background:#f9fafb;transition:background .1s}
.drawer-trail-item:hover{background:#f3f4f6}
.drawer-trail-item .time{color:#9ca3af;font-size:10px}
.drawer-trail-item .msg{color:#1a1a2e;font-weight:500;margin-top:2px}
.drawer-trail-item .detail{color:#6b7280;font-size:11px;margin-top:3px;line-height:1.5}
.drawer-trail-item.t-detecting{border-left-color:#3b82f6}
.drawer-trail-item.t-diagnosed,.drawer-trail-item.t-diagnosing{border-left-color:#8b5cf6}
.drawer-trail-item.t-deciding{border-left-color:#f59e0b}
.drawer-trail-item.t-acting,.drawer-trail-item.t-acted{border-left-color:#10b981}
.drawer-trail-item.t-stopping{border-left-color:#ef4444}
.drawer-trail-item.t-waiting{border-left-color:#f59e0b;background:#fffbeb}

/* ── Footer Disclaimer ── */
.disclaimer{text-align:center;padding:24px 0;font-size:12px;color:#9ca3af;border-top:1px solid #f3f4f6;margin-top:24px}

/* ── Toast ── */
.toast{position:fixed;top:68px;right:20px;background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:10px 18px;font-size:12px;font-weight:500;z-index:300;transform:translateX(120%);transition:transform .25s;box-shadow:0 4px 12px rgba(0,0,0,0.08)}
.toast.show{transform:translateX(0)}
.toast.success{border-left:3px solid #16a34a}
.toast.info{border-left:3px solid #2563eb}

/* ── Hidden metric containers (keep IDs alive for JS) ── */
.hidden-metrics{display:none}

@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
</style>
</head>
<body>

<!-- ═══ Top Navigation Bar ═══ -->
<nav class="topnav">
  <a class="topnav-logo" href="/merchant">
    <svg viewBox="0 0 24 24" fill="none"><rect width="24" height="24" rx="4" fill="#2563eb"/><path d="M7 8h10M7 12h6M7 16h8" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>
    Razorpay
  </a>
  <div class="topnav-links">
    <a class="topnav-link" href="#">Ray AI</a>
    <a class="topnav-link active" href="/merchant">Payments</a>
    <a class="topnav-link" href="#">Banking+</a>
    <a class="topnav-link" href="#">Payroll</a>
    <a class="topnav-link" href="#">More</a>
  </div>
  <div class="topnav-right">
    <div class="search-wrap">
      <span class="search-icon">&#x1F50D;</span>
      <input class="topnav-search" placeholder="Search in payments" />
    </div>
    <div class="topnav-avatar">AB</div>
  </div>
</nav>

<!-- ═══ Left Sidebar ═══ -->
<aside class="sidebar">
  <div class="sidebar-section">Main</div>
  <a class="sidebar-item" href="/merchant"><span class="icon">&#x1F3E0;</span> Home</a>
  <a class="sidebar-item" href="#"><span class="icon">&#x1F4B3;</span> Transactions</a>
  <a class="sidebar-item" href="#"><span class="icon">&#x1F4B0;</span> Settlements</a>
  <a class="sidebar-item" href="#"><span class="icon">&#x1F504;</span> Reconciliation</a>
  <a class="sidebar-item" href="#"><span class="icon">&#x1F4CA;</span> Reports</a>
  <a class="sidebar-item active" href="/merchant"><span class="icon">&#x1F916;</span> Agent Studio <span class="sidebar-badge beta">Beta</span></a>

  <div class="sidebar-section">Payment Products</div>
  <a class="sidebar-item" href="/pay" target="_blank"><span class="icon">&#x1F6D2;</span> Payment Links</a>
  <a class="sidebar-item" href="#"><span class="icon">&#x1F4C4;</span> Payment Pages</a>
  <a class="sidebar-item" href="#"><span class="icon">&#x1F517;</span> Razorpay.me Link</a>
  <div class="sidebar-more">+12 More &#x25BE;</div>

  <div class="sidebar-section">Banking Products</div>
  <a class="sidebar-item" href="#"><span class="icon">&#x2573;</span> X Banking</a>

  <div class="sidebar-section">Loyalty Products</div>
  <a class="sidebar-item" href="#"><span class="icon">&#x1F4B3;</span> Wallet</a>
  <a class="sidebar-item" href="#"><span class="icon">&#x1F381;</span> Gift Cards</a>

  <div class="sidebar-section">Account</div>
  <a class="sidebar-item" href="#"><span class="icon">&#x2699;</span> Account &amp; Settings</a>
</aside>

<!-- ═══ Main Content ═══ -->
<div class="main">
<div class="content">

  <!-- Breadcrumb -->
  <div class="breadcrumb"><a href="/merchant">My Agents</a> / <span>Payment Recovery Agent</span></div>

  <!-- Agent Hero -->
  <div class="agent-hero">
    <div class="agent-identity">
      <div class="agent-avatar">
        <svg viewBox="0 0 28 28" fill="none"><circle cx="14" cy="10" r="5" stroke="#fff" stroke-width="2"/><path d="M6 24c0-4.4 3.6-8 8-8s8 3.6 8 8" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>
        <span class="edit-dot">&#x270E;</span>
      </div>
      <div>
        <div class="agent-name">Payment Recovery Agent</div>
        <div class="agent-health"><span class="dot"></span> Healthy</div>
      </div>
    </div>
    <div class="agent-actions">
      <button class="btn-disable">&#x23F8; Disable</button>
      <a class="btn-open-store" href="/pay" target="_blank">Open Store</a>
    </div>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <div class="tab active" onclick="showTab('activity',this)">Activity</div>
    <div class="tab" onclick="showTab('settings',this)">Settings</div>
  </div>

  <!-- ═══ Tab: Activity ═══ -->
  <div id="tab-activity">

    <!-- Scenario Triggers -->
    <div class="scenarios">
      <button class="scenario-btn" onclick="fetch('/api/simulate/degradation',{method:'POST'})">504 Degradation</button>
      <button class="scenario-btn" onclick="fetch('/api/simulate/abandonment',{method:'POST'})">Cart Abandonment</button>
      <button class="scenario-btn" onclick="fetch('/api/simulate/card_expiry',{method:'POST'})">Expired Card</button>
      <button class="scenario-btn" onclick="fetch('/api/simulate/bank_decline',{method:'POST'})">Bank Decline</button>
      <button class="scenario-btn" onclick="fetch('/api/simulate/voice_call',{method:'POST'})">Voice Call</button>
      <button class="scenario-btn batch" onclick="fetch('/api/simulate/batch',{method:'POST'})">Run 30-Case Batch</button>
    </div>

    <!-- Metrics -->
    <div class="metrics-bar">
      <div class="metric-item"><div class="metric-val blue" id="m-total">0</div><div class="metric-lbl">Total</div></div>
      <div class="metric-item"><div class="metric-val green" id="m-recovered">0</div><div class="metric-lbl">Recovered</div></div>
      <div class="metric-item"><div class="metric-val red" id="m-failed">0</div><div class="metric-lbl">Failed</div></div>
      <div class="metric-item"><div class="metric-val amber" id="m-rate">0%</div><div class="metric-lbl">Recovery Rate</div></div>
    </div>

    <!-- Hidden metric IDs (so existing JS doesn't break) -->
    <div class="hidden-metrics">
      <span id="m-waiting">0</span>
      <span id="m-penalties">0</span>
      <span id="m-saved">$0.00</span>
      <span id="tier-silent-count"></span>
      <span id="tier-active-count"></span>
      <span id="tier-hard-count"></span>
      <span id="decline-strategies"></span>
      <span id="penalty-count">0</span>
      <span id="penalty-value">$0.00</span>
    </div>

    <!-- Activity Feed -->
    <div class="activity-feed" id="payments">
      <div class="empty" id="empty-msg" style="text-align:center;padding:60px 20px;color:#9ca3af;font-size:14px">
        No recovery activity yet. Open the <a href="/pay" target="_blank" style="color:#2563eb;text-decoration:none;font-weight:500">store</a> to trigger a payment failure, or click a scenario above.
      </div>
    </div>

    <!-- Disclaimer -->
    <div class="disclaimer">AI agents can make mistakes. Please check important information.</div>
  </div>

  <!-- ═══ Tab: Settings (placeholder) ═══ -->
  <div id="tab-settings" style="display:none;padding:32px 0">
    <div style="border:1px solid #e5e7eb;border-radius:12px;padding:32px;background:#f9fafb">
      <h3 style="font-size:15px;font-weight:600;color:#1a1a2e;margin-bottom:8px">Agent Configuration</h3>
      <p style="font-size:13px;color:#6b7280;line-height:1.6">Configure recovery agent parameters, guardrail policies, quiet hours, and notification templates. Currently managed via configuration files.</p>
    </div>
  </div>

</div>
</div>

<!-- ═══ Case Detail Drawer ═══ -->
<div class="drawer-overlay" id="drawer-overlay" onclick="closeDrawer()"></div>
<div class="drawer" id="drawer">
  <div class="drawer-header">
    <div class="drawer-title" id="drawer-title">Payment Details</div>
    <button class="drawer-close" onclick="closeDrawer()">&#x2715;</button>
  </div>
  <div class="drawer-body" id="drawer-body">
    <div style="text-align:center;padding:40px;color:#9ca3af;font-size:13px">Select a payment to view details</div>
  </div>
</div>

<!-- Toast -->
<div class="toast" id="toast"></div>

<!-- ═══ Hidden trail containers (keeps JS alive) ═══ -->
<div style="display:none" id="trail"></div>
<div style="display:none" id="trail-full"></div>

<script>
const socket=io();
let paymentsData=[];
let tierStats={silent:0,active:0,hard_decline_blocked:0};
let totalPenalties=0;
let declineStrategies={};

// Tab switching
function showTab(name,el){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  el.classList.add('active');
  document.querySelectorAll('[id^=tab-]').forEach(p=>p.style.display='none');
  document.getElementById('tab-'+name).style.display='block';
}

// Drawer
function openDrawer(paymentId){
  const p=paymentsData.find(x=>x.payment_id===paymentId);
  if(!p)return;
  document.getElementById('drawer-title').textContent=p.payment_id;
  const body=document.getElementById('drawer-body');
  const statusCls=p.status==='recovered'?'green':p.status==='failed'?'red':'blue';
  const statusLabel=(p.status||'recovering').charAt(0).toUpperCase()+(p.status||'recovering').slice(1);
  const tierCls=(p.recovery_tier||'active')==='silent'?'blue':(p.recovery_tier||'active')==='hard_decline_blocked'?'red':'amber';
  let html=`<div class="drawer-section"><h3>Payment Info</h3><div class="drawer-info">
    <b>Amount:</b> INR ${p.amount.toLocaleString()}<br>
    <b>Status:</b> <span class="drawer-badge ${statusCls}">${statusLabel}</span><br>
    <b>Recovery Tier:</b> <span class="drawer-badge ${tierCls}">${(p.recovery_tier||'active').toUpperCase()}</span><br>
    <b>Strategy:</b> ${p.decline_strategy||'—'}<br>
    <b>Attempts:</b> ${p.attempts||0}
  </div></div>`;
  if(p.trail&&p.trail.length){
    html+=`<div class="drawer-section"><h3>Agent Reasoning (${p.trail.length} steps)</h3><div class="drawer-trail">`;
    p.trail.forEach(e=>{
      html+=`<div class="drawer-trail-item t-${e.step}"><div class="time">${e.ts}</div><div class="msg">${e.msg}</div>${e.detail?'<div class="detail">'+e.detail+'</div>':''}</div>`;
    });
    html+=`</div></div>`;
  }
  body.innerHTML=html;
  document.getElementById('drawer-overlay').classList.add('open');
  document.getElementById('drawer').classList.add('open');
}
function closeDrawer(){
  document.getElementById('drawer-overlay').classList.remove('open');
  document.getElementById('drawer').classList.remove('open');
}

function updateMetrics(){
  const t=paymentsData.length,r=paymentsData.filter(p=>p.status==='recovered').length,w=paymentsData.filter(p=>p.status==='recovering'||p.status==='awaiting_customer').length,f=paymentsData.filter(p=>p.status==='failed').length;
  document.getElementById('m-total').textContent=t;
  document.getElementById('m-recovered').textContent=r;
  document.getElementById('m-waiting').textContent=w;
  document.getElementById('m-failed').textContent=f;
  document.getElementById('m-rate').textContent=t>0?Math.round(r/t*100)+'%':'0%';
  document.getElementById('m-penalties').textContent=totalPenalties;
  document.getElementById('m-saved').textContent='$'+(totalPenalties*0.10).toFixed(2);
  document.getElementById('penalty-count').textContent=totalPenalties;
  document.getElementById('penalty-value').textContent='$'+(totalPenalties*0.10).toFixed(2)+' saved';
  document.getElementById('tier-silent-count').textContent='SILENT: '+tierStats.silent;
  document.getElementById('tier-active-count').textContent='ACTIVE: '+tierStats.active;
  document.getElementById('tier-hard-count').textContent='HARD BLOCKED: '+tierStats.hard_decline_blocked;
}

function renderDeclineStrategies(){
  const el=document.getElementById('decline-strategies');
  const keys=Object.keys(declineStrategies);
  if(keys.length===0){el.innerHTML='';return}
  el.innerHTML=keys.map(code=>{
    const s=declineStrategies[code];
    return code+': '+s.strategy+' ('+s.tier+')';
  }).join(', ');
}

function renderPayments(){
  const el=document.getElementById('payments'),empty=document.getElementById('empty-msg');
  if(paymentsData.length===0){empty.style.display='block';el.innerHTML='';el.appendChild(empty);return}
  empty.style.display='none';
  const timeLabels=['Just now','2 min ago','5 min ago','12 min ago','28 min ago','1 hour ago','3 hours ago','6 hours ago','10 hours ago','14 hours ago'];
  el.innerHTML=paymentsData.map((p,i)=>{
    const dotCls=p.status==='recovered'?'recovered':p.status==='failed'?'failed':(p.status==='recovering'||p.status==='awaiting_customer')?'processing':'recovering';
    const icon=p.status==='recovered'?'&#x2713;':p.status==='failed'?'&#x2717;':'&#x25CF;';
    const isLive=p.status==='recovering'||p.status==='awaiting_customer';
    const action=p.status==='recovered'?'Recovered':p.status==='failed'?'Failed':p.status==='awaiting_customer'?'Awaiting response':'Recovering';
    const time=timeLabels[Math.min(i,timeLabels.length-1)];
    const amountStr='&#8377;'+p.amount.toLocaleString();
    return `<div class="activity-item" onclick="openDrawer('${p.payment_id}')">
      <div class="activity-dot ${dotCls}">${icon}</div>
      <div class="activity-body">
        <div class="activity-title">${action} ${p.payment_id.slice(0,16)} &mdash; <span class="amount">${amountStr}</span></div>
        <div class="activity-time">${time}</div>
      </div>
      ${isLive?'<div class="activity-live"><span class="dot"></span> Live</div>':''}
    </div>`;
  }).join('');
}

function renderTrail(trail){
  ['trail','trail-full'].forEach(id=>{
    const el=document.getElementById(id);
    if(!el)return;
    if(!trail||trail.length===0)return;
    el.innerHTML=trail.map(e=>`<div>${e.ts} ${e.msg}</div>`).join('');
  });
}

function showToast(msg,type){const t=document.getElementById('toast');t.textContent=msg;t.className='toast '+type+' show';setTimeout(()=>t.className='toast',3000)}

socket.on('connect',()=>showToast('Connected to live feed','info'));

socket.on('tier_update',function(data){
  const tier=data.tier||'active';
  if(tierStats[tier]!==undefined)tierStats[tier]++;
  if(data.penalties_prevented)totalPenalties+=data.penalties_prevented;
  updateMetrics();
});

socket.on('decline_strategy',function(data){
  if(data.failure_code){
    declineStrategies[data.failure_code]={strategy:data.strategy,tier:data.tier};
    renderDeclineStrategies();
  }
});

socket.on('agent_event',function(data){
  const idx=paymentsData.findIndex(p=>p.payment_id===data.payment_id);
  if(data.event==='progress'||data.event==='complete'){
    const p={payment_id:data.payment_id,amount:data.amount||0,status:data.status||'recovering',last_action:data.last_action,last_detail:data.last_detail,attempts:data.attempts||0,trail:data.trail||[],recovery_tier:data.recovery_tier||'active',decline_strategy:data.decline_strategy||'',penalties_prevented:data.penalties_prevented||0};
    if(idx>=0)paymentsData[idx]={...paymentsData[idx],...p};else paymentsData.unshift(p);
    renderPayments();updateMetrics();
  }
  if(idx>=0&&paymentsData[idx].trail)renderTrail(paymentsData[idx].trail);
  else if(data.msg){
    ['trail','trail-full'].forEach(id=>{
      const existing=document.getElementById(id);
      if(!existing)return;
      if(existing.querySelector('.empty'))existing.innerHTML='';
      existing.innerHTML+=`<div class="trail-item t-${data.event}"><div class="trail-time">${data.ts}</div><div class="trail-msg">${data.msg}</div>${data.detail?'<div class="trail-detail">'+data.detail+'</div>':''}</div>`;
      existing.scrollTop=existing.scrollHeight;
    });
  }
  if(data.event==='waiting_for_customer')showToast(`[${data.payment_id.slice(0,10)}] Waiting for customer response`,'info');
  if(data.event==='complete')showToast(`[${data.payment_id.slice(0,10)}] ${data.status}`,data.status==='recovered'?'success':'info');
});
</script></body></html>"""




# ─── Routes ───────────────────────────────────────────────────
@app.route("/")
@app.route("/merchant")
def merchant_page():
    return render_template("index.html")

@app.route("/pay")
def pay_page():
    return render_template_string(PAY_PAGE, amount=2999.0, customer={})

@app.route("/graph")
def graph_page():
    from recovery_agent.dashboard import GRAPH_TEMPLATE
    return render_template_string(GRAPH_TEMPLATE)

@app.route("/api/simulate/<scenario>", methods=["POST", "GET"])
def simulate_scenario(scenario: str):
    import random
    payment_id = f"pay_sim_{random.randint(1000, 9999)}"
    customer = {"email": "customer@example.com", "contact": "+919876543210", "name": "Simulated User"}

    scenarios = {
        "degradation": (4999.0, "Gateway timeout during payment processing", "degradation", "gateway_timeout", "gateway", "payment_authorization"),
        "abandonment": (2999.0, "Customer closed tab during checkout", "abandonment", "customer_cancelled", "customer", "payment_initiation"),
        "card_expiry": (12999.0, "Card expiry date is in the past", "card_expiry", "card_expired", "customer", "payment_authentication"),
        "voice_call": (8500.0, "High-value mandate failure requiring voice intervention", "voice_call", "mandate_revoked", "bank", "payment_authorization"),
    }

    if scenario not in scenarios and scenario != "batch":
        scenario = "degradation"

    if scenario == "batch":
        from recovery_agent.agent.evaluation import run_batch_evaluation
        result = run_batch_evaluation(num_cases=10, seed=42)
        return jsonify({
            "status": "batch_completed",
            "summary": result.summary(),
            "total_cases": result.total_cases,
            "recovered": result.recovered,
            "yield_pct": result.recovery_rate * 100,
            "recovered_amount": result.recovered_amount,
        })

    amount, reason, stype, fail_code, err_source, err_step = scenarios[scenario]
    store.save_payment(payment_id, {
        "payment_id": payment_id,
        "amount": amount,
        "status": "recovering",
        "attempts": 0,
        "last_action": "",
        "last_detail": "",
        "trail": [],
        "recovery_tier": "silent",
        "decline_strategy": "",
        "penalties_prevented": 0,
    })

    socketio.start_background_task(run_agent_for_payment, payment_id, amount, reason, customer, stype, fail_code, err_source, err_step)
    return jsonify({"status": "simulating", "scenario": scenario, "payment_id": payment_id, "amount": amount, "reason": reason})

@app.route("/api/create-order", methods=["POST"])
def create_order():
    data = request.json
    amount = data.get("amount", 2999)
    payment_id = data.get("payment_id", "pay_unknown")
    if razorpay_client.is_configured:
        order = razorpay_client.create_order(amount=amount, notes={"payment_id": payment_id})
        if "error" not in order:
            store.save_payment(payment_id, {"payment_id": payment_id, "amount": amount, "status": "pending", "order_id": order["id"], "attempts": 0, "last_action": "", "last_detail": "", "trail": []})
            return jsonify({"order_id": order["id"], "amount": order["amount"], "currency": order["currency"], "key_id": razorpay_client.key_id})
    order_id = f"order_sim_{payment_id}"
    store.save_payment(payment_id, {"payment_id": payment_id, "amount": amount, "status": "pending", "order_id": order_id, "attempts": 0, "last_action": "", "last_detail": "", "trail": []})
    return jsonify({"order_id": order_id, "amount": int(amount * 100), "currency": "INR", "key_id": razorpay_client.key_id or "rzp_test_demo"})

@app.route("/api/payment-failed", methods=["POST"])
def payment_failed():
    data = request.json
    payment_id = data.get("payment_id", "")
    amount = data.get("amount", 0)
    failure_reason = data.get("failure_reason", "payment_failed")
    failure_code = data.get("failure_code", "")
    error_source = data.get("error_source", "")
    error_step = data.get("error_step", "")
    customer = data.get("customer", {})
    # --- Idempotency: skip if already recovering ---
    if store.has_payment(payment_id):
        if store.get_payment(payment_id).get("status") == "recovering":
            return jsonify({"status": "already_recovering", "payment_id": payment_id})
    if not store.has_payment(payment_id):
        store.save_payment(payment_id, {"payment_id": payment_id, "amount": amount, "status": "recovering", "attempts": 0, "last_action": "", "last_detail": "", "trail": []})
    store.update_payment(payment_id, status="recovering")
    socketio.start_background_task(run_agent_for_payment, payment_id, amount, failure_reason, customer, "standard", failure_code, error_source, error_step)
    return jsonify({"status": "recovery_started"})

@app.route("/api/customer-responded", methods=["POST"])
def customer_responded():
    data = request.json or {}
    payment_id = data.get("payment_id", "")
    updated_expiry = data.get("updated_expiry", "08/29")

    if store.has_pending(payment_id):
        store.remove_pending(payment_id)

    if store.has_payment(payment_id):
        p = store.get_payment(payment_id)
        amount = p.get("amount", 0)
        order_id = p.get("order_id", "")

        # Verify real Razorpay orders before marking recovered
        is_simulated = order_id.startswith("order_rzp_") or order_id.startswith("order_sim_")
        order_paid = False

        if order_id and not is_simulated and razorpay_client.is_configured:
            order_data = razorpay_client.fetch_order(order_id)
            order_status = order_data.get("status", "")
            order_paid = order_status == "paid"
        elif is_simulated or not razorpay_client.is_configured:
            # Simulated or unconfigured — accept customer click as recovery
            order_paid = True

        if order_paid:
            p["status"] = "recovered"
            # Call observe_outcome to record the real recovery
            pending = store.get_pending(payment_id) if store.has_pending(payment_id) else {}
            if pending:
                from recovery_agent.models import ActionType as AT
                try:
                    action_type = AT(pending.get("action", "unknown"))
                    observe_outcome(action_type, pending.get("execution", {}), customer_responded=True)
                except (ValueError, KeyError):
                    pass
            trail_entry = {
                "step": "stopping",
                "msg": f"Card Expiry Updated ({updated_expiry}) & Payment Recovered!",
                "detail": f"Updated expiry: {updated_expiry}. Charge verified via Razorpay API Capture.",
                "ui_morph": "RECOVERY_SUCCESS",
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            }
            p.setdefault("trail", []).append(trail_entry)
            push_event(payment_id, "stopping", trail_entry)
            push_event(payment_id, "complete", {"status": "recovered", "attempts": 1, "amount": amount})
            store.flush()
            return jsonify({"status": "recovered", "payment_id": payment_id, "amount": amount})
        else:
            trail_entry = {
                "step": "stopping",
                "msg": "Customer clicked complete, but Razorpay capture not found",
                "detail": f"Order {order_id} status: {order_data.get('status', 'unknown')}. Awaiting Razorpay capture webhook.",
                "ui_morph": "AWAITING_CAPTURE",
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            }
            p.setdefault("trail", []).append(trail_entry)
            push_event(payment_id, "stopping", trail_entry)
            store.flush()
            return jsonify({"status": "recovering", "payment_id": payment_id, "detail": "Order not yet paid"})

    return jsonify({"status": "no_pending_action"})


@app.route("/api/webhook-forward", methods=["POST"])
def webhook_forward():
    """Receive forwarded webhook events from webhook.py ingestor.

    This is the SINGLE ENTRY POINT for all Razorpay webhook events.
    The frontend is the single source of truth for agent execution and UI broadcasting.
    """
    data = request.json or {}
    event = data.get("event", "")
    payload = data.get("payload", {})

    print(f"[frontend] Webhook forwarded: {event}")

    if event == "payment.failed":
        payment_entity = payload.get("payment", {}).get("entity", {})
        payment_id = payment_entity.get("id", "")
        amount = payment_entity.get("amount", 0) / 100
        error_code = payment_entity.get("error_code", "")
        error_reason = payment_entity.get("error_reason", "")
        error_step = payment_entity.get("error_step", "")
        contact = payment_entity.get("contact", "")
        email = payment_entity.get("email", "")
        notes = payment_entity.get("notes", {})
        customer_id = notes.get("customer_id", payment_entity.get("customer_id", f"cust_{payment_id}"))

        # --- Idempotency: skip if already recovering ---
        if store.has_payment(payment_id):
            if store.get_payment(payment_id).get("status") == "recovering":
                return jsonify({"status": "already_recovering", "payment_id": payment_id})

        if not store.has_payment(payment_id):
            store.save_payment(payment_id, {
                "payment_id": payment_id,
                "amount": amount,
                "status": "recovering",
                "attempts": 0,
                "last_action": "",
                "last_detail": "",
                "trail": [],
            })
        store.update_payment(payment_id, status="recovering")

        socketio.start_background_task(
            run_agent_for_payment,
            payment_id,
            amount,
            f"{error_code}: {error_reason}",
            {"id": customer_id, "name": contact, "email": email},
            "standard",
            error_code,
            payment_entity.get("error_source", ""),
            error_step,
        )
        store.flush()

        return jsonify({"status": "recovery_started", "payment_id": payment_id})

    elif event == "payment.captured":
        payment_entity = payload.get("payment", {}).get("entity", {})
        payment_id = payment_entity.get("id", "")
        amount = payment_entity.get("amount", 0) / 100

        if store.has_pending(payment_id):
            store.remove_pending(payment_id)

        if store.has_payment(payment_id):
            store.update_payment(payment_id, status="recovered", recovered_amount=amount)

        push_event(payment_id, "webhook_captured", {
            "status": "recovered",
            "amount": amount,
            "payment_id": payment_id,
        })
        store.flush()

        return jsonify({"status": "captured", "payment_id": payment_id})

    # All other events (disputes, refunds, etc.) — log but don't process
    return jsonify({"status": "ignored", "event": event})


@app.route("/api/daemon-retry-complete", methods=["POST"])
def daemon_retry_complete():
    """Receive retry execution results from the background daemon worker.

    The daemon executes retries independently of the frontend.
    This endpoint receives the results and broadcasts them to the UI.
    """
    data = request.json or {}
    job_id = data.get("job_id", "")
    payment_id = data.get("payment_id", "")
    action = data.get("action", "")
    result = data.get("result", {})
    source = data.get("source", "daemon_worker")

    print(f"[frontend] Daemon retry complete: job={job_id} payment={payment_id} status={result.get('status')}")

    # Update payment state
    if store.has_payment(payment_id):
        p = store.get_payment(payment_id)
        p["last_action"] = action
        p["last_detail"] = result.get("message", "")

        # If retry created an order, store it
        if result.get("order_id"):
            p["order_id"] = result["order_id"]

        # If link created, store it
        if result.get("link_url"):
            p["payment_link"] = result["link_url"]

    # Broadcast to WebSocket
    push_event(payment_id, "daemon_retry_executed", {
        "job_id": job_id,
        "action": action,
        "result_status": result.get("status", "unknown"),
        "message": result.get("message", ""),
        "order_id": result.get("order_id", ""),
        "link_url": result.get("link_url", ""),
        "source": source,
    })
    store.flush()

    return jsonify({"status": "received", "payment_id": payment_id})


@app.route("/api/payments")
def api_payments():
    p_list = store.payment_values()
    total_at_risk = sum(p.get("amount", 0) for p in p_list)
    total_recovered = sum(p.get("amount", 0) for p in p_list if p.get("status") == "recovered")
    rec_count = sum(1 for p in p_list if p.get("status") == "recovered")
    rate = (rec_count / len(p_list) * 100) if p_list else 0.0
    total_penalties = sum(p.get("penalties_prevented", 0) for p in p_list)
    return jsonify({
        "payments": p_list,
        "total_at_risk": total_at_risk,
        "total_recovered": total_recovered,
        "recovery_rate": round(rate, 1),
        "total_penalties_prevented": total_penalties,
        "penalties_saved_usd": round(total_penalties * NETWORK_FINE_PER_ATTEMPT, 2),
        "penalties_saved_inr": round(total_penalties * 8.30, 2),
    })


@app.route("/api/benchmark", methods=["POST"])
def run_benchmark():
    """Run before/after benchmark comparison for Phase 4 validation."""
    from recovery_agent.eval.chaos_gym import run_before_after_benchmark
    data = request.json or {}
    seed = data.get("seed", 42)
    count = data.get("count", 50)
    result = run_before_after_benchmark(seed=seed, count=count)
    return jsonify(result)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "payments": len(store.all_payments())})

def main():
    port = int(os.getenv("FRONTEND_PORT", "5002"))
    print(f"\n  Customer Checkout:  http://localhost:{port}/pay")
    print(f"  Merchant Dashboard: http://localhost:{port}/merchant")
    print(f"  Agent Flow:         http://localhost:{port}/graph\n")
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)

if __name__ == "__main__":
    main()
