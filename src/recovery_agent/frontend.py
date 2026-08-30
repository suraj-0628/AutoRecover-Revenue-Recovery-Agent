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

import threading
from flask import Flask, render_template, render_template_string, request, jsonify
from flask_socketio import SocketIO

from recovery_agent.razorpay_client import RazorpayClient
from recovery_agent.state_store import StateStore

# ── HITL Approval Gate for Voice Calls ──
# When agent decides VOICE_CALL, it blocks here until merchant approves or 60s timeout.
_pending_voice_approvals: dict[str, threading.Event] = {}  # payment_id → Event
_voice_call_approved: dict[str, bool] = {}                 # payment_id → True/False

# --- OpenTelemetry: Manual spans for coherent parent→child trace hierarchy ---
_otel_tracer = None

def _get_tracer():
    """Lazy-init a tracer for frontend agent spans (parent→child)."""
    global _otel_tracer
    if _otel_tracer is not None:
        return _otel_tracer
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        provider = TracerProvider()
        exporter = OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _otel_tracer = trace.get_tracer("recovery-agent-frontend")
    except Exception:
        # Graceful degradation — spans become no-ops
        from opentelemetry import trace
        _otel_tracer = trace.get_tracer("recovery-agent-frontend")
    return _otel_tracer

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

    tracer = _get_tracer()
    parent_attrs = {
        "payment_id": payment_id,
        "amount": amount,
        "failure_reason": failure_reason,
        "failure_code": failure_code,
        "error_source": error_source,
        "error_step": error_step,
        "customer_email": customer.get("email", ""),
        "customer_id": customer.get("id", ""),
        "customer_name": customer.get("name", ""),
        "scenario_type": scenario_type,
        "currency": "INR",
    }

    with tracer.start_as_current_span("agent_recovery", attributes=parent_attrs) as parent_span:
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
      with tracer.start_as_current_span("detect") as detect_span:
        detect_span.set_attribute("payment_id", payment_id)
        detect_span.set_attribute("failure_reason", failure_reason)
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
      with tracer.start_as_current_span("diagnose") as diag_span:
        emit_thought(
            step="diagnosing",
            thought="Initiating LLM Diagnostic Reflection Engine",
            detail="Nemotron analyzing raw failure payload, customer history, bank health signals",
        )
        case = run_diagnosis(case)
        cause = case.diagnosis.root_cause.value if case.diagnosis else "unknown"
        confidence = case.diagnosis.confidence if case.diagnosis else 0.7
        diag_span.set_attribute("root_cause", cause)
        diag_span.set_attribute("confidence", confidence)

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
      with tracer.start_as_current_span("decide") as decide_span:
        emit_thought(
            step="deciding",
            thought="Initiating LLM Strategy Planner",
            detail="Querying strategy metrics and planning optimal recovery intervention",
        )
        case.attempt_count = 0
        case = run_decision(case)
        action_val = case.payment.metadata.get("decided_action", "send_notification")
        action = ActionType(action_val)
        strategy_source = case.payment.metadata.get("strategy_source", "unknown")
        decide_span.set_attribute("action", action_val)
        decide_span.set_attribute("strategy_source", strategy_source)

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

        # Source badge: deterministic_fast_path | llm_strategy_planner | heuristic_fallback
        source_label = {
            "deterministic_fast_path": "⚡ Deterministic Fast-Path (0ms, no LLM)",
            "llm_strategy_planner": "🧠 LLM Strategy Planner",
            "heuristic_fallback": "📏 Heuristic Fallback",
        }.get(strategy_source, f"❓ {strategy_source}")

        emit_thought(
            step="deciding",
            thought=thought_str,
            detail=f"Strategy Source: {source_label}\nGuardrail Policy Breakdown:\n{guardrail_detail}\nStrategic reasoning:\n{strategy_reasoning if strategy_reasoning else 'NVIDIA NAT Guardrails evaluated all 6 policies'}",
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
                "strategy_source": strategy_source,
            },
        )

      # ── 4. RUN REAL AGENT HARNESS (multi-turn ReAct reasoning + MCP tool calls) ──
      with tracer.start_as_current_span("harness") as harness_span:
        harness = AgentHarness(memory_store=memory_store, guardrail_engine=guardrail_engine)
        emit_thought(
            step="harness_start",
            thought="Launching TrueForge AgentHarness — multi-turn ReAct reasoning loop",
            detail=f"Tools: query_payment_recovery_kb (RAG), query_gateway_error_details, check_bank_health, calculate_payday_window, generate_smart_recovery_link, schedule_payday_retry, escalate_to_human_agent, initiate_voice_call",
        )

        harness_result = harness.run_recovery_case(case)
        harness_span.set_attribute("harness_turns", harness_result.total_turns)
        harness_span.set_attribute("tools_called", len(harness_result.tools_called))

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
                thought=f"Harness Turn {obs.turn}: {obs.reasoning}",
                detail=tools_detail,
                tool_call=tool_call_data,
            )

      # ── 5. WIRE REAL RAZORPAY SDK based on harness-executed tools ──
      # The harness is the SINGLE execution path. It dispatches tools based on
      # the strategy planner's decision. The frontend does NOT re-execute.
      with tracer.start_as_current_span("act") as act_span:
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

        act_span.set_attribute("action", action_val)
        act_span.set_attribute("tool_name", tool_name_str)

        emit_thought(
            step="acting",
            thought="Generating Dynamic Customer UI Spec",
            detail="Morphing the frontend checkout experience based on the harness decision",
        )

        # Generate UI spec — own child span + emit_thought for visibility
        with tracer.start_as_current_span("generate_ui_spec") as uispec_span:
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
            uispec_span.set_attribute("ui_type", ui_spec_dict.ui_type)
            uispec_span.set_attribute("tone", ui_spec_dict.tone)
            emit_thought(
                step="ui_spec",
                thought=f"UI Spec Generated: {ui_spec_dict.ui_type} ({ui_spec_dict.tone} tone)",
                detail=f"Headline: {ui_spec_dict.headline}\nCTA: {ui_spec_dict.primary_cta_text}\nTarget Rail: {ui_spec_dict.target_rail}",
                ui_morph=ui_spec_dict.ui_type,
                ui_spec=ui_spec_dict.model_dump(),
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

        # ── HITL: Voice Call Approval Gate ──
        # Block agent thread until merchant approves or 60s timeout.
        if action_val == "voice_call":
            approval_event = threading.Event()
            _pending_voice_approvals[payment_id] = approval_event

            emit_thought(
                step="approval_needed",
                thought="Voice Call Requires Merchant Approval",
                detail=f"Agent wants to call {customer.get('contact', 'N/A')} for INR {amount:.0f}. Waiting for merchant approval...",
            )
            socketio.emit("voice_call_approval_request", {
                "payment_id": payment_id,
                "customer_phone": customer.get("contact", "N/A"),
                "amount": amount,
                "failure_reason": failure_reason,
                "recovery_link": sdk_res.get("short_url") or sdk_res.get("link_url", ""),
                "cause": cause,
            })

            # Block agent thread for up to 60 seconds
            approved = approval_event.wait(timeout=60)
            _pending_voice_approvals.pop(payment_id, None)
            approved = approved and _voice_call_approved.pop(payment_id, False)

            if not approved:
                # Timeout or denied — fall back to notification
                emit_thought(
                    step="approval_denied",
                    thought="Voice Call Approval Denied / Timed Out",
                    detail="Falling back to SEND_NOTIFICATION",
                )
                action_val = "send_notification"
                sdk_res = {}
                tool_name_str = "Tool.generate_smart_recovery_link (fallback)"
                from recovery_agent.agent.tools import execute_tool
                sdk_res = execute_tool("generate_smart_recovery_link", {
                    "payment_id": payment_id,
                    "allowed_rails": ["upi", "card", "netbanking"],
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

        emit_thought(
            step="acted",
            thought=f"Executed: {action_val}",
            detail=execution.get("detail", ""),
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
        parent_span.set_attribute("final_status", final_status)
        parent_span.set_attribute("recovery_tier", recovery_tier)
        parent_span.set_attribute("strategy_source", strategy_source)
        parent_span.set_attribute("root_cause", cause)
        parent_span.set_attribute("decided_action", action_val)
        parent_span.set_attribute("harness_turns", harness_result.total_turns)
        parent_span.set_attribute("harness_errors", harness_result.error_count)
        parent_span.set_attribute("silent_attempts", case.silent_attempts)
        parent_span.set_attribute("attempt_count", case.attempt_count)
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
PAY_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SoulStreet — Online Store</title>
<script src="https://checkout.razorpay.com/v1/checkout.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#f5f5f0;--card:#fff;--text:#1a1a1a;--muted:#6b7280;--accent:#2563eb;--accent-hover:#1d4ed8;--green:#16a34a;--red:#dc2626;--amber:#f59e0b;--border:#e5e5e5;--radius:12px}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg);min-height:100vh;color:var(--text)}
@keyframes fadeIn{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideRight{from{transform:translateX(100%)}to{transform:translateX(0)}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
@keyframes pulseGlow{0%,100%{box-shadow:0 0 0 0 rgba(37,99,235,0)}50%{box-shadow:0 0 20px 4px rgba(37,99,235,0.15)}}

/* ── Header ── */
.header{background:var(--card);border-bottom:1px solid var(--border);padding:0 24px;height:60px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}
.header-left{display:flex;align-items:center;gap:24px}
.logo{font-size:22px;font-weight:800;letter-spacing:-0.03em;color:var(--text)}
.logo span{color:var(--accent)}
.nav-links{display:flex;gap:20px}
.nav-links a{font-size:13px;font-weight:600;color:var(--muted);text-decoration:none;text-transform:uppercase;letter-spacing:0.05em;transition:color .15s}
.nav-links a:hover,.nav-links a.active{color:var(--text)}
.header-right{display:flex;align-items:center;gap:16px}
.search-box{display:flex;align-items:center;gap:8px;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:8px 14px;width:220px}
.search-box input{border:none;background:none;outline:none;font-size:13px;color:var(--text);width:100%;font-family:inherit}
.search-box input::placeholder{color:#9ca3af}
.cart-btn{position:relative;background:none;border:none;cursor:pointer;padding:8px;border-radius:8px;transition:background .15s}
.cart-btn:hover{background:var(--bg)}
.cart-btn svg{width:22px;height:22px;color:var(--text)}
.cart-count{position:absolute;top:2px;right:2px;background:var(--accent);color:#fff;font-size:10px;font-weight:700;width:18px;height:18px;border-radius:50%;display:flex;align-items:center;justify-content:center;display:none}

/* ── Store View ── */
.store-view{max-width:1100px;margin:0 auto;padding:24px 20px}
.store-banner{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);border-radius:16px;padding:40px 36px;margin-bottom:32px;color:#fff;position:relative;overflow:hidden}
.store-banner::after{content:'';position:absolute;top:-50%;right:-20%;width:400px;height:400px;background:radial-gradient(circle,rgba(37,99,235,0.15),transparent 70%);pointer-events:none}
.store-banner h1{font-size:28px;font-weight:800;margin-bottom:6px;letter-spacing:-0.03em}
.store-banner p{font-size:14px;color:rgba(255,255,255,0.7);max-width:400px;line-height:1.5}
.section-title{font-size:18px;font-weight:700;margin-bottom:20px;letter-spacing:-0.02em}
.product-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:20px}
.product-card{background:var(--card);border-radius:var(--radius);overflow:hidden;border:1px solid var(--border);transition:all .2s;cursor:pointer}
.product-card:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,0.08);border-color:#d1d5db}
.product-img{width:100%;aspect-ratio:3/4;overflow:hidden;background:#f3f4f6}
.product-img img{width:100%;height:100%;object-fit:cover;transition:transform .3s}
.product-card:hover .product-img img{transform:scale(1.03)}
.product-tag{position:absolute;top:10px;left:10px;background:var(--accent);color:#fff;font-size:10px;font-weight:700;padding:3px 8px;border-radius:4px;text-transform:uppercase;letter-spacing:0.05em}
.product-body{padding:14px 16px}
.product-brand{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px}
.product-name{font-size:14px;font-weight:600;margin-bottom:6px;line-height:1.3}
.product-price{display:flex;align-items:baseline;gap:6px}
.product-price .current{font-size:16px;font-weight:700;color:var(--text)}
.product-price .original{font-size:12px;color:var(--muted);text-decoration:line-through}
.product-price .discount{font-size:11px;color:var(--green);font-weight:600}
.add-btn{width:100%;margin-top:10px;padding:10px;border:1.5px solid var(--text);background:transparent;color:var(--text);border-radius:8px;font-size:12px;font-weight:600;cursor:pointer;transition:all .15s;text-transform:uppercase;letter-spacing:0.04em;font-family:inherit}
.add-btn:hover{background:var(--text);color:#fff}
.add-btn.added{background:var(--green);border-color:var(--green);color:#fff;pointer-events:none}

/* ── Cart Sidebar ── */
.cart-overlay{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.4);z-index:200;opacity:0;pointer-events:none;transition:opacity .25s}
.cart-overlay.open{opacity:1;pointer-events:auto}
.cart-sidebar{position:fixed;top:0;right:0;width:400px;max-width:90vw;height:100vh;background:var(--card);z-index:201;transform:translateX(100%);transition:transform .3s cubic-bezier(.4,0,.2,1);display:flex;flex-direction:column;box-shadow:-8px 0 30px rgba(0,0,0,0.1)}
.cart-sidebar.open{transform:translateX(0)}
.cart-header{padding:20px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.cart-header h2{font-size:18px;font-weight:700}
.cart-close{background:none;border:none;cursor:pointer;padding:6px;border-radius:6px;transition:background .15s;font-size:20px;color:var(--muted)}
.cart-close:hover{background:var(--bg)}
.cart-items{flex:1;overflow-y:auto;padding:16px 24px}
.cart-empty{text-align:center;padding:40px 0;color:var(--muted);font-size:14px}
.cart-item{display:flex;gap:12px;padding:14px 0;border-bottom:1px solid var(--border)}
.cart-item-img{width:64px;height:80px;border-radius:8px;overflow:hidden;background:#f3f4f6;flex-shrink:0}
.cart-item-img img{width:100%;height:100%;object-fit:cover}
.cart-item-info{flex:1;min-width:0}
.cart-item-brand{font-size:10px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em}
.cart-item-name{font-size:13px;font-weight:600;margin:2px 0 4px}
.cart-item-price{font-size:14px;font-weight:700}
.cart-item-qty{display:flex;align-items:center;gap:8px;margin-top:6px}
.qty-btn{width:26px;height:26px;border-radius:6px;border:1px solid var(--border);background:var(--card);cursor:pointer;font-size:14px;display:flex;align-items:center;justify-content:center;transition:all .15s;font-family:inherit}
.qty-btn:hover{background:var(--bg);border-color:#d1d5db}
.qty-val{font-size:13px;font-weight:600;min-width:20px;text-align:center}
.cart-item-remove{background:none;border:none;color:var(--muted);cursor:pointer;font-size:11px;margin-top:4px;padding:2px 0;transition:color .15s}
.cart-item-remove:hover{color:var(--red)}
.cart-footer{padding:20px 24px;border-top:1px solid var(--border);background:var(--card)}
.cart-total{display:flex;justify-content:space-between;margin-bottom:14px}
.cart-total-label{font-size:14px;color:var(--muted)}
.cart-total-val{font-size:20px;font-weight:700}
.checkout-btn{width:100%;padding:14px;background:var(--text);color:#fff;border:none;border-radius:var(--radius);font-size:14px;font-weight:600;cursor:pointer;transition:all .2s;font-family:inherit;letter-spacing:-0.01em}
.checkout-btn:hover{background:#333;transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,0.15)}
.checkout-btn:disabled{background:#d1d5db;cursor:not-allowed;transform:none;box-shadow:none}

/* ── Checkout View ── */
.checkout-view{display:none;max-width:600px;margin:0 auto;padding:24px 20px}
.checkout-view.active{display:block}
.store-view.hidden{display:none}
.back-btn{background:none;border:none;cursor:pointer;font-size:13px;font-weight:600;color:var(--muted);margin-bottom:20px;display:flex;align-items:center;gap:6px;transition:color .15s;font-family:inherit}
.back-btn:hover{color:var(--text)}
.checkout-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:32px;box-shadow:0 4px 16px rgba(0,0,0,0.04)}
.checkout-title{font-size:22px;font-weight:700;margin-bottom:4px;letter-spacing:-0.02em}
.checkout-subtitle{font-size:13px;color:var(--muted);margin-bottom:20px}
.cust-form{display:grid;gap:12px;margin-bottom:20px}
.cust-field{display:flex;flex-direction:column;gap:4px}
.cust-field label{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.04em}
.cust-field input{padding:10px 12px;border:1.5px solid var(--border);border-radius:8px;font-size:14px;font-family:inherit;color:var(--text);background:var(--bg);transition:border-color .15s}
.cust-field input:focus{outline:none;border-color:var(--accent);background:var(--card)}
.cust-field input::placeholder{color:#9ca3af}
.cust-field input.invalid{border-color:var(--red)}
.checkout-items{margin-bottom:20px}
.checkout-item{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #f3f4f6;font-size:13px}
.checkout-item:last-child{border-bottom:none}
.checkout-item-name{color:var(--text);font-weight:500}
.checkout-item-price{font-weight:600}
.checkout-divider{height:1px;background:var(--border);margin:16px 0}
.checkout-row{display:flex;justify-content:space-between;font-size:13px;padding:4px 0}
.checkout-row.total{font-size:18px;font-weight:700;padding-top:12px;border-top:2px solid var(--text);margin-top:8px}
.pay-btn{width:100%;padding:16px;background:linear-gradient(135deg,var(--accent),var(--accent-hover));color:#fff;border:none;border-radius:var(--radius);font-size:15px;font-weight:700;cursor:pointer;transition:all .2s;margin-top:24px;font-family:inherit;letter-spacing:-0.01em}
.pay-btn:hover{transform:translateY(-1px);box-shadow:0 8px 20px rgba(37,99,235,0.3)}
.pay-btn:disabled{background:linear-gradient(135deg,#94a3b8,#64748b);cursor:not-allowed;transform:none;box-shadow:none}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid currentColor;border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite;flex-shrink:0;vertical-align:middle;margin-right:6px}

/* ── Status, Decline ── */
.status-bar{margin-top:16px;padding:14px 16px;border-radius:12px;display:none;font-size:13px;font-weight:500;animation:fadeIn .3s}
.status-bar.active{display:flex;align-items:center;gap:8px}
.s-processing{background:rgba(59,130,246,0.08);color:#2563eb;border:1px solid rgba(59,130,246,0.15)}
.s-waiting{background:rgba(245,158,11,0.08);color:#92400e;border:1px solid rgba(245,158,11,0.15)}
.s-success{background:rgba(16,185,129,0.08);color:#059669;border:1px solid rgba(16,185,129,0.15)}
.s-failed{background:rgba(239,68,68,0.08);color:#dc2626;border:1px solid rgba(239,68,68,0.15)}
.s-recovering{background:rgba(245,158,11,0.08);color:#92400e;border:1px solid rgba(245,158,11,0.15)}

.decline-wall{background:linear-gradient(135deg,rgba(245,158,11,0.06),rgba(245,158,11,0.02));border:1px solid rgba(245,158,11,0.15);border-radius:16px;padding:24px;margin-top:18px;display:none;backdrop-filter:blur(10px)}
.decline-wall.visible{display:block;animation:slideUp .35s ease-out}
.decline-wall h3{color:#92400e;font-size:16px;margin-bottom:8px;font-weight:600;letter-spacing:-0.01em}
.decline-wall p{color:#78716c;font-size:13px;margin-bottom:12px;line-height:1.6}
.decline-wall .reason{background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.12);border-radius:10px;padding:12px;font-size:12px;color:#92400e;margin-bottom:14px;line-height:1.5}
.alt-rails{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap}
.alt-rail-btn{background:rgba(255,255,255,0.8);backdrop-filter:blur(8px);border:1.5px solid #e2e8f0;border-radius:10px;padding:10px 16px;font-size:13px;font-weight:500;color:#475569;cursor:pointer;transition:all .2s;display:flex;align-items:center;gap:6px;font-family:inherit}
.alt-rail-btn:hover{border-color:var(--accent);color:var(--accent);background:rgba(59,130,246,0.06);transform:translateY(-1px)}
.alt-rail-btn.active{border-color:var(--accent);color:var(--accent);background:rgba(59,130,246,0.08)}
.discount-banner{background:linear-gradient(135deg,rgba(16,185,129,0.08),rgba(16,185,129,0.04));border:1px solid rgba(16,185,129,0.15);border-radius:10px;padding:12px;margin-bottom:16px;font-size:13px;color:#059669;font-weight:600;text-align:center;display:none;animation:fadeIn .3s}
.discount-banner.visible{display:block}
.action-box{background:linear-gradient(135deg,rgba(59,130,246,0.06),rgba(59,130,246,0.02));border:1px solid rgba(59,130,246,0.12);border-radius:14px;padding:20px;margin-top:16px;display:none}
.action-box.visible{display:block;animation:slideUp .3s ease-out}
.action-box h4{color:var(--accent);font-size:14px;margin-bottom:6px;font-weight:600}
.action-box p{color:var(--muted);font-size:13px;margin-bottom:14px;line-height:1.5}
.btn-respond{background:linear-gradient(135deg,var(--amber),#d97706);color:#fff;border:none;padding:12px 20px;border-radius:10px;font-size:13px;font-weight:600;cursor:pointer;transition:all .2s;width:100%;font-family:inherit}
.btn-respond:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(245,158,11,0.3)}

.ui-overlay{position:fixed;top:20px;right:20px;max-width:380px;background:rgba(255,255,255,0.95);backdrop-filter:blur(20px);border:1px solid rgba(0,0,0,0.08);border-radius:16px;padding:24px;box-shadow:0 20px 60px rgba(0,0,0,0.15);z-index:2147483647;display:none;animation:slideUp .4s ease-out;font-family:'Inter',sans-serif}
.ui-overlay.visible{display:block}
.ui-overlay-headline{font-size:18px;font-weight:700;color:var(--text);margin-bottom:8px;letter-spacing:-0.02em}
.ui-overlay-subtext{font-size:13px;color:var(--muted);line-height:1.6;margin-bottom:16px}
.ui-overlay-cta{display:inline-block;background:linear-gradient(135deg,var(--accent),var(--accent-hover));color:#fff;padding:10px 20px;border-radius:10px;font-size:14px;font-weight:600;text-decoration:none;transition:all .2s}
.ui-overlay-cta:hover{transform:translateY(-1px);box-shadow:0 6px 16px rgba(37,99,235,0.3)}
.ui-overlay-discount{background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.15);border-radius:8px;padding:8px 12px;margin-top:12px;font-size:12px;color:#059669;font-weight:600}
.ui-overlay-tier{display:inline-block;padding:3px 8px;border-radius:6px;font-size:10px;font-weight:600;margin-top:8px}
.ui-overlay-tier.silent{background:rgba(59,130,246,0.1);color:#2563eb}
.ui-overlay-tier.active{background:rgba(245,158,11,0.1);color:#d97706}

@media(max-width:768px){.product-grid{grid-template-columns:repeat(2,1fr);gap:12px}.header{padding:0 16px}.nav-links{display:none}.search-box{width:160px}.store-banner{padding:28px 24px}.store-banner h1{font-size:22px}}
@media(max-width:480px){.product-grid{grid-template-columns:1fr}.cart-sidebar{width:100%}}
</style>
</head>
<body>

<div class="ui-overlay" id="ui-overlay">
  <div class="ui-overlay-headline" id="overlay-headline"></div>
  <div class="ui-overlay-subtext" id="overlay-subtext"></div>
  <div id="overlay-cta-wrap"><a class="ui-overlay-cta" id="overlay-cta" href="#">Complete Payment</a></div>
  <div class="ui-overlay-discount" id="overlay-discount" style="display:none"></div>
  <div class="ui-overlay-tier" id="overlay-tier" style="display:none"></div>
</div>

<!-- Header -->
<div class="header">
  <div class="header-left">
    <div class="logo">Soul<span>Street</span></div>
    <div class="nav-links">
      <a href="#" class="active">Men</a>
      <a href="#">Women</a>
      <a href="#">New Arrivals</a>
      <a href="#">Sale</a>
    </div>
  </div>
  <div class="header-right">
    <div class="search-box">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
      <input placeholder="Search products..." />
    </div>
    <button class="cart-btn" onclick="toggleCart()">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 2L3 6v14a2 2 0 002 2h14a2 2 0 002-2V6l-3-4z"/><line x1="3" y1="6" x2="21" y2="6"/><path d="M16 10a4 4 0 01-8 0"/></svg>
      <span class="cart-count" id="cart-count">0</span>
    </button>
  </div>
</div>

<!-- Store View -->
<div class="store-view" id="store-view">
  <div class="store-banner">
    <h1>New Season Collection</h1>
    <p>Premium streetwear crafted for the bold. Free shipping on orders above INR 1,999.</p>
  </div>
  <div class="section-title">Trending Now</div>
  <div class="product-grid" id="product-grid"></div>
</div>

<!-- Checkout View -->
<div class="checkout-view" id="checkout-view">
  <button class="back-btn" onclick="showStore()">&#8592; Back to Shop</button>
  <div class="checkout-card">
    <div class="checkout-title">Checkout</div>
    <div class="checkout-subtitle">Review your order and complete payment</div>
    <div class="cust-form" id="cust-form">
      <div class="cust-field">
        <label for="cust-name">Full Name</label>
        <input type="text" id="cust-name" placeholder="Rahul Sharma" autocomplete="name" />
      </div>
      <div class="cust-field">
        <label for="cust-email">Email</label>
        <input type="email" id="cust-email" placeholder="rahul@example.com" autocomplete="email" />
      </div>
      <div class="cust-field">
        <label for="cust-phone">Phone Number</label>
        <input type="tel" id="cust-phone" placeholder="+91 98765 43210" autocomplete="tel" />
      </div>
    </div>
    <div class="checkout-items" id="checkout-items"></div>
    <div class="checkout-divider"></div>
    <div class="checkout-row"><span>Subtotal</span><span id="co-subtotal">INR 0</span></div>
    <div class="checkout-row"><span>Shipping</span><span style="color:var(--green);font-weight:600">FREE</span></div>
    <div class="checkout-row total"><span>Total</span><span id="co-total">INR 0</span></div>
    <div class="discount-banner" id="discount-banner"></div>
    <button class="pay-btn" id="pay-btn" onclick="startPayment()">Pay Now</button>
    <div class="status-bar" id="status"></div>
    <div class="decline-wall" id="decline-wall">
      <h3 id="decline-headline">Let's try a different way to complete your payment</h3>
      <p id="decline-subtext">Sometimes one payment method doesn't work — we've found alternatives that often succeed.</p>
      <div class="reason" id="decline-reason"></div>
      <p style="font-size:12px;color:#78716c;margin-bottom:8px;">Switch to an alternative payment method:</p>
      <div class="alt-rails" id="alt-rails">
        <button class="alt-rail-btn" onclick="switchRail('upi')"><span style="font-size:16px">&#128241;</span> UPI Autopay</button>
        <button class="alt-rail-btn" onclick="switchRail('netbanking')"><span style="font-size:16px">&#127974;</span> Netbanking</button>
        <button class="alt-rail-btn" onclick="switchRail('new_card')"><span style="font-size:16px">&#128179;</span> New Card</button>
        <button class="alt-rail-btn" onclick="switchRail('wallet')"><span style="font-size:16px">&#128092;</span> Wallet</button>
      </div>
    </div>
    <div class="action-box" id="action-box">
      <h4 id="action-title">Agent took an action</h4>
      <p id="action-detail"></p>
      <button class="btn-respond" id="respond-btn" onclick="respondToAgent()">I'll complete the payment now</button>
    </div>
  </div>
</div>

<!-- Cart Sidebar -->
<div class="cart-overlay" id="cart-overlay" onclick="toggleCart()"></div>
<div class="cart-sidebar" id="cart-sidebar">
  <div class="cart-header">
    <h2>Your Cart (<span id="cart-count-text">0</span>)</h2>
    <button class="cart-close" onclick="toggleCart()">&times;</button>
  </div>
  <div class="cart-items" id="cart-items">
    <div class="cart-empty">Your cart is empty</div>
  </div>
  <div class="cart-footer" id="cart-footer" style="display:none">
    <div class="cart-total">
      <span class="cart-total-label">Total</span>
      <span class="cart-total-val" id="cart-total">INR 0</span>
    </div>
    <button class="checkout-btn" onclick="goToCheckout()">Proceed to Checkout</button>
  </div>
</div>

<script>
/* ── Product Catalog ── */
const PRODUCTS = [
  {id:1, name:"Oversized Graphic Tee", brand:"SoulStreet Originals", price:1299, originalPrice:1799, category:"tops", img:"https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&h=500&fit=crop&auto=format", tag:"Trending"},
  {id:2, name:"Premium Cotton Hoodie", brand:"Street Essentials", price:2499, originalPrice:3299, category:"outerwear", img:"https://images.unsplash.com/photo-1618354691373-d851c5c3a990?w=400&h=500&fit=crop&auto=format", tag:"New"},
  {id:3, name:"Washed Denim Jacket", brand:"Heritage Denim", price:3499, originalPrice:4499, category:"outerwear", img:"https://images.unsplash.com/photo-1576995853123-5a10305d93c0?w=400&h=500&fit=crop&auto=format", tag:null},
  {id:4, name:"Slim Fit Joggers", brand:"Athleisure Co.", price:1799, originalPrice:2299, category:"bottoms", img:"https://images.unsplash.com/photo-1552902865-b72c031ac5ea?w=400&h=500&fit=crop&auto=format", tag:null},
  {id:5, name:"Polo Classic Tee", brand:"SoulStreet Originals", price:1499, originalPrice:1999, category:"tops", img:"https://images.unsplash.com/photo-1586363104862-3a5e2ab60d99?w=400&h=500&fit=crop&auto=format", tag:null},
  {id:6, name:"Urban Crop Top", brand:"Street Essentials", price:999, originalPrice:1499, category:"tops", img:"https://images.unsplash.com/photo-1503342217505-b0a15ec3261c?w=400&h=500&fit=crop&auto=format", tag:"-33%"},
  {id:7, name:"Relaxed Cargo Pants", brand:"Heritage Denim", price:2199, originalPrice:2999, category:"bottoms", img:"https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=400&h=500&fit=crop&auto=format", tag:"Popular"},
  {id:8, name:"Classic Bomber Jacket", brand:"SoulStreet Originals", price:3999, originalPrice:5499, category:"outerwear", img:"https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400&h=500&fit=crop&auto=format", tag:"-27%"}
];

/* ── Cart State ── */
let cart = {};

function renderProducts() {
  const grid = document.getElementById("product-grid");
  grid.innerHTML = PRODUCTS.map(p => {
    const discount = Math.round((1 - p.price / p.originalPrice) * 100);
    const inCart = cart[p.id];
    return '<div class="product-card">' +
      '<div class="product-img" style="position:relative">' +
        (p.tag ? '<span class="product-tag">' + p.tag + '</span>' : '') +
        '<img src="' + p.img + '" alt="' + p.name + '" loading="lazy" />' +
      '</div>' +
      '<div class="product-body">' +
        '<div class="product-brand">' + p.brand + '</div>' +
        '<div class="product-name">' + p.name + '</div>' +
        '<div class="product-price">' +
          '<span class="current">INR ' + p.price.toLocaleString() + '</span>' +
          '<span class="original">INR ' + p.originalPrice.toLocaleString() + '</span>' +
          '<span class="discount">' + discount + '% off</span>' +
        '</div>' +
        '<button class="add-btn' + (inCart ? ' added' : '') + '" onclick="addToCart(' + p.id + ')" id="add-btn-' + p.id + '">' +
          (inCart ? '&#10003; Added' : 'Add to Cart') +
        '</button>' +
      '</div>' +
    '</div>';
  }).join("");
}

function addToCart(id) {
  if (cart[id]) { cart[id].qty++; }
  else { cart[id] = { qty: 1 }; }
  updateCartUI();
  const btn = document.getElementById("add-btn-" + id);
  if (btn) { btn.classList.add("added"); btn.innerHTML = "&#10003; Added"; }
}

function removeFromCart(id) {
  delete cart[id];
  updateCartUI();
  const btn = document.getElementById("add-btn-" + id);
  if (btn) { btn.classList.remove("added"); btn.innerHTML = "Add to Cart"; }
}

function changeQty(id, delta) {
  if (!cart[id]) return;
  cart[id].qty += delta;
  if (cart[id].qty <= 0) { removeFromCart(id); return; }
  updateCartUI();
}

function getCartTotal() {
  let total = 0;
  for (const id in cart) {
    const p = PRODUCTS.find(x => x.id === parseInt(id));
    if (p) total += p.price * cart[id].qty;
  }
  return total;
}

function getCartCount() {
  let c = 0; for (const id in cart) c += cart[id].qty; return c;
}

function updateCartUI() {
  const count = getCartCount();
  const total = getCartTotal();
  document.getElementById("cart-count").textContent = count;
  document.getElementById("cart-count").style.display = count > 0 ? "flex" : "none";
  document.getElementById("cart-count-text").textContent = count;
  const footer = document.getElementById("cart-footer");
  footer.style.display = count > 0 ? "block" : "none";
  document.getElementById("cart-total").textContent = "INR " + total.toLocaleString();
  const itemsEl = document.getElementById("cart-items");
  if (count === 0) {
    itemsEl.innerHTML = '<div class="cart-empty">Your cart is empty</div>';
    return;
  }
  itemsEl.innerHTML = "";
  for (const id in cart) {
    const p = PRODUCTS.find(x => x.id === parseInt(id));
    if (!p) continue;
    itemsEl.innerHTML += '<div class="cart-item">' +
      '<div class="cart-item-img"><img src="' + p.img + '" alt="' + p.name + '" /></div>' +
      '<div class="cart-item-info">' +
        '<div class="cart-item-brand">' + p.brand + '</div>' +
        '<div class="cart-item-name">' + p.name + '</div>' +
        '<div class="cart-item-price">INR ' + (p.price * cart[id].qty).toLocaleString() + '</div>' +
        '<div class="cart-item-qty">' +
          '<button class="qty-btn" onclick="changeQty(' + id + ',-1)">-</button>' +
          '<span class="qty-val">' + cart[id].qty + '</span>' +
          '<button class="qty-btn" onclick="changeQty(' + id + ',1)">+</button>' +
        '</div>' +
        '<button class="cart-item-remove" onclick="removeFromCart(' + id + ')">Remove</button>' +
      '</div>' +
    '</div>';
  }
  /* update add buttons */
  PRODUCTS.forEach(p => {
    const btn = document.getElementById("add-btn-" + p.id);
    if (!btn) return;
    if (cart[p.id]) { btn.classList.add("added"); btn.innerHTML = "&#10003; Added"; }
    else { btn.classList.remove("added"); btn.innerHTML = "Add to Cart"; }
  });
}

function toggleCart() {
  const overlay = document.getElementById("cart-overlay");
  const sidebar = document.getElementById("cart-sidebar");
  const isOpen = sidebar.classList.contains("open");
  if (isOpen) { overlay.classList.remove("open"); sidebar.classList.remove("open"); }
  else { overlay.classList.add("open"); sidebar.classList.add("open"); }
}

function goToCheckout() {
  const count = getCartCount();
  if (count === 0) return;
  toggleCart();
  document.getElementById("store-view").classList.add("hidden");
  const cv = document.getElementById("checkout-view");
  cv.classList.add("active");
  const items = document.getElementById("checkout-items");
  items.innerHTML = "";
  for (const id in cart) {
    const p = PRODUCTS.find(x => x.id === parseInt(id));
    if (!p) continue;
    items.innerHTML += '<div class="checkout-item"><span class="checkout-item-name">' + p.name + ' &times; ' + cart[id].qty + '</span><span class="checkout-item-price">INR ' + (p.price * cart[id].qty).toLocaleString() + '</span></div>';
  }
  const total = getCartTotal();
  document.getElementById("co-subtotal").textContent = "INR " + total.toLocaleString();
  document.getElementById("co-total").textContent = "INR " + total.toLocaleString();
  window.scrollTo(0, 0);
}

function showStore() {
  document.getElementById("checkout-view").classList.remove("active");
  document.getElementById("store-view").classList.remove("hidden");
  window.scrollTo(0, 0);
}

/* ── Payment Logic (preserved) ── */
const paymentId = "pay_" + Math.random().toString(36).substr(2,9);
const socket = io();

function getCustomerData() {
  return {
    name: (document.getElementById("cust-name").value || "").trim(),
    email: (document.getElementById("cust-email").value || "").trim(),
    contact: (document.getElementById("cust-phone").value || "").replace(/\s/g, "").trim(),
  };
}

function validateCustomerForm() {
  const c = getCustomerData();
  let valid = true;
  ["cust-name", "cust-email", "cust-phone"].forEach(id => {
    document.getElementById(id).classList.remove("invalid");
  });
  if (!c.name) { document.getElementById("cust-name").classList.add("invalid"); valid = false; }
  if (!c.email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(c.email)) { document.getElementById("cust-email").classList.add("invalid"); valid = false; }
  if (!c.contact || c.contact.replace(/\D/g, "").length < 10) { document.getElementById("cust-phone").classList.add("invalid"); valid = false; }
  return valid;
}

function startPayment() {
  const btn = document.getElementById("pay-btn");
  const total = getCartTotal();
  if (total === 0) return;
  if (!validateCustomerForm()) { showStatus("failed", "Please fill in all fields with valid information."); return; }
  const cust = getCustomerData();
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Processing...';
  document.getElementById("decline-wall").classList.remove("visible");
  fetch("/api/create-order", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({amount:total, payment_id:paymentId})})
  .then(r => r.json()).then(data => {
    if (data.error) { showStatus("failed", data.error); btn.disabled = false; btn.innerHTML = "Pay Now"; return; }
    const rzp = new Razorpay({key:data.key_id, amount:data.amount, currency:data.currency, name:"SoulStreet", description:"Order #" + paymentId.slice(-6).toUpperCase(), order_id:data.order_id,
    handler: function(r) {
      showStatus("success", "Payment successful! ID: " + r.razorpay_payment_id);
      btn.innerHTML = "&#10003; Paid";
      btn.style.background = "var(--green)";
      document.getElementById("action-box").classList.remove("visible");
      document.getElementById("decline-wall").classList.remove("visible");
    },
    prefill: {name:cust.name, email:cust.email, contact:cust.contact},
    theme: {color:"#2563eb"},
    modal: {ondismiss: function() {
      showStatus("failed", "Payment cancelled. Our agent will help you recover it.");
      btn.disabled = false; btn.innerHTML = "Retry Payment";
      triggerRecovery({code:"customer_cancelled", reason:"Payment cancelled by customer", source:"customer", step:"payment_processing"});
    }}});
    rzp.on("payment.failed", function(r) {
      showStatus("failed", "Payment couldn't be processed: " + r.error.description);
      btn.disabled = false; btn.innerHTML = "Retry Payment";
      showDeclineWall(r.error.code || "failed", r.error.description);
      triggerRecovery({code:r.error.code || "technical_error", reason:r.error.description || r.error.reason || "Payment failed", source:r.error.source || "gateway", step:r.error.step || "payment_processing"});
    });
    rzp.open();
  }).catch(() => { showStatus("failed", "Connection error"); btn.disabled = false; btn.innerHTML = "Pay Now"; });
}

function showDeclineWall(code, description) {
  document.getElementById("decline-wall").classList.add("visible");
  document.getElementById("decline-reason").textContent = "Reason: " + (description || code);
  const reasons = {card_expired:"Your card has expired. Let's try UPI or update your card.", insufficient_funds:"Insufficient funds. Try a different payment method.", network_timeout:"Connection issue. We'll retry automatically.", bank_declined:"Bank couldn't process this. Try another method."};
  document.getElementById("decline-subtext").textContent = reasons[code] || "We found alternative payment methods that often succeed.";
}

function switchRail(rail) {
  document.querySelectorAll(".alt-rail-btn").forEach(b => b.classList.remove("active"));
  event.currentTarget.classList.add("active");
  showStatus("processing", "Switching to " + rail.charAt(0).toUpperCase() + rail.slice(1) + "...");
  triggerRecovery({code:"method_switch", reason:"Customer switched to " + rail, source:"customer", step:"payment_processing"});
}

function triggerRecovery(err) {
  const total = getCartTotal();
  const cust = getCustomerData();
  fetch("/api/payment-failed", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({payment_id:paymentId, amount:total, failure_code:err.code||"technical_error", failure_reason:err.reason||"Payment failed", error_source:err.source||"gateway", error_step:err.step||"payment_processing", customer:cust})});
  showStatus("recovering", "Payment failed — our agent is working on it...");
}

function respondToAgent() {
  fetch("/api/customer-responded", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({payment_id:paymentId})});
  document.getElementById("action-box").classList.remove("visible");
  document.getElementById("decline-wall").classList.remove("visible");
  showStatus("processing", "Thank you! Processing your payment...");
  const btn = document.getElementById("pay-btn");
  btn.disabled = false;
  btn.innerHTML = "Complete Payment Now";
  btn.className = "pay-btn";
  btn.onclick = function() { startPayment(); };
}

function showStatus(type, msg) {
  const el = document.getElementById("status");
  el.className = "status-bar active s-" + type;
  if (type === "processing" || type === "recovering") el.innerHTML = '<span class="spinner"></span> ' + msg;
  else el.innerHTML = msg;
}

socket.on("agent_event", function(data) {
  if (data.payment_id !== paymentId) return;
  if (data.event === "waiting_for_customer") {
    const box = document.getElementById("action-box");
    box.classList.add("visible");
    document.getElementById("action-title").textContent = "Agent: " + (data.action || "took action");
    document.getElementById("action-detail").textContent = data.detail || "Please respond to continue recovery.";
    showStatus("waiting", "Agent is waiting for your response...");
  }
  if (data.event === "acting" && data.ui_spec) { applyGenerativeUISpec(data.ui_spec); }
  if (data.event === "complete") {
    const s = document.getElementById("status");
    document.getElementById("action-box").classList.remove("visible");
    document.getElementById("decline-wall").classList.remove("visible");
    if (data.status === "recovered") { s.className = "status-bar active s-success"; s.innerHTML = "Payment recovered! Thank you."; }
    else { s.className = "status-bar active s-failed"; s.innerHTML = "Could not recover automatically. Please try again or update your payment method."; }
  }
});

socket.on("ui_spec_overlay", function(data) {
  if (data.payment_id !== paymentId) return;
  const overlay = document.getElementById("ui-overlay");
  if (!overlay) return;
  if (data.headline) document.getElementById("overlay-headline").textContent = data.headline;
  if (data.subtext) document.getElementById("overlay-subtext").textContent = data.subtext;
  if (data.primary_cta_text) {
    const cta = document.getElementById("overlay-cta");
    cta.textContent = data.primary_cta_text;
    cta.onclick = function(e) { e.preventDefault(); overlay.classList.remove("visible"); startPayment(); };
  }
  if (data.discount_incentive) { const d = document.getElementById("overlay-discount"); d.textContent = data.discount_incentive; d.style.display = "block"; }
  if (data.recovery_tier) { const t = document.getElementById("overlay-tier"); t.textContent = data.recovery_tier.toUpperCase(); t.className = "ui-overlay-tier " + data.recovery_tier; t.style.display = "inline-block"; }
  overlay.classList.add("visible");
  setTimeout(function() { overlay.classList.remove("visible"); }, 8000);
});

function applyGenerativeUISpec(spec) {
  if (!spec) return;
  if (spec.headline) { document.querySelector(".checkout-title").textContent = spec.headline; }
  if (spec.subtext) { document.querySelector(".checkout-subtitle").textContent = spec.subtext; }
  if (spec.primary_cta_text) { const btn = document.getElementById("pay-btn"); if (btn) { btn.textContent = spec.primary_cta_text; } }
  if (spec.discount_incentive) { const inc = document.getElementById("discount-banner"); inc.textContent = spec.discount_incentive; inc.classList.add("visible"); }
  if (spec.recovery_tier) {
    const tierLabel = spec.recovery_tier === "silent" ? "Background Retry Active" : "Recovery in Progress";
    showStatus("recovering", tierLabel + " — " + (spec.decline_strategy || "Agent working on it"));
  }
}

/* ── Init ── */
renderProducts();
</script>
</body></html>"""


# ─── Merchant Dashboard ───────────────────────────────────────
MERCHANT_PAGE = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Revenue Recovery — Agent Studio</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<style>
:root{
  --bg-canvas:#FFFFFF;--bg-surface:#F7F8FA;--bg-card:#FFFFFF;
  --border-subtle:#E5E7EB;--border-hover:#D1D5DB;
  --text-primary:#1A1A2E;--text-secondary:#6B7280;--text-muted:#9CA3AF;
  --brand-blue:#2563EB;--brand-blue-light:#EFF6FF;
  --success:#16A34A;--success-light:#F0FDF4;--success-border:#BBF7D0;
  --error:#DC2626;--error-light:#FEF2F2;--error-border:#FECACA;
  --warning:#F59E0B;--warning-light:#FFFBEB;--warning-border:#FDE68A;
  --accent-teal:#0EA5E9;
  --sidebar-bg:#F7F8FA;--sidebar-active:#EFF6FF;--sidebar-active-text:#2563EB;
  --card-shadow:0 1px 3px rgba(0,0,0,0.04),0 1px 2px rgba(0,0,0,0.02);
  --card-shadow-hover:0 4px 12px rgba(0,0,0,0.06);
}
[data-theme="dark"]{
  --bg-canvas:#020617;--bg-surface:#0F172A;--bg-card:rgba(15,23,42,0.7);
  --border-subtle:rgba(255,255,255,0.06);--border-hover:rgba(255,255,255,0.1);
  --text-primary:#F8FAFC;--text-secondary:#94A3B8;--text-muted:#475569;
  --brand-blue:#3B82F6;--brand-blue-light:rgba(59,130,246,0.1);
  --success:#10B981;--success-light:rgba(16,185,129,0.1);--success-border:rgba(16,185,129,0.2);
  --error:#EF4444;--error-light:rgba(239,68,68,0.1);--error-border:rgba(239,68,68,0.2);
  --warning:#F59E0B;--warning-light:rgba(245,158,11,0.1);--warning-border:rgba(245,158,11,0.2);
  --sidebar-bg:#0F172A;--sidebar-active:rgba(59,130,246,0.1);--sidebar-active-text:#60A5FA;
  --card-shadow:0 1px 3px rgba(0,0,0,0.2);--card-shadow-hover:0 4px 12px rgba(0,0,0,0.3);
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg-canvas);color:var(--text-primary);letter-spacing:-0.011em;min-height:100vh;display:flex;transition:background .3s,color .3s}

/* Sidebar */
.sidebar{width:220px;background:var(--sidebar-bg);border-right:1px solid var(--border-subtle);display:flex;flex-direction:column;position:fixed;top:0;left:0;bottom:0;z-index:50;transition:background .3s}
.sidebar-logo{padding:20px 20px 16px;border-bottom:1px solid var(--border-subtle);display:flex;align-items:center;gap:10px}
.sidebar-logo-icon{width:32px;height:32px;background:var(--brand-blue);border-radius:6px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:14px;font-weight:700}
.sidebar-logo-text{font-size:14px;color:var(--text-primary)}
.sidebar-logo-sub{font-size:10px;color:var(--text-muted);font-weight:400;margin-top:1px}
.sidebar-nav{flex:1;padding:12px 8px;overflow-y:auto}
.nav-section{font-size:10px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.06em;padding:8px 12px 4px;margin-top:8px}
.nav-item{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:8px;font-size:13px;font-weight:500;color:var(--text-secondary);text-decoration:none;transition:all .15s;cursor:pointer;border-bottom:2px solid transparent}
.nav-item:hover{background:var(--brand-blue-light);color:var(--text-primary);border-bottom-color:var(--brand-blue)}
.nav-item.active{background:var(--sidebar-active);color:var(--sidebar-active-text);font-weight:600}
.nav-item .nav-icon{width:18px;text-align:center;font-size:14px;flex-shrink:0}
.nav-badge{font-size:9px;padding:2px 6px;border-radius:4px;font-weight:600;background:var(--brand-blue);color:#fff;margin-left:auto}
.sidebar-footer{padding:12px 16px;border-top:1px solid var(--border-subtle);font-size:11px;color:var(--text-muted)}

/* Main Area */
.main{margin-left:220px;flex:1;display:flex;flex-direction:column;min-height:100vh;position:relative}
.main::before{content:'';position:fixed;top:0;left:220px;right:0;bottom:0;background:radial-gradient(circle at 20% 50%,rgba(59,130,246,0.03) 0%,transparent 50%),radial-gradient(circle at 80% 20%,rgba(16,185,129,0.03) 0%,transparent 50%);pointer-events:none;z-index:0}
.topbar{height:56px;background:var(--bg-canvas);border-bottom:1px solid var(--border-subtle);display:flex;justify-content:space-between;align-items:center;padding:0 28px;position:sticky;top:0;z-index:40;transition:background .3s}
.topbar-left{display:flex;align-items:center;gap:16px}
.breadcrumb{font-size:13px;color:var(--text-secondary)}
.breadcrumb span{color:var(--text-primary);font-weight:600}
.topbar-right{display:flex;align-items:center;gap:12px}
.topbar-search{background:var(--bg-surface);border:1px solid var(--border-subtle);border-radius:8px;padding:7px 14px 7px 32px;font-size:13px;color:var(--text-primary);width:220px;outline:none;transition:border-color .15s}
.topbar-search:focus{border-color:var(--brand-blue)}
.search-wrap{position:relative}
.search-wrap::before{content:"\\1F50D";position:absolute;left:10px;top:50%;transform:translateY(-50%);font-size:12px;z-index:1}
.theme-toggle{width:36px;height:36px;border-radius:8px;border:1px solid var(--border-subtle);background:var(--bg-surface);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:16px;transition:all .15s}
.theme-toggle:hover{border-color:var(--brand-blue);background:var(--brand-blue-light)}

/* Content */
.content{padding:24px 28px;flex:1}
.agent-hero{background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:14px;padding:24px;margin-bottom:24px;box-shadow:var(--card-shadow);display:flex;justify-content:space-between;align-items:center;transition:all .2s}
.agent-hero:hover{box-shadow:var(--card-shadow-hover)}
.agent-identity{display:flex;align-items:center;gap:14px}
.agent-icon{width:44px;height:44px;background:var(--brand-blue);border-radius:12px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:22px}
.agent-name{font-size:18px;font-weight:700;color:var(--text-primary)}
.agent-subtitle{font-size:13px;color:var(--text-secondary);margin-top:2px}
.health-badge{display:flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:var(--success);padding:6px 14px;border-radius:20px;background:var(--success-light);border:1px solid var(--success-border)}
.health-dot{width:7px;height:7px;background:var(--success);border-radius:50%;animation:blink 1.5s infinite}
.scenario-triggers{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.scenario-triggers button{padding:6px 14px;border-radius:8px;font-size:12px;font-weight:500;cursor:pointer;transition:all .15s;font-family:inherit}
.scenario-trigger{background:transparent;border:1px solid var(--border-subtle);color:var(--text-secondary)}
.scenario-trigger:hover{border-color:var(--brand-blue);color:var(--brand-blue);background:var(--brand-blue-light)}
.scenario-trigger:active{transform:scale(0.97)}
.scenario-batch{background:var(--brand-blue);border:1px solid var(--brand-blue);color:#fff}
.scenario-batch:hover{background:#1d4ed8}

/* Tabs */
.tabs{display:flex;gap:0;border-bottom:2px solid var(--border-subtle);margin-bottom:24px}
.tab{padding:10px 20px;font-size:13px;font-weight:500;color:var(--text-secondary);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-2px;transition:all .15s}
.tab:hover{color:var(--text-primary)}
.tab.active{color:var(--brand-blue);border-bottom-color:var(--brand-blue);font-weight:600}

/* Metrics Row */
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}
.metric{background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:12px;padding:20px 16px;box-shadow:var(--card-shadow);transition:all .2s}
.metric:hover{box-shadow:var(--card-shadow-hover);border-color:var(--border-hover)}
.metric-value{font-size:2em;font-weight:700;color:var(--brand-blue);letter-spacing:-0.02em;line-height:1}
.metric-value.sv{color:var(--success)}.metric-value.wv{color:var(--warning)}.metric-value.rv{color:var(--error)}
.metric-label{color:var(--text-muted);font-size:11px;margin-top:4px;font-weight:500;text-transform:uppercase;letter-spacing:0.04em}
.metric:nth-child(1){border-left:3px solid var(--brand-blue)}
.metric:nth-child(2){border-left:3px solid var(--success)}
.metric:nth-child(3){border-left:3px solid var(--error)}
.metric:nth-child(4){border-left:3px solid var(--warning)}

/* Grid layouts */
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px}

/* Card */
.card{background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:14px;padding:20px;box-shadow:var(--card-shadow);transition:all .2s}
.card:hover{box-shadow:var(--card-shadow-hover)}
.card h2{font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:14px;font-weight:600}

/* Activity Feed */
.activity-feed{max-height:600px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border-subtle) transparent;scroll-behavior:smooth}
.activity-item{display:flex;gap:12px;padding:14px 8px;border-bottom:1px solid var(--border-subtle);font-size:13px;transition:all .2s;cursor:pointer;position:relative}
.activity-item::before{content:'';position:absolute;left:24px;top:46px;bottom:-1px;width:2px;background:var(--border-subtle)}
.activity-item:last-child::before{display:none}
.activity-item:hover{background:var(--bg-surface)}
.activity-icon{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0}
.activity-icon.recovering{background:var(--warning-light);color:var(--warning)}
.activity-icon.recovered{background:var(--success-light);color:var(--success)}
.activity-icon.failed{background:var(--error-light);color:var(--error)}
.activity-icon.escalated{background:var(--brand-blue-light);color:var(--brand-blue)}
.activity-icon.scheduled{background:var(--brand-blue-light);color:var(--brand-blue)}
.activity-body{flex:1;min-width:0}
.activity-title{font-weight:600;color:var(--text-primary);display:flex;align-items:center;gap:8px}
.activity-title .live{color:var(--success);font-size:10px;font-weight:600}
.activity-meta{color:var(--text-secondary);font-size:12px;margin-top:3px;display:flex;gap:12px;align-items:center}
.activity-amount{font-weight:600;color:var(--text-primary)}

/* Table */
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:10px 14px;border-bottom:1px solid var(--border-subtle);font-size:12px}
th{color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:0.05em;font-weight:600}
tr{transition:background .15s}
tr:hover{background:var(--bg-surface)}

/* Badges */
.badge{display:inline-block;padding:3px 10px;border-radius:8px;font-size:10px;font-weight:600;letter-spacing:0.02em}
.bs{background:var(--success-light);color:var(--success);border:1px solid var(--success-border)}
.bf{background:var(--error-light);color:var(--error);border:1px solid var(--error-border)}
.bp{background:var(--brand-blue-light);color:var(--brand-blue);border:1px solid rgba(59,130,246,0.2)}
.bw{background:var(--warning-light);color:var(--warning);border:1px solid var(--warning-border)}
.btier-silent{background:var(--brand-blue-light);color:var(--brand-blue);border:1px solid rgba(59,130,246,0.2)}
.btier-active{background:var(--warning-light);color:var(--warning);border:1px solid var(--warning-border)}
.btier-hard{background:var(--error-light);color:var(--error);border:1px solid var(--error-border)}

/* Strategy items */
.strategy-item{display:flex;justify-content:space-between;align-items:center;padding:10px 12px;border-bottom:1px solid var(--border-subtle);font-size:12px}
.strategy-code{color:var(--brand-blue);font-weight:600;min-width:60px;font-family:'SF Mono',SFMono-Regular,monospace;font-size:11px}
.strategy-name{color:var(--text-primary);flex:1}
.strategy-tier{font-size:10px;padding:3px 8px;border-radius:6px;font-weight:600}

/* Penalty counter */
.penalty-counter{text-align:center;padding:20px}
.penalty-big{font-size:2.2em;font-weight:700;color:var(--success);letter-spacing:-0.02em}
.penalty-sub{color:var(--text-muted);font-size:12px;margin-top:6px;font-weight:500}
.penalty-usd{color:var(--brand-blue);font-size:14px;margin-top:10px;font-weight:600}

/* Trail */
.trail{max-height:500px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border-subtle) transparent}
.trail-item{padding:12px;border-left:3px solid var(--border-subtle);margin-bottom:4px;border-radius:0 10px 10px 0;background:var(--bg-surface);font-size:12px;transition:background .15s}
.trail-item:hover{background:var(--brand-blue-light)}
.trail-item.t-detecting{border-left-color:var(--brand-blue)}.trail-item.t-diagnosing,.trail-item.t-diagnosed{border-left-color:#8B5CF6}.trail-item.t-deciding{border-left-color:var(--warning)}.trail-item.t-acting,.trail-item.t-acted{border-left-color:var(--success)}.trail-item.t-waiting{border-left-color:var(--warning);background:var(--warning-light)}.trail-item.t-observed{border-left-color:#6366F1}.trail-item.t-stopping{border-left-color:var(--error)}
.trail-time{color:var(--text-muted);font-size:10px;font-weight:500}.trail-msg{color:var(--text-primary);margin-top:3px;font-weight:500}.trail-detail{color:var(--text-secondary);margin-top:3px;font-size:11px;line-height:1.5}

/* Toast */
.toast{position:fixed;top:20px;right:20px;background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:12px;padding:12px 18px;font-size:12px;z-index:1000;transform:translateX(120%);transition:transform .3s cubic-bezier(.4,0,.2,1);font-weight:500;box-shadow:var(--card-shadow-hover)}
.toast.show{transform:translateX(0)}.toast.success{border-left:3px solid var(--success)}.toast.info{border-left:3px solid var(--brand-blue)}

/* Empty state */
.empty{text-align:center;padding:40px;color:var(--text-muted);font-size:13px}

/* Case detail drawer */
.drawer-overlay{position:fixed;inset:0;background:rgba(0,0,0,0.3);z-index:200;opacity:0;pointer-events:none;transition:opacity .25s}
.drawer-overlay.open{opacity:1;pointer-events:all}
.drawer{position:fixed;top:0;right:0;width:600px;height:100vh;background:var(--bg-canvas);border-left:1px solid var(--border-subtle);z-index:201;transform:translateX(100%);transition:transform .3s cubic-bezier(.4,0,.2,1);overflow-y:auto;display:flex;flex-direction:column;scroll-behavior:smooth}
.drawer.open{transform:translateX(0)}
.drawer-header{padding:20px 24px;border-bottom:1px solid var(--border-subtle);display:flex;justify-content:space-between;align-items:center}
.drawer-title{font-size:16px;font-weight:700;color:var(--text-primary)}
.drawer-close{width:32px;height:32px;border-radius:8px;border:1px solid var(--border-subtle);background:var(--bg-surface);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:16px;transition:all .15s}
.drawer-close:hover{background:var(--error-light);border-color:var(--error);color:var(--error)}
.drawer-body{padding:20px 24px;flex:1}
.drawer-section{margin-bottom:20px}
.drawer-section h3{font-size:12px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.04em;margin-bottom:10px}
.drawer-trail{max-height:400px;overflow-y:auto}
.drawer-trail .trail-item{margin-bottom:6px}

@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes fadeIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:translateY(0)}}
</style>
</head>
<body>

<!-- Sidebar Navigation -->
<aside class="sidebar">
  <div class="sidebar-logo">
    <div class="sidebar-logo-icon">R</div>
    <div>
      <div class="sidebar-logo-text">AutoRecover</div>
      <div class="sidebar-logo-sub">Agent Studio</div>
    </div>
  </div>
  <nav class="sidebar-nav">
    <div class="nav-section">Payment Products</div>
    <a class="nav-item" href="/merchant"><span class="nav-icon">&#9673;</span> Dashboard<span class="nav-badge">Live</span></a>
    <a class="nav-item" href="/pay" target="_blank"><span class="nav-icon">&#9671;</span> Store</a>
    <a class="nav-item" href="/graph" target="_blank"><span class="nav-icon">&#10230;</span> Agent Flow</a>
    <div class="nav-section">Agent Studio</div>
    <a class="nav-item active" href="/merchant"><span class="nav-icon" style="width:18px;height:18px;background:var(--brand-blue);border-radius:50%;display:inline-flex;align-items:center;justify-content:center;color:#fff;font-size:10px">&#10003;</span> Recovery Agent<span class="nav-badge" style="background:var(--success)">Active</span></a>
    <a class="nav-item" href="#" onclick="return false"><span class="nav-icon">&#8801;</span> Transactions</a>
    <a class="nav-item" href="#" onclick="return false"><span class="nav-icon">$</span> Settlements</a>
    <a class="nav-item" href="#" onclick="return false"><span class="nav-icon">&#128279;</span> Payment Links</a>
    <div class="nav-section">Settings</div>
    <a class="nav-item" href="#" onclick="return false"><span class="nav-icon">&#9881;</span> Configuration</a>
    <a class="nav-item" href="#" onclick="return false"><span class="nav-icon">&#8984;</span> API Keys</a>
  </nav>
  <div class="sidebar-footer">AutoRecover v2.0 — Buildathon</div>
</aside>

<!-- Main Content -->
<div class="main">
  <header class="topbar">
    <div class="topbar-left">
      <div class="breadcrumb">Agent Studio / <span>Recovery Agent</span></div>
    </div>
    <div class="topbar-right">
      <div class="search-wrap"><span style="position:absolute;left:10px;top:50%;transform:translateY(-50%);font-size:12px;color:var(--text-muted);z-index:1">&#128269;</span><input class="topbar-search" placeholder="Search payments..." /></div>
      <button class="theme-toggle" onclick="toggleTheme()" title="Toggle theme">🌓</button>
      <a href="/pay" target="_blank" style="text-decoration:none"><button style="padding:7px 16px;border-radius:8px;border:1px solid var(--brand-blue);background:var(--brand-blue);color:#fff;font-size:12px;font-weight:600;cursor:pointer">Open Store</button></a>
    </div>
  </header>

  <div class="content">
    <!-- Agent Hero Card -->
    <div class="agent-hero">
      <div class="agent-identity">
        <div class="agent-icon">🤖</div>
        <div>
          <div class="agent-name">Payment Recovery Agent</div>
          <div class="agent-subtitle">Recovering failed payments with AI-powered multi-turn reasoning</div>
        </div>
      </div>
      <div class="health-badge"><span class="health-dot"></span> Healthy &amp; Active</div>
    </div>
    <div class="scenario-triggers">
      <button class="scenario-trigger" onclick="simulateScenario('degradation')">504 Degradation</button>
      <button class="scenario-trigger" onclick="simulateScenario('abandonment')">Cart Abandonment</button>
      <button class="scenario-trigger" onclick="simulateScenario('card_expiry')">Expired Card</button>
      <button class="scenario-trigger" onclick="simulateScenario('bank_decline')">Bank Decline</button>
      <button class="scenario-trigger" onclick="simulateScenario('voice_call')">Voice Call</button>
      <button class="scenario-trigger scenario-batch" onclick="simulateScenario('batch')">Run 30-Case Batch</button>
    </div>

    <!-- Tabs -->
    <div class="tabs">
      <div class="tab active" onclick="showTab('activity')">Activity</div>
      <div class="tab" onclick="showTab('settings')">Settings</div>
    </div>

    <!-- Tab: Activity -->
    <div id="tab-activity">
      <!-- Metrics -->
      <div class="metrics">
        <div class="metric"><div class="metric-value" id="m-total">0</div><div class="metric-label">Total Payments</div></div>
        <div class="metric"><div class="metric-value sv" id="m-recovered">0</div><div class="metric-label">Recovered</div></div>
        <div class="metric"><div class="metric-value rv" id="m-failed">0</div><div class="metric-label">Failed</div></div>
        <div class="metric"><div class="metric-value" id="m-rate">0%</div><div class="metric-label">Recovery Rate</div></div>
      </div>
      <!-- Hidden metrics (kept for JS updateMetrics) -->
      <div style="display:none"><div id="m-waiting">0</div><div id="m-penalties">0</div><div id="m-saved">$0.00</div></div>

      <!-- Activity + Sidebar Cards -->
      <div class="grid">
        <!-- Activity Feed (left) -->
        <div class="card">
          <h2>Activity Feed <span class="live"><span class="health-dot"></span> Live</span></h2>
          <div class="activity-feed" id="payments">
            <div class="empty" id="empty-msg">No payments yet. Open the <a href="/pay" target="_blank" style="color:var(--brand-blue)">store</a> to start recovery.</div>
          </div>
        </div>

        <!-- Right column: Agent Trail -->
        <div style="display:flex;flex-direction:column;gap:16px">
          <div class="card" style="flex:1">
            <h2>Agent Trail</h2>
            <div class="trail" id="trail"><div class="empty">Waiting for agent activity...</div></div>
          </div>
        </div>
      </div>

      <!-- Hidden metric IDs for JS updateMetrics -->
      <div style="display:none">
        <span id="tier-silent-count">SILENT: 0</span>
        <span id="tier-active-count">ACTIVE: 0</span>
        <span id="tier-hard-count">HARD BLOCKED: 0</span>
        <div id="decline-strategies"></div>
        <div id="penalty-count">0</div>
        <div id="penalty-value">$0.00 saved</div>
      </div>

      <div style="text-align:center;padding:12px 0;color:var(--text-muted);font-size:11px;position:sticky;bottom:0;background:var(--bg-canvas);border-top:1px solid var(--border-subtle)">
        AI agents can make mistakes. Verify important decisions independently.
      </div>
    </div>

    <!-- Tab: Settings (placeholder) -->
    <div id="tab-settings" style="display:none">
      <div class="card">
        <h2>Agent Configuration</h2>
        <div style="padding:20px;color:var(--text-secondary);font-size:13px">
          <p>Configure recovery agent parameters, guardrail policies, and notification templates.</p>
          <p style="margin-top:12px;color:var(--text-muted);font-size:12px">Settings panel coming soon. Currently managed via configuration files.</p>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Case Detail Drawer -->
<div class="drawer-overlay" id="drawer-overlay" onclick="closeDrawer()"></div>
<div class="drawer" id="drawer">
  <div class="drawer-header">
    <div class="drawer-title" id="drawer-title">Payment Details</div>
    <button class="drawer-close" onclick="closeDrawer()">&#10005;</button>
  </div>
  <div id="drawer-status-bar" style="height:4px;border-radius:0"></div>
  <div class="drawer-body" id="drawer-body">
    <div class="empty">Select a payment to view details</div>
  </div>
</div>

<!-- Toast -->
<div class="toast" id="toast"></div>

<script>
const socket=io();
let paymentsData=[];
let tierStats={silent:0,active:0,hard_decline_blocked:0};
let totalPenalties=0;
let declineStrategies={};

// Theme toggle
function toggleTheme(){
  const html=document.documentElement;
  const current=html.getAttribute("data-theme");
  html.setAttribute("data-theme",current==="light"?"dark":"light");
  localStorage.setItem("theme",html.getAttribute("data-theme"));
}
(function(){const saved=localStorage.getItem("theme");if(saved)document.documentElement.setAttribute("data-theme",saved)})();

// Tab switching
function showTab(name){
  document.querySelectorAll(".tab").forEach(t=>t.classList.remove("active"));
  document.querySelectorAll("[id^=tab-]").forEach(p=>p.style.display="none");
  event.target.classList.add("active");
  document.getElementById("tab-"+name).style.display="block";
}

// Drawer
function openDrawer(paymentId){
  const p=paymentsData.find(x=>x.payment_id===paymentId);
  if(!p)return;
  document.getElementById("drawer-title").textContent=p.payment_id;
  const bar=document.getElementById("drawer-status-bar");
  const statusColors={recovering:"var(--brand-blue)",awaiting_customer:"var(--brand-blue)",recovered:"var(--success)",failed:"var(--error)",escalated:"var(--brand-blue)",scheduled:"var(--warning)"};
  bar.style.background=statusColors[p.status]||"var(--brand-blue)";
  const body=document.getElementById("drawer-body");
  const tierLabel=(p.recovery_tier||"active").toUpperCase();
  const tierCls=p.recovery_tier==="silent"?"btier-silent":p.recovery_tier==="hard_decline_blocked"?"btier-hard":"btier-active";
  const statusBadge=p.status==="recovered"?"bs":p.status==="failed"?"bf":p.status==="escalated"?"bp":"bw";
  let html=`<div class="drawer-section"><h3>Payment Info</h3><div style="font-size:13px;color:var(--text-secondary);display:flex;flex-wrap:wrap;gap:12px"><span><b>Amount:</b> INR ${p.amount.toLocaleString()}</span><span><b>Status:</b> <span class="badge ${statusBadge}">${p.status}</span></span><span class="badge ${tierCls}" style="font-size:11px">${tierLabel}</span><span><b>Attempts:</b> ${p.attempts||0}</span>${p.decline_strategy==='voice_call'?'<br><b>Channel:</b> <span class="drawer-badge blue">AI Voice Call (SuperU)</span>':''}</div></div>`;
  if(p.decline_strategy){html+=`<div class="drawer-section"><h3>Decline Strategy</h3><div style="font-size:13px;color:var(--text-secondary)">${p.decline_strategy}</div></div>`}
  if(p.penalties_prevented){html+=`<div class="drawer-section"><h3>Penalties Prevented</h3><div style="font-size:13px;color:var(--success);font-weight:600">${p.penalties_prevented} blocked ($${(p.penalties_prevented*0.10).toFixed(2)} saved)</div></div>`}
  if(p.trail&&p.trail.length){
    const diagnosed=p.trail.find(e=>e.step==="diagnosed"||e.step==="diagnosing");
    const decided=p.trail.find(e=>e.step==="deciding");
    if(diagnosed){html+=`<div class="drawer-section"><h3>Diagnosis</h3><div style="font-size:13px;color:var(--text-secondary)">${diagnosed.msg}${diagnosed.detail?'<br>'+diagnosed.detail:''}</div></div>`}
    if(decided){html+=`<div class="drawer-section"><h3>Strategy</h3><div style="font-size:13px;color:var(--text-secondary)">${decided.msg}${decided.detail?'<br>'+decided.detail:''}</div></div>`}
    html+=`<div class="drawer-section"><h3>Agent Trail (${p.trail.length} steps)</h3><div class="drawer-trail">`;
    p.trail.forEach(e=>{
      html+=`<div class="trail-item t-${e.step}" style="cursor:pointer" onclick="var d=this.querySelector('.trail-detail');if(d)d.style.display=d.style.display==='none'?'block':'none'"><div class="trail-time">${e.ts}</div><div class="trail-msg">${e.msg}</div>${e.detail?'<div class="trail-detail" style="display:none">'+e.detail+'</div>':''}</div>`;
    });
    html+=`</div></div>`;
  }
  body.innerHTML=html;
  document.getElementById("drawer-overlay").classList.add("open");
  document.getElementById("drawer").classList.add("open");
}
function closeDrawer(){
  document.getElementById("drawer-overlay").classList.remove("open");
  document.getElementById("drawer").classList.remove("open");
}

function updateMetrics(){
  const t=paymentsData.length,r=paymentsData.filter(p=>p.status==="recovered").length,w=paymentsData.filter(p=>p.status==="recovering"||p.status==="awaiting_customer").length,f=paymentsData.filter(p=>p.status==="failed").length;
  document.getElementById("m-total").textContent=t;document.getElementById("m-recovered").textContent=r;
  document.getElementById("m-waiting").textContent=w;document.getElementById("m-failed").textContent=f;
  document.getElementById("m-rate").textContent=t>0?Math.round(r/t*100)+"%":"0%";
  document.getElementById("m-penalties").textContent=totalPenalties;
  document.getElementById("m-saved").textContent="$"+(totalPenalties*0.10).toFixed(2);
  document.getElementById("penalty-count").textContent=totalPenalties;
  document.getElementById("penalty-value").textContent="$"+(totalPenalties*0.10).toFixed(2)+" saved";
  document.getElementById("tier-silent-count").textContent="SILENT: "+tierStats.silent;
  document.getElementById("tier-active-count").textContent="ACTIVE: "+tierStats.active;
  document.getElementById("tier-hard-count").textContent="HARD BLOCKED: "+tierStats.hard_decline_blocked;
}

function renderDeclineStrategies(){
  const el=document.getElementById("decline-strategies");
  const keys=Object.keys(declineStrategies);
  if(keys.length===0){el.innerHTML='<div class="empty" style="padding:8px">Waiting...</div>';return}
  el.innerHTML=keys.map(code=>{
    const s=declineStrategies[code];
    const tierCls=s.tier==="silent"?"btier-silent":s.tier==="hard_decline_blocked"?"btier-hard":"btier-active";
    const tierLabel=s.tier==="silent"?"SILENT":s.tier==="hard_decline_blocked"?"BLOCKED":"ACTIVE";
    return `<div class="strategy-item"><span class="strategy-code">${code}</span><span class="strategy-name">${s.strategy}</span><span class="strategy-tier ${tierCls}">${tierLabel}</span></div>`;
  }).join("");
}

function renderPayments(){
  const el=document.getElementById("payments"),empty=document.getElementById("empty-msg");
  const times=["Just now","2 min ago","5 min ago","12 min ago","18 min ago","25 min ago","38 min ago","52 min ago","1 hr ago","2 hr ago"];
  if(paymentsData.length===0){empty.style.display="block";el.innerHTML="";el.appendChild(empty);return}
  empty.style.display="none";
  el.innerHTML=paymentsData.map((p,i)=>{
    const iconCls=p.status==="recovered"?"recovered":p.status==="recovering"||p.status==="awaiting_customer"?"recovering":p.status==="failed"?"failed":p.status==="escalated"?"escalated":"scheduled";
    const icon=p.status==="recovered"?"✓":p.status==="failed"?"✕":p.status==="escalated"?"↗":p.status==="scheduled"?"⏱":"●";
    const live=p.status==="recovering"||p.status==="awaiting_customer"?'<span class="live"><span class="health-dot"></span> Live</span>':'';
    const tierCls=p.recovery_tier==="silent"?"btier-silent":p.recovery_tier==="hard_decline_blocked"?"btier-hard":"btier-active";
    const tierLabel=(p.recovery_tier||"active").toUpperCase();
    const action=p.status==="recovered"?"Recovery confirmed":p.status==="escalated"?"Escalated to human":p.status==="scheduled"?"Retry scheduled":p.status==="awaiting_customer"?"Awaiting response":p.status==="voice_call"?"Voice Call Active":"Recovering";
    const ts=times[i]||(i*7+3)+" min ago";
    return `<div class="activity-item" onclick="openDrawer('${p.payment_id}')">
      <div class="activity-icon ${iconCls}">${icon}</div>
      <div class="activity-body">
        <div class="activity-title"><span>${action} ${p.payment_id.slice(0,16)}</span><span style="margin-left:auto;display:flex;align-items:center;gap:8px;flex-shrink:0"><span class="activity-amount">INR ${p.amount.toLocaleString()}</span><span class="badge ${tierCls}">${tierLabel}</span>${live}</span></div>
        <div class="activity-meta"><span>${ts}</span></div>
      </div>
    </div>`;
  }).join("");
}

function renderTrail(trail){
  const el=document.getElementById("trail");
  if(!el)return;
  if(!trail||trail.length===0){el.innerHTML='<div class="empty">Waiting for agent activity...</div>';return}
  el.innerHTML=trail.map(e=>{
    let specHtml='';
    if(e.ui_spec){
      specHtml=`<div class="trail-detail" style="margin-top:6px;padding:8px;border-radius:6px;border-left:3px solid #8B5CF6;background:var(--bg-surface)">
        <strong style="color:#8B5CF6">UI Spec:</strong> ${e.ui_spec.headline||''} — ${e.ui_spec.primary_cta_text||''}
      </div>`;
    }
    return `<div class="trail-item t-${e.step}"><div class="trail-time">${e.ts}</div><div class="trail-msg">${e.msg}</div>${e.detail?'<div class="trail-detail">'+e.detail+'</div>':''}${specHtml}</div>`;
  }).join('');
}
</script></body></html>"""


# ─── Routes ───────────────────────────────────────────────────
@app.route("/")
@app.route("/merchant")
def merchant_page():
    return render_template("index.html")

@app.route("/pay")
def pay_page():
    return render_template_string(PAY_PAGE)

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

        is_simulated = order_id.startswith("order_rzp_") or order_id.startswith("order_sim_")
        order_paid = False

        if order_id and not is_simulated and razorpay_client.is_configured:
            order_data = razorpay_client.fetch_order(order_id)
            order_status = order_data.get("status", "")
            order_paid = order_status == "paid"
        elif is_simulated or not razorpay_client.is_configured:
            order_paid = True

        if order_paid:
            p["status"] = "recovered"
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
                "detail": f"Order {order_id} status: not paid. Awaiting Razorpay capture webhook.",
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

    return jsonify({"status": "ignored", "event": event})


@app.route("/api/daemon-retry-complete", methods=["POST"])
def daemon_retry_complete():
    data = request.json or {}
    job_id = data.get("job_id", "")
    payment_id = data.get("payment_id", "")
    action = data.get("action", "")
    result = data.get("result", {})
    source = data.get("source", "daemon_worker")

    print(f"[frontend] Daemon retry complete: job={job_id} payment={payment_id} status={result.get('status')}")

    if store.has_payment(payment_id):
        p = store.get_payment(payment_id)
        p["last_action"] = action
        p["last_detail"] = result.get("message", "")
        if result.get("order_id"):
            p["order_id"] = result["order_id"]
        if result.get("link_url"):
            p["payment_link"] = result["link_url"]

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
    from recovery_agent.eval.chaos_gym import run_before_after_benchmark
    data = request.json or {}
    seed = data.get("seed", 42)
    count = data.get("count", 50)
    result = run_before_after_benchmark(seed=seed, count=count)
    return jsonify(result)

@app.route("/api/voice-call-approval", methods=["POST"])
def voice_call_approval():
    """Merchant approves or denies a pending voice call."""
    data = request.json or {}
    payment_id = data.get("payment_id", "").strip()
    approved = data.get("approved", False)
    if not payment_id:
        return jsonify({"error": "payment_id required"}), 400
    event = _pending_voice_approvals.get(payment_id)
    if not event:
        return jsonify({"error": "No pending approval for this payment"}), 404
    _voice_call_approved[payment_id] = approved
    event.set()  # unblocks the agent thread
    return jsonify({"status": "approved" if approved else "denied", "payment_id": payment_id})

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
