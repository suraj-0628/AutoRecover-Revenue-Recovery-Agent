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

    # BUG FIX: Add timeout to prevent hung threads from accumulating
    import threading
    _AGENT_TIMEOUT_SECONDS = 120  # 2 minutes max per agent execution
    _agent_timer = None

    def _timeout_handler():
        print(f"[Frontend] Agent timeout for {payment_id} after {_AGENT_TIMEOUT_SECONDS}s")
        try:
            push_event(payment_id, "error", {
                "step": "timeout",
                "msg": f"Agent execution timed out after {_AGENT_TIMEOUT_SECONDS}s",
                "detail": "The agent thread was terminated due to timeout. Possible LLM hang or network issue.",
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            })
        except Exception:
            pass
        if store.has_payment(payment_id):
            p = store.get_payment(payment_id)
            p["status"] = "timeout"
            store.flush()

    try:
        _agent_timer = threading.Timer(_AGENT_TIMEOUT_SECONDS, _timeout_handler)
        _agent_timer.daemon = True
        _agent_timer.start()

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
        if _agent_timer:
            _agent_timer.cancel()
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
        # BUG FIX: Pass memory context to decision layer (was passing nothing)
        from recovery_agent.agent.strategy_metrics import StrategyMetricsStore, ThompsonBandit
        try:
            strategy_metrics = StrategyMetricsStore()
            bandit = ThompsonBandit(strategy_metrics)
        except Exception:
            strategy_metrics = None
            bandit = None

        case = run_decision(
            case, profile=cust_profile, memory=memory_store,
            strategy_metrics=strategy_metrics, bandit=bandit,
        )
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

        # BUG FIX: The harness already executed the tool — do NOT re-execute.
        # Use the harness result directly instead of calling execute_action() again.
        # This prevents duplicate Razorpay orders, duplicate emails, and double payment links.
        execution = {
            "action": action_val,
            "detail": f"Harness executed {tool_name_str} after {harness_result.total_turns} turns",
            "observable": sdk_res.get("observable", "none"),
        }

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
const urlParams = new URLSearchParams(window.location.search);
const paymentId = urlParams.get("payment_id") || ("pay_" + Math.random().toString(36).substr(2,9));
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

/* FLAW-38: Real-time payment status update (webhook push, not polling) */
socket.on("payment_update", function(data) {
  if (data.payment_id !== paymentId) return;
  const s = document.getElementById("status");
  if (data.status === "captured") {
    s.className = "status-bar active s-success";
    s.innerHTML = "Payment of INR " + (data.amount || 0).toLocaleString() + " captured successfully! Thank you.";
    document.getElementById("action-box").classList.remove("visible");
  } else if (data.status === "failed") {
    s.className = "status-bar active s-failed";
    s.innerHTML = "Payment could not be processed. Please try again or update your payment method.";
  }
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



# ─── Routes ───────────────────────────────────────────────────
@app.route("/")
@app.route("/merchant")
def merchant_page():
    return render_template("index.html")

@app.route("/pay")
def pay_page():
    return render_template_string(PAY_PAGE)

@app.route("/api/ui-spec", methods=["POST"])
def api_ui_spec():
    """FLAW-35: Generate LLM-powered UI morphing spec for customer checkout."""
    data = request.get_json(force=True, silent=True) or {}
    payment_id = data.get("payment_id", "")
    amount = data.get("amount", 0)
    failure_type = data.get("failure_type", "network_error")
    failure_reason = data.get("failure_reason", "")

    try:
        from recovery_agent.models import FailureType
        from recovery_agent.agent.llm_client import invoke_llm_json
        ft = FailureType(failure_type) if failure_type in [e.value for e in FailureType] else FailureType.NETWORK_ERROR
        spec = _generate_ui_spec(
            llm_fn=invoke_llm_json,
            cause=ft.value,
            amount=amount,
            failure_reason=failure_reason,
            recommended_rail="payment_link",
            customer_name="",
            action="send_notification",
            scenario_type="standard",
        )
        return jsonify({"status": "ok", "spec": spec.model_dump()})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/graph")
def graph_page():
    from recovery_agent.dashboard import GRAPH_TEMPLATE
    return render_template_string(GRAPH_TEMPLATE)

@app.route("/api/agent-trace/<payment_id>")
def api_agent_trace(payment_id: str):
    """FLAW-37: Return full agent decision trace for merchant debugging."""
    trail = store.get_trail(payment_id)
    payment = store.get_payment(payment_id)
    return jsonify({
        "payment_id": payment_id,
        "status": payment.get("status", "unknown") if payment else "not_found",
        "trail": trail or [],
        "total_steps": len(trail or []),
    })

@app.route("/api/eval/run", methods=["POST"])
def api_eval_run():
    """FLAW-45: Trigger batch evaluation via API (automated eval pipeline)."""
    data = request.get_json(force=True, silent=True) or {}
    num_cases = data.get("num_cases", 30)

    try:
        from recovery_agent.agent.evaluation import (
            run_batch_evaluation,
            detect_regression,
        )
        result = run_batch_evaluation(num_cases=num_cases)
        regression = detect_regression(result)

        return jsonify({
            "status": "ok",
            "recovery_rate": result.recovery_rate,
            "recovered": result.recovered,
            "total_cases": result.total_cases,
            "recovered_amount": result.recovered_amount,
            "regression": {
                "is_regression": regression.is_regression,
                "baseline": regression.baseline_value,
                "current": regression.current_value,
                "change": regression.change,
                "reason": regression.reason,
            },
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

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
        "status": "pending",
        "attempts": 0,
        "last_action": "",
        "last_detail": "",
        "trail": [],
        "recovery_tier": "silent",
        "decline_strategy": "",
        "penalties_prevented": 0,
    })

    # Agent starts ONLY when customer attempts payment and it fails (via /api/payment-failed).
    # This ensures the agent runs with real payment context, not before the customer acts.
    store.flush()
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
        # Only update status — preserve order_id and other fields from create_order()
        store.update_payment(payment_id, status="recovering")
    else:
        store.save_payment(payment_id, {"payment_id": payment_id, "amount": amount, "status": "recovering", "attempts": 0, "last_action": "", "last_detail": "", "trail": []})
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

        # BUG FIX: Always verify via Razorpay API when configured, even for simulated orders
        # Previously, simulated orders were auto-marked as paid without verification
        if razorpay_client.is_configured:
            if order_id:
                order_data = razorpay_client.fetch_order(order_id)
                order_status = order_data.get("status", "")
                order_paid = order_status == "paid"
            # Also try fetching the payment directly
            if not order_paid:
                payment_data = razorpay_client.fetch_payment(payment_id)
                if payment_data.get("status") == "captured":
                    order_paid = True
        elif is_simulated:
            # Only trust simulated orders when Razorpay is NOT configured (dev mode)
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
        # FLAW-38: Emit real-time payment_update for customer checkout
        socketio.emit("payment_update", {
            "payment_id": payment_id,
            "status": "captured",
            "amount": amount,
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
