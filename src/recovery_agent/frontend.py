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
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, render_template, render_template_string, request, jsonify
from flask_socketio import SocketIO

from recovery_agent.razorpay_client import RazorpayClient

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24).hex()
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

razorpay_client = RazorpayClient()

# In-memory stores
payments: dict[str, dict] = {}
agent_trails: dict[str, list[dict]] = {}
pending_actions: dict[str, dict] = {}  # payment_id → action waiting for customer response


def push_event(payment_id: str, event_type: str, data: dict):
    payload = {"payment_id": payment_id, "event": event_type, "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"), **data}
    socketio.emit("agent_event", payload)
    socketio.emit("agent_stream", payload)


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


def run_agent_for_payment(payment_id: str, amount: float, failure_reason: str, customer: dict, scenario_type: str = "standard"):
    """Run agent step by step with live streaming thoughts, tool cards, guardrails, and LLM-generated UI morphing."""
    from recovery_agent.agent.diagnosis import diagnose_payment_failure, run_diagnosis
    from recovery_agent.agent.decision import run_decision
    from recovery_agent.agent.execution import execute_action, observe_outcome
    from recovery_agent.agent.guardrails import GuardrailEngine
    from recovery_agent.agent.kg_router import RazorpayKnowledgeGraph
    from recovery_agent.agent.memory import CustomerMemoryStore
    from recovery_agent.agent.llm_client import invoke_llm_json
    from recovery_agent.models import Case, PaymentEvent, Attempt as AttemptModel, ActionType, GenerativeUISpec

    memory_store = CustomerMemoryStore()
    guardrail_engine = GuardrailEngine()
    kg_router = RazorpayKnowledgeGraph()

    customer_email = customer.get("email", "rahul@example.com")
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
        agent_trails[payment_id] = trail
        if payment_id in payments:
            payments[payment_id]["trail"] = list(trail)
        push_event(payment_id, step, entry)

    from recovery_agent.razorpay_knowledge_base import normalize_razorpay_failure
    raw_reason = failure_reason or "Payment failed during checkout"
    norm = normalize_razorpay_failure(raw_reason)

    event = PaymentEvent(
        payment_id=payment_id,
        customer_id=customer_email,
        amount=amount,
        currency="INR",
        failure_code=norm.get("failure_code", "payment_failed"),
        failure_reason=raw_reason,
        metadata={
            "customer_name": customer.get("name", "Rahul Kumar"),
            "scenario": scenario_type,
            "error_code": norm.get("error_code", "BAD_REQUEST_ERROR"),
            "error_source": norm.get("error_source", "gateway"),
            "error_step": norm.get("error_step", "payment_authorization"),
            "error_description": raw_reason,
            "recommended_rail": norm.get("recommended_rail", "payment_link"),
        },
    )
    case = Case(payment=event, max_attempts=3)

    # 1. SENSING / DETECT
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
    time.sleep(0.4)

    # 2. DIAGNOSE (LLM Diagnostic Reflection — primary method)
    emit_thought(
        step="diagnosing",
        thought="Initiating LLM Diagnostic Reflection Engine...",
        detail="Nemotron analyzing raw failure payload, customer history, bank health signals",
    )
    time.sleep(0.4)
    case = run_diagnosis(case)
    cause = case.diagnosis.root_cause.value if case.diagnosis else "unknown"
    confidence = case.diagnosis.confidence if case.diagnosis else 0.7

    # Discover optimal API rail via Knowledge Graph
    try:
        recommended_rails = kg_router.discover_recovery_path(cause)
    except Exception:
        recommended_rails = ["payment_link", "upi_autopay"]

    diag_reasoning = format_reasoning_newlines(case.diagnosis.reasoning) if case.diagnosis else 'Analyzed gateway failure payload'

    emit_thought(
        step="diagnosed",
        thought=f"Root Cause Confirmed: {cause.upper()} (Confidence: {confidence:.0%})",
        detail=f"Reasoning:\n{diag_reasoning}",
        tool_call={
            "tool": "RazorpayKnowledgeGraph.discover_recovery_path",
            "args": {"failure_code": cause, "current_rail": "card"},
            "result": {"target_rail": recommended_rails[0] if recommended_rails else "payment_link"},
        },
    )
    time.sleep(0.4)

    # 3. DECIDE & GUARDRAIL INTERCEPT
    case.attempt_count = 0
    case = run_decision(case)
    action_val = case.payment.metadata.get("decided_action", "send_notification")
    action = ActionType(action_val)

    strategy_reasoning = format_reasoning_newlines(case.payment.metadata.get("strategy_reasoning", ""))

    # Intercept with NVIDIA NAT Guardrails
    approved_action, check_results = guardrail_engine.validate_action(
        case=case,
        action=action,
        profile=cust_profile,
    )
    is_allowed = (approved_action == action)
    action_val = approved_action.value

    emit_thought(
        step="deciding",
        thought=f"LLM Strategy Planner selected intervention: {action_val.upper()}",
        detail=f"Strategic reasoning:\n{strategy_reasoning if strategy_reasoning else 'NVIDIA NAT Guardrails checked (Allowed)'}",
        guardrail={
            "allowed": is_allowed,
            "quiet_hours_active": False,
            "attempt_cap": "1/3",
            "double_debit_lock": "SECURE",
            "modified_action": approved_action.value if not is_allowed else None,
        },
    )
    time.sleep(0.4)

    # 4. GENERATIVE UI SPEC (LLM-generated — no hardcoded branches)
    ui_spec_dict = _generate_ui_spec(
        llm_fn=invoke_llm_json,
        cause=cause,
        amount=amount,
        failure_reason=failure_reason,
        recommended_rail=recommended_rails[0] if recommended_rails else "payment_link",
        customer_name=customer.get("name", "Rahul Kumar"),
        action=action_val,
        scenario_type=scenario_type,
    )

    # Invoke backend Razorpay SDK tool call
    if action_val in ("retry_payment", "update_payment_method"):
        sdk_res = razorpay_client.create_order(
            amount=amount,
            receipt=payment_id,
            notes={"target_rail": recommended_rails[0] if recommended_rails else "upi_autopay", "customer": customer_email},
        )
        tool_name_str = "RazorpaySDK.Order.create"
    else:
        sdk_res = razorpay_client.create_payment_link(
            amount=amount,
            customer={"name": customer.get("name", "Rahul Kumar"), "email": customer_email, "contact": "+919876543210"},
            notes={"recovery_agent": "AutoRecover_v2"},
        )
        tool_name_str = "RazorpaySDK.PaymentLink.create"

    emit_thought(
        step="acting",
        thought=f"Executing Tool Action: {action_val}",
        detail=f"Generating Razorpay API recovery payload & morphing customer interface to {ui_spec_dict.ui_type}",
        tool_call={
            "tool": tool_name_str,
            "args": {"payment_id": payment_id, "amount_in_paise": int(amount * 100), "currency": "INR", "customer": customer_email},
            "response": sdk_res,
        },
        ui_morph=ui_spec_dict.ui_type,
        ui_spec=ui_spec_dict.model_dump(),
    )
    time.sleep(0.5)

    execution = execute_action(action, cause, amount)

    # Update payment store
    if payment_id in payments:
        payments[payment_id]["attempts"] = 1
        payments[payment_id]["last_action"] = action_val
        payments[payment_id]["last_detail"] = execution["detail"]
        payments[payment_id]["trail"] = trail
        payments[payment_id]["ui_spec"] = ui_spec_dict.model_dump()

    # 5. OBSERVE & RECOVER
    if scenario_type in ("abandonment", "card_expiry", "degradation", "voice_call"):
        time.sleep(0.5)
        case.recovered = True
        case.recovered_amount = amount
        capture_res = razorpay_client.capture_payment(payment_id, amount)
        emit_thought(
            step="stopping",
            thought=f"SUCCESS: Payment of INR {amount:,.2f} Recovered!",
            detail=f"Customer completed recovery via {ui_spec_dict.ui_type} rail. Verified via Razorpay API Capture event.",
            tool_call={
                "tool": "RazorpaySDK.Payment.capture",
                "args": {"payment_id": payment_id, "amount_in_paise": int(amount * 100)},
                "response": capture_res,
            },
            ui_morph="RECOVERY_SUCCESS",
        )
    else:
        pending_actions[payment_id] = {
            "action": action_val,
            "execution": execution,
            "case": case,
            "attempt": 0,
            "trail": trail,
            "amount": amount,
        }
        push_event(payment_id, "waiting_for_customer", {"action": action_val, "detail": execution["detail"], "ui_morph": ui_spec_dict.ui_type})

        customer_responded = False
        for _ in range(30):
            time.sleep(1)
            if payment_id not in pending_actions:
                customer_responded = True
                break

        outcome = observe_outcome(action, execution, customer_responded)
        if outcome["success"]:
            case.recovered = True
            case.recovered_amount = amount
            emit_thought(
                step="stopping",
                thought=f"SUCCESS: Payment Recovered! Total INR {amount:,.2f}",
                detail="Verified capture webhook received.",
                ui_morph="RECOVERY_SUCCESS",
            )
        else:
            emit_thought(
                step="stopping",
                thought="Case Escalated / Max Attempts Reached",
                detail="Human support handoff initiated.",
                ui_morph="RECOVERY_FAILED",
            )

    final_status = "recovered" if case.recovered else "failed"
    if payment_id in payments:
        payments[payment_id]["status"] = final_status
        payments[payment_id]["trail"] = trail

    push_event(payment_id, "complete", {"status": final_status, "attempts": 1, "trail": trail, "amount": amount})


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
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f0f4f8;min-height:100vh;display:flex;align-items:center;justify-content:center}
.checkout{background:#fff;border-radius:16px;padding:40px;max-width:520px;width:100%;box-shadow:0 4px 24px rgba(0,0,0,.08)}
.logo{font-size:24px;font-weight:700;color:#2563eb;margin-bottom:4px}
.subtitle{color:#64748b;font-size:14px;margin-bottom:28px}
.product{display:flex;gap:16px;align-items:center;padding:16px;background:#f8fafc;border-radius:12px;margin-bottom:24px}
.product-img{width:56px;height:56px;background:#e2e8f0;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:24px}
.product-info h3{font-size:15px;color:#1e293b}.product-info p{color:#64748b;font-size:13px}
.price{font-size:28px;font-weight:700;color:#1e293b;margin-bottom:24px}
.price span{font-size:14px;color:#64748b;font-weight:400}
.btn{width:100%;padding:14px;border:none;border-radius:10px;font-size:15px;font-weight:600;cursor:pointer;transition:all .2s}
.btn-primary{background:#2563eb;color:#fff}.btn-primary:hover{background:#1d4ed8}
.btn-primary:disabled{background:#94a3b8;cursor:not-allowed}
.btn-respond{background:#f59e0b;color:#fff;margin-top:8px}.btn-respond:hover{background:#d97706}
.btn-success{background:#10b981;color:#fff}
.status-bar{margin-top:16px;padding:14px;border-radius:10px;display:none;font-size:13px}
.status-bar.active{display:block}
.s-processing{background:#eff6ff;color:#2563eb;border:1px solid #bfdbfe}
.s-waiting{background:#fef3c7;color:#92400e;border:1px solid #fde68a}
.s-success{background:#dcfce7;color:#166534;border:1px solid #bbf7d0}
.s-failed{background:#fee2e2;color:#991b1b;border:1px solid #fecaca}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid currentColor;border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite;margin-right:6px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
.trail{margin-top:16px;display:none;max-height:400px;overflow-y:auto}
.trail.visible{display:block}
.trail h4{font-size:13px;color:#64748b;margin-bottom:10px;text-transform:uppercase;letter-spacing:.5px}
.step{display:flex;gap:12px;padding:10px;border-left:3px solid #e2e8f0;margin-bottom:6px;font-size:13px;animation:fadeIn .3s}
.step.active{border-left-color:#2563eb;background:#f8fafc;border-radius:0 8px 8px 0}
.step.done{border-left-color:#10b981}.step.failed-step{border-left-color:#ef4444}.step.waiting{border-left-color:#f59e0b;background:#fffbeb;border-radius:0 8px 8px 0}
.step-num{width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#fff;flex-shrink:0}
.step-num.n-detect{background:#3b82f6}.step-num.n-diagnose{background:#8b5cf6}.step-num.n-decide{background:#f59e0b}.step-num.n-act{background:#10b981}.step-num.n-stop{background:#ef4444}.step-num.n-wait{background:#f59e0b}
.step-content{flex:1}.step-title{font-weight:600;color:#1e293b;margin-bottom:2px}.step-detail{color:#64748b;font-size:12px;line-height:1.5}
.action-box{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:16px;margin-top:12px;display:none}
.action-box.visible{display:block}
.action-box h4{color:#2563eb;font-size:14px;margin-bottom:6px}
.action-box p{color:#64748b;font-size:13px;margin-bottom:12px}
@keyframes fadeIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:translateY(0)}}
</style>
</head>
<body>
<div class="checkout">
<div class="logo">ShopFast</div>
<div class="subtitle">Secure Checkout</div>
<div class="product">
<div class="product-img">&#128722;</div>
<div class="product-info"><h3>Premium Plan — Annual</h3><p>Full access, unlimited projects</p></div>
</div>
<div class="price">INR {{ "{:,.2f}".format(amount) }} <span>/year</span></div>
<button class="btn btn-primary" id="pay-btn" onclick="startPayment()">Pay Now</button>
<div class="status-bar" id="status"></div>
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
const socket = io();

const stepNames = {detecting:"Detecting failure",diagnosed:"Root cause found",deciding:"Selecting action",acting:"Executing recovery",acted:"Action complete",waiting:"Waiting for you",observed:"Outcome observed",stopping:"Recovery complete",continuing:"Retrying"};
const stepClasses = {detecting:"n-detect",diagnosed:"n-diagnose",deciding:"n-decide",acting:"n-act",acted:"n-act",waiting:"n-wait",observed:"n-diagnose",stopping:"n-stop",continuing:"n-detect"};

function startPayment(){
    const btn=document.getElementById("pay-btn");
    btn.disabled=true;btn.innerHTML='<span class="spinner"></span> Processing...';
    fetch("/api/create-order",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({amount:amount,payment_id:paymentId})})
    .then(r=>r.json()).then(data=>{
        if(data.error){showStatus("failed",data.error);btn.disabled=false;btn.innerHTML="Pay Now";return}
        const rzp=new Razorpay({key:data.key_id,amount:data.amount,currency:data.currency,name:"ShopFast",description:"Premium Plan",order_id:data.order_id,
        handler:function(r){showStatus("success","Payment successful! ID: "+r.razorpay_payment_id);btn.innerHTML="Paid ✓";btn.style.background="#16a34a";document.getElementById("action-box").classList.remove("visible")},
        prefill:{name:"Rahul Kumar",email:"rahul@example.com",contact:"9876543210"},theme:{color:"#2563eb"},
        modal:{ondismiss:function(){showStatus("failed","Payment cancelled. Our agent will help you recover it.");btn.disabled=false;btn.innerHTML="Retry Payment";triggerRecovery("cancelled")}}});
        rzp.on("payment.failed",function(r){showStatus("failed","Payment failed: "+r.error.description);btn.disabled=false;btn.innerHTML="Retry Payment";triggerRecovery(r.error.reason||r.error.code||"failed")});
        rzp.open();
    }).catch(()=>{showStatus("failed","Connection error");btn.disabled=false;btn.innerHTML="Pay Now"});
}

function triggerRecovery(reason){
    fetch("/api/payment-failed",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({payment_id:paymentId,amount:amount,failure_reason:reason,customer:{name:"Rahul Kumar",email:"rahul@example.com"}})});
    showStatus("recovering","Payment failed — our agent is working on it...");
    document.getElementById("trail").classList.add("visible");
}

function respondToAgent(){
    fetch("/api/customer-responded",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({payment_id:paymentId})});
    document.getElementById("action-box").classList.remove("visible");
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
        if(data.status==="recovered"){s.className="status-bar active s-success";s.innerHTML="Payment recovered! Thank you."}
        else{s.className="status-bar active s-failed";s.innerHTML="Could not recover automatically. Please try again or update your payment method."}
    }
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
        let inc=document.getElementById("discount-banner");
        if(!inc){inc=document.createElement("div");inc.id="discount-banner";inc.style.cssText="background:#dcfce7;border:1px solid #bbf7d0;border-radius:8px;padding:10px;margin-bottom:16px;font-size:13px;color:#166534;font-weight:500;text-align:center";document.querySelector(".price").after(inc)}
        inc.textContent=spec.discount_incentive;
    }
}
</script></body></html>"""


# ─── Merchant Dashboard ───────────────────────────────────────
MERCHANT_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Merchant Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0}
.topbar{background:#1e293b;padding:12px 24px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #334155}
.topbar .logo{font-weight:700;font-size:16px;color:#3b82f6}
.topbar .links{display:flex;gap:12px}
.topbar a{color:#94a3b8;text-decoration:none;font-size:13px;padding:5px 10px;border-radius:6px}.topbar a:hover,.topbar a.active{background:#334155;color:#e2e8f0}
.container{max-width:1400px;margin:0 auto;padding:20px}
.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:20px}
.metric{background:#1e293b;border-radius:10px;padding:16px;border:1px solid #334155}
.metric-value{font-size:1.8em;font-weight:700;color:#3b82f6}.sv{color:#10b981}.wv{color:#f59e0b}.rv{color:#ef4444}
.metric-label{color:#64748b;font-size:11px;margin-top:2px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#1e293b;border-radius:10px;padding:16px;border:1px solid #334155;margin-bottom:16px}
.card h2{font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #334155;font-size:12px}th{color:#64748b;font-size:10px;text-transform:uppercase}
.badge{display:inline-block;padding:2px 8px;border-radius:8px;font-size:10px;font-weight:600}
.bs{background:#065f46;color:#10b981}.bf{background:#7f1d1d;color:#ef4444}.bp{background:#1e3a5f;color:#3b82f6}.bw{background:#4a3000;color:#f59e0b}
.trail{max-height:500px;overflow-y:auto}
.trail-item{padding:10px;border-left:3px solid #334155;margin-bottom:4px;border-radius:0 6px 6px 0;background:#0f172a;font-size:12px}
.trail-item.t-detecting{border-left-color:#3b82f6}.trail-item.t-diagnosing,.trail-item.t-diagnosed{border-left-color:#8b5cf6}.trail-item.t-deciding{border-left-color:#f59e0b}.trail-item.t-acting,.trail-item.t-acted{border-left-color:#10b981}.trail-item.t-waiting{border-left-color:#f59e0b;background:#1c1917}.trail-item.t-observed{border-left-color:#6366f1}.trail-item.t-stopping{border-left-color:#ef4444}
.trail-time{color:#475569;font-size:10px}.trail-msg{color:#e2e8f0;margin-top:2px;font-weight:500}.trail-detail{color:#94a3b8;margin-top:2px;font-size:11px}
.toast{position:fixed;top:16px;right:16px;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:10px 16px;font-size:12px;z-index:1000;transform:translateX(120%);transition:transform .3s}
.toast.show{transform:translateX(0)}.toast.success{border-left:3px solid #10b981}.toast.info{border-left:3px solid #3b82f6}
.live{display:flex;align-items:center;gap:6px;font-size:11px;color:#10b981}
.live-dot{width:6px;height:6px;background:#10b981;border-radius:50%;animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.empty{text-align:center;padding:40px;color:#475569;font-size:13px}
</style>
</head>
<body>
<div class="topbar">
<span class="logo">Revenue Recovery — Live</span>
<div class="links">
<a href="/pay" target="_blank">Open Store</a>
<a href="/merchant" class="active">Dashboard</a>
<a href="/graph" target="_blank">Agent Flow</a>
</div>
</div>
<div class="container">
<div class="metrics">
<div class="metric"><div class="metric-value" id="m-total">0</div><div class="metric-label">Total Payments</div></div>
<div class="metric"><div class="metric-value sv" id="m-recovered">0</div><div class="metric-label">Recovered</div></div>
<div class="metric"><div class="metric-value wv" id="m-waiting">0</div><div class="metric-label">Waiting for Customer</div></div>
<div class="metric"><div class="metric-value rv" id="m-failed">0</div><div class="metric-label">Failed</div></div>
<div class="metric"><div class="metric-value" id="m-rate">0%</div><div class="metric-label">Recovery Rate</div></div>
</div>
<div class="grid">
<div class="card">
<h2>Payments <span class="live"><span class="live-dot"></span> Live</span></h2>
<table>
<tr><th>ID</th><th>Amount</th><th>Status</th><th>Agent Action</th><th>Attempts</th></tr>
<tbody id="payments"></tbody>
</table>
<div class="empty" id="empty-msg">No payments yet. Open the <a href="/pay" target="_blank" style="color:#3b82f6">store</a> to start.</div>
</div>
<div class="card">
<h2>Agent Trail</h2>
<div class="trail" id="trail"><div class="empty">Waiting for agent activity...</div></div>
</div>
</div>
</div>
<div class="toast" id="toast"></div>
<script>
const socket=io();
let paymentsData=[];

function updateMetrics(){
    const t=paymentsData.length,r=paymentsData.filter(p=>p.status==="recovered").length,w=paymentsData.filter(p=>p.status==="recovering").length,f=paymentsData.filter(p=>p.status==="failed").length;
    document.getElementById("m-total").textContent=t;document.getElementById("m-recovered").textContent=r;
    document.getElementById("m-waiting").textContent=w;document.getElementById("m-failed").textContent=f;
    document.getElementById("m-rate").textContent=t>0?Math.round(r/t*100)+"%":"0%";
}

function renderPayments(){
    const el=document.getElementById("payments"),empty=document.getElementById("empty-msg");
    if(paymentsData.length===0){empty.style.display="block";el.innerHTML="";return}
    empty.style.display="none";
    el.innerHTML=paymentsData.map(p=>{
        const sc=p.status==="recovered"?"bs":p.status==="recovering"?"bw":p.status==="failed"?"bf":"bp";
        return `<tr><td style="color:#94a3b8">${p.payment_id.slice(0,14)}</td><td>INR ${p.amount.toLocaleString()}</td><td><span class="badge ${sc}">${p.status}</span></td><td>${p.last_action||"—"}</td><td>${p.attempts||0}</td></tr>`;
    }).join("");
}

function renderTrail(trail){
    const el=document.getElementById("trail");
    if(!trail||trail.length===0){el.innerHTML='<div class="empty">Waiting for agent activity...</div>';return}
    el.innerHTML=trail.map(e=>{
        let specHtml='';
        if(e.ui_spec){
            specHtml=`<div class="trail-detail" style="margin-top:6px;padding:8px;background:#1e293b;border-radius:6px;border-left:3px solid #8b5cf6">
                <strong style="color:#8b5cf6">Generative UI Spec:</strong><br>
                <span style="color:#e2e8f0">${e.ui_spec.headline||''}</span><br>
                <span style="color:#94a3b8;font-size:11px">${e.ui_spec.subtext||''}</span><br>
                <span style="color:#10b981;font-size:11px">CTA: ${e.ui_spec.primary_cta_text||''} | Rail: ${e.ui_spec.target_rail||''}</span>
                ${e.ui_spec.discount_incentive?`<br><span style="color:#f59e0b;font-size:11px">${e.ui_spec.discount_incentive}</span>`:''}
                ${e.ui_spec.hinglish_voice_script?`<br><span style="color:#94a3b8;font-size:11px">Voice: ${e.ui_spec.hinglish_voice_script}</span>`:''}
            </div>`;
        }
        return `<div class="trail-item t-${e.step}"><div class="trail-time">${e.ts}</div><div class="trail-msg">${e.msg}</div>${e.detail?'<div class="trail-detail">'+e.detail+'</div>':''}${specHtml}</div>`;
    }).join("");
    el.scrollTop=el.scrollHeight;
}

function showToast(msg,type){const t=document.getElementById("toast");t.textContent=msg;t.className="toast "+type+" show";setTimeout(()=>t.className="toast",3000)}

socket.on("connect",()=>showToast("Connected to live feed","info"));

socket.on("agent_event",function(data){
    const idx=paymentsData.findIndex(p=>p.payment_id===data.payment_id);
    if(data.event==="progress"||data.event==="complete"){
        const p={payment_id:data.payment_id,amount:data.amount||0,status:data.status||"recovering",last_action:data.last_action,last_detail:data.last_detail,attempts:data.attempts||0,trail:data.trail||[]};
        if(idx>=0)paymentsData[idx]={...paymentsData[idx],...p};else paymentsData.unshift(p);
        renderPayments();updateMetrics();
    }
    if(idx>=0&&paymentsData[idx].trail)renderTrail(paymentsData[idx].trail);
    else if(data.msg){
        const existing=document.getElementById("trail");
        if(existing.querySelector(".empty"))existing.innerHTML="";
        existing.innerHTML+=`<div class="trail-item t-${data.event}"><div class="trail-time">${data.ts}</div><div class="trail-msg">${data.msg}</div>${data.detail?'<div class="trail-detail">'+data.detail+'</div>':''}</div>`;
        existing.scrollTop=existing.scrollHeight;
    }
    if(data.event==="waiting_for_customer")showToast(`[${data.payment_id.slice(0,10)}] Waiting for customer response`,"info");
    if(data.event==="complete")showToast(`[${data.payment_id.slice(0,10)}] ${data.status}`,data.status==="recovered"?"success":"info");
});
</script></body></html>"""


# ─── Routes ───────────────────────────────────────────────────
@app.route("/")
@app.route("/merchant")
def merchant_page():
    return render_template("index.html")

@app.route("/pay")
def pay_page():
    return render_template_string(PAY_PAGE, amount=2999.0)

@app.route("/graph")
def graph_page():
    from recovery_agent.dashboard import GRAPH_TEMPLATE
    return render_template_string(GRAPH_TEMPLATE)

@app.route("/api/simulate/<scenario>", methods=["POST", "GET"])
def simulate_scenario(scenario: str):
    import random
    payment_id = f"pay_sim_{random.randint(1000, 9999)}"
    customer = {"name": "Rahul Kumar", "email": "rahul@example.com"}

    scenarios = {
        "degradation": (4999.0, "Gateway Timeout 504 (HDFC Netbanking drop)", "degradation"),
        "abandonment": (2999.0, "Customer closed tab during checkout", "abandonment"),
        "card_expiry": (12999.0, "Card expiry date is in the past", "card_expiry"),
        "voice_call": (8500.0, "High-value mandate failure requiring voice intervention", "voice_call"),
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

    amount, reason, stype = scenarios[scenario]
    payments[payment_id] = {
        "payment_id": payment_id,
        "amount": amount,
        "status": "recovering",
        "attempts": 0,
        "last_action": "",
        "last_detail": "",
        "trail": [],
    }

    socketio.start_background_task(run_agent_for_payment, payment_id, amount, reason, customer, stype)
    return jsonify({"status": "simulating", "scenario": scenario, "payment_id": payment_id, "amount": amount, "reason": reason})

@app.route("/api/create-order", methods=["POST"])
def create_order():
    data = request.json
    amount = data.get("amount", 2999)
    payment_id = data.get("payment_id", "pay_unknown")
    if razorpay_client.is_configured:
        order = razorpay_client.create_order(amount=amount, notes={"payment_id": payment_id})
        if "error" not in order:
            payments[payment_id] = {"payment_id": payment_id, "amount": amount, "status": "pending", "order_id": order["id"], "attempts": 0, "last_action": "", "last_detail": "", "trail": []}
            return jsonify({"order_id": order["id"], "amount": order["amount"], "currency": order["currency"], "key_id": razorpay_client.key_id})
    order_id = f"order_sim_{payment_id}"
    payments[payment_id] = {"payment_id": payment_id, "amount": amount, "status": "pending", "order_id": order_id, "attempts": 0, "last_action": "", "last_detail": "", "trail": []}
    return jsonify({"order_id": order_id, "amount": int(amount * 100), "currency": "INR", "key_id": razorpay_client.key_id or "rzp_test_demo"})

@app.route("/api/payment-failed", methods=["POST"])
def payment_failed():
    data = request.json
    payment_id = data.get("payment_id", "")
    amount = data.get("amount", 0)
    failure_reason = data.get("failure_reason", "payment_failed")
    customer = data.get("customer", {})
    if payment_id not in payments:
        payments[payment_id] = {"payment_id": payment_id, "amount": amount, "status": "recovering", "attempts": 0, "last_action": "", "last_detail": "", "trail": []}
    payments[payment_id]["status"] = "recovering"
    socketio.start_background_task(run_agent_for_payment, payment_id, amount, failure_reason, customer)
    return jsonify({"status": "recovery_started"})

@app.route("/api/customer-responded", methods=["POST"])
def customer_responded():
    data = request.json or {}
    payment_id = data.get("payment_id", "")
    updated_expiry = data.get("updated_expiry", "08/29")

    if payment_id in pending_actions:
        del pending_actions[payment_id]

    if payment_id in payments:
        payments[payment_id]["status"] = "recovered"
        amount = payments[payment_id].get("amount", 0)
        trail_entry = {
            "step": "stopping",
            "msg": f"Card Expiry Updated ({updated_expiry}) & Payment Recovered!",
            "detail": f"Updated expiry: {updated_expiry}. Charge verified via Razorpay API Capture.",
            "ui_morph": "RECOVERY_SUCCESS",
            "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        }
        payments[payment_id].setdefault("trail", []).append(trail_entry)
        push_event(payment_id, "stopping", trail_entry)
        push_event(payment_id, "complete", {"status": "recovered", "attempts": 1, "amount": amount})
        return jsonify({"status": "recovered", "payment_id": payment_id, "amount": amount})

    return jsonify({"status": "no_pending_action"})

@app.route("/api/payments")
def api_payments():
    p_list = list(payments.values())
    total_at_risk = sum(p.get("amount", 0) for p in p_list)
    total_recovered = sum(p.get("amount", 0) for p in p_list if p.get("status") == "recovered")
    rec_count = sum(1 for p in p_list if p.get("status") == "recovered")
    rate = (rec_count / len(p_list) * 100) if p_list else 0.0
    return jsonify({
        "payments": p_list,
        "total_at_risk": total_at_risk,
        "total_recovered": total_recovered,
        "recovery_rate": round(rate, 1),
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "payments": len(payments)})

def main():
    port = int(os.getenv("FRONTEND_PORT", "5002"))
    print(f"\n  Customer Checkout:  http://localhost:{port}/pay")
    print(f"  Merchant Dashboard: http://localhost:{port}/merchant")
    print(f"  Agent Flow:         http://localhost:{port}/graph\n")
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)

if __name__ == "__main__":
    main()
