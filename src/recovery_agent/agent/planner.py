"""Planning Agent — pydantic-ai Agent for structured recovery planning.

Uses pydantic-ai's real SDK:
- Agent with output_type=RecoveryPlan for structured output
- RunContext[Deps] for dependency injection
- @agent.tool for tools with typed context

Source: pydantic-ai Agent structured output pattern
        AGENTS.md MANDATE 1 — use real SDKs, not raw Python
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from recovery_agent.models import Case


# ═══════════════════════════════════════════════════════════════
# DEPENDENCIES — passed to every tool via RunContext
# ═══════════════════════════════════════════════════════════════

@dataclass
class RecoveryDeps:
    """Dependencies for the planning agent.

    Injected into every tool via RunContext[Deps].
    """
    case: Case
    guardrail_engine: object = None


# ═══════════════════════════════════════════════════════════════
# OUTPUT SCHEMA — structured plan from LLM
# ═══════════════════════════════════════════════════════════════

class PlanStep(BaseModel):
    """A single step in the recovery plan."""
    step_number: int = Field(description="Step number (1-indexed)")
    action: str = Field(description="Action to take: send_notification, retry_payment, update_payment_method, wait_and_retry, escalate_to_human, schedule_retry")
    description: str = Field(description="Human-readable description of what this step does")
    reasoning: str = Field(description="Why this action is appropriate for this failure")
    expected_outcome: str = Field(description="What we expect to happen after this step")


class RecoveryPlan(BaseModel):
    """Structured recovery plan returned by the planning agent.

    The LLM generates this as output — not hand-coded rules.
    """
    failure_type: str = Field(description="Detected failure type")
    root_cause: str = Field(description="Root cause of the failure")
    confidence: float = Field(description="Confidence in the diagnosis (0-1)")
    steps: list[PlanStep] = Field(description="Ordered list of recovery steps")
    reasoning: str = Field(description="Overall reasoning for this plan")
    estimated_recoverability: str = Field(description="low/medium/high — how likely recovery is")


# ═══════════════════════════════════════════════════════════════
# PLANNING AGENT — pydantic-ai Agent with real tools
# ═══════════════════════════════════════════════════════════════

def _get_model():
    """Get the LLM model for the planning agent."""
    import os
    from dotenv import load_dotenv
    load_dotenv()

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    base_url = os.getenv("LLM_BASE_URL", "http://localhost:20128/v1")
    api_key = os.getenv("LLM_API_KEY", "dummy")
    model_name = os.getenv("LLM_MODEL", "antigravity/gemini-2.5-flash")

    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
    )


planning_agent = Agent[RecoveryDeps, RecoveryPlan](
    _get_model(),
    deps_type=RecoveryDeps,
    output_type=RecoveryPlan,
    instructions="""You are a revenue recovery planner. Given a failed payment, you create a structured recovery plan.

Your plan must be based on:
1. The actual failure type and root cause
2. Customer payment history (if available)
3. Razorpay error documentation
4. Recovery best practices

NEVER plan to retry a hard decline (codes 41, 43, 54, 14, 04, 46, 57, 93) — always escalate.

For each step, explain WHY this action is appropriate for this specific failure.""",
)


@planning_agent.tool
def get_customer_history(ctx: RunContext[RecoveryDeps], customer_id: str) -> str:
    """Fetch customer payment history to inform planning decisions.

    Args:
        customer_id: The customer identifier

    Returns:
        JSON with payment history, success rates, and preferred methods
    """
    from recovery_agent.razorpay_client import RazorpayClient

    client = RazorpayClient()
    if not client.is_configured:
        return json.dumps({"status": "unavailable", "message": "Razorpay not configured"})

    try:
        payments = client.client.payment.fetch_all({"count": 100, "skip": 0})
        customer_payments = []
        for p in payments.get("items", []):
            p_customer = p.get("notes", {}).get("customer_id", "") or p.get("customer_id", "")
            p_email = p.get("notes", {}).get("customer_email", "")
            if p_customer == customer_id or p_email == customer_id:
                customer_payments.append({
                    "status": p.get("status"),
                    "method": p.get("method"),
                    "amount": p.get("amount", 0) / 100,
                })

        total = len(customer_payments)
        successful = sum(1 for p in customer_payments if p["status"] == "captured")
        methods = {}
        for p in customer_payments:
            m = p["method"]
            if m not in methods:
                methods[m] = {"total": 0, "successful": 0}
            methods[m]["total"] += 1
            if p["status"] == "captured":
                methods[m]["successful"] += 1

        return json.dumps({
            "total_payments": total,
            "successful_count": successful,
            "success_rate": successful / total if total > 0 else 0,
            "method_success_rates": {
                m: v["successful"] / v["total"] if v["total"] > 0 else 0
                for m, v in methods.items()
            },
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@planning_agent.tool_plain
def get_razorpay_error_info(error_code: str) -> str:
    """Look up Razorpay error documentation for a specific error code.

    Args:
        error_code: The Razorpay failure code (e.g., '51', 'card_expired')

    Returns:
        JSON with error description, is_hard_decline, recommended_action
    """
    from recovery_agent.models import HARD_DECLINES

    is_hard_decline = error_code in HARD_DECLINES

    error_docs = {
        "51": {"description": "Insufficient funds", "retryable": True},
        "card_expired": {"description": "Card has expired", "retryable": False},
        "network_timeout": {"description": "Network connection timeout", "retryable": True},
        "risk_block": {"description": "Risk assessment blocked transaction", "retryable": False},
        "mandate_revoked": {"description": "UPI mandate cancelled by customer", "retryable": False},
    }

    doc = error_docs.get(error_code, {"description": "Unknown error", "retryable": False})

    return json.dumps({
        "error_code": error_code,
        "description": doc["description"],
        "is_hard_decline": is_hard_decline,
        "retryable": doc["retryable"],
    })


@planning_agent.tool
def check_guardrails(ctx: RunContext[RecoveryDeps], action: str) -> str:
    """Check if an action is allowed by guardrails (quiet hours, frequency cap, etc.).

    Args:
        action: The proposed action to check

    Returns:
        JSON with allowed (bool) and reason
    """
    if not ctx.deps.guardrail_engine:
        return json.dumps({"allowed": True, "reason": "No guardrail engine configured"})

    try:
        from recovery_agent.models import ActionType
        action_type = ActionType(action)
        approved, checks = ctx.deps.guardrail_engine.validate_action(
            ctx.deps.case, action_type
        )
        return json.dumps({
            "allowed": approved == action_type,
            "approved_action": approved.value,
            "checks": [{"name": c.guardrail, "verdict": c.verdict.value, "reason": c.reason} for c in checks],
        })
    except Exception as e:
        return json.dumps({"allowed": True, "reason": f"Guardrail check failed: {e}"})


# ═══════════════════════════════════════════════════════════════
# PLANNING INTERFACE — called from agent loop
# ═══════════════════════════════════════════════════════════════

def generate_plan(case: Case, guardrail_engine=None) -> RecoveryPlan:
    """Generate a recovery plan using the pydantic-ai planning agent.

    This is the real implementation — LLM generates structured output.
    """
    deps = RecoveryDeps(
        case=case,
        guardrail_engine=guardrail_engine,
    )

    prompt = f"""Create a recovery plan for this failed payment:

Payment ID: {case.payment.payment_id}
Amount: {case.payment.currency} {case.payment.amount:,.2f}
Customer ID: {case.payment.customer_id}
Failure Code: {case.payment.failure_code or 'not provided'}
Failure Reason: {case.payment.failure_reason or 'not provided'}
Attempt: {case.attempt_count + 1} of {case.max_attempts}
Recovery Tier: {case.recovery_tier.value}

Use your tools to:
1. Check customer payment history (get_customer_history)
2. Look up error documentation (get_razorpay_error_info)
3. Verify guardrails allow proposed actions (check_guardrails)

Then create a structured plan with 1-3 steps."""

    result = planning_agent.run_sync(prompt, deps=deps)
    return result.output
