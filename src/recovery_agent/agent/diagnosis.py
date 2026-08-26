"""Diagnosis engine — real-time LLM diagnostic reflection.

Uses LLM (Nemotron via OmniRoute) as the PRIMARY diagnosis method.
The LLM performs structured reflection over raw payment failure payloads,
customer history, and bank health signals to classify root causes.

Cascade: Razorpay API data → LLM reflection → UNKNOWN fallback

Source: Reflection pattern from Agentic AI (Andrew Ng), Module 2
        Router decomposition from Evaluating AI Agents
"""
from __future__ import annotations

from recovery_agent.agent.llm_client import invoke_llm_json
from recovery_agent.models import (
    Case,
    CaseStatus,
    Diagnosis,
    FailureType,
)
from recovery_agent.razorpay_client import RazorpayClient, diagnose_from_razorpay_error

# Quick-check mapping for Razorpay API error codes (Layer 1)
FAILURE_CODE_MAP: dict[str, FailureType] = {
    "card_expired": FailureType.CARD_EXPIRED,
    "insufficient_funds": FailureType.INSUFFICIENT_FUNDS,
    "do_not_honor": FailureType.BANK_DECLINED,
    "generic_decline": FailureType.BANK_DECLINED,
    "network_error": FailureType.NETWORK_TIMEOUT,
    "network_timeout": FailureType.NETWORK_TIMEOUT,
    "gateway_timeout": FailureType.NETWORK_TIMEOUT,
    "timeout": FailureType.NETWORK_TIMEOUT,
    "risk_check_failed": FailureType.RISK_BLOCK,
    "fraud_suspected": FailureType.RISK_BLOCK,
    "mandate_inactive": FailureType.MANDATE_REVOKED,
    "mandate_revoked": FailureType.MANDATE_REVOKED,
}

DIAGNOSIS_SYSTEM_PROMPT = """You are an expert payment failure diagnostician for Razorpay.

Your task: Analyze a raw payment failure payload and classify the root cause.
You MUST reason step-by-step through the evidence before concluding:
- Step 1 — Failure code & raw error message analysis
- Step 2 — Error step (initiation, authentication, authorization) & error source (customer, bank, gateway, risk)
- Step 3 — Contextual signals (transaction amount, customer history, bank health)
- Step 4 — Final confidence assessment & root cause classification

Available failure categories:
- card_expired: Card on file has expired (expiry date in the past)
- insufficient_funds: Customer's account lacks sufficient balance
- bank_declined: Bank rejected the transaction (limit, security, intl block)
- network_timeout: Gateway timeout, connection drop, HTTP 5xx, 3DS OTP timeout
- risk_block: Fraud/risk system blocked the payment
- mandate_revoked: UPI autopay mandate was cancelled or expired
- unknown: Insufficient evidence to classify

CRITICAL: Start your response immediately with the character { . Do NOT output any preamble or markdown explanation before the JSON object.

You must output EXACTLY this JSON format:
{
  "root_cause": "<one of the 7 categories above>",
  "confidence": <0.70 to 0.98>,
  "reasoning": "Step 1 — Failure code analysis: ... Step 2 — Failure reason analysis: ... Step 3 — Contextual signal analysis: ... Step 4 — Confidence assessment: ..."
}"""


def _build_diagnosis_prompt(case: Case) -> str:
    """Build the diagnostic prompt from raw payment failure data."""
    payment = case.payment
    attempt_history = ""
    if case.attempts:
        attempt_history = f"\nPrevious attempt history ({len(case.attempts)} attempts):\n"
        for i, a in enumerate(case.attempts[-3:], 1):
            attempt_history += (
                f"  {i}. Action: {a.action_type.value}, "
                f"Result: {a.result}, "
                f"Detail: {a.action_details.get('detail', 'none')}\n"
            )

    customer_context = ""
    if case.payment.metadata.get("customer_name"):
        customer_context += f"\nCustomer: {payment.metadata['customer_name']}"
    if payment.metadata.get("salary_window"):
        customer_context += f"\nSalary window active: {payment.metadata['salary_window']}"
    if payment.metadata.get("promise_to_pay"):
        customer_context += f"\nPromise-to-pay: {payment.metadata['promise_to_pay']}"
    if payment.metadata.get("bank_health"):
        customer_context += f"\nBank health signal: {payment.metadata['bank_health']}"

    code_str = payment.failure_code or payment.metadata.get('error_code') or 'unmapped_code'
    reason_str = payment.failure_reason or payment.metadata.get('error_description') or 'unmapped_reason'

    return f"""Diagnose this payment failure using raw diagnostic signals:

FAILURE PAYMENT DATA:
  Payment ID: {payment.payment_id}
  Customer ID: {payment.customer_id}
  Amount: INR {payment.amount:,.2f}
  Failure Reason: {reason_str}
  Failure Code: {code_str}
  Error Source: {payment.metadata.get('error_source', 'unknown')}
  Error Step: {payment.metadata.get('error_step', 'unknown')}
  Error Description: {payment.metadata.get('error_description', reason_str)}
  Currency: {payment.currency}
{attempt_history}{customer_context}

Analyze the raw failure payload above. Reason step-by-step:
1. Step 1 — Failure Code Analysis: What failure code or error string is provided?
2. Step 2 — Failure Reason Analysis: What exact issue does the text describe?
3. Step 3 — Contextual Signal Analysis: What do amount, customer, and gateway details imply?
4. Step 4 — Confidence Assessment: Conclude the root cause category and confidence score.

Output your diagnosis strictly as JSON:"""


def diagnose_with_llm(case: Case) -> Diagnosis | None:
    """Use LLM reflection to diagnose the root cause of a payment failure.

    This is the PRIMARY diagnosis method — not a fallback.
    The LLM reasons step-by-step over raw failure payload, customer history,
    and bank health signals.

    Source: Router decomposition pattern from Evaluating AI Agents
    https://www.deeplearning.ai/courses/evaluating-ai-agents
    """
    prompt = _build_diagnosis_prompt(case)
    result = invoke_llm_json(
        prompt=prompt,
        system=DIAGNOSIS_SYSTEM_PROMPT,
        temperature=0,
        max_tokens=1024,
    )

    if result is None:
        return None

    # Parse root cause
    cause_str = result.get("root_cause", "unknown").lower().strip()
    cause_map = {
        "card_expired": FailureType.CARD_EXPIRED,
        "insufficient_funds": FailureType.INSUFFICIENT_FUNDS,
        "bank_declined": FailureType.BANK_DECLINED,
        "network_timeout": FailureType.NETWORK_TIMEOUT,
        "risk_block": FailureType.RISK_BLOCK,
        "mandate_revoked": FailureType.MANDATE_REVOKED,
        "unknown": FailureType.UNKNOWN,
    }
    cause = cause_map.get(cause_str, FailureType.UNKNOWN)

    # Parse confidence
    confidence = float(result.get("confidence", 0.7))
    confidence = max(0.0, min(1.0, confidence))

    # Extract reasoning
    reasoning = result.get("reasoning", "LLM diagnostic reflection")
    if isinstance(reasoning, list):
        reasoning = " ".join(str(r) for r in reasoning)

    return Diagnosis(
        root_cause=cause,
        confidence=confidence,
        reasoning=reasoning,
        category=f"payment_failure_{cause.value}",
    )


def diagnose_from_razorpay(case: Case) -> Diagnosis | None:
    """Diagnose using Razorpay payment error data if available.

    Highest confidence — real API data.

    Source: Razorpay error structure
    https://razorpay.com/docs/errors/payments/list/
    """
    rp_error = case.payment.metadata.get("razorpay_error")
    if not rp_error:
        rp_payment_id = case.payment.metadata.get("razorpay_payment_id")
        if rp_payment_id:
            client = RazorpayClient()
            if client.is_configured:
                payment_data = client.fetch_payment(rp_payment_id)
                if "error" not in payment_data:
                    rp_error = payment_data

    if not rp_error:
        return None

    cause, reasoning = diagnose_from_razorpay_error(rp_error)

    return Diagnosis(
        root_cause=cause,
        confidence=0.95,
        reasoning=reasoning,
        category=f"payment_failure_{cause.value}",
    )


def diagnose_payment_failure(case: Case) -> Diagnosis:
    """Rule-based fallback diagnosis — used only when LLM is unavailable.

    Preserves the original logic as a safety net.
    """
    payment = case.payment
    root_cause = FailureType.UNKNOWN
    confidence = 0.3
    reasoning_parts = []

    # Try failure code mapping
    if payment.failure_code:
        mapped = FAILURE_CODE_MAP.get(payment.failure_code.lower())
        if mapped:
            root_cause = mapped
            confidence = 0.9
            reasoning_parts.append(
                f"Failure code '{payment.failure_code}' maps to {mapped.value}"
            )

    # Razorpay Knowledge Base normalizer fallback
    if root_cause == FailureType.UNKNOWN:
        from recovery_agent.razorpay_knowledge_base import normalize_razorpay_failure
        raw_signal = f"{payment.failure_code or ''} {payment.failure_reason or ''} {payment.metadata.get('error_description', '')}"
        norm = normalize_razorpay_failure(raw_signal)
        root_cause = norm["failure_type"]
        confidence = 0.90
        reasoning_parts.append(f"Razorpay Official Error Catalog matched '{norm['error_code']}' -> {root_cause.value}")

    if root_cause == FailureType.UNKNOWN:
        if payment.amount > 100000:
            root_cause = FailureType.RISK_BLOCK
            confidence = 0.4
            reasoning_parts.append(f"High amount ({payment.amount}) suggests risk block")
        else:
            reasoning_parts.append("No clear signal — classified as unknown")

    reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Rule-based fallback"

    return Diagnosis(
        root_cause=root_cause,
        confidence=confidence,
        reasoning=reasoning,
        category=f"payment_failure_{root_cause.value}",
    )


def run_diagnosis(case: Case) -> Case:
    """Run diagnosis on a case — LLM Diagnostic Reflection is Layer 1.

    Cascade:
      Layer 1: LLM diagnostic reflection (PRIMARY intelligence — reasons step-by-step over raw payload)
      Layer 2: Razorpay API error data
      Layer 3: Rule-based fallback (safety net when LLM unavailable)
    """
    case.status = CaseStatus.DIAGNOSING

    # Layer 1: LLM diagnostic reflection (PRIMARY METHOD — ALWAYS RUN FIRST)
    llm_diagnosis = diagnose_with_llm(case)
    if llm_diagnosis and llm_diagnosis.root_cause != FailureType.UNKNOWN:
        case.diagnosis = llm_diagnosis
        return case

    # Layer 2: Razorpay API error data
    rp_diagnosis = diagnose_from_razorpay(case)
    if rp_diagnosis:
        case.diagnosis = rp_diagnosis
        return case

    # Layer 3: Safety net fallback
    if llm_diagnosis:
        case.diagnosis = llm_diagnosis
    else:
        case.diagnosis = diagnose_payment_failure(case)
    return case
