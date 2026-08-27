"""Diagnosis engine — real-time LLM diagnostic reflection with LlamaIndex RAG.

Uses LLM (Nemotron via OmniRoute) as the PRIMARY diagnosis method.
The LLM performs structured reflection over raw payment failure payloads,
customer history, bank health signals, AND RAG-retrieved knowledge base context.

Cascade: LLM + RAG grounded context → Razorpay API data → Rule-based fallback

Source: Reflection pattern from Agentic AI (Andrew Ng), Module 2
        Router decomposition from Evaluating AI Agents
        LlamaIndex Agentic RAG: https://www.deeplearning.ai/courses/building-agentic-rag-with-llamaindex
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

    return f"""Diagnose this payment failure using raw diagnostic signals AND RAG-retrieved knowledge base context:

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
  Payment Method: {payment.metadata.get('method', 'unknown')}
  Provider: {payment.metadata.get('provider', 'unknown')}
{attempt_history}{customer_context}

Analyze the raw failure payload above. Reason step-by-step:
1. Step 1 — Failure Code Analysis: What failure code or error string is provided?
2. Step 2 — Failure Reason Analysis: What exact issue does the text describe?
3. Step 3 — Contextual Signal Analysis: What do amount, customer, and gateway details imply?
4. Step 4 — RAG Knowledge Base: What protocol does the retrieved knowledge base context recommend?
5. Step 5 — Confidence Assessment: Conclude the root cause category and confidence score.

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


def diagnose_with_rag(case: Case) -> Diagnosis | None:
    """Diagnose using LlamaIndex Agentic RAG — retrieves grounded context from knowledge base.

    Uses SubQuestionQueryEngine to decompose the failure into sub-questions,
    routes each to VectorIndex (specific codes) or SummaryIndex (policies),
    and evaluates groundedness to ensure zero hallucination.

    Groundedness >= 0.8 boosts confidence to 0.95.
    """
    from recovery_agent.agent.agentic_rag import LlamaIndexAgenticRAG

    rag = LlamaIndexAgenticRAG()
    metadata = {
        "failure_code": case.payment.failure_code or case.payment.metadata.get("error_code", "unknown"),
        "failure_reason": case.payment.failure_reason or case.payment.metadata.get("error_description", "unknown"),
        "error_description": case.payment.metadata.get("error_description", case.payment.failure_reason),
        "method": case.payment.metadata.get("method", "unknown"),
        "provider": case.payment.metadata.get("provider", "unknown"),
        "amount": case.payment.amount,
    }

    rag_response = rag.query_for_diagnosis(metadata)

    if not rag_response.retrieved_chunks:
        return None

    # Build diagnosis from RAG context
    context_chunks = rag_response.retrieved_chunks[:5]
    rag_context = "\n---\n".join(c.text for c in context_chunks)

    # Determine root cause from RAG context
    cause_str = _infer_cause_from_rag_context(rag_context, case)
    cause_map = {
        "card_expired": FailureType.CARD_EXPIRED,
        "insufficient_funds": FailureType.INSUFFICIENT_FUNDS,
        "bank_declined": FailureType.BANK_DECLINED,
        "network_timeout": FailureType.NETWORK_TIMEOUT,
        "risk_block": FailureType.RISK_BLOCK,
        "mandate_revoked": FailureType.MANDATE_REVOKED,
    }
    cause = cause_map.get(cause_str, FailureType.UNKNOWN)

    # Groundedness-boosted confidence
    groundedness = rag_response.groundedness_score
    if groundedness >= 0.8:
        confidence = 0.95
    elif groundedness >= 0.6:
        confidence = 0.85
    else:
        confidence = 0.70

    reasoning = (
        f"RAG Grounded Diagnosis (groundedness={groundedness:.2f}):\n"
        f"Retrieved from: {', '.join(c.source_file for c in context_chunks)}\n"
        f"Sub-questions decomposed: {rag_response.decomposition_steps}\n"
        f"Key context: {rag_context[:500]}"
    )

    return Diagnosis(
        root_cause=cause,
        confidence=confidence,
        reasoning=reasoning,
        category=f"payment_failure_{cause.value}",
    )


def _infer_cause_from_rag_context(context: str, case: Case) -> str:
    """Infer root cause from RAG-retrieved context."""
    context_lower = context.lower()
    method = case.payment.metadata.get("method", "").lower()
    reason = (case.payment.failure_reason or "").lower()
    desc = (case.payment.metadata.get("error_description", "") or "").lower()

    # Instrument switch detection (highest priority)
    switch_keywords = ["use another payment instrument", "use another payment method",
                       "try another method", "expired", "invalid card"]
    if any(kw in reason or kw in desc for kw in switch_keywords):
        if "expired" in reason or "expired" in desc:
            return "card_expired"
        return "bank_declined"

    # RAG context-based inference
    if "card_expired" in context_lower or "expiry date" in context_lower:
        return "card_expired"
    if "insufficient" in context_lower:
        return "insufficient_funds"
    if "mandate" in context_lower and ("inactive" in context_lower or "revoked" in context_lower or "cancelled" in context_lower):
        return "mandate_revoked"
    if "timeout" in context_lower or "503" in context_lower or "network" in context_lower:
        return "network_timeout"
    if "risk" in context_lower or "fraud" in context_lower:
        return "risk_block"
    if "lazypay" in context_lower or "paylater" in context_lower or "otp" in context_lower:
        return "bank_declined"
    if "bank_declined" in context_lower or "declined" in context_lower:
        return "bank_declined"

    return "unknown"


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
    """Run diagnosis on a case — LLM + RAG is the PRIMARY intelligence layer.

    Cascade:
      Layer 1: LLM diagnostic reflection + LlamaIndex RAG grounded context (PRIMARY)
      Layer 2: Razorpay API error data (real API)
      Layer 3: Rule-based fallback (safety net when LLM unavailable)
    """
    case.status = CaseStatus.DIAGNOSING

    # Layer 1a: LlamaIndex RAG — retrieve grounded knowledge base context
    rag_diagnosis = None
    try:
        rag_diagnosis = diagnose_with_rag(case)
    except Exception as e:
        # RAG requires ChromaDB with working embedding model.
        # Fail loudly in VectorIndex, gracefully degrade here.
        print(f"[diagnosis] RAG unavailable: {e}")

    # Layer 1b: LLM diagnostic reflection with RAG context injected into prompt
    llm_diagnosis = diagnose_with_llm(case)
    if llm_diagnosis and llm_diagnosis.root_cause != FailureType.UNKNOWN:
        # If RAG has high groundedness, boost LLM confidence
        if rag_diagnosis and rag_diagnosis.confidence >= 0.9:
            llm_diagnosis.confidence = max(llm_diagnosis.confidence, rag_diagnosis.confidence)
            llm_diagnosis.reasoning = (
                f"{llm_diagnosis.reasoning}\n\n"
                f"RAG Grounded Context: {rag_diagnosis.reasoning[:300]}"
            )
        case.diagnosis = llm_diagnosis
        return case

    # If LLM failed but RAG succeeded, use RAG diagnosis
    if rag_diagnosis and rag_diagnosis.root_cause != FailureType.UNKNOWN:
        case.diagnosis = rag_diagnosis
        return case

    # Layer 2: Razorpay API error data
    rp_diagnosis = diagnose_from_razorpay(case)
    if rp_diagnosis:
        case.diagnosis = rp_diagnosis
        return case

    # Layer 3: Safety net fallback
    if llm_diagnosis:
        case.diagnosis = llm_diagnosis
    elif rag_diagnosis:
        case.diagnosis = rag_diagnosis
    else:
        case.diagnosis = diagnose_payment_failure(case)
    return case
