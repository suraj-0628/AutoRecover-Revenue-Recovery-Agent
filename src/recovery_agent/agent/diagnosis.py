"""Diagnosis engine — classifies payment failure root causes.

Uses LLM (via OmniRoute local API) for nuanced classification,
with Razorpay error mapping and rule-based fallback.

Source: Reflection pattern from Agentic AI (Andrew Ng), Module 2
        Router decomposition from Evaluating AI Agents
"""
from __future__ import annotations

import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from recovery_agent.models import (
    Case,
    CaseStatus,
    Diagnosis,
    FailureType,
)
from recovery_agent.razorpay_client import RazorpayClient, diagnose_from_razorpay_error

# Mapping of Razorpay failure codes to our failure types
FAILURE_CODE_MAP: dict[str, FailureType] = {
    "card_expired": FailureType.CARD_EXPIRED,
    "insufficient_funds": FailureType.INSUFFICIENT_FUNDS,
    "do_not_honor": FailureType.BANK_DECLINED,
    "generic_decline": FailureType.BANK_DECLINED,
    "network_error": FailureType.NETWORK_TIMEOUT,
    "timeout": FailureType.NETWORK_TIMEOUT,
    "risk_check_failed": FailureType.RISK_BLOCK,
    "fraud_suspected": FailureType.RISK_BLOCK,
    "mandate_inactive": FailureType.MANDATE_REVOKED,
    "mandate_revoked": FailureType.MANDATE_REVOKED,
}

# Keywords in failure reasons that hint at root cause
REASON_KEYWORDS: dict[str, FailureType] = {
    "expired": FailureType.CARD_EXPIRED,
    "expir": FailureType.CARD_EXPIRED,
    "insufficient": FailureType.INSUFFICIENT_FUNDS,
    "not enough": FailureType.INSUFFICIENT_FUNDS,
    "balance": FailureType.INSUFFICIENT_FUNDS,
    "declined": FailureType.BANK_DECLINED,
    "rejected": FailureType.BANK_DECLINED,
    "blocked": FailureType.RISK_BLOCK,
    "fraud": FailureType.RISK_BLOCK,
    "risk": FailureType.RISK_BLOCK,
    "timeout": FailureType.NETWORK_TIMEOUT,
    "network": FailureType.NETWORK_TIMEOUT,
    "connection": FailureType.NETWORK_TIMEOUT,
    "mandate": FailureType.MANDATE_REVOKED,
    "revoked": FailureType.MANDATE_REVOKED,
    "cancelled": FailureType.MANDATE_REVOKED,
}


def diagnose_payment_failure(case: Case) -> Diagnosis:
    """Classify the root cause of a payment failure.

    Uses failure code first, then keyword matching on reason text.
    Falls back to UNKNOWN if nothing matches.

    Source: Router decomposition pattern from Evaluating AI Agents
    https://www.deeplearning.ai/courses/evaluating-ai-agents
    """
    payment = case.payment
    root_cause = FailureType.UNKNOWN
    confidence = 0.3
    reasoning_parts = []

    # Try failure code mapping first (highest confidence)
    if payment.failure_code:
        mapped = FAILURE_CODE_MAP.get(payment.failure_code.lower())
        if mapped:
            root_cause = mapped
            confidence = 0.9
            reasoning_parts.append(
                f"Failure code '{payment.failure_code}' maps to {mapped.value}"
            )

    # Try keyword matching on failure reason
    if root_cause == FailureType.UNKNOWN and payment.failure_reason:
        reason_lower = payment.failure_reason.lower()
        for keyword, cause in REASON_KEYWORDS.items():
            if keyword in reason_lower:
                root_cause = cause
                confidence = 0.7
                reasoning_parts.append(
                    f"Keyword '{keyword}' in reason '{payment.failure_reason}' -> {cause.value}"
                )
                break

    # If still unknown, check if amount gives hints
    if root_cause == FailureType.UNKNOWN:
        if payment.amount > 100000:  # > 1 lakh
            root_cause = FailureType.RISK_BLOCK
            confidence = 0.4
            reasoning_parts.append(f"High amount ({payment.amount}) suggests risk block")
        else:
            reasoning_parts.append("No clear signal from failure code or reason text")

    category = f"payment_failure_{root_cause.value}"
    reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Classified by default"

    return Diagnosis(
        root_cause=root_cause,
        confidence=confidence,
        reasoning=reasoning,
        category=category,
    )


def run_diagnosis(case: Case) -> Case:
    """Run diagnosis on a case and update its state.

    Tries: Razorpay error mapping -> LLM -> rule-based fallback

    Source: Reflection pattern — agent reviews before acting
    https://www.deeplearning.ai/courses/agentic-ai (Module 2)
    """
    case.status = CaseStatus.DIAGNOSING

    # Try Razorpay error mapping first (highest confidence for real API data)
    rp_diagnosis = diagnose_from_razorpay(case)
    if rp_diagnosis:
        case.diagnosis = rp_diagnosis
        return case

    # Try LLM diagnosis
    llm_diagnosis = diagnose_with_llm(case)
    if llm_diagnosis:
        case.diagnosis = llm_diagnosis
    else:
        # Fallback to rule-based
        case.diagnosis = diagnose_payment_failure(case)

    return case


def diagnose_from_razorpay(case: Case) -> Diagnosis | None:
    """Diagnose using Razorpay payment error data if available.

    Source: Razorpay error structure
    https://razorpay.com/docs/errors/payments/list/
    """
    # Check if we have Razorpay error data in payment metadata
    rp_error = case.payment.metadata.get("razorpay_error")
    if not rp_error:
        # If payment has a Razorpay payment_id, try fetching from API
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
        confidence=0.95,  # High confidence for real API data
        reasoning=reasoning,
        category=f"payment_failure_{cause.value}",
    )


def diagnose_with_llm(case: Case) -> Diagnosis | None:
    """Use LLM to diagnose the root cause of a payment failure.

    Source: Router decomposition pattern from Evaluating AI Agents
    https://www.deeplearning.ai/courses/evaluating-ai-agents
    """
    # Check if LLM is configured
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:20128/v1")
    model = os.getenv("LLM_MODEL", "oc/nemotron-3-ultra-free")
    api_key = os.getenv("LLM_API_KEY", "dummy")

    try:
        llm = ChatOpenAI(
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=0,
            max_tokens=100,
        )

        prompt = f"""Classify this payment failure as exactly one word from this list: card_expired, insufficient_funds, bank_declined, network_timeout, risk_block, mandate_revoked, unknown.
Failure: {case.payment.failure_reason}
Answer:"""

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip().lower()

        # Map response to FailureType
        cause_map = {
            "card_expired": FailureType.CARD_EXPIRED,
            "insufficient_funds": FailureType.INSUFFICIENT_FUNDS,
            "bank_declined": FailureType.BANK_DECLINED,
            "network_timeout": FailureType.NETWORK_TIMEOUT,
            "risk_block": FailureType.RISK_BLOCK,
            "mandate_revoked": FailureType.MANDATE_REVOKED,
            "unknown": FailureType.UNKNOWN,
        }

        # Extract the first word that matches a category
        words = content.split()
        cause = FailureType.UNKNOWN
        for word in words:
            word = word.strip(".,;:!?\"'`")
            if word in cause_map:
                cause = cause_map[word]
                break
        confidence = 0.85 if cause != FailureType.UNKNOWN else 0.4

        return Diagnosis(
            root_cause=cause,
            confidence=confidence,
            reasoning=f"LLM classified as {cause.value}",
            category=f"payment_failure_{cause.value}",
        )

    except Exception:
        # LLM unavailable — fall back to rules
        return None
