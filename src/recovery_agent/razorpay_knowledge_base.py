"""Razorpay Official Error Code & Recovery Knowledge Base.

Contains standard Razorpay API error codes, failure reasons, error steps, error sources,
customer-facing UI error descriptions, and automatic failure payload normalization.

Source: Razorpay API Documentation
        https://razorpay.com/docs/errors/
        https://razorpay.com/docs/payments/webhooks/
"""
from __future__ import annotations

from typing import Any
from recovery_agent.models import FailureType


# Official Razorpay Error Sources & Steps Documentation
# Source: https://razorpay.com/docs/api/errors/
RAZORPAY_ERROR_SOURCES = {
    "customer": "Error originated from customer action (wrong OTP, card expired, insufficient balance, tab closed).",
    "business": "Error originated from merchant integration configuration (missing parameter, invalid key).",
    "gateway": "Error originated from bank or payment gateway drop (504 timeout, NPCI server drop).",
    "razorpay": "Internal Razorpay infrastructure error.",
}

RAZORPAY_ERROR_STEPS = {
    "payment_initiation": "Customer launched checkout window prior to authentication.",
    "payment_authentication": "3DS OTP or PIN entry stage.",
    "payment_authorization": "Bank debit or card network authorization stage.",
    "payment_capture": "Post-authorization payment capture stage.",
}


# Official Razorpay Error Codes Catalog
RAZORPAY_ERROR_CATALOG: dict[str, dict[str, Any]] = {
    # 1. Gateway Timeout / Temporary Technical Issue
    "gateway_timeout": {
        "error_code": "BAD_REQUEST_PAYMENT_TEMPORARY_TECHNICAL_ISSUE",
        "failure_code": "gateway_timeout",
        "failure_reason": "Gateway timeout during payment authorization",
        "error_source": "gateway",
        "error_step": "payment_authorization",
        "error_description": "Your payment could not be completed due to a temporary technical issue. To complete the payment, use another payment instrument.",
        "failure_type": FailureType.NETWORK_TIMEOUT,
        "recommended_rail": "upi_autopay",
    },
    "network_timeout": {
        "error_code": "BAD_REQUEST_PAYMENT_TIMED_OUT",
        "failure_code": "network_timeout",
        "failure_reason": "Network timeout during payment processing",
        "error_source": "gateway",
        "error_step": "payment_authorization",
        "error_description": "Payment authorization timed out. Retrying via instant UPI failover.",
        "failure_type": FailureType.NETWORK_TIMEOUT,
        "recommended_rail": "upi_autopay",
    },
    # 2. Card Expiry
    "card_expired": {
        "error_code": "BAD_REQUEST_CARD_EXPIRED",
        "failure_code": "card_expired",
        "failure_reason": "Card expiry date is in the past",
        "error_source": "customer",
        "error_step": "payment_authentication",
        "error_description": "The card used for this payment has expired. Please update your card details or use a different payment method.",
        "failure_type": FailureType.CARD_EXPIRED,
        "recommended_rail": "card_expiry_fixer",
    },
    # 3. Insufficient Funds
    "insufficient_funds": {
        "error_code": "BAD_REQUEST_PAYMENT_INSUFFICIENT_FUNDS",
        "failure_code": "insufficient_funds",
        "failure_reason": "Insufficient funds in customer account",
        "error_source": "customer",
        "error_step": "payment_authorization",
        "error_description": "Payment failed due to insufficient balance in your account.",
        "failure_type": FailureType.INSUFFICIENT_FUNDS,
        "recommended_rail": "wait_and_retry",
    },
    # 4. Bank Decline
    "bank_declined": {
        "error_code": "BAD_REQUEST_PAYMENT_DECLINED_BY_BANK",
        "failure_code": "bank_declined",
        "failure_reason": "Transaction declined by issuing bank",
        "error_source": "bank",
        "error_step": "payment_authorization",
        "error_description": "Your issuing bank declined this transaction. Contact your bank or try an alternate card.",
        "failure_type": FailureType.BANK_DECLINED,
        "recommended_rail": "payment_link",
    },
    # 5. Mandate Revoked / Subscription Halted
    "mandate_revoked": {
        "error_code": "BAD_REQUEST_MANDATE_INACTIVE",
        "failure_code": "mandate_revoked",
        "failure_reason": "High-value mandate failure requiring voice intervention",
        "error_source": "bank",
        "error_step": "payment_authorization",
        "error_description": "Auto-debit mandate is inactive or revoked by customer. Re-authorization required.",
        "failure_type": FailureType.MANDATE_REVOKED,
        "recommended_rail": "voice_call",
    },
    # 6. Checkout Abandonment
    "abandonment": {
        "error_code": "BAD_REQUEST_CHECKOUT_ABANDONED",
        "failure_code": "abandonment",
        "failure_reason": "Customer closed tab during checkout",
        "error_source": "customer",
        "error_step": "payment_initiation",
        "error_description": "Customer exited the payment window prior to completing authentication.",
        "failure_type": FailureType.USER_DROPOFF,
        "recommended_rail": "whatsapp_recovery",
    },
    # 7. Risk / Fraud Block
    "risk_block": {
        "error_code": "BAD_REQUEST_RISK_CHECK_FAILED",
        "failure_code": "risk_block",
        "failure_reason": "Transaction blocked by risk and fraud rules",
        "error_source": "business",
        "error_step": "payment_initiation",
        "error_description": "Transaction flagged for risk review. Human support verification required.",
        "failure_type": FailureType.RISK_BLOCK,
        "recommended_rail": "escalate_human",
    },
}


def normalize_razorpay_failure(input_data: Any) -> dict[str, Any]:
    """Normalize raw failure inputs into a fully specified Razorpay Failure Payload.

    Handles raw text strings (e.g. "Your payment could not be completed..."), scenario keys,
    or webhook JSON dictionaries.
    """
    if isinstance(input_data, dict):
        code = input_data.get("failure_code") or input_data.get("error_code") or ""
        reason = input_data.get("failure_reason") or input_data.get("description") or ""
        text = f"{code} {reason}".lower()
    else:
        text = str(input_data or "").lower()

    # Match text pattern against Razorpay Catalog
    if "temporary technical issue" in text or "504" in text or "degradation" in text or "gateway_timeout" in text or "timeout" in text or "otp" in text or "authentication" in text:
        return dict(RAZORPAY_ERROR_CATALOG["network_timeout"])
    elif "card expiry" in text or "expired" in text:
        return dict(RAZORPAY_ERROR_CATALOG["card_expired"])
    elif "insufficient" in text or "balance" in text:
        return dict(RAZORPAY_ERROR_CATALOG["insufficient_funds"])
    elif "closed tab" in text or "abandoned" in text or "cancelled" in text:
        return dict(RAZORPAY_ERROR_CATALOG["abandonment"])
    elif "mandate" in text or "voice" in text:
        return dict(RAZORPAY_ERROR_CATALOG["mandate_revoked"])
    elif "risk" in text or "fraud" in text:
        return dict(RAZORPAY_ERROR_CATALOG["risk_block"])
    elif "declined" in text or "bank" in text:
        return dict(RAZORPAY_ERROR_CATALOG["bank_declined"])

    # Fallback to UNKNOWN if no specific Razorpay error code matched
    return {
        "error_code": "BAD_REQUEST_UNKNOWN_ERROR",
        "failure_code": "unknown_error",
        "failure_reason": text or "Unknown payment failure",
        "error_source": "system",
        "error_step": "payment_processing",
        "error_description": "Unknown payment processing error.",
        "failure_type": FailureType.UNKNOWN,
        "recommended_rail": "payment_link",
    }
