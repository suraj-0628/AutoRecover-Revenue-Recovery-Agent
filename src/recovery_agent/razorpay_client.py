"""Razorpay client wrapper — real API integration for test mode.

Source: Razorpay Python SDK docs
        https://razorpay.com/docs/payments/server-integration/python/integration-steps/
"""
from __future__ import annotations

import asyncio
import os
from functools import partial
from typing import Any

import razorpay
from razorpay.errors import BadRequestError, GatewayError, ServerError

from recovery_agent.models import FailureType


# Razorpay error codes -> our failure types
# Source: https://razorpay.com/docs/errors/payments/list/
RAZORPAY_ERROR_MAP: dict[str, FailureType] = {
    # Card errors
    "card_expired": FailureType.CARD_EXPIRED,
    "invalid_card": FailureType.CARD_EXPIRED,
    "expired_card": FailureType.CARD_EXPIRED,
    "do_not_honor": FailureType.BANK_DECLINED,
    "generic_decline": FailureType.BANK_DECLINED,
    "card_declined": FailureType.BANK_DECLINED,
    "insufficient_funds": FailureType.INSUFFICIENT_FUNDS,
    "low_balance": FailureType.INSUFFICIENT_FUNDS,
    # Network/timeout errors
    "network_error": FailureType.NETWORK_TIMEOUT,
    "timeout": FailureType.NETWORK_TIMEOUT,
    "gateway_timeout": FailureType.NETWORK_TIMEOUT,
    # Risk/fraud errors
    "risk_check_failed": FailureType.RISK_BLOCK,
    "fraud_suspected": FailureType.RISK_BLOCK,
    "transaction_not_permitted": FailureType.RISK_BLOCK,
    # UPI errors
    "upi_payer_declined": FailureType.BANK_DECLINED,
    "upi_auth_failure": FailureType.BANK_DECLINED,
    # Mandate errors
    "mandate_inactive": FailureType.MANDATE_REVOKED,
    "mandate_already_exists": FailureType.MANDATE_REVOKED,
}

# Razorpay payment statuses that indicate failure
FAILED_STATUSES = {"failed", "refunded", "created"}

# Razorpay error steps -> more context
ERROR_STEP_MAP: dict[str, str] = {
    "payment_authorization": "Bank authorization step",
    "payment_capture": "Payment capture step",
    "payment_authentication": "Customer authentication step (3DS/OTP)",
    "payment_processing": "Payment processing step",
}


#: Creating a payment link spends one of an account's thirty, for its lifetime.
#: Cancelling does not give it back and rotating the API key does not reset it,
#: so the only way to get more is a new signup.
#:
#: A promise not to create them during development is not a control. Twenty-eight
#: of one account's thirty were spent by verification scripts that redirected the
#: state store to a temp directory but left the Razorpay client pointed at the
#: live key — every one of them a real link on the user's real account, against
#: an explicit instruction not to.
#:
#: So writes are refused unless the process was started as a service. `start.sh`
#: exports RAZORPAY_WRITES_OK=1; an ad-hoc script does not have it and therefore
#: cannot spend the quota, whatever it calls.
def _writes_allowed() -> bool:
    return os.getenv("RAZORPAY_WRITES_OK", "").strip().lower() in ("1", "true", "yes", "on")


class _WritesDisabled(RuntimeError):
    pass


class RazorpayClient:
    """Wrapper around Razorpay SDK for payment operations.

    Source: Razorpay Python SDK
    https://github.com/razorpay/razorpay-python
    """

    def __init__(
        self,
        key_id: str | None = None,
        key_secret: str | None = None,
    ):
        self.key_id = key_id or os.getenv("RAZORPAY_KEY_ID", "")
        self.key_secret = key_secret or os.getenv("RAZORPAY_KEY_SECRET", "")

        if self.key_id and self.key_secret:
            self.client = razorpay.Client(auth=(self.key_id, self.key_secret))
        else:
            self.client = None

    @property
    def is_configured(self) -> bool:
        return self.client is not None and bool(self.key_id and self.key_secret)

    def create_order(
        self,
        amount: float,
        currency: str = "INR",
        receipt: str | None = None,
        notes: dict | None = None,
    ) -> dict[str, Any]:
        """Create a Razorpay order.

        `amount` is in RUPEES, despite the older docstring here claiming paise —
        the body multiplies by 100. Do NOT pre-convert.
        """
        if not self.is_configured:
            import time
            clean_rcpt = receipt or f"rcpt_{os.urandom(4).hex()}"
            return {
                "id": f"order_rzp_{clean_rcpt[-8:]}",
                "entity": "order",
                "amount": int(amount * 100),
                "amount_paid": 0,
                "amount_due": int(amount * 100),
                "currency": currency,
                "receipt": clean_rcpt,
                "status": "created",
                "attempts": 1,
                "notes": notes or {"recovery_agent": "AutoRecover_v2"},
                "created_at": int(time.time()),
            }

        try:
            order = self.client.order.create({
                "amount": int(amount * 100),  # Convert to paise
                "currency": currency,
                "receipt": receipt or f"rcpt_{os.urandom(4).hex()}",
                "notes": notes or {},
            })
            return order
        except (BadRequestError, GatewayError, ServerError) as e:
            return {"error": str(e)}

    def fetch_order(self, order_id: str) -> dict[str, Any]:
        """Fetch order details from Razorpay.

        Returns full order object including status, amount_paid, amount_due.
        """
        if not self.is_configured:
            import time
            return {
                "id": order_id,
                "entity": "order",
                "amount": 0,
                "amount_paid": 0,
                "amount_due": 0,
                "currency": "INR",
                "status": "created",
                "created_at": int(time.time()),
            }

        try:
            order = self.client.order.fetch(order_id)
            return order
        except (BadRequestError, GatewayError, ServerError) as e:
            return {"error": str(e)}

    def create_payment_link(
        self,
        amount: float,
        currency: str = "INR",
        customer: dict | None = None,
        notes: dict | None = None,
        expire_by: int | None = None,
    ) -> dict[str, Any]:
        """Create a Razorpay Payment Link.

        `amount` is in RUPEES. This method converts to paise itself — do NOT
        pre-convert. Passing paise here bills the customer 100x the debt, which
        shipped and reached the live account before it was caught.
        """
        if not _writes_allowed():
            raise _WritesDisabled(
                "refusing to create a payment link: RAZORPAY_WRITES_OK is not set, "
                "so this process was not started as a service. Each link is one of "
                "thirty for the account's lifetime and cannot be reclaimed."
            )

        if not self.is_configured:
            import time
            ref_id = f"plink_rzp_{os.urandom(4).hex()}"
            return {
                "id": ref_id,
                "entity": "payment_link",
                "amount": int(amount * 100),
                "currency": currency,
                "status": "created",
                "short_url": f"https://rzp.io/i/{ref_id[-8:]}",
                "customer": customer or {},
                "notes": notes or {"recovery_agent": "AutoRecover_v2"},
                "created_at": int(time.time()),
            }

        try:
            params = {
                "amount": int(amount * 100),
                "currency": currency,
                "customer": customer or {},
                "notes": notes or {},
            }
            if expire_by:
                params["expire_by"] = int(expire_by)
            return self.client.payment_link.create(params)
        except Exception as e:
            return {"error": str(e)}

    def fetch_payment(self, payment_id: str) -> dict[str, Any]:
        """Fetch payment details from Razorpay.

        Returns full payment object including error_code, error_reason, etc.
        """
        if not self.is_configured:
            import time
            return {
                "id": payment_id,
                "entity": "payment",
                "amount": 0,
                "currency": "INR",
                "status": "failed",
                "method": "card",
                "error_code": "gateway_timeout",
                "error_reason": "Payment gateway timeout during authorization",
                "error_step": "payment_authorization",
                "created_at": int(time.time()),
            }

        try:
            payment = self.client.payment.fetch(payment_id)
            return payment
        except (BadRequestError, GatewayError, ServerError) as e:
            return {"error": str(e)}


def diagnose_from_razorpay_error(payment_data: dict[str, Any]) -> tuple[FailureType, str]:
    """Map Razorpay payment error to our FailureType.

    Uses error_code, error_reason, and error_step from Razorpay response.

    Source: Razorpay error structure
    https://razorpay.com/docs/errors/payments/list/
    """
    error_code = (payment_data.get("error_code") or "").lower()
    error_reason = (payment_data.get("error_reason") or "").lower()
    error_step = (payment_data.get("error_step") or "").lower()

    # Try exact error_code match
    if error_code in RAZORPAY_ERROR_MAP:
        return RAZORPAY_ERROR_MAP[error_code], f"Error code: {error_code}"

    # Try error_reason keywords
    reason_keywords = {
        "expired": FailureType.CARD_EXPIRED,
        "insufficient": FailureType.INSUFFICIENT_FUNDS,
        "balance": FailureType.INSUFFICIENT_FUNDS,
        "declined": FailureType.BANK_DECLINED,
        "blocked": FailureType.RISK_BLOCK,
        "fraud": FailureType.RISK_BLOCK,
        "risk": FailureType.RISK_BLOCK,
        "timeout": FailureType.NETWORK_TIMEOUT,
        "network": FailureType.NETWORK_TIMEOUT,
        "mandate": FailureType.MANDATE_REVOKED,
        "revoked": FailureType.MANDATE_REVOKED,
    }

    for keyword, cause in reason_keywords.items():
        if keyword in error_reason:
            return cause, f"Reason keyword '{keyword}' in '{error_reason}'"

    # Check error_step for context
    if "authentication" in error_step:
        return FailureType.BANK_DECLINED, f"Auth failure at step: {error_step}"
    if "timeout" in error_step or "network" in error_step:
        return FailureType.NETWORK_TIMEOUT, f"Network issue at step: {error_step}"

    return FailureType.UNKNOWN, f"No match: code={error_code}, reason={error_reason}, step={error_step}"
