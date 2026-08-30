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

    @staticmethod
    def check_error(result: dict[str, Any]) -> dict[str, Any]:
        """BUG FIX: Check if a Razorpay SDK result contains an error.

        Raises RuntimeError if the result contains an error field.
        This prevents silent error propagation where callers assume success.
        """
        if "error" in result:
            raise RuntimeError(f"Razorpay API error: {result['error']}")
        return result

    def create_order(
        self,
        amount: float,
        currency: str = "INR",
        receipt: str | None = None,
        notes: dict | None = None,
    ) -> dict[str, Any]:
        """Create a Razorpay order.

        Amount in paise (INR). For INR 500, pass 50000.
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
                "_simulated": True,  # BUG FIX: Mark as simulated for downstream detection
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
    ) -> dict[str, Any]:
        """Create a Razorpay Payment Link."""
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

    def fetch_all_payments(
        self,
        from_timestamp: int | None = None,
        to_timestamp: int | None = None,
        count: int = 10,
        skip: int = 0,
    ) -> dict[str, Any]:
        """Fetch all payments with optional time range."""
        if not self.is_configured:
            return {"error": "Razorpay not configured"}

        try:
            params = {"count": count, "skip": skip}
            if from_timestamp:
                params["from"] = from_timestamp
            if to_timestamp:
                params["to"] = to_timestamp

            payments = self.client.payment.all(params)
            return payments
        except (BadRequestError, GatewayError, ServerError) as e:
            return {"error": str(e)}

    def capture_payment(self, payment_id: str, amount: float) -> dict[str, Any]:
        """Capture an authorized payment.

        Amount in paise.
        """
        if not self.is_configured:
            import time
            clean_id = payment_id[-8:] if len(payment_id) >= 8 else payment_id
            return {
                "id": f"pay_rzp_{clean_id}",
                "entity": "payment",
                "amount": int(amount * 100),
                "currency": "INR",
                "status": "captured",
                "order_id": f"order_rzp_{clean_id}",
                "captured": True,
                "fee": int(amount * 2),
                "tax": int(amount * 0.36),
                "error_code": None,
                "error_description": None,
                "created_at": int(time.time()),
            }

        try:
            result = self.client.payment.capture(
                payment_id,
                int(amount * 100),
            )
            return result
        except Exception as e:
            import time
            clean_id = payment_id[-8:] if len(payment_id) >= 8 else payment_id
            return {
                "id": f"pay_rzp_{clean_id}",
                "entity": "payment",
                "amount": int(amount * 100) if amount else 0,
                "currency": "INR",
                "status": "failed",
                "order_id": f"order_rzp_{clean_id}",
                "captured": False,
                "fee": 0,
                "tax": 0,
                "error_code": "CAPTURE_FAILED",
                "error_description": str(e),
                "created_at": int(time.time()),
            }

    def create_refund(
        self,
        payment_id: str,
        amount: float | None = None,
        notes: dict | None = None,
    ) -> dict[str, Any]:
        """Create a refund for a captured payment."""
        if not self.is_configured:
            return {"error": "Razorpay not configured"}

        try:
            params: dict[str, Any] = {}
            if amount is not None and amount > 0:
                params["amount"] = int(amount * 100)
            # BUG FIX: amount=0.0 or None means full refund (Razorpay default)
            if notes:
                params["notes"] = notes

            refund = self.client.payment.refund(payment_id, params)
            return refund
        except (BadRequestError, GatewayError, ServerError) as e:
            return {"error": str(e)}

    def create_subscription(
        self,
        plan_id: str,
        customer_id: str | None = None,
        total_count: int = 12,
        start_at: int | None = None,
    ) -> dict[str, Any]:
        """Create a subscription."""
        if not self.is_configured:
            return {"error": "Razorpay not configured"}

        try:
            params: dict[str, Any] = {
                "plan_id": plan_id,
                "total_count": total_count,
            }
            if customer_id:
                params["customer_id"] = customer_id
            if start_at:
                params["start_at"] = start_at

            subscription = self.client.subscription.create(params)
            return subscription
        except (BadRequestError, GatewayError, ServerError) as e:
            return {"error": str(e)}

    def cancel_subscription(self, subscription_id: str) -> dict[str, Any]:
        """Cancel a subscription."""
        if not self.is_configured:
            return {"error": "Razorpay not configured"}

        try:
            result = self.client.subscription.cancel(subscription_id)
            return result
        except (BadRequestError, GatewayError, ServerError) as e:
            return {"error": str(e)}

    # ═══════════════════════════════════════════════════════════════════
    # Async wrappers — offload blocking SDK calls to thread pool
    # ═══════════════════════════════════════════════════════════════════

    async def create_order_async(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Non-blocking create_order via thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.create_order, *args, **kwargs))

    async def create_payment_link_async(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Non-blocking create_payment_link via thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.create_payment_link, *args, **kwargs))

    async def fetch_payment_async(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Non-blocking fetch_payment via thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.fetch_payment, *args, **kwargs))

    async def fetch_all_payments_async(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Non-blocking fetch_all_payments via thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.fetch_all_payments, *args, **kwargs))

    async def capture_payment_async(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Non-blocking capture_payment via thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.capture_payment, *args, **kwargs))

    async def create_refund_async(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Non-blocking create_refund via thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.create_refund, *args, **kwargs))

    async def create_subscription_async(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Non-blocking create_subscription via thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.create_subscription, *args, **kwargs))

    async def cancel_subscription_async(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Non-blocking cancel_subscription via thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.cancel_subscription, *args, **kwargs))


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
