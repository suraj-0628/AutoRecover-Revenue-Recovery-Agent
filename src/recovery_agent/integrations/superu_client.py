"""SuperU AI Voice Agent integration for payment recovery calls.

SuperU is an AI voice calling platform used by Razorpay in production
for abandoned cart recovery and payment failure outreach.

API Reference: https://docs.superu.ai
Free Tier: 100 AI calls at https://workspace.superu.ai

CRITICAL: Never invoke during tests. Voice calls cost real credits.
Pytest sets PYTEST_CURRENT_TEST automatically — this blocks all test invocations.
"""
from __future__ import annotations

import os
import logging
import sys

import requests
from dotenv import load_dotenv
from typing import Any

load_dotenv()

logger = logging.getLogger(__name__)

SUPERU_API_URL = "https://voip-middlware.superu.ai/campaign/outbound/create_call/superu"


def _is_test_environment() -> bool:
    """Detect if running under pytest or any test runner."""
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True
    if os.getenv("SUPERU_DISABLE_IN_TESTS", "").strip() == "1":
        return True
    if "pytest" in sys.modules:
        return True
    if any(arg.endswith("pytest") or arg.endswith("py.test") for arg in sys.argv):
        return True
    return False


class SuperUClient:
    """Client for SuperU AI outbound voice calling API."""

    def __init__(self):
        self.api_key = os.getenv("SUPERU_API_KEY", "")
        self.assistant_id = os.getenv("SUPERU_ASSISTANT_ID", "")
        self.from_phone = os.getenv("SUPERU_FROM_PHONE", "")
        self._enabled = bool(self.api_key and self.assistant_id and self.from_phone)

    @property
    def is_enabled(self) -> bool:
        """Check if SuperU is configured with valid credentials."""
        return self._enabled

    def initiate_recovery_call(
        self,
        payment_id: str,
        customer_name: str,
        customer_phone: str,
        amount: float,
        failure_reason: str = "",
        recovery_link: str = "",
    ) -> dict[str, Any]:
        """Initiate an AI voice call to recover a failed payment.

        The AI voice agent will:
        1. Call the customer
        2. Explain the payment failure empathetically
        3. Offer alternative payment methods
        4. Send a Razorpay payment link during the call

        Args:
            payment_id: Razorpay payment ID
            customer_name: Customer's name for personalization
            customer_phone: Customer's phone number (with country code)
            amount: Payment amount in INR
            failure_reason: Human-readable failure reason
            recovery_link: Razorpay payment link URL

        Returns:
            dict with call status, campaign_id, and metadata
        """
        # STRICT: Never make real API calls during tests — credits cost real money
        if _is_test_environment():
            logger.warning("[SuperU] BLOCKED: Test environment detected — refusing to make voice call for %s", payment_id)
            return {
                "status": "skipped",
                "reason": "test_environment",
                "detail": "Voice calls are blocked during tests to preserve credits. Only real customer interactions invoke SuperU.",
            }

        if not self._enabled:
            missing = []
            if not self.api_key:
                missing.append("SUPERU_API_KEY")
            if not self.assistant_id:
                missing.append("SUPERU_ASSISTANT_ID")
            if not self.from_phone:
                missing.append("SUPERU_FROM_PHONE")
            logger.warning("SuperU not configured — missing: %s", ", ".join(missing))
            return {
                "status": "skipped",
                "reason": "superu_not_configured",
                "detail": f"Missing env vars: {', '.join(missing)}. Purchase a phone number at https://workspace.superu.ai to enable.",
            }

        # Normalize phone number — ensure it has country code
        phone = customer_phone.strip()
        if not phone.startswith("+"):
            phone = "+91" + phone  # Default to India

        # Build payload per SuperU API spec
        payload = {
            "assistant_id": self.assistant_id,
            "to": phone,
            "from": self.from_phone,
            "campaign_id": f"recovery_{payment_id}",
            "customer_name": customer_name,
            "customer_id": payment_id,
        }

        # Add optional variables for the agent script
        if recovery_link:
            payload["variable_values"] = {
                "recovery_link": recovery_link,
                "amount": f"{amount:.0f}",
                "failure_reason": failure_reason,
            }

        headers = {
            "superU-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            logger.info(
                "[SuperU] Initiating voice call for %s to %s (INR %.2f)",
                payment_id, phone, amount,
            )
            response = requests.post(
                SUPERU_API_URL,
                json=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            logger.info("[SuperU] Call initiated successfully: %s", result)

            return {
                "status": "call_initiated",
                "call_uuid": result.get("call_uuid", result.get("call_id", "")),
                "campaign_id": result.get("campaign_id", f"recovery_{payment_id}"),
                "phone": phone,
                "payment_id": payment_id,
                "response": result,
            }

        except requests.exceptions.HTTPError as e:
            error_body = ""
            try:
                error_body = e.response.json()
            except Exception:
                error_body = e.response.text if e.response else str(e)
            logger.error("[SuperU] Voice call failed for %s: %s — %s", payment_id, e, error_body)
            return {
                "status": "call_failed",
                "error": str(e),
                "detail": error_body,
                "payment_id": payment_id,
            }

        except requests.exceptions.RequestException as e:
            logger.error("[SuperU] Voice call failed for %s: %s", payment_id, e)
            return {
                "status": "call_failed",
                "error": str(e),
                "payment_id": payment_id,
            }


# Singleton for reuse across the application
_client: SuperUClient | None = None


def get_superu_client() -> SuperUClient:
    """Get or create the singleton SuperU client."""
    global _client
    if _client is None:
        _client = SuperUClient()
    return _client
