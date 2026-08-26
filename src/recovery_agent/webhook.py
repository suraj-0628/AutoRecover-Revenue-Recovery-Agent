"""Webhook listener for Razorpay payment events.

Real-time failure detection via Razorpay webhooks.
Source: https://razorpay.com/docs/payments/webhooks/

Usage:
    python -m recovery_agent.main webhook
"""
from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

from flask import Flask, request, jsonify

from recovery_agent.agent import RecoveryAgent
from recovery_agent.models import Case, PaymentEvent

app = Flask(__name__)

WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")
active_cases: dict[str, Case] = {}


def verify_signature(payload: bytes, signature: str) -> bool:
    """Verify Razorpay webhook HMAC SHA256 signature."""
    if not WEBHOOK_SECRET:
        return True
    expected = hmac.new(WEBHOOK_SECRET.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def process_webhook_payload(payload: dict) -> dict:
    """Process a raw webhook event dictionary programmatically."""
    event = payload.get("event", "")
    data = payload.get("payload", {})

    if event == "payment.failed":
        response, status = _handle_payment_failed(data)
        return {"event": event, "status": "processed", "status_code": status}
    elif event == "payment.captured":
        response, status = _handle_payment_captured(data)
        return {"event": event, "status": "processed", "status_code": status}
    elif "dispute" in event:
        return {"event": event, "status": "handled", "status_code": 200}

    return {"event": event, "status": "ignored", "status_code": 200}


@app.route("/webhook", methods=["POST"])
def handle_webhook():
    signature = request.headers.get("X-Razorpay-Signature", "")
    if not verify_signature(request.data, signature):
        return jsonify({"error": "Invalid signature"}), 400

    payload = request.json
    res = process_webhook_payload(payload)
    return jsonify(res), res.get("status_code", 200)


def _handle_payment_failed(data: dict) -> tuple[Any, int]:
    payment = data.get("payment", {}).get("entity", {})
    payment_id = payment.get("id", "")
    amount = payment.get("amount", 0) / 100
    error_code = payment.get("error_code", "")
    error_reason = payment.get("error_reason", "")
    error_step = payment.get("error_step", "")
    contact = payment.get("contact", "")
    email = payment.get("email", "")
    notes = payment.get("notes", {})
    customer_id = notes.get("customer_id", payment.get("customer_id", f"cust_{payment_id}"))

    print(f"[webhook] Payment failed: {payment_id} — {error_code}: {error_reason}")

    event = PaymentEvent(
        event_type="payment_failed",
        payment_id=payment_id,
        customer_id=customer_id,
        amount=amount,
        currency=payment.get("currency", "INR"),
        status="failed",
        failure_reason=f"{error_code}: {error_reason}",
        failure_code=error_code or "BAD_REQUEST_PAYMENT_FAILED",
        metadata={
            "order_id": payment.get("order_id", ""),
            "error_code": error_code,
            "error_reason": error_reason,
            "error_step": error_step,
            "contact": contact,
            "email": email,
            "razorpay_payment_id": payment_id,
            "razorpay_error": {
                "error_code": error_code,
                "error_reason": error_reason,
                "error_step": error_step,
            },
        },
    )

    case = Case(payment=event)
    agent = RecoveryAgent()
    final_case = agent.run(case)

    active_cases[payment_id] = final_case

    return {
        "status": "processed",
        "case_id": final_case.id,
        "recovered": final_case.recovered,
        "recovered_amount": final_case.recovered_amount,
    }, 200


def _handle_payment_captured(data: dict) -> tuple[Any, int]:
    payment = data.get("payment", {}).get("entity", {})
    payment_id = payment.get("id", "")
    amount = payment.get("amount", 0) / 100

    print(f"[webhook] Payment captured: {payment_id} — INR {amount:,.2f}")

    if payment_id in active_cases:
        case = active_cases[payment_id]
        case.recovered = True
        case.recovered_amount = amount
        del active_cases[payment_id]

    return {"status": "processed"}, 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "active_cases": len(active_cases)})


@app.route("/cases", methods=["GET"])
def list_cases():
    cases = [{"payment_id": k, "case_id": v.id, "recovered": v.recovered} for k, v in active_cases.items()]
    return jsonify({"cases": cases, "total": len(cases)})


def main():
    port = int(os.getenv("WEBHOOK_PORT", "5000"))
    print(f"Webhook listener: http://localhost:{port}/webhook")
    print(f"Health check: http://localhost:{port}/health")
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
