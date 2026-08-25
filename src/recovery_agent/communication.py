"""Customer communication engine — personalized recovery messages.

Generates context-aware messages for different failure types
and customer segments.

Source: https://www.deeplearning.ai/courses/agentic-ai (Module 5)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recovery_agent.models import FailureType


@dataclass
class RecoveryMessage:
    """A personalized recovery message."""
    channel: str  # sms, email, push, in_app
    subject: str | None
    body: str
    tone: str  # friendly, urgent, supportive
    cta: str | None  # call to action
    priority: int  # 1=highest


# Message templates by failure type and channel
MESSAGES: dict[FailureType, dict[str, list[RecoveryMessage]]] = {
    FailureType.CARD_EXPIRED: {
        "email": [
            RecoveryMessage(
                channel="email",
                subject="Action needed: Your card on file has expired",
                body=(
                    "Hi {name},\n\n"
                    "We noticed your {card_last4} card ending in {card_last4} has expired. "
                    "To continue your subscription without interruption, please update your "
                    "payment method.\n\n"
                    "It only takes 30 seconds:\n{update_link}\n\n"
                    "Need help? Reply to this email or call us at 1800-XXX-XXXX.\n\n"
                    "Best,\nRazorpay Recovery Team"
                ),
                tone="supportive",
                cta="Update Payment Method",
                priority=1,
            ),
            RecoveryMessage(
                channel="email",
                subject="Your payment method needs updating",
                body=(
                    "Hi {name},\n\n"
                    "Your card {card_last4} has expired. We'll retry your payment of "
                    "INR {amount} in 24 hours, but updating now ensures no interruption.\n\n"
                    "{update_link}\n\n"
                    "Thanks,\nRazorpay"
                ),
                tone="friendly",
                cta="Update Now",
                priority=2,
            ),
        ],
        "sms": [
            RecoveryMessage(
                channel="sms",
                subject=None,
                body=(
                    "Razorpay: Your card {card_last4} has expired. "
                    "Update now to avoid service interruption: {update_link}"
                ),
                tone="urgent",
                cta=None,
                priority=1,
            ),
        ],
    },
    FailureType.INSUFFICIENT_FUNDS: {
        "email": [
            RecoveryMessage(
                channel="email",
                subject="Payment of INR {amount} couldn't go through",
                body=(
                    "Hi {name},\n\n"
                    "Your payment of INR {amount} couldn't be processed due to "
                    "insufficient funds. We'll automatically retry in {retry_hours} hours.\n\n"
                    "No action needed from you — we'll handle it.\n\n"
                    "If you have questions, we're here to help.\n\n"
                    "Best,\nRazorpay"
                ),
                tone="supportive",
                cta=None,
                priority=1,
            ),
        ],
        "sms": [
            RecoveryMessage(
                channel="sms",
                subject=None,
                body=(
                    "Razorpay: Payment of INR {amount} failed due to insufficient funds. "
                    "We'll retry automatically. No action needed."
                ),
                tone="supportive",
                cta=None,
                priority=1,
            ),
        ],
    },
    FailureType.BANK_DECLINED: {
        "email": [
            RecoveryMessage(
                channel="email",
                subject="Payment issue — bank declined the transaction",
                body=(
                    "Hi {name},\n\n"
                    "Your bank declined the payment of INR {amount}. This can happen "
                    "for several reasons:\n"
                    "- Daily transaction limit reached\n"
                    "- International transaction not enabled\n"
                    "- Bank security hold\n\n"
                    "Try these steps:\n"
                    "1. Check with your bank\n"
                    "2. Use a different payment method\n"
                    "3. We'll retry in 24 hours\n\n"
                    "{update_link}\n\n"
                    "Best,\nRazorpay"
                ),
                tone="supportive",
                cta="Try Different Method",
                priority=1,
            ),
        ],
        "sms": [
            RecoveryMessage(
                channel="sms",
                subject=None,
                body=(
                    "Razorpay: Payment of INR {amount} declined by bank. "
                    "Try another method: {update_link} or we'll retry in 24h."
                ),
                tone="friendly",
                cta=None,
                priority=1,
            ),
        ],
    },
    FailureType.NETWORK_TIMEOUT: {
        "email": [
            RecoveryMessage(
                channel="email",
                subject="Payment interrupted — retrying now",
                body=(
                    "Hi {name},\n\n"
                    "Your payment of INR {amount} was interrupted due to a network "
                    "issue. Don't worry — we're retrying automatically.\n\n"
                    "If it doesn't go through this time, we'll keep trying for "
                    "the next 24 hours.\n\n"
                    "Best,\nRazorpay"
                ),
                tone="reassuring",
                cta=None,
                priority=1,
            ),
        ],
        "sms": [
            RecoveryMessage(
                channel="sms",
                subject=None,
                body=(
                    "Razorpay: Payment of INR {amount} interrupted. "
                    "Retrying automatically — no action needed."
                ),
                tone="reassuring",
                cta=None,
                priority=1,
            ),
        ],
    },
    FailureType.RISK_BLOCK: {
        "email": [
            RecoveryMessage(
                channel="email",
                subject="Action needed: Payment verification required",
                body=(
                    "Hi {name},\n\n"
                    "We need to verify your payment of INR {amount} for security "
                    "purposes. This is to protect your account.\n\n"
                    "Please verify your identity:\n{verify_link}\n\n"
                    "Once verified, your payment will be processed immediately.\n\n"
                    "Best,\nRazorpay Security Team"
                ),
                tone="formal",
                cta="Verify Now",
                priority=1,
            ),
        ],
        "sms": [
            RecoveryMessage(
                channel="sms",
                subject=None,
                body=(
                    "Razorpay: Payment verification needed for INR {amount}. "
                    "Verify here: {verify_link}"
                ),
                tone="urgent",
                cta=None,
                priority=1,
            ),
        ],
    },
    FailureType.MANDATE_REVOKED: {
        "email": [
            RecoveryMessage(
                channel="email",
                subject="Your autopay has been disabled",
                body=(
                    "Hi {name},\n\n"
                    "Your autopay for {service_name} has been disabled. "
                    "To continue your subscription without interruption, "
                    "please re-authorize autopay.\n\n"
                    "{reauthorize_link}\n\n"
                    "Or you can pay manually each month.\n\n"
                    "Best,\nRazorpay"
                ),
                tone="supportive",
                cta="Re-authorize Autopay",
                priority=1,
            ),
        ],
        "sms": [
            RecoveryMessage(
                channel="sms",
                subject=None,
                body=(
                    "Razorpay: Your autopay is disabled. "
                    "Re-authorize to continue: {reauthorize_link}"
                ),
                tone="friendly",
                cta=None,
                priority=1,
            ),
        ],
    },
}

# Fallback message for unknown failure types
FALLBACK_MESSAGE = RecoveryMessage(
    channel="email",
    subject="Payment issue — action needed",
    body=(
        "Hi {name},\n\n"
        "Your payment of INR {amount} couldn't be processed. "
        "Please try again or update your payment method.\n\n"
        "{update_link}\n\n"
        "Best,\nRazorpay"
    ),
    tone="supportive",
    cta="Try Again",
    priority=1,
)


def generate_recovery_message(
    failure_type: FailureType,
    channel: str,
    customer_name: str = "Customer",
    amount: float = 0,
    card_last4: str = "XXXX",
    retry_hours: int = 24,
    attempt_count: int = 1,
) -> RecoveryMessage | None:
    """Generate a personalized recovery message.

    Args:
        failure_type: Root cause of payment failure
        channel: Communication channel (email, sms, push, in_app)
        customer_name: Customer's name
        amount: Payment amount
        card_last4: Last 4 digits of card
        retry_hours: Hours until next retry
        attempt_count: How many attempts made

    Returns:
        RecoveryMessage or None if no message for this channel
    """
    templates = MESSAGES.get(failure_type, {})
    channel_messages = templates.get(channel, [])

    if not channel_messages:
        if channel == "email":
            return _customize_message(FALLBACK_MESSAGE, customer_name, amount, card_last4, retry_hours)
        return None

    # Pick the best message based on attempt count
    # First attempt: urgent/friendly, subsequent: supportive
    if attempt_count == 1:
        # Pick highest priority message
        msg = min(channel_messages, key=lambda m: m.priority)
    else:
        # Pick lower priority (more supportive) message
        msg = max(channel_messages, key=lambda m: m.priority)

    return _customize_message(msg, customer_name, amount, card_last4, retry_hours)


def _customize_message(
    msg: RecoveryMessage,
    customer_name: str,
    amount: float,
    card_last4: str,
    retry_hours: int,
) -> RecoveryMessage:
    """Fill template variables in a message."""
    replacements = {
        "{name}": customer_name,
        "{amount}": f"{amount:,.2f}",
        "{card_last4}": card_last4,
        "{retry_hours}": str(retry_hours),
        "{update_link}": "https://dashboard.razorpay.com/update-payment",
        "{verify_link}": "https://dashboard.razorpay.com/verify",
        "{reauthorize_link}": "https://dashboard.razorpay.com/autopay",
        "{service_name}": "your subscription",
    }

    body = msg.body
    subject = msg.subject
    for key, value in replacements.items():
        body = body.replace(key, value)
        if subject:
            subject = subject.replace(key, value)

    return RecoveryMessage(
        channel=msg.channel,
        subject=subject,
        body=body,
        tone=msg.tone,
        cta=msg.cta,
        priority=msg.priority,
    )


def get_message_sequence(
    failure_type: FailureType,
    attempt_count: int,
    customer_name: str = "Customer",
    amount: float = 0,
) -> list[dict[str, Any]]:
    """Get the recommended message sequence for a failure type.

    Returns a list of messages to send across channels over time.
    """
    messages = []

    # First attempt: immediate SMS + email
    if attempt_count == 1:
        sms = generate_recovery_message(failure_type, "sms", customer_name, amount)
        email = generate_recovery_message(failure_type, "email", customer_name, amount)

        if sms:
            messages.append({
                "delay_hours": 0,
                "channel": "sms",
                "message": sms.body,
                "tone": sms.tone,
            })
        if email:
            messages.append({
                "delay_hours": 0,
                "channel": "email",
                "subject": email.subject,
                "message": email.body,
                "tone": email.tone,
            })

    # Second attempt: email reminder
    elif attempt_count == 2:
        email = generate_recovery_message(failure_type, "email", customer_name, amount, attempt_count=2)
        if email:
            messages.append({
                "delay_hours": 24,
                "channel": "email",
                "subject": email.subject,
                "message": email.body,
                "tone": email.tone,
            })

    # Third attempt: final warning
    elif attempt_count == 3:
        sms = generate_recovery_message(failure_type, "sms", customer_name, amount, attempt_count=3)
        if sms:
            messages.append({
                "delay_hours": 48,
                "channel": "sms",
                "message": sms.body,
                "tone": "urgent",
            })

    return messages
