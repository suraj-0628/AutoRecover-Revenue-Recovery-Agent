"""Customer communication engine — LLM-generated personalized recovery messages.

Replaces hardcoded templates with dynamic LLM generation.
The LLM drafts personalized messages based on failure type, customer persona,
tone preferences, and channel constraints.

Source: https://www.deeplearning.ai/courses/agentic-ai (Module 5)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recovery_agent.agent.llm_client import invoke_llm_json
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


MESSAGE_SYSTEM_PROMPT = """You are a Razorpay customer communication specialist.

Your task: Draft a personalized recovery message for a customer whose payment failed.
The message must be empathetic, clear, and drive action.

Rules:
- SMS: Max 160 characters, no subject, no formatting
- Email: Include subject line, professional but warm, max 200 words
- WhatsApp/In-App: Casual, conversational, can use Hinglish (Hindi + English mix)
- Tone must match the urgency: friendly (low urgency), supportive (medium), urgent (high)
- Always include a clear call-to-action when appropriate
- For Hinglish: mix Hindi and naturally, like "Aapka payment fail ho gaya hai"
- Never blame the customer
- Reference specific details (amount, failure reason) to feel personal

You must output EXACTLY this JSON format:
{
  "subject": "<email subject or null for SMS>",
  "body": "<the message body>",
  "tone": "<friendly|supportive|urgent>",
  "cta": "<call to action text or null>",
  "priority": <1 for first attempt, 2 for follow-up>
}"""


def _build_message_prompt(
    failure_type: FailureType,
    channel: str,
    customer_name: str,
    amount: float,
    attempt_count: int,
    card_last4: str,
    failure_reason: str,
    persona: str,
    language_tone: str,
) -> str:
    """Build the message generation prompt."""
    persona_context = ""
    if persona == "salary_dependent":
        persona_context = "\nCustomer persona: Salary-dependent — may have temporary cash flow issues. Be extra supportive."
    elif persona == "busy_executive":
        persona_context = "\nCustomer persona: Busy executive — value their time, be concise and direct."
    elif persona == "frustrated_subscriber":
        persona_context = "\nCustomer persona: Frustrated subscriber — they've had failed payments before. Be empathetic and offer solutions."
    elif persona == "b2b_ap":
        persona_context = "\nCustomer persona: Business account — formal tone, emphasize business continuity."

    attempt_context = ""
    if attempt_count > 1:
        attempt_context = f"\nThis is attempt #{attempt_count}. Previous attempts failed. Be more urgent but not pushy."

    return f"""Draft a {channel} recovery message:

FAILURE DETAILS:
  Type: {failure_type.value}
  Amount: INR {amount:,.2f}
  Failure reason: {failure_reason}
  Card last 4: {card_last4}

CUSTOMER:
  Name: {customer_name}
  Attempt: #{attempt_count}{persona_context}{attempt_context}

LANGUAGE TONE: {language_tone}
  - "hinglish": Mix Hindi and English naturally (e.g., "Aapka payment fail ho gaya hai")
  - "supportive": Warm, empathetic, reassuring
  - "formal": Professional, business-like
  - "urgent": Direct, action-oriented, time-sensitive

CHANNEL: {channel}
  {"SMS: Max 160 chars, no subject, no formatting" if channel == "sms" else "Email: Include subject, professional, max 200 words" if channel == "email" else "WhatsApp/In-App: Casual, conversational"}

Generate the message as JSON:"""


def generate_recovery_message(
    failure_type: FailureType,
    channel: str,
    customer_name: str = "Customer",
    amount: float = 0,
    card_last4: str = "XXXX",
    retry_hours: int = 24,
    attempt_count: int = 1,
    persona: str = "",
    language_tone: str = "supportive",
    failure_reason: str = "",
) -> RecoveryMessage | None:
    """Generate a personalized recovery message using LLM.

    Falls back to a minimal default message if LLM is unavailable.

    Args:
        failure_type: Root cause of payment failure
        channel: Communication channel (email, sms, push, in_app, whatsapp)
        customer_name: Customer's name
        amount: Payment amount
        card_last4: Last 4 digits of card
        retry_hours: Hours until next retry
        attempt_count: How many attempts made
        persona: Customer persona (salary_dependent, busy_executive, etc.)
        language_tone: Tone (hinglish, supportive, formal, urgent)
        failure_reason: Raw failure reason text

    Returns:
        RecoveryMessage or None if channel not supported
    """
    prompt = _build_message_prompt(
        failure_type, channel, customer_name, amount, attempt_count,
        card_last4, failure_reason, persona, language_tone,
    )

    result = invoke_llm_json(
        prompt=prompt,
        system=MESSAGE_SYSTEM_PROMPT,
        temperature=0.3,
        max_tokens=400,
    )

    if result is None:
        # Minimal fallback — no hardcoded templates
        return _fallback_message(failure_type, channel, amount)

    subject = result.get("subject")
    if channel == "sms":
        subject = None

    body = result.get("body", f"Payment of INR {amount:,.2f} failed. Please retry.")
    # Enforce SMS length limit
    if channel == "sms" and len(body) > 160:
        body = body[:157] + "..."

    return RecoveryMessage(
        channel=channel,
        subject=subject,
        body=body,
        tone=result.get("tone", "supportive"),
        cta=result.get("cta"),
        priority=result.get("priority", 1 if attempt_count == 1 else 2),
    )


def _fallback_message(
    failure_type: FailureType,
    channel: str,
    amount: float,
) -> RecoveryMessage:
    """Minimal fallback when LLM is unavailable."""
    if channel == "sms":
        return RecoveryMessage(
            channel="sms",
            subject=None,
            body=f"Razorpay: Payment of INR {amount:,.2f} failed. Please retry or update your payment method.",
            tone="supportive",
            cta=None,
            priority=1,
        )
    return RecoveryMessage(
        channel="email",
        subject=f"Payment of INR {amount:,.2f} — action needed",
        body=(
            f"Hi,\n\nYour payment of INR {amount:,.2f} could not be processed "
            f"due to {failure_type.value.replace('_', ' ')}. "
            f"Please try again or update your payment method.\n\nBest,\nRazorpay"
        ),
        tone="supportive",
        cta="Retry Payment",
        priority=1,
    )


def get_message_sequence(
    failure_type: FailureType,
    attempt_count: int,
    customer_name: str = "Customer",
    amount: float = 0,
    persona: str = "",
    failure_reason: str = "",
) -> list[dict[str, Any]]:
    """Get the recommended message sequence for a failure type.

    Uses LLM to generate each message in the sequence.
    Returns a list of messages to send across channels over time.
    """
    messages = []

    if attempt_count == 1:
        # First attempt: immediate SMS + email
        sms = generate_recovery_message(
            failure_type, "sms", customer_name, amount,
            persona=persona, failure_reason=failure_reason,
        )
        email = generate_recovery_message(
            failure_type, "email", customer_name, amount,
            persona=persona, failure_reason=failure_reason,
        )

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

    elif attempt_count == 2:
        # Second attempt: email reminder
        email = generate_recovery_message(
            failure_type, "email", customer_name, amount,
            attempt_count=2, persona=persona, failure_reason=failure_reason,
        )
        if email:
            messages.append({
                "delay_hours": 24,
                "channel": "email",
                "subject": email.subject,
                "message": email.body,
                "tone": email.tone,
            })

    elif attempt_count == 3:
        # Third attempt: final SMS warning
        sms = generate_recovery_message(
            failure_type, "sms", customer_name, amount,
            attempt_count=3, persona=persona, language_tone="urgent",
            failure_reason=failure_reason,
        )
        if sms:
            messages.append({
                "delay_hours": 48,
                "channel": "sms",
                "message": sms.body,
                "tone": "urgent",
            })

    return messages
