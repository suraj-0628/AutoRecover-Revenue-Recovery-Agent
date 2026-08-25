"""Smart retry scheduler — predict optimal retry timing.

Uses failure type, customer behavior patterns, and time-of-day
to schedule retries at the highest-probability windows.

Source: https://razorpay.com/docs/payments/subscriptions/retry/
        https://www.deeplearning.ai/courses/ai-agents-in-langgraph (Module 5)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from recovery_agent.models import FailureType


@dataclass
class RetryWindow:
    """A recommended retry window."""
    start_time: datetime
    end_time: datetime
    confidence: float
    reason: str


# Optimal retry windows by failure type
# Based on Razorpay retry patterns and payment industry data
# Source: https://razorpay.com/docs/payments/subscriptions/retry/
RETRY_WINDOWS: dict[FailureType, list[RetryWindow]] = {
    FailureType.INSUFFICIENT_FUNDS: [
        # Salary credit days (1st and 15th of month)
        # Peak: 10am-2pm on salary days
        RetryWindow(
            start_time=datetime.now(timezone.utc).replace(hour=10, minute=0),
            end_time=datetime.now(timezone.utc).replace(hour=14, minute=0),
            confidence=0.65,
            reason="Peak salary credit window (10am-2pm)",
        ),
        # End of month — people get paid
        RetryWindow(
            start_time=datetime.now(timezone.utc).replace(hour=9, minute=0),
            end_time=datetime.now(timezone.utc).replace(hour=18, minute=0),
            confidence=0.55,
            reason="End-of-month salary credit window",
        ),
    ],
    FailureType.NETWORK_TIMEOUT: [
        # Network issues are transient — retry immediately
        RetryWindow(
            start_time=datetime.now(timezone.utc) + timedelta(minutes=5),
            end_time=datetime.now(timezone.utc) + timedelta(minutes=15),
            confidence=0.70,
            reason="Immediate retry — network issues are transient",
        ),
        # Off-peak hours for better connectivity
        RetryWindow(
            start_time=datetime.now(timezone.utc).replace(hour=22, minute=0),
            end_time=datetime.now(timezone.utc).replace(hour=6, minute=0),
            confidence=0.50,
            reason="Off-peak hours — less network congestion",
        ),
    ],
    FailureType.BANK_DECLINED: [
        # Banks process at night — retry next morning
        RetryWindow(
            start_time=datetime.now(timezone.utc).replace(hour=10, minute=0) + timedelta(days=1),
            end_time=datetime.now(timezone.utc).replace(hour=14, minute=0) + timedelta(days=1),
            confidence=0.55,
            reason="Next morning — bank systems refreshed",
        ),
    ],
    FailureType.CARD_EXPIRED: [
        # Card update takes 24-48h to process
        RetryWindow(
            start_time=datetime.now(timezone.utc) + timedelta(hours=24),
            end_time=datetime.now(timezone.utc) + timedelta(hours=48),
            confidence=0.40,
            reason="Wait for customer to update card details",
        ),
    ],
    FailureType.RISK_BLOCK: [
        # Risk blocks need manual review — 24h minimum
        RetryWindow(
            start_time=datetime.now(timezone.utc) + timedelta(hours=24),
            end_time=datetime.now(timezone.utc) + timedelta(hours=72),
            confidence=0.30,
            reason="Risk block — wait for manual review",
        ),
    ],
    FailureType.MANDATE_REVOKED: [
        # Mandate revocation needs customer action — 48h
        RetryWindow(
            start_time=datetime.now(timezone.utc) + timedelta(hours=48),
            end_time=datetime.now(timezone.utc) + timedelta(hours=168),
            confidence=0.35,
            reason="Mandate needs re-authorization by customer",
        ),
    ],
}

# Customer behavior patterns
# Based on Indian payment market data
BEHAVIOR_PATTERNS: dict[str, dict[str, Any]] = {
    "salary_credit_days": [1, 15],  # 1st and 15th of month
    "peak_payment_hours": (10, 14),  # 10am-2pm
    "weekend_decline_rate": 0.15,   # 15% higher decline on weekends
    "late_night_success_rate": 0.45,  # 45% success rate late night
}


def get_retry_windows(
    failure_type: FailureType,
    attempt_count: int = 1,
    amount: float = 0,
    customer_email: str = "",
) -> list[RetryWindow]:
    """Get recommended retry windows for a failure type.

    Args:
        failure_type: Root cause of payment failure
        attempt_count: How many retries attempted
        amount: Payment amount (higher amounts = more conservative)
        customer_email: For future personalization

    Returns:
        List of retry windows sorted by confidence
    """
    windows = RETRY_WINDOWS.get(failure_type, [])

    # Adjust confidence based on attempt count
    # Diminishing returns with each retry
    adjusted = []
    for w in windows:
        adjusted_confidence = w.confidence * (0.8 ** (attempt_count - 1))
        adjusted.append(RetryWindow(
            start_time=w.start_time,
            end_time=w.end_time,
            confidence=adjusted_confidence,
            reason=w.reason,
        ))

    # Adjust for high-value payments (more conservative)
    if amount > 10000:  # INR 10,000+
        for w in adjusted:
            w.confidence *= 0.9
            w.reason += " (high-value — conservative)"

    # Sort by confidence
    adjusted.sort(key=lambda x: x.confidence, reverse=True)

    return adjusted[:3]  # Top 3 windows


def should_retry_now(
    failure_type: FailureType,
    attempt_count: int,
    last_attempt: datetime | None = None,
) -> tuple[bool, str]:
    """Check if we should retry now based on timing rules.

    Returns:
        (should_retry, reason)
    """
    windows = get_retry_windows(failure_type, attempt_count)
    now = datetime.now(timezone.utc)

    # Network timeout — always retry quickly
    if failure_type == FailureType.NETWORK_TIMEOUT and attempt_count <= 2:
        return True, "Network timeout — immediate retry recommended"

    # Card expired — never retry without customer action
    if failure_type == FailureType.CARD_EXPIRED:
        return False, "Card expired — customer must update payment method"

    # Mandate revoked — never retry without customer action
    if failure_type == FailureType.MANDATE_REVOKED:
        return False, "Mandate revoked — customer must re-authorize"

    # Check if we're in a retry window
    for window in windows:
        if window.start_time <= now <= window.end_time:
            return True, f"In retry window: {window.reason}"

    # If no window matches, use conservative delay
    if last_attempt:
        hours_since = (now - last_attempt).total_seconds() / 3600
        min_delay = 4 if attempt_count <= 2 else 12
        if hours_since < min_delay:
            return False, f"Too soon — wait {min_delay - hours_since:.1f} more hours"

    return False, "No retry window active"


def get_next_retry_time(
    failure_type: FailureType,
    attempt_count: int,
) -> datetime | None:
    """Get the next recommended retry time.

    Returns None if no retry should be attempted.
    """
    windows = get_retry_windows(failure_type, attempt_count)
    now = datetime.now(timezone.utc)

    # Find the next window that starts in the future
    future_windows = [w for w in windows if w.start_time > now]

    if future_windows:
        # Return the start of the next window
        return future_windows[0].start_time

    # If all windows are in the past, schedule for tomorrow
    return now.replace(hour=10, minute=0) + timedelta(days=1)


def format_retry_schedule(failure_type: FailureType, attempt_count: int) -> str:
    """Format a human-readable retry schedule."""
    windows = get_retry_windows(failure_type, attempt_count)
    now = datetime.now(timezone.utc)

    lines = [f"Retry schedule for {failure_type.value} (attempt #{attempt_count}):"]
    lines.append(f"Current time: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    for i, w in enumerate(windows, 1):
        in_window = w.start_time <= now <= w.end_time
        status = " (ACTIVE)" if in_window else ""
        lines.append(f"  {i}. {w.reason}{status}")
        lines.append(f"     Window: {w.start_time.strftime('%Y-%m-%d %H:%M')} — {w.end_time.strftime('%H:%M UTC')}")
        lines.append(f"     Confidence: {w.confidence:.0%}")
        lines.append("")

    return "\n".join(lines)
