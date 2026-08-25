"""Test case generator — creates synthetic payment failure scenarios.

Source: Gold-standard datasets from NeMo Agent Toolkit
        Evaluation framework from Evaluating AI Agents
"""
from __future__ import annotations

import random
from datetime import datetime, timezone

from recovery_agent.models import FailureType, PaymentEvent


# Templates for realistic payment failure scenarios
FAILURE_SCENARIOS: list[dict] = [
    {
        "failure_type": FailureType.CARD_EXPIRED,
        "failure_code": "card_expired",
        "failure_reasons": [
            "Card has expired",
            "Expired card details",
            "Card expiry date is in the past",
        ],
        "amount_range": (500, 50000),
        "recoverable": True,  # Recoverable if customer updates card
    },
    {
        "failure_type": FailureType.INSUFFICIENT_FUNDS,
        "failure_code": "insufficient_funds",
        "failure_reasons": [
            "Insufficient funds in account",
            "Not enough balance",
            "Account has insufficient funds",
        ],
        "amount_range": (1000, 200000),
        "recoverable": True,  # Recoverable on retry
    },
    {
        "failure_type": FailureType.BANK_DECLINED,
        "failure_code": "do_not_honor",
        "failure_reasons": [
            "Transaction declined by bank",
            "Do not honor",
            "Bank declined the transaction",
        ],
        "amount_range": (200, 100000),
        "recoverable": True,  # Sometimes transient
    },
    {
        "failure_type": FailureType.NETWORK_TIMEOUT,
        "failure_code": "network_error",
        "failure_reasons": [
            "Network timeout during processing",
            "Connection lost",
            "Gateway timeout",
        ],
        "amount_range": (100, 50000),
        "recoverable": True,  # Usually works on retry
    },
    {
        "failure_type": FailureType.RISK_BLOCK,
        "failure_code": "risk_check_failed",
        "failure_reasons": [
            "Transaction flagged by risk engine",
            "Suspicious activity detected",
            "Risk check failed",
        ],
        "amount_range": (10000, 500000),
        "recoverable": False,  # Needs human review
    },
    {
        "failure_type": FailureType.MANDATE_REVOKED,
        "failure_code": "mandate_inactive",
        "failure_reasons": [
            "Customer has revoked the mandate",
            "Auto-pay mandate is inactive",
            "NPCI mandate cancelled",
        ],
        "amount_range": (500, 100000),
        "recoverable": False,  # Customer actively cancelled
    },
]


def generate_payment_event(
    scenario: dict | None = None,
    payment_id: str | None = None,
    customer_id: str | None = None,
) -> PaymentEvent:
    """Generate a single synthetic payment failure event."""
    if scenario is None:
        scenario = random.choice(FAILURE_SCENARIOS)

    amount = round(random.uniform(*scenario["amount_range"]), 2)
    reason = random.choice(scenario["failure_reasons"])

    return PaymentEvent(
        payment_id=payment_id or f"pay_{random.randint(100000, 999999)}",
        customer_id=customer_id or f"cust_{random.randint(1000, 9999)}",
        amount=amount,
        currency="INR",
        failure_reason=reason,
        failure_code=scenario["failure_code"],
        created_at=datetime.now(timezone.utc),
        metadata={
            "failure_type": scenario["failure_type"].value,
            "recoverable": scenario["recoverable"],
        },
    )


def generate_batch(num_cases: int = 30) -> list[PaymentEvent]:
    """Generate a batch of diverse payment failure events.

    Distributes evenly across failure types for balanced evaluation.
    """
    events = []
    scenarios_per_type = max(1, num_cases // len(FAILURE_SCENARIOS))

    for scenario in FAILURE_SCENARIOS:
        for i in range(scenarios_per_type):
            if len(events) >= num_cases:
                break
            events.append(generate_payment_event(scenario=scenario))

    # Fill remaining with random
    while len(events) < num_cases:
        events.append(generate_payment_event())

    random.shuffle(events)
    return events[:num_cases]


def get_known_outcomes(events: list[PaymentEvent]) -> dict[str, dict]:
    """Return known properties for each event (for evaluation)."""
    outcomes = {}
    for event in events:
        outcomes[event.payment_id] = {
            "failure_type": event.metadata.get("failure_type", "unknown"),
            "recoverable": event.metadata.get("recoverable", False),
            "amount": event.amount,
        }
    return outcomes
