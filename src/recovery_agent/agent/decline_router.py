"""Decline Code Router — per-code recovery strategy routing.

Routes each Razorpay decline code to a completely different recovery strategy.
Inspired by:
- Redux per-code strategies (Code 51, 05, 19)
- Slicker issuer network intelligence
- Stripe hard decline handling

Source: Industry research — INDUSTRY-RESEARCH.md
"""
from __future__ import annotations

from enum import Enum

from recovery_agent.models import ActionType, FailureType, HARD_DECLINES


class DeclineStrategy(str, Enum):
    """Strategy categories for decline code routing."""
    PAYDAY_TIMING = "payday_timing"                # Code 51: Insufficient funds
    METADATA_ENRICHMENT = "metadata_enrichment"    # Code 05: Do not honor
    BANK_HEALTH_MONITORING = "bank_health"          # Code 19: Try again later
    COOLING_OFF = "cooling_off"                    # Code 61: Withdrawal limit
    VELOCITY_CHECK = "velocity_check"              # Code 65: Activity count
    CARD_UPDATE_FLOW = "card_update"               # Code 54: Expired card
    HARD_BLOCK = "hard_block"                      # Codes 41/43/04/46/57/93
    IMMEDIATE_ESCALATION = "immediate_escalation"  # Risk blocks
    RETRY_IMMEDIATE = "retry_immediate"            # Transient network errors
    GENERIC_SOFT = "generic_soft"                  # Unclassified soft declines


# Razorpay decline code → strategy mapping
# Source: Razorpay API Error Codes + Redux/Stripe per-code research
DECLINE_CODE_MAP: dict[str, dict] = {
    "51": {
        "strategy": DeclineStrategy.PAYDAY_TIMING,
        "description": "Insufficient funds — retry at payday window",
        "default_tier": "silent",
        "default_action": ActionType.WAIT_AND_RETRY,
        "max_retries": 3,
        "cooldown_hours": 24,
        "notes": "First-in-line advantage: retry at 12:01 AM on payday",
    },
    "05": {
        "strategy": DeclineStrategy.METADATA_ENRICHMENT,
        "description": "Do not honor — optimize transaction shape",
        "default_tier": "silent",
        "default_action": ActionType.WAIT_AND_RETRY,
        "max_retries": 2,
        "cooldown_hours": 4,
        "notes": "Banks use generic code when fraud models lack clean metadata",
    },
    "19": {
        "strategy": DeclineStrategy.BANK_HEALTH_MONITORING,
        "description": "Try again later — monitor bank health",
        "default_tier": "silent",
        "default_action": ActionType.WAIT_AND_RETRY,
        "max_retries": 3,
        "cooldown_hours": 2,
        "notes": "Bank may have predictable offline windows",
    },
    "61": {
        "strategy": DeclineStrategy.COOLING_OFF,
        "description": "Withdrawal limit exceeded — cooling off period",
        "default_tier": "silent",
        "default_action": ActionType.WAIT_AND_RETRY,
        "max_retries": 2,
        "cooldown_hours": 12,
    },
    "65": {
        "strategy": DeclineStrategy.VELOCITY_CHECK,
        "description": "Activity count exceeded — wait for velocity reset",
        "default_tier": "silent",
        "default_action": ActionType.WAIT_AND_RETRY,
        "max_retries": 2,
        "cooldown_hours": 6,
    },
    "54": {
        "strategy": DeclineStrategy.CARD_UPDATE_FLOW,
        "description": "Expired card — route to card update",
        "default_tier": "active",
        "default_action": ActionType.UPDATE_PAYMENT_METHOD,
        "max_retries": 0,
        "cooldown_hours": 0,
        "notes": "Never retry same expired card",
    },
    "14": {
        "strategy": DeclineStrategy.CARD_UPDATE_FLOW,
        "description": "Invalid card number — route to card update",
        "default_tier": "active",
        "default_action": ActionType.UPDATE_PAYMENT_METHOD,
        "max_retries": 0,
        "cooldown_hours": 0,
    },
    # Hard declines — NEVER retry
    "41": {
        "strategy": DeclineStrategy.HARD_BLOCK,
        "description": "Lost card — never retry",
        "default_tier": "active",
        "default_action": ActionType.ESCALATE_TO_HUMAN,
        "max_retries": 0,
        "cooldown_hours": 0,
    },
    "43": {
        "strategy": DeclineStrategy.HARD_BLOCK,
        "description": "Stolen card — never retry",
        "default_tier": "active",
        "default_action": ActionType.ESCALATE_TO_HUMAN,
        "max_retries": 0,
        "cooldown_hours": 0,
    },
    "04": {
        "strategy": DeclineStrategy.HARD_BLOCK,
        "description": "Pick up card (fraud) — never retry",
        "default_tier": "active",
        "default_action": ActionType.ESCALATE_TO_HUMAN,
        "max_retries": 0,
        "cooldown_hours": 0,
    },
    "46": {
        "strategy": DeclineStrategy.HARD_BLOCK,
        "description": "Closed account — never retry",
        "default_tier": "active",
        "default_action": ActionType.ESCALATE_TO_HUMAN,
        "max_retries": 0,
        "cooldown_hours": 0,
    },
    "57": {
        "strategy": DeclineStrategy.HARD_BLOCK,
        "description": "Transaction not permitted — never retry",
        "default_tier": "active",
        "default_action": ActionType.ESCALATE_TO_HUMAN,
        "max_retries": 0,
        "cooldown_hours": 0,
    },
    "93": {
        "strategy": DeclineStrategy.HARD_BLOCK,
        "description": "Transaction cannot be completed — never retry",
        "default_tier": "active",
        "default_action": ActionType.ESCALATE_TO_HUMAN,
        "max_retries": 0,
        "cooldown_hours": 0,
    },
}


# FailureType → DeclineStrategy mapping (for when we only have the failure type, not the code)
FAILURE_TYPE_STRATEGY: dict[FailureType, DeclineStrategy] = {
    FailureType.INSUFFICIENT_FUNDS: DeclineStrategy.PAYDAY_TIMING,
    FailureType.BANK_DECLINED: DeclineStrategy.BANK_HEALTH_MONITORING,
    FailureType.NETWORK_TIMEOUT: DeclineStrategy.RETRY_IMMEDIATE,
    FailureType.CARD_EXPIRED: DeclineStrategy.CARD_UPDATE_FLOW,
    FailureType.MANDATE_REVOKED: DeclineStrategy.IMMEDIATE_ESCALATION,
    FailureType.RISK_BLOCK: DeclineStrategy.IMMEDIATE_ESCALATION,
}


class DeclineCodeRouter:
    """Routes recovery strategy based on specific Razorpay decline code.

    Each code gets a completely different strategy — not one-size-fits-all.

    Usage:
        router = DeclineCodeRouter()
        strategy = router.get_strategy("51")  # PAYDAY_TIMING
        action = router.get_default_action("51")  # WAIT_AND_RETRY
        is_hard = router.is_hard_decline("41")  # True
    """

    def __init__(self) -> None:
        self.code_map = DECLINE_CODE_MAP
        self.failure_type_map = FAILURE_TYPE_STRATEGY

    def get_strategy(self, code: str) -> DeclineStrategy:
        """Get the recovery strategy for a decline code."""
        entry = self.code_map.get(code)
        if entry:
            return entry["strategy"]
        return DeclineStrategy.GENERIC_SOFT

    def get_default_action(self, code: str) -> ActionType | None:
        """Get the default action for a decline code."""
        entry = self.code_map.get(code)
        if entry:
            return entry["default_action"]
        return None

    def get_max_retries(self, code: str) -> int:
        """Get max retries allowed for a decline code."""
        entry = self.code_map.get(code)
        if entry:
            return entry["max_retries"]
        return 2  # default for unclassified

    def get_cooldown_hours(self, code: str) -> int:
        """Get cooldown period in hours for a decline code."""
        entry = self.code_map.get(code)
        if entry:
            return entry["cooldown_hours"]
        return 2  # default

    def is_hard_decline(self, code: str) -> bool:
        """Check if a decline code is a hard decline (never retry)."""
        return code in HARD_DECLINES

    def get_tier(self, code: str) -> str:
        """Get the default tier (silent/active) for a decline code."""
        entry = self.code_map.get(code)
        if entry:
            return entry["default_tier"]
        return "silent"

    def get_strategy_for_failure_type(self, failure_type: FailureType) -> DeclineStrategy:
        """Get strategy from FailureType when raw code is unavailable."""
        return self.failure_type_map.get(failure_type, DeclineStrategy.GENERIC_SOFT)

    def should_suppress_communication(self, code: str, attempt_count: int) -> bool:
        """Determine if customer communication should be suppressed.

        Silent tier: suppress communications for first N retries.
        Active tier: allow communications.
        """
        entry = self.code_map.get(code)
        if not entry:
            return attempt_count < 2

        strategy = entry["strategy"]

        # Hard declines: never suppress (need immediate escalation)
        if strategy == DeclineStrategy.HARD_BLOCK:
            return False

        # Card update flows: never suppress (customer must act)
        if strategy == DeclineStrategy.CARD_UPDATE_FLOW:
            return False

        # Payday timing, bank health, cooling off: suppress during silent phase
        max_silent = entry.get("max_retries", 2)
        return attempt_count <= max_silent

    def get_code_info(self, code: str) -> dict:
        """Get full info for a decline code."""
        return self.code_map.get(code, {
            "strategy": DeclineStrategy.GENERIC_SOFT,
            "description": f"Unclassified code: {code}",
            "default_tier": "silent",
            "default_action": ActionType.RETRY_PAYMENT,
            "max_retries": 2,
            "cooldown_hours": 2,
        })
