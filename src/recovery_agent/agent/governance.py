"""Agent Governance — Tool Access Control + Data Masking (MANDATE 0 compliant).

Governing AI Agents course (4 pillars):
    1. Security: Least-privilege tool access per tier/amount
    2. Security: PII masking in tool outputs and responses
    3. Risk Management: Configurable policies per agent role
    4. Lifecycle: Version tracking per episode

MANDATE 0: This is NOT a pipeline — policies are checked at runtime by the
guardrail node, and the LLM still decides which tool to call. The governance
engine only RESTRICTS the available tool set, it doesn't decide strategy.

MANDATE 1: Uses pydantic (real SDK) for policy schemas, re (stdlib) for PII masking.
MANDATE 2: No stubs — real masking, real access control.
MANDATE 3: No hardcoded decisions — policies are configurable, not if/else chains.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# AGENT VERSION — lifecycle tracking (Governing AI Agents P1)
# ═══════════════════════════════════════════════════════════════

AGENT_VERSION = "2.0.0"


# ═══════════════════════════════════════════════════════════════
# TOOL ACCESS POLICIES — least-privilege per tier (P0)
# ═══════════════════════════════════════════════════════════════

class AgentTier(str, Enum):
    """Agent operating tiers — higher tier = more tool access."""
    SILENT = "silent"
    ACTIVE = "active"
    ESCALATED = "escalated"


class ToolAccessPolicy(BaseModel):
    """Defines which tools are allowed per tier and amount threshold.

    MANDATE 1: pydantic BaseModel (real SDK).
    MANDATE 3: Policies are configurable data, not hardcoded decisions.
    """
    # Tools allowed per tier (union of all tiers = full tool set)
    # Every name here MUST exist in tools.TOOLS_BY_NAME. A name that does not
    # is silently dropped when tools are bound, so the LLM never sees the tool —
    # which is why `retry_in_hours` (listed here as the non-existent
    # "schedule_retry") could never be called, making Tier 1 silent retry
    # impossible. `test_policy_names_match_registry` now guards this.
    silent_tools: list[str] = Field(default_factory=lambda: [
        "check_payment_status",
        "diagnose_payment_failure",
        "get_customer_payment_history",
        "query_knowledge_base",
        "discover_recovery_rail",
        "manage_memory",
        "search_memory",
        "generate_recovery_payment_link",
        "get_recovery_offer",
        "send_page_push",
        "show_page_offer",
        "send_recovery_notification",
        "retry_in_hours",
        "escalate_to_human",
        "wait_for_customer",
    ])
    active_tools: list[str] = Field(default_factory=lambda: [
        "send_page_push",
        "show_page_offer",
        "send_recovery_notification",
        "retry_in_hours",
        "generate_recovery_payment_link",
        "get_recovery_offer",
        "initiate_voice_call",
    ])
    escalated_tools: list[str] = Field(default_factory=lambda: [
        "escalate_to_human",
        "initiate_voice_call",
        "wait_for_customer",
    ])
    # Amount threshold requiring escalation (in paise)
    escalation_threshold_paise: int = 5_000_000  # ₹50,000


# Default policy — configurable via env vars or constructor
_DEFAULT_POLICY = ToolAccessPolicy()


def get_allowed_tools(
    tier: str | AgentTier,
    amount_paise: int = 0,
    policy: ToolAccessPolicy | None = None,
) -> list[str]:
    """Return list of tool names allowed for the given tier and amount.

    MANDATE 3: This is access control (acceptable guardrail), not decision-making.
    The LLM still decides WHICH allowed tool to call.
    MANDATE 1: pydantic validation via ToolAccessPolicy.
    """
    p = policy or _DEFAULT_POLICY
    t = AgentTier(tier) if isinstance(tier, str) else tier

    allowed = list(p.silent_tools)

    if t in (AgentTier.ACTIVE, AgentTier.ESCALATED):
        allowed.extend(p.active_tools)

    if t == AgentTier.ESCALATED:
        allowed.extend(p.escalated_tools)

    # High-value transactions require escalated tier for financial tools
    if amount_paise > p.escalation_threshold_paise:
        financial_tools = {"generate_recovery_payment_link"}
        allowed = [t for t in allowed if t not in financial_tools]
        if "escalate_to_human" not in allowed:
            allowed.append("escalate_to_human")

    return list(set(allowed))  # deduplicate


# ═══════════════════════════════════════════════════════════════
# DATA MASKING — PII protection (Governing AI Agents P0)
# ═══════════════════════════════════════════════════════════════

# Regex patterns for PII detection — MANDATE 1: re (stdlib, acceptable)
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("card_number", re.compile(r"\b(\d{4})[\s-]?(\d{4})[\s-]?(\d{4})[\s-]?(\d{4})\b")),
    ("upi_id", re.compile(r"\b([\w.\-]+)@([\w]+)\b")),
    ("email", re.compile(r"\b([\w.\-]+)@([\w.\-]+\.\w+)\b")),
    ("phone", re.compile(r"\b(\+?91[\s-]?)?(\d{5})[\s-]?(\d{5})\b")),
    ("aadhaar", re.compile(r"\b(\d{4})[\s-]?(\d{4})[\s-]?(\d{4})\b")),
    ("pan", re.compile(r"\b([A-Z]{5})(\d{4})([A-Z])\b")),
]


def mask_pii(text: str) -> str:
    """Mask PII in text — card numbers, UPI IDs, emails, phone numbers.

    Governing AI Agents: "mask personal information, build tools that
    provide only the data needed."

    MANDATE 1: re (stdlib) for pattern matching — no external PII library needed.
    MANDATE 2: Real masking, not stubs.

    Examples:
        "Card 4111 1111 1111 1111" → "Card 4111 **** **** 1111"
        "user@upi" → "u***@upi"
        "test@email.com" → "t***@email.com"
    """
    masked = text

    # Mask card numbers: show first 4 and last 4
    def _mask_card(m: re.Match) -> str:
        return f"{m.group(1)} **** **** {m.group(4)}"

    masked = _PII_PATTERNS[0][1].sub(_mask_card, masked)

    # Mask UPI IDs: show first char and domain
    def _mask_upi(m: re.Match) -> str:
        user = m.group(1)
        domain = m.group(2)
        return f"{user[0]}***@{domain}"

    masked = _PII_PATTERNS[1][1].sub(_mask_upi, masked)

    # Mask emails: show first char and domain
    def _mask_email(m: re.Match) -> str:
        user = m.group(1)
        domain = m.group(2)
        return f"{user[0]}***@{domain}"

    masked = _PII_PATTERNS[2][1].sub(_mask_email, masked)

    # Mask phone numbers: show last 4 digits
    def _mask_phone(m: re.Match) -> str:
        return f"******{m.group(3)}"

    masked = _PII_PATTERNS[3][1].sub(_mask_phone, masked)

    return masked


def mask_tool_output(tool_name: str, output: str) -> str:
    """Mask PII in tool output based on tool type.

    Governing AI Agents: "restrict agent's access to sensitive data."
    Different tools expose different PII levels.
    """
    # Financial tools: always mask card numbers
    if tool_name in ("check_payment_status", "get_customer_payment_history",
                      "generate_recovery_payment_link", "initiate_refund"):
        return mask_pii(output)

    # Communication tools: mask contact info
    if tool_name in ("send_recovery_notification", "send_email", "send_sms",
                      "initiate_voice_call"):
        return mask_pii(output)

    # Knowledge/memory tools: no PII in output
    if tool_name in ("query_knowledge_base", "search_similar_episodes",
                      "manage_memory", "search_memory", "discover_recovery_rail"):
        return output

    # Default: mask everything
    return mask_pii(output)


# ═══════════════════════════════════════════════════════════════
# TIER-BASED POLICIES — configurable per tier (Governing AI Agents P1)
# ═══════════════════════════════════════════════════════════════

class TierPolicy(BaseModel):
    """Configurable policies per agent tier.

    Governing AI Agents: "Define allowed actions, set guardrails."
    MANDATE 1: pydantic BaseModel (real SDK).
    MANDATE 3: Policies are data, not hardcoded decisions.
    """
    max_communications_per_day: int = 3
    allow_financial_tools: bool = False
    require_approval_above_paise: int = 5_000_000
    allowed_failure_categories: list[str] = Field(default_factory=lambda: [
        "transient", "permanent", "unknown"
    ])
    can_escalate: bool = False


TIER_POLICIES: dict[str, TierPolicy] = {
    "silent": TierPolicy(
        max_communications_per_day=2,
        allow_financial_tools=False,
        require_approval_above_paise=0,
        allowed_failure_categories=["transient"],
        can_escalate=False,
    ),
    "active": TierPolicy(
        max_communications_per_day=3,
        allow_financial_tools=False,
        require_approval_above_paise=5_000_000,
        allowed_failure_categories=["transient", "permanent", "unknown"],
        can_escalate=True,
    ),
    "escalated": TierPolicy(
        max_communications_per_day=5,
        allow_financial_tools=True,
        require_approval_above_paise=10_000_000,
        allowed_failure_categories=["transient", "permanent", "unknown"],
        can_escalate=True,
    ),
}


def get_tier_policy(tier: str) -> TierPolicy:
    """Get the policy for a given tier.
    MANDATE 1: pydantic validation via TierPolicy.
    """
    return TIER_POLICIES.get(tier, TIER_POLICIES["silent"])
