"""Data models for the recovery agent.

Defines Case, Attempt, Diagnosis, AuditEntry, and AgentState.
Source: Data model design from ARCHITECTURE.md
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CaseStatus(str, Enum):
    """Operational state of a recovery case.

    Each state answers one question: *what is the system waiting on?*
    OPEN/DIAGNOSING/DIAGNOSED/ACTING wait on the agent; AWAITING_CUSTOMER waits
    on a human; SCHEDULED waits on a clock; the rest are terminal.

    SCHEDULED and AWAITING_CUSTOMER are deliberately distinct: the sensor polls
    the latter, the scheduler wakes the former. Conflating them is why the old
    retry subsystem had nowhere to live (AUDIT-FINDINGS S1-1).
    """
    OPEN = "open"
    DIAGNOSING = "diagnosing"
    DIAGNOSED = "diagnosed"
    ACTING = "acting"
    AWAITING_CUSTOMER = "awaiting_customer"   # waiting on a person
    SCHEDULED = "scheduled"                   # waiting on a timer (silent retry)
    RECOVERED = "recovered"
    STOPPED = "stopped"
    ESCALATED = "escalated"


class FailureType(str, Enum):
    CARD_EXPIRED = "card_expired"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    BANK_DECLINED = "bank_declined"
    NETWORK_TIMEOUT = "network_timeout"
    RISK_BLOCK = "risk_block"
    MANDATE_REVOKED = "mandate_revoked"
    USER_DROPOFF = "user_dropoff"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    RETRY_PAYMENT = "retry_payment"
    SEND_NOTIFICATION = "send_notification"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    UPDATE_PAYMENT_METHOD = "update_payment_method"
    WAIT_AND_RETRY = "wait_and_retry"
    ABANDON = "abandon"
    VOICE_CALL = "voice_call"


class RecoveryTier(str, Enum):
    """Recovery tier — silent (background) vs active (customer-contacting).

    Inspired by Redux 'Silent First' architecture:
    "Every unnecessary email is a cancellation opportunity."
    """
    SILENT = "silent"       # Background retries, no customer contact
    ACTIVE = "active"       # Customer-facing: emails, SMS, payment link updates


# Actions that do NOT contact the customer (safe for silent tier)
SILENT_ACTIONS = frozenset({
    ActionType.RETRY_PAYMENT,
    ActionType.WAIT_AND_RETRY,
})

# Actions that DO contact the customer (active tier only)
ACTIVE_ACTIONS = frozenset({
    ActionType.SEND_NOTIFICATION,
    ActionType.UPDATE_PAYMENT_METHOD,
    ActionType.VOICE_CALL,
})

# Hard decline codes — NEVER retry (Visa/MC network penalty prevention)
# Source: Redux hard decline handling, Stripe Schedule+Skip behavior
HARD_DECLINES = frozenset({
    "41",  # Lost card
    "43",  # Stolen card
    "54",  # Expired card
    "14",  # Invalid card number
    "04",  # Pick up card (fraud)
    "46",  # Closed account
    "57",  # Transaction not permitted
    "93",  # Transaction cannot be completed
})


class AuditStep(str, Enum):
    DETECT = "detect"
    DIAGNOSE = "diagnose"
    DECIDE = "decide"
    ACT = "act"
    OBSERVE = "observe"
    STOP = "stop"


class AuditEntry(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    step: AuditStep
    input_data: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    output_data: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = 0


class Diagnosis(BaseModel):
    root_cause: FailureType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    category: str = ""


class Attempt(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action_type: ActionType
    action_details: dict[str, Any] = Field(default_factory=dict)
    result: str = "pending"  # success, failed, pending
    error: str = ""
    signals: dict[str, Any] = Field(default_factory=dict)
    tier: RecoveryTier = RecoveryTier.ACTIVE


class PaymentEvent(BaseModel):
    """Raw payment failure event — input to the agent."""
    payment_id: str
    customer_id: str
    amount: float
    currency: str = "INR"
    failure_reason: str = ""
    failure_code: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Case(BaseModel):
    """A revenue-at-risk case — the core unit the agent works on."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payment: PaymentEvent
    status: CaseStatus = CaseStatus.OPEN
    attempts: list[Attempt] = Field(default_factory=list)
    diagnosis: Diagnosis | None = None
    audit_log: list[AuditEntry] = Field(default_factory=list)
    attempt_count: int = 0
    max_attempts: int = 5
    recovered: bool = False
    recovered_amount: float = 0.0
    recovery_tier: RecoveryTier = RecoveryTier.SILENT
    silent_attempts: int = 0
    max_silent_attempts: int = 3
    penalties_prevented: int = 0


class AgentState(BaseModel):
    """State passed through the LangGraph."""
    case: Case
    current_step: AuditStep = AuditStep.DETECT
    should_stop: bool = False
    stop_reason: str = ""
    loop_count: int = 0
    max_loops: int = 10


# --- Chaos Gym Models ---

class CustomerMood(str, Enum):
    COOPERATIVE = "cooperative"
    FRUSTRATED = "frustrated"
    NON_RESPONSIVE = "non_responsive"


class BankHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class ChaosAnomaly(str, Enum):
    NONE = "none"
    GATEWAY_DEGRADATION_SPIKE = "gateway_degradation_spike"
    OUT_OF_ORDER_WEBHOOK = "out_of_order_webhook"
    CARD_EXPIRY_SURGE = "card_expiry_surge"


class CustomerPersona(str, Enum):
    SALARY_DEPENDENT = "salary_dependent"
    BUSY_EXECUTIVE = "busy_executive"
    FRUSTRATED_SUBSCRIBER = "frustrated_subscriber"
    B2B_AP = "b2b_ap"


class GymState(BaseModel):
    """Current state of the Gym environment."""
    case: Case
    environment_time: int = 0
    bank_health: BankHealth = BankHealth.HEALTHY
    customer_mood: CustomerMood = CustomerMood.COOPERATIVE
    customer_persona: CustomerPersona = CustomerPersona.SALARY_DEPENDENT
    attempt_count: int = 0
    reward_score: float = 0.0
    chaos_anomaly: ChaosAnomaly = ChaosAnomaly.NONE
    messages_sent: int = 0
    policy_violations: int = 0
    done: bool = False


class RedTeamAction(BaseModel):
    """Event generated by the Red Team Chaos Engine."""
    persona: CustomerPersona
    failure_type: FailureType
    amount: float = Field(ge=0)
    webhook_latency_ms: int = 0
    chaos_anomaly: ChaosAnomaly = ChaosAnomaly.NONE
    customer_mood: CustomerMood = CustomerMood.COOPERATIVE
    bank_health: BankHealth = BankHealth.HEALTHY
    metadata: dict[str, Any] = Field(default_factory=dict)


class GymStepResult(BaseModel):
    """Output of env.step(action) — (next_state, reward, done, info)."""
    next_state: GymState
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


# --- Long-Term Memory Models ---

class PaymentRecord(BaseModel):
    """Historical payment attempt record for a customer."""
    payment_id: str
    amount: float
    channel_used: str
    status: str  # success, failed, pending
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    failure_type: str = ""


class PromiseToPay(BaseModel):
    """Customer commitment to pay by a specific date."""
    promise_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    amount: float
    promised_date: str  # ISO date string YYYY-MM-DD
    fulfilled: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SalaryWindow(BaseModel):
    """Tracks customer salary credit timing for liquidity-aware retries."""
    typical_pay_day: int = Field(ge=0, le=31, default=0)
    last_salary_date: str = ""  # ISO date string
    is_salary_due: bool = False
    salary_history: list[str] = Field(default_factory=list)  # ISO dates


class CustomerProfile(BaseModel):
    """Persistent memory profile for a customer across sessions."""
    customer_id: str
    payment_history: list[PaymentRecord] = Field(default_factory=list)
    salary_window: SalaryWindow = Field(default_factory=SalaryWindow)
    promises: list[PromiseToPay] = Field(default_factory=list)
    preferred_channel: str = ""
    total_recovered: float = 0.0
    total_attempts: int = 0
    channel_success_rates: dict[str, float] = Field(default_factory=dict)
    last_contacted: datetime | None = None
    failure_type_counts: dict[str, int] = Field(default_factory=dict)
    opt_out: bool = False


# --- Generative UI Spec ---

