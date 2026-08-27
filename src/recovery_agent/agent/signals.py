"""Payment Signals — 40+ feature extraction for recovery optimization.

Expands feature set from ~15 to 40+ signals inspired by:
- Stripe: 500+ features (multimodal embeddings, temporal, seasonal)
- Redux: 100+ signals (decline code, card brand, issuer, timezone, payday)
- Slicker: 40+ variables (BIN, MCC, geography, tenure)

Source: Industry research — INDUSTRY-RESEARCH.md
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class PaymentSignals(BaseModel):
    """Enriched payment signals — 40+ features for recovery optimization.

    These signals are extracted from Razorpay response data, customer profiles,
    and temporal context to feed into LLM diagnosis and strategy planning.
    """
    # === Payment basics (3 signals) ===
    failure_code: str = ""
    amount: float = 0.0
    currency: str = "INR"

    # === Card signals (7 signals) ===
    card_brand: str = ""              # visa, mastercard, amex, rupay
    card_type: str = ""               # debit, credit, prepaid
    card_issuer_bank: str = ""        # HDFC, ICICI, SBI, etc.
    card_last4: str = ""              # last 4 digits
    bin_number: str = ""              # first 6-8 digits
    network_token_status: str = ""    # active, inactive, none
    card_expiry_status: str = ""      # valid, expired, expiring_soon

    # === Customer signals (6 signals) ===
    customer_id: str = ""
    customer_timezone: str = "Asia/Kolkata"
    customer_country: str = "IN"
    customer_tenure_days: int = 0     # days since first payment
    card_velocity_1h: int = 0         # transactions with this card in last hour
    card_velocity_24h: int = 0        # transactions with this card in last 24h

    # === Bank health signals (4 signals) ===
    bank_health_score: float = 1.0    # 0.0=down, 0.5=degraded, 1.0=healthy
    bank_last_outage_hours: float = -1  # hours since last bank outage
    bank_approval_rate: float = 0.9   # historical approval rate for this bank
    issuer_response_code: str = ""    # raw issuer response code

    # === Temporal signals (7 signals) ===
    hour_of_day: int = 0
    day_of_week: int = 0             # 0=Monday, 6=Sunday
    day_of_month: int = 0
    is_weekend: bool = False
    is_payday: bool = False
    is_bank_holiday: bool = False
    is_month_end: bool = False

    # === Context signals (5 signals) ===
    merchant_descriptor: str = ""
    mcc_code: str = ""               # merchant category code
    amount_vs_avg: float = 1.0       # this amount / customer's avg amount
    first_payment: bool = False      # first payment for this customer
    subscription_frequency: str = ""  # monthly, annual, one-time

    # === History signals (4 signals) ===
    previous_failures_7d: int = 0    # failures in last 7 days
    previous_successes_30d: int = 0  # successes in last 30 days
    total_lifetime_payments: int = 0
    avg_time_between_payments: float = 0.0  # days

    # === Risk signals (3 signals) ===
    avs_match_result: str = ""       # full, partial, none, unchecked
    cvv_match_result: str = ""       # match, mismatch, unchecked
    risk_score: float = 0.0          # 0.0=low, 1.0=high

    class Config:
        """Allow extra fields for forward compatibility."""
        extra = "allow"


class SignalEnricher:
    """Extracts 40+ signals from Razorpay response data and context.

    Usage:
        enricher = SignalEnricher()
        signals = enricher.extract_from_response(raw_response, customer_profile)
        signals = enricher.extract_from_payment_event(event, metadata)
    """

    def extract_from_response(
        self,
        raw_response: dict[str, Any],
        customer_profile: dict[str, Any] | None = None,
    ) -> PaymentSignals:
        """Extract signals from a raw Razorpay API response."""
        profile = customer_profile or {}
        now = datetime.now(timezone.utc)

        # Card signals from response
        card = raw_response.get("card", {})
        payment_method = raw_response.get("payment_method", "")

        # Bank health from metadata
        bank_health = raw_response.get("bank_health", {})
        health_score = 1.0
        if isinstance(bank_health, dict):
            status = bank_health.get("status", "healthy")
            health_score = {"healthy": 1.0, "degraded": 0.5, "down": 0.0}.get(status, 1.0)

        return PaymentSignals(
            # Payment basics
            failure_code=str(raw_response.get("failure_code", "")),
            amount=float(raw_response.get("amount", 0)) / 100,  # Razorpay returns paise
            currency=raw_response.get("currency", "INR"),

            # Card signals
            card_brand=card.get("network", payment_method),
            card_type=card.get("type", ""),
            card_issuer_bank=card.get("issuer_bank", ""),
            card_last4=card.get("last4", ""),
            bin_number=str(card.get("bin", "")),
            network_token_status=card.get("tokenization_status", ""),
            card_expiry_status=self._check_expiry_status(card.get("expiry_month"), card.get("expiry_year")),

            # Customer signals
            customer_id=profile.get("customer_id", ""),
            customer_timezone=profile.get("timezone", "Asia/Kolkata"),
            customer_country=profile.get("country", "IN"),
            customer_tenure_days=profile.get("tenure_days", 0),
            card_velocity_1h=profile.get("card_velocity_1h", 0),
            card_velocity_24h=profile.get("card_velocity_24h", 0),

            # Bank health signals
            bank_health_score=health_score,
            bank_last_outage_hours=bank_health.get("last_outage_hours", -1),
            bank_approval_rate=bank_health.get("approval_rate", 0.9),
            issuer_response_code=str(raw_response.get("issuer_response_code", "")),

            # Temporal signals
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            day_of_month=now.day,
            is_weekend=now.weekday() >= 5,
            is_payday=self._check_is_payday(profile.get("country", "IN"), now),
            is_bank_holiday=False,  # Would need holiday calendar
            is_month_end=now.day >= 28,

            # Context signals
            merchant_descriptor=raw_response.get("merchant_descriptor", ""),
            mcc_code=raw_response.get("mcc", ""),
            amount_vs_avg=self._calc_amount_vs_avg(
                float(raw_response.get("amount", 0)) / 100,
                profile.get("avg_payment_amount", 0),
            ),
            first_payment=profile.get("total_payments", 0) == 0,
            subscription_frequency=profile.get("subscription_frequency", ""),

            # History signals
            previous_failures_7d=profile.get("failures_7d", 0),
            previous_successes_30d=profile.get("successes_30d", 0),
            total_lifetime_payments=profile.get("total_payments", 0),
            avg_time_between_payments=profile.get("avg_days_between_payments", 0.0),

            # Risk signals
            avs_match_result=raw_response.get("avs_result", ""),
            cvv_match_result=raw_response.get("cvv_result", ""),
            risk_score=float(raw_response.get("risk_score", 0.0)),
        )

    def extract_from_payment_event(
        self,
        event: Any,
        metadata: dict[str, Any] | None = None,
    ) -> PaymentSignals:
        """Extract signals from a PaymentEvent model."""
        meta = metadata or {}
        now = datetime.now(timezone.utc)

        # Extract card info from metadata if available
        card_info = meta.get("card_info", {})
        bank_health = meta.get("bank_health", {})
        profile_data = meta.get("customer_profile", {})

        health_score = 1.0
        if isinstance(bank_health, dict):
            status = bank_health.get("status", "healthy")
            health_score = {"healthy": 1.0, "degraded": 0.5, "down": 0.0}.get(status, 1.0)

        return PaymentSignals(
            failure_code=event.failure_code,
            amount=event.amount,
            currency=event.currency,

            card_brand=card_info.get("brand", ""),
            card_type=card_info.get("type", ""),
            card_issuer_bank=card_info.get("issuer_bank", ""),
            card_last4=card_info.get("last4", ""),
            bin_number=card_info.get("bin", ""),
            network_token_status=card_info.get("token_status", ""),
            card_expiry_status=card_info.get("expiry_status", ""),

            customer_id=event.customer_id,
            customer_timezone=profile_data.get("timezone", "Asia/Kolkata"),
            customer_country=profile_data.get("country", "IN"),
            customer_tenure_days=profile_data.get("tenure_days", 0),
            card_velocity_1h=meta.get("card_velocity_1h", 0),
            card_velocity_24h=meta.get("card_velocity_24h", 0),

            bank_health_score=health_score,
            bank_last_outage_hours=bank_health.get("last_outage_hours", -1),
            bank_approval_rate=bank_health.get("approval_rate", 0.9),
            issuer_response_code=meta.get("issuer_response_code", ""),

            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            day_of_month=now.day,
            is_weekend=now.weekday() >= 5,
            is_payday=self._check_is_payday(profile_data.get("country", "IN"), now),
            is_bank_holiday=False,
            is_month_end=now.day >= 28,

            merchant_descriptor=meta.get("merchant_descriptor", ""),
            mcc_code=meta.get("mcc", ""),
            amount_vs_avg=self._calc_amount_vs_avg(
                event.amount,
                profile_data.get("avg_payment_amount", 0),
            ),
            first_payment=profile_data.get("total_payments", 0) == 0,
            subscription_frequency=profile_data.get("subscription_frequency", ""),

            previous_failures_7d=profile_data.get("failures_7d", 0),
            previous_successes_30d=profile_data.get("successes_30d", 0),
            total_lifetime_payments=profile_data.get("total_payments", 0),
            avg_time_between_payments=profile_data.get("avg_days_between_payments", 0.0),

            avs_match_result=meta.get("avs_result", ""),
            cvv_match_result=meta.get("cvv_result", ""),
            risk_score=float(meta.get("risk_score", 0.0)),
        )

    def _check_expiry_status(self, month: int | None, year: int | None) -> str:
        """Check card expiry status."""
        if not month or not year:
            return "unknown"
        now = datetime.now(timezone.utc)
        if year < now.year or (year == now.year and month < now.month):
            return "expired"
        if year == now.year and month <= now.month + 1:
            return "expiring_soon"
        return "valid"

    def _check_is_payday(self, country_code: str, now: datetime) -> bool:
        """Quick payday check based on region."""
        typical_days = {
            "IN": [28],
            "US": [1, 15],
            "GB": [28],
        }.get(country_code.upper(), [])

        return any(abs(now.day - day) <= 2 for day in typical_days)

    def _calc_amount_vs_avg(self, amount: float, avg_amount: float) -> float:
        """Calculate amount relative to customer's average."""
        if avg_amount <= 0:
            return 1.0
        return round(amount / avg_amount, 2)
