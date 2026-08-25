"""Long-Term Memory Engine — persistent customer profiles for liquidity-aware retries.

Tracks payment history, salary credit windows, promise-to-pay commitments,
and channel success rates across sessions to optimize retry timing.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from recovery_agent.models import (
    CustomerProfile,
    PaymentRecord,
    PromiseToPay,
    SalaryWindow,
)


class CustomerMemoryStore:
    """In-memory & JSON-backed persistent store for customer profiles.

    Provides memory-aware decision support:
    - Salary window detection for liquidity-aware retries
    - Channel optimization based on historical success rates
    - Promise-to-pay tracking for commitment follow-through
    """

    def __init__(self, persist_dir: str | None = None):
        self._profiles: dict[str, CustomerProfile] = {}
        self._persist_dir = Path(persist_dir) if persist_dir else None
        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_all()

    # --- Profile Access ---

    def get_or_create_profile(self, customer_id: str) -> CustomerProfile:
        """Fetch existing memory or initialize new profile."""
        if customer_id in self._profiles:
            return self._profiles[customer_id]
        profile = CustomerProfile(customer_id=customer_id)
        self._profiles[customer_id] = profile
        return profile

    def save_profile(self, profile: CustomerProfile) -> None:
        """Persist profile to memory and optionally to disk."""
        self._profiles[profile.customer_id] = profile
        if self._persist_dir:
            path = self._persist_dir / f"{profile.customer_id}.json"
            path.write_text(profile.model_dump_json(indent=2))

    # --- Memory Updates ---

    def update_profile_after_attempt(
        self,
        customer_id: str,
        attempt: dict,
        success: bool,
        channel: str = "",
    ) -> CustomerProfile:
        """Update channel response rates and payment history after an attempt."""
        profile = self.get_or_create_profile(customer_id)
        profile.total_attempts += 1

        if success:
            profile.total_recovered += attempt.get("amount", 0)

        # Update channel success rates
        if channel:
            if channel not in profile.channel_success_rates:
                profile.channel_success_rates[channel] = 0.0
            total_for_channel = sum(
                1 for p in profile.payment_history if p.channel_used == channel
            ) + 1
            successes_for_channel = sum(
                1 for p in profile.payment_history
                if p.channel_used == channel and p.status == "success"
            ) + (1 if success else 0)
            profile.channel_success_rates[channel] = successes_for_channel / total_for_channel

            # Update preferred channel
            if profile.channel_success_rates:
                profile.preferred_channel = max(
                    profile.channel_success_rates,
                    key=profile.channel_success_rates.get,  # type: ignore[arg-type]
                )

        # Add payment record
        record = PaymentRecord(
            payment_id=attempt.get("payment_id", ""),
            amount=attempt.get("amount", 0),
            channel_used=channel,
            status="success" if success else "failed",
            failure_type=attempt.get("failure_type", ""),
        )
        profile.payment_history.append(record)

        # Track failure type frequency
        ft = attempt.get("failure_type", "")
        if ft:
            profile.failure_type_counts[ft] = profile.failure_type_counts.get(ft, 0) + 1

        profile.last_contacted = datetime.now(timezone.utc)
        self.save_profile(profile)
        return profile

    def record_promise_to_pay(
        self,
        customer_id: str,
        amount: float,
        promised_date: str,
    ) -> PromiseToPay:
        """Record a customer commitment to pay by a specific date."""
        profile = self.get_or_create_profile(customer_id)
        promise = PromiseToPay(amount=amount, promised_date=promised_date)
        profile.promises.append(promise)
        self.save_profile(profile)
        return promise

    def fulfill_promise(self, customer_id: str, promise_id: str) -> bool:
        """Mark a promise as fulfilled. Returns True if found."""
        profile = self.get_or_create_profile(customer_id)
        for promise in profile.promises:
            if promise.promise_id == promise_id:
                promise.fulfilled = True
                self.save_profile(profile)
                return True
        return False

    def check_fulfilled_promises(self, customer_id: str) -> list[PromiseToPay]:
        """Return all fulfilled promises for a customer."""
        profile = self.get_or_create_profile(customer_id)
        return [p for p in profile.promises if p.fulfilled]

    # --- Salary Window ---

    def update_salary_window(
        self,
        customer_id: str,
        pay_day: int,
        last_salary_date: str = "",
    ) -> SalaryWindow:
        """Update salary window information for a customer."""
        profile = self.get_or_create_profile(customer_id)
        profile.salary_window.typical_pay_day = pay_day
        if last_salary_date:
            profile.salary_window.last_salary_date = last_salary_date
            if last_salary_date not in profile.salary_window.salary_history:
                profile.salary_window.salary_history.append(last_salary_date)
        self.save_profile(profile)
        return profile.salary_window

    def check_salary_liquidity(self, customer_id: str, current_day: int) -> bool:
        """Determine if customer is currently in high-liquidity salary window.

        Returns True if the current day is within ±2 days of the typical pay day
        or if salary was credited in the last 3 days.
        """
        profile = self.get_or_create_profile(customer_id)
        sw = profile.salary_window

        if not sw.typical_pay_day:
            return False

        # Check if current day is within salary window (±2 days of pay day)
        pay_day = sw.typical_pay_day
        days_diff = abs(current_day - pay_day)
        # Handle month wrap-around (e.g., pay_day=28, current_day=3 -> diff=25, but actual is 5)
        if days_diff > 15:
            days_diff = 30 - days_diff

        if days_diff <= 2:
            return True

        # Check if salary was credited recently (last 3 days)
        if sw.last_salary_date:
            try:
                last = datetime.fromisoformat(sw.last_salary_date)
                now = datetime.now(timezone.utc)
                days_since = (now - last).days
                if days_since <= 3:
                    return True
            except (ValueError, TypeError):
                pass

        return False

    # --- Channel Optimization ---

    def get_optimal_channel(self, customer_id: str) -> str:
        """Returns channel with highest historical conversion rate for this customer.

        Falls back to 'sms' if no history exists.
        """
        profile = self.get_or_create_profile(customer_id)
        if profile.preferred_channel and profile.preferred_channel in profile.channel_success_rates:
            return profile.preferred_channel
        return "sms"

    def get_channel_confidence(self, customer_id: str, channel: str) -> float:
        """Return success rate confidence for a specific channel."""
        profile = self.get_or_create_profile(customer_id)
        return profile.channel_success_rates.get(channel, 0.5)

    # --- Persistence Helpers ---

    def _load_all(self) -> None:
        """Load all profiles from disk."""
        if not self._persist_dir:
            return
        for path in self._persist_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                profile = CustomerProfile(**data)
                self._profiles[profile.customer_id] = profile
            except Exception:
                continue

    def list_profiles(self) -> list[str]:
        """Return all tracked customer IDs."""
        return list(self._profiles.keys())

    def get_stats(self) -> dict:
        """Return aggregate memory statistics."""
        total = len(self._profiles)
        total_recovered = sum(p.total_recovered for p in self._profiles.values())
        total_promises = sum(len(p.promises) for p in self._profiles.values())
        fulfilled = sum(
            sum(1 for p in prof.promises if p.fulfilled)
            for prof in self._profiles.values()
        )
        return {
            "total_customers": total,
            "total_recovered": total_recovered,
            "total_promises": total_promises,
            "fulfilled_promises": fulfilled,
            "promise_fulfillment_rate": fulfilled / total_promises if total_promises > 0 else 0.0,
        }
