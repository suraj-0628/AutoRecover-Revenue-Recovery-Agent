"""Payday Scheduler — regional payroll cycle detection for retry timing.

Calculates optimal retry timestamps based on regional payroll cycles.
Inspired by Redux Code 51 strategy — "first-in-line advantage":
retry at the exact instant liquidity returns, before other subscriptions
and utility bills clear the balance.

Source: Industry research — INDUSTRY-RESEARCH.md
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum


class PayrollCycle(str, Enum):
    """Regional payroll cycle types."""
    MONTHLY = "monthly"
    BIWEEKLY = "biweekly"
    SEMIMONTHLY = "semimonthly"
    WEEKLY = "weekly"
    UNKNOWN = "unknown"


# Regional payroll cycle configurations
# Source: Redux Code 51 research, Slicker regional pay cycles
REGIONAL_PAYROLL: dict[str, dict] = {
    "IN": {
        "cycle": PayrollCycle.MONTHLY,
        "typical_days": [28],  # Last working day of month
        "tolerance_days": 2,
        "timezone": "Asia/Kolkata",
        "notes": "Indian salaries typically credited on last working day",
    },
    "US": {
        "cycle": PayrollCycle.BIWEEKLY,
        "typical_days": [1, 15],  # 1st and 15th of month
        "tolerance_days": 1,
        "timezone": "America/New_York",
        "notes": "US bi-weekly payroll centered on 1st and 15th",
    },
    "GB": {
        "cycle": PayrollCycle.MONTHLY,
        "typical_days": [28],
        "tolerance_days": 2,
        "timezone": "Europe/London",
        "notes": "UK monthly payroll, end of month",
    },
    "DE": {
        "cycle": PayrollCycle.MONTHLY,
        "typical_days": [27],
        "tolerance_days": 3,
        "timezone": "Europe/Berlin",
        "notes": "German monthly payroll, varies by employer",
    },
    "EU": {
        "cycle": PayrollCycle.MONTHLY,
        "typical_days": [27],
        "tolerance_days": 3,
        "timezone": "Europe/Paris",
        "notes": "European monthly payroll, varies by country",
    },
    "AU": {
        "cycle": PayrollCycle.BIWEEKLY,
        "typical_days": [1, 15],
        "tolerance_days": 2,
        "timezone": "Australia/Sydney",
        "notes": "Australian fortnightly payroll",
    },
}


class PaydayScheduler:
    """Calculates optimal retry time based on regional payroll cycles.

    "First-in-line advantage": retry at 12:01 AM local time on payday,
    before other subscriptions/utility bills clear the balance.

    Usage:
        scheduler = PaydayScheduler()
        target = scheduler.calculate_next_payday("IN", current_time=datetime.now(timezone.utc))
        is_window = scheduler.is_in_payday_window("IN", current_time=datetime.now(timezone.utc))
    """

    def __init__(self) -> None:
        self.regional_config = REGIONAL_PAYROLL

    def get_config(self, country_code: str) -> dict:
        """Get payroll configuration for a country."""
        code = country_code.upper()
        return self.regional_config.get(code, {
            "cycle": PayrollCycle.UNKNOWN,
            "typical_days": [],
            "tolerance_days": 2,
            "timezone": "UTC",
            "notes": "Unknown region, using default 2-day tolerance",
        })

    def calculate_next_payday(
        self,
        country_code: str,
        current_time: datetime | None = None,
    ) -> datetime | None:
        """Calculate the next payday timestamp for a region.

        Returns a datetime at 12:01 AM local time on the next payday.
        Returns None if no payroll cycle is configured.
        """
        config = self.get_config(country_code)
        typical_days = config.get("typical_days", [])

        if not typical_days:
            return None

        now = current_time or datetime.now(timezone.utc)
        cycle = config["cycle"]

        if cycle == PayrollCycle.MONTHLY:
            return self._next_monthly_payday(now, typical_days)
        elif cycle == PayrollCycle.BIWEEKLY:
            return self._next_biweekly_payday(now, typical_days)
        elif cycle == PayrollCycle.WEEKLY:
            return self._next_weekly_payday(now, typical_days[0] if typical_days else 5)
        else:
            return None

    def _next_monthly_payday(self, now: datetime, typical_days: list[int]) -> datetime:
        """Find next monthly payday (typically last few days of month)."""
        year = now.year
        month = now.month

        for day in sorted(typical_days, reverse=True):
            try:
                target = datetime(year, month, day, 0, 1, tzinfo=timezone.utc)
            except ValueError:
                continue

            if target > now:
                return target

        # If all days this month have passed, try next month
        if month == 12:
            month = 1
            year += 1
        else:
            month += 1

        for day in sorted(typical_days, reverse=True):
            try:
                target = datetime(year, month, day, 0, 1, tzinfo=timezone.utc)
                return target
            except ValueError:
                continue

        return None

    def _next_biweekly_payday(self, now: datetime, typical_days: list[int]) -> datetime:
        """Find next biweekly payday (typically 1st and 15th)."""
        year = now.year
        month = now.month

        # Check remaining days this month
        for day in sorted(typical_days):
            try:
                target = datetime(year, month, day, 0, 1, tzinfo=timezone.utc)
            except ValueError:
                continue
            if target > now:
                return target

        # Check next month
        if month == 12:
            month = 1
            year += 1
        else:
            month += 1

        for day in sorted(typical_days):
            try:
                target = datetime(year, month, day, 0, 1, tzinfo=timezone.utc)
                return target
            except ValueError:
                continue

        return None

    def _next_weekly_payday(self, now: datetime, weekday: int) -> datetime:
        """Find next weekly payday (on a specific day of week)."""
        days_ahead = weekday - now.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        target = now + timedelta(days=days_ahead)
        return target.replace(hour=0, minute=1, second=0, microsecond=0)

    def is_in_payday_window(
        self,
        country_code: str,
        current_time: datetime | None = None,
    ) -> bool:
        """Check if current time is within the payday liquidity window.

        The window is ±tolerance_days from the typical payday.
        """
        config = self.get_config(country_code)
        tolerance = config.get("tolerance_days", 2)
        typical_days = config.get("typical_days", [])

        if not typical_days:
            return False

        now = current_time or datetime.now(timezone.utc)
        current_day = now.day

        return any(
            abs(current_day - day) <= tolerance
            for day in typical_days
        )

    def hours_until_payday(
        self,
        country_code: str,
        current_time: datetime | None = None,
    ) -> float:
        """Calculate hours until next payday.

        Returns negative value if we're currently in a payday window.
        """
        next_payday = self.calculate_next_payday(country_code, current_time)
        if not next_payday:
            return -1.0

        now = current_time or datetime.now(timezone.utc)
        diff = next_payday - now
        return diff.total_seconds() / 3600

    def get_payday_info(self, country_code: str = "IN") -> dict:
        """Get full payday information for a region."""
        config = self.get_config(country_code)
        now = datetime.now(timezone.utc)

        return {
            "country": country_code.upper(),
            "cycle": config["cycle"].value,
            "typical_days": config["typical_days"],
            "tolerance_days": config["tolerance_days"],
            "timezone": config["timezone"],
            "in_window": self.is_in_payday_window(country_code, now),
            "hours_until_payday": round(self.hours_until_payday(country_code, now), 1),
            "next_payday": self.calculate_next_payday(country_code, now).isoformat()
                if self.calculate_next_payday(country_code, now) else None,
            "notes": config.get("notes", ""),
        }
