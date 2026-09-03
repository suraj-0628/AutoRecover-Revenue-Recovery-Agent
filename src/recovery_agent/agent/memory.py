"""Long-Term Memory Engine — persistent customer profiles with file locking.

Single unified JSON store with cross-platform file locking.
Persists across process restarts and tracks rate limits accurately.

Thread Safety:
  - Uses filelock.FileLock for cross-platform atomic file access
  - Works on POSIX, Windows, and macOS without platform-specific code
  - Single unified memory_store.json (no per-customer files)
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from recovery_agent.models import (
    CustomerProfile,
    PaymentRecord,
    PromiseToPay,
    SalaryWindow,
)


class CustomerMemoryStore:
    """Persistent store for customer profiles with cross-platform file locking.

    Uses a single unified JSON file with filelock.FileLock for atomic access.
    Thread-safe and process-safe across POSIX and Windows.
    """

    def __init__(self, persist_dir: str | None = None):
        self._profiles: dict[str, CustomerProfile] = {}
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._store_path: Path | None = None
        self._lock = threading.Lock()

        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._store_path = self._persist_dir / "memory_store.json"
            self._load_all()

    @contextmanager
    def _file_lock(self):
        """Acquire cross-platform file lock for atomic read/write."""
        if self._store_path is None:
            yield
            return

        lock_path = self._store_path.with_suffix(".lock")
        lock = FileLock(str(lock_path), timeout=10)
        try:
            with lock:
                yield
        except Timeout:
            print(f"[memory] WARNING: Could not acquire file lock within 10s: {lock_path}")
            raise RuntimeError(
                f"File lock timeout after 10s on {lock_path}. "
                "Another process may be stuck. Retry or check for stale locks."
            )

    # --- Profile Access ---

    def get_or_create_profile(self, customer_id: str) -> CustomerProfile:
        """Fetch existing memory or initialize new profile."""
        with self._lock:
            if customer_id in self._profiles:
                return self._profiles[customer_id]
            profile = CustomerProfile(customer_id=customer_id)
            self._profiles[customer_id] = profile
            return profile

    def save_profile(self, profile: CustomerProfile) -> None:
        """Persist profile to memory and to disk with file locking."""
        with self._lock:
            self._profiles[profile.customer_id] = profile
            self._persist_to_disk()

    def _persist_to_disk(self) -> None:
        """Write all profiles to the unified JSON store with file locking."""
        if not self._store_path:
            return

        data = {}
        for cid, profile in self._profiles.items():
            try:
                data[cid] = profile.model_dump(mode="json")
            except Exception:
                continue

        with self._file_lock():
            try:
                # Write to temp file first, then atomic rename
                tmp_path = self._store_path.with_suffix(".tmp")
                tmp_path.write_text(json.dumps(data, indent=2, default=str))
                tmp_path.rename(self._store_path)
            except Exception as e:
                print(f"[memory] Error persisting to disk: {e}")

    def _load_all(self) -> None:
        """Load all profiles from the unified JSON store with file locking."""
        if not self._store_path or not self._store_path.exists():
            return

        with self._file_lock():
            try:
                raw = self._store_path.read_text()
                data = json.loads(raw)
                for cid, profile_data in data.items():
                    try:
                        profile = CustomerProfile(**profile_data)
                        self._profiles[cid] = profile
                    except Exception:
                        continue
            except Exception as e:
                print(f"[memory] Error loading from disk: {e}")

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

            if profile.channel_success_rates:
                profile.preferred_channel = max(
                    profile.channel_success_rates,
                    key=profile.channel_success_rates.get,  # type: ignore[arg-type]
                )

        record = PaymentRecord(
            payment_id=attempt.get("payment_id", ""),
            amount=attempt.get("amount", 0),
            channel_used=channel,
            status="success" if success else "failed",
            failure_type=attempt.get("failure_type", ""),
        )
        profile.payment_history.append(record)

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
        profile = self.get_or_create_profile(customer_id)
        promise = PromiseToPay(amount=amount, promised_date=promised_date)
        profile.promises.append(promise)
        self.save_profile(profile)
        return promise

    def fulfill_promise(self, customer_id: str, promise_id: str) -> bool:
        profile = self.get_or_create_profile(customer_id)
        for promise in profile.promises:
            if promise.promise_id == promise_id:
                promise.fulfilled = True
                self.save_profile(profile)
                return True
        return False

    def check_fulfilled_promises(self, customer_id: str) -> list[PromiseToPay]:
        profile = self.get_or_create_profile(customer_id)
        return [p for p in profile.promises if p.fulfilled]

    # --- Salary Window ---

    def update_salary_window(
        self,
        customer_id: str,
        pay_day: int,
        last_salary_date: str = "",
    ) -> SalaryWindow:
        profile = self.get_or_create_profile(customer_id)
        profile.salary_window.typical_pay_day = pay_day
        if last_salary_date:
            profile.salary_window.last_salary_date = last_salary_date
            if last_salary_date not in profile.salary_window.salary_history:
                profile.salary_window.salary_history.append(last_salary_date)
        self.save_profile(profile)
        return profile.salary_window

    def check_salary_liquidity(self, customer_id: str, current_day: int) -> bool:
        profile = self.get_or_create_profile(customer_id)
        sw = profile.salary_window

        if not sw.typical_pay_day:
            return False

        pay_day = sw.typical_pay_day
        days_diff = abs(current_day - pay_day)
        if days_diff > 15:
            days_diff = 30 - days_diff

        if days_diff <= 2:
            return True

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
        profile = self.get_or_create_profile(customer_id)
        if profile.preferred_channel and profile.preferred_channel in profile.channel_success_rates:
            return profile.preferred_channel
        return "sms"

    def get_channel_confidence(self, customer_id: str, channel: str) -> float:
        profile = self.get_or_create_profile(customer_id)
        return profile.channel_success_rates.get(channel, 0.5)

    # --- Rate Limit Tracking ---

    def get_communication_count_24h(self, customer_id: str) -> int:
        """Count communications in the last 24 hours for frequency cap enforcement."""
        profile = self.get_or_create_profile(customer_id)
        now = datetime.now(timezone.utc)
        count = 0
        for record in profile.payment_history:
            if record.channel_used and record.status in ("success", "failed"):
                time_diff = (now - record.timestamp).total_seconds()
                if time_diff < 86400:
                    count += 1
        return count

    def get_last_communication_time(self, customer_id: str) -> datetime | None:
        """Get the timestamp of the last communication."""
        profile = self.get_or_create_profile(customer_id)
        if profile.last_contacted:
            return profile.last_contacted
        return None

    # --- Persistence Helpers ---

    def list_profiles(self) -> list[str]:
        return list(self._profiles.keys())

    def get_stats(self) -> dict:
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
            "store_path": str(self._store_path) if self._store_path else "in-memory",
        }
