"""Brevo, read-only: how much email allowance is left, and what happened to it.

Email is the cheapest channel and the easiest to over-spend: the free plan
allows 300 transactional sends a day, and an agent working a batch of two
hundred failed payments can eat that without noticing. Nothing in this system
knew the number.

Two reads answer it. `GET /v3/account` carries the plan and its credits;
`GET /v3/smtp/statistics/reports` carries per-day requests, delivered and
bounces — the authoritative view, including mail this system did not send.
Both are reads: they deliver nothing and consume no allowance.

The SMTP relay credentials in `SMTP_PASS` (`xsmtpsib-…`) are NOT an API key
and will not authenticate here. The v3 API needs an `xkeysib-…` key from
Brevo → SMTP & API → API Keys, in `BREVO_API_KEY`. Without it every function
degrades to "unknown" and the local meter in `email_quota` carries on alone.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

BREVO_BASE_URL = os.getenv("BREVO_BASE_URL", "https://api.brevo.com/v3")


class BrevoClient:
    """Read-only client for account allowance and send statistics."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key if api_key is not None else os.getenv("BREVO_API_KEY", "")

    @property
    def can_read(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, params: dict | None = None,
             timeout: int = 15) -> dict[str, Any]:
        if not self.can_read:
            return {"status": "skipped", "reason": "brevo_not_configured",
                    "detail": "BREVO_API_KEY is not set (the SMTP relay "
                              "password is a different credential and will "
                              "not work here)."}
        try:
            response = requests.get(
                f"{BREVO_BASE_URL}{path}",
                headers={"api-key": self.api_key, "accept": "application/json"},
                params=params or {}, timeout=timeout)
            response.raise_for_status()
            return {"status": "ok", "data": response.json()}
        except Exception as e:
            logger.warning("[Brevo] read %s failed: %s", path, e)
            return {"status": "error", "error": str(e)[:200]}

    def get_account(self) -> dict[str, Any]:
        """Plan and remaining credits.

        Brevo reports credits per plan entry; a free transactional plan often
        reports its allowance as a daily limit rather than a credit balance,
        so callers must treat a missing number as unknown, not as zero.
        """
        out = self._get("/account")
        if out.get("status") != "ok":
            return out
        data = out["data"] or {}
        plans = data.get("plan") or []
        email_credits = None
        for p in plans:
            if str(p.get("creditsType", "")).lower().startswith("send"):
                email_credits = p.get("credits")
                break
        return {
            "status": "ok",
            "email": data.get("email", ""),
            "company": data.get("companyName", ""),
            "plans": [{"type": p.get("type"), "credits_type": p.get("creditsType"),
                       "credits": p.get("credits")} for p in plans],
            "email_credits": email_credits,
        }

    def get_smtp_statistics(self, days: int = 7) -> dict[str, Any]:
        """Per-day transactional stats. Brevo caps the window at 30 days."""
        out = self._get("/smtp/statistics/reports",
                        {"days": max(1, min(int(days), 30))})
        if out.get("status") != "ok":
            return out
        data = out["data"] or {}
        reports = data.get("reports") or []
        return {
            "status": "ok",
            "reports": [{
                "date": r.get("date"),
                "requests": int(r.get("requests") or 0),
                "delivered": int(r.get("delivered") or 0),
                "hard_bounces": int(r.get("hardBounces") or 0),
                "soft_bounces": int(r.get("softBounces") or 0),
                "blocked": int(r.get("blocked") or 0),
            } for r in reports],
        }


_client: BrevoClient | None = None


def get_brevo_client() -> BrevoClient:
    global _client
    if _client is None:
        _client = BrevoClient()
    return _client


def reset_client() -> None:
    global _client
    _client = None
