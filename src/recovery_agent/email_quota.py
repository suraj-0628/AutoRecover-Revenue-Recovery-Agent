"""How much email allowance is left today — and whether to spend it.

The free Brevo plan sends 300 transactional emails a day. An agent working a
batch of two hundred failed payments can exhaust that in one click, and the
three hundred and first send does not fail loudly: it simply never arrives,
so a customer who was promised a payment link never gets one and the case
stalls looking like the customer ignored it.

Two sources, deliberately:

  local     counted from our own dispatch log, which records every delivery
            this system actually made. Always available, exact for OUR sends,
            blind to anything else using the account.
  provider  Brevo's own per-day statistics. Authoritative and complete, but
            only when BREVO_API_KEY is set.

The remaining figure takes the WORSE of the two, because a cap you have
already hit elsewhere is still hit. A reserve is held back so the demo — or
whatever matters most — is never the send that discovers the ceiling.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def daily_limit() -> int:
    try:
        return int(os.getenv("BREVO_DAILY_LIMIT", "") or 300)
    except (TypeError, ValueError):
        return 300


def reserve() -> int:
    """Sends held back from the agent. Zero disables the budget guardrail."""
    try:
        return max(0, int(os.getenv("EMAIL_QUOTA_RESERVE", "") or 20))
    except (TypeError, ValueError):
        return 20


def _dispatch_log_path(state_dir: str | None = None) -> Path:
    outbox = os.getenv("OUTBOX_DIR")
    if outbox and state_dir is None:
        return Path(outbox) / "dispatch_log.jsonl"
    base = Path(state_dir or os.getenv("STATE_DIR", "data"))
    return base / "outbox" / "dispatch_log.jsonl"


def sent_today(state_dir: str | None = None, now: datetime | None = None) -> int:
    """Emails this system delivered today, from its own dispatch log.

    Counts only results marked delivered — an attempt that failed reached
    nobody and consumed no allowance. Never raises: an unreadable log means
    an unknown count, and an unknown count must not block recovery, so it
    reads as zero and the provider figure (when present) covers for it.
    """
    path = _dispatch_log_path(state_dir)
    if not path.exists():
        return 0
    today = (now or datetime.now(timezone.utc)).date()
    count = 0
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            for result in entry.get("results") or []:
                if result.get("channel") != "email" or not result.get("delivered"):
                    continue
                stamp = str(result.get("timestamp") or "")
                try:
                    when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue
                if when.astimezone(timezone.utc).date() == today:
                    count += 1
    except Exception:
        return count
    return count


def _provider_sent_today(now: datetime | None = None) -> tuple[int | None, dict]:
    """Brevo's own count for today, plus the raw report. (None, {}) if unknown."""
    try:
        from recovery_agent.integrations.brevo_client import get_brevo_client
        stats = get_brevo_client().get_smtp_statistics(days=1)
        if stats.get("status") != "ok":
            return None, {"status": stats.get("status"),
                          "reason": stats.get("reason") or stats.get("error", "")}
        today = (now or datetime.now(timezone.utc)).date().isoformat()
        for report in stats.get("reports") or []:
            if str(report.get("date")) == today:
                return int(report.get("requests") or 0), report
        # A day with no sends yet simply has no row.
        return 0, {}
    except Exception as exc:
        return None, {"status": "error", "reason": str(exc)[:120]}


def status(state_dir: str | None = None, now: datetime | None = None) -> dict:
    """Everything the ops view and the budget guardrail need. Never raises."""
    limit = daily_limit()
    local = sent_today(state_dir, now)
    provider, provider_detail = _provider_sent_today(now)

    # The worse of the two: allowance spent by anything else on this account
    # is still spent, and our own log is authoritative for our own sends.
    used = local if provider is None else max(local, provider)
    remaining = max(0, limit - used)
    held = reserve()
    return {
        "limit": limit,
        "sent_today": used,
        "sent_today_local": local,
        "sent_today_provider": provider,
        "remaining": remaining,
        "reserve": held,
        "spendable": max(0, remaining - held),
        "exhausted": remaining - held <= 0,
        "source": "provider+local" if provider is not None else "local",
        "provider_detail": provider_detail,
    }


def may_send(state_dir: str | None = None, now: datetime | None = None) -> tuple[bool, str]:
    """(allowed, why not). The reserve is what makes this a budget rather
    than a wall hit at the worst possible moment."""
    if reserve() == 0:
        return True, ""
    s = status(state_dir, now)
    if s["exhausted"]:
        return False, (f"today's email allowance is spent: {s['sent_today']} of "
                       f"{s['limit']} sent, and the last {s['reserve']} are "
                       f"held in reserve")
    return True, ""
