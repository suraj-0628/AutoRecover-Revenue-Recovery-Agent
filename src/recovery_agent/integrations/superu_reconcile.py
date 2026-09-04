"""Turn SuperU's call records into billed cost against the right case.

The voice channel was the one place this system spent real money and could
only guess what it spent: `VOICE_COST_INR_PER_CALL`, a constant somebody
chose. SuperU knows the actual figure per call and will hand it over on a
read, and every call we place is tagged `campaign_id="recovery_{payment_id}"`
— so the invoice line joins straight back to the case that caused it.

Two meters come back, and SuperU labels neither with a currency. Their
published pricing settles which one is the invoice:

    cost                 what we are charged. SuperU bills a flat
                         **$0.02 per connected minute**, and the field matches
                         exactly — 0.02 for a 3s call and for a 27s call,
                         0.04 once a call passes one minute.
    telecom_total_cost   ~0.02 per SECOND, i.e. about 4x the platform charge.
                         NOT ours to pay: SuperU states that telephony is
                         included in the per-minute price, so this is their
                         own carrier cost, exposed for transparency. Counting
                         it would overstate a merchant's spend ~4x.

So only `cost` is billed, converted at USD_TO_INR. `telecom_total_cost` is
carried on the ledger entry as evidence, not added to the total; set
SUPERU_INCLUDE_TELECOM=1 if a future plan really does pass it through.
Both raw figures are stored verbatim, so a change of mind is a re-derive
rather than a re-fetch.

Reconciliation is idempotent on the call's own uuid, so this is safe to run
on a timer, twice, or after a crash.
"""
from __future__ import annotations

import os
from typing import Any

from recovery_agent import cost_ledger

#: What the two SuperU meters are denominated in, and the rate to convert.
#: Override once you have confirmed against the SuperU dashboard.
def _env(name: str, default: str) -> str:
    return (os.getenv(name, "") or default).strip().upper()


def _rate(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def usd_to_inr() -> float:
    return _rate("USD_TO_INR", 88.0)


def to_inr(amount: float, currency: str) -> float:
    amount = float(amount or 0)
    return amount * usd_to_inr() if currency == "USD" else amount


def include_telecom() -> bool:
    return os.getenv("SUPERU_INCLUDE_TELECOM", "").strip().lower() in (
        "1", "true", "yes", "on")


def call_cost_inr(call: dict) -> tuple[float, dict]:
    """What this call cost US in rupees, plus the raw evidence for it.

    Only the platform charge counts. SuperU's published price is $0.02 per
    connected minute with telephony included, and `cost` matches that meter
    exactly — so adding `telecom_total_cost` on top would bill a merchant for
    SuperU's own carrier bill, roughly quadrupling the figure.
    """
    platform_raw = float(call.get("cost") or 0)
    telecom_raw = float(call.get("telecom_total_cost")
                        or call.get("telecom_cost") or 0)
    platform_ccy = _env("SUPERU_COST_CURRENCY", "USD")
    telecom_ccy = _env("SUPERU_TELECOM_CURRENCY", "INR")

    inr = to_inr(platform_raw, platform_ccy)
    if include_telecom():
        inr += to_inr(telecom_raw, telecom_ccy)

    return round(inr, 4), {
        "cost": platform_raw,
        "cost_currency": platform_ccy,
        "rate_note": "SuperU published price: $0.02 per connected minute, "
                     "telephony included",
        "telecom_total_cost": telecom_raw,
        "telecom_currency": telecom_ccy,
        "telecom_billed_to_us": include_telecom(),
        "usd_to_inr": usd_to_inr(),
        "call_duration_seconds": call.get("call_duration_seconds"),
        "status": call.get("status"),
        "ended_reason": call.get("endedReason"),
    }


def payment_id_from_campaign(campaign_id: str) -> str:
    """`recovery_pay_abc` -> `pay_abc`. Anything else is not ours."""
    cid = str(campaign_id or "")
    return cid[len("recovery_"):] if cid.startswith("recovery_") else ""


def reconcile(client: Any = None, limit: int = 100,
              state_dir: str | None = None) -> dict:
    """Read SuperU's call log and record what has not been recorded yet.

    Never raises, never places a call, and never double-counts: each entry is
    keyed on the call's own uuid.
    """
    if client is None:
        from recovery_agent.integrations.superu_client import get_superu_client
        client = get_superu_client()

    result = client.get_call_logs(limit=limit)
    if result.get("status") != "ok":
        return {"status": result.get("status", "error"),
                "reason": result.get("reason") or result.get("error", ""),
                "recorded": 0, "already_known": 0, "unmatched": 0, "inr": 0.0}

    recorded = already = unmatched = 0
    total_inr = 0.0
    rows: list[dict] = []
    for call in result.get("calls") or []:
        call_id = str(call.get("id") or "")
        if not call_id:
            continue
        payment_id = payment_id_from_campaign(call.get("campaign_id"))
        if not payment_id:
            # A call this system did not place (a manual test from the SuperU
            # console, say). It cost money, but not on any case's behalf.
            unmatched += 1
            continue
        inr, raw = call_cost_inr(call)
        wrote = cost_ledger.record(
            surface=cost_ledger.SURFACE_VOICE,
            inr=inr,
            provenance=cost_ledger.BILLED,
            payment_id=payment_id,
            source_ref=call_id,
            qty=float(call.get("call_duration_seconds") or 0),
            unit="seconds",
            raw=raw,
            state_dir=state_dir,
        )
        if wrote:
            recorded += 1
            total_inr += inr
            rows.append({"payment_id": payment_id, "call_id": call_id,
                         "inr": inr, "seconds": raw["call_duration_seconds"]})
        else:
            already += 1

    return {"status": "ok", "recorded": recorded, "already_known": already,
            "unmatched": unmatched, "inr": round(total_inr, 2),
            "rows": rows,
            "provider_total_cost": result.get("total_cost"),
            "provider_calls": result.get("total")}
