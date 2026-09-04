"""Unit economics — what one recovered rupee costs to recover.

Every agent turn burns tokens, every voice call burns SuperU credits, every
discount is revenue deliberately given away, and every payment link spends a
unit of a 30-per-lifetime account quota. None of that was on a screen: the HUD
showed "Recovered: ₹X" with no idea whether recovering it cost ₹2 or ₹200.

This module is the ledger and the arithmetic, nothing else. Token usage is
recorded by the graph as each LLM call returns; contacts, calls and links are
recorded by the tools that verified delivery; the discount is the gap between
what was owed and what was accepted. Prices are env-tunable estimates and the
API labels them as such — the point is the ORDER of the numbers, per failure
kind, next to the revenue they bought back.

Reads and writes go through StateStore, so the same figures come out of the
live stack and the integration rig without either knowing about this module.
"""
from __future__ import annotations

import os
from typing import Any, Iterable

# ── prices (INR, env-tunable estimates) ────────────────────────────────

def _price(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def prices() -> dict[str, float]:
    return {
        # Flash-class token pricing in INR per million tokens.
        "llm_in_per_mtok": _price("LLM_COST_INR_PER_MTOK_IN", 25.0),
        "llm_out_per_mtok": _price("LLM_COST_INR_PER_MTOK_OUT", 75.0),
        # SuperU credits are real money; this is the per-call estimate.
        "voice_per_call": _price("VOICE_COST_INR_PER_CALL", 15.0),
        # Email goes through SMTP we already pay for; SMS is file-only in this
        # deployment, so its honest marginal cost is zero until it is real.
        "email_per_msg": _price("EMAIL_COST_INR", 0.0),
        "sms_per_msg": _price("SMS_COST_INR", 0.0),
    }


# ── recording (called from the graph and the frontend) ─────────────────

def record_llm_usage(payment_id: str, model: str, usage: dict | None) -> None:
    """Fold one LLM response's token usage into the case record. Never raises.

    `usage` is LangChain's AIMessage.usage_metadata: input_tokens,
    output_tokens, total_tokens. A response without it still counts the call —
    a call with unknown tokens is a call, not a free one.
    """
    try:
        from recovery_agent.state_store import StateStore
        store = StateStore()
        rec = store.get_payment(payment_id)
        if rec is None:
            return
        u = dict(rec.get("llm_usage") or {})
        u["calls"] = int(u.get("calls") or 0) + 1
        u["input_tokens"] = int(u.get("input_tokens") or 0) + int((usage or {}).get("input_tokens") or 0)
        u["output_tokens"] = int(u.get("output_tokens") or 0) + int((usage or {}).get("output_tokens") or 0)
        by_model = dict(u.get("by_model") or {})
        m = str(model or "unknown")
        by_model[m] = int(by_model.get(m) or 0) + 1
        u["by_model"] = by_model
        store.update_payment(payment_id, llm_usage=u)
        store.flush()
    except Exception:
        pass


def record_run_wall(payment_id: str, seconds: float) -> None:
    """Accumulate wall-clock time the agent spent on this case. Never raises."""
    try:
        from recovery_agent.state_store import StateStore
        store = StateStore()
        rec = store.get_payment(payment_id)
        if rec is None:
            return
        u = dict(rec.get("llm_usage") or {})
        u["wall_seconds"] = round(float(u.get("wall_seconds") or 0) + max(0.0, seconds), 1)
        u["runs"] = int(u.get("runs") or 0) + 1
        store.update_payment(payment_id, llm_usage=u)
        store.flush()
    except Exception:
        pass


# ── arithmetic (pure) ──────────────────────────────────────────────────

def _f(v: Any) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


#: The two populations on this dashboard, which must never be averaged
#: together without saying so.
ORIGIN_LIVE = "live"
ORIGIN_SEEDED = "seeded"


def case_origin(rec: dict) -> str:
    """Did the agent work this case for real, or is it volume we put there?

    LIVE means all three: the agent actually ran (it burned tokens), a real
    Razorpay order exists behind it, and it did not come from the demo bar.
    That is a case driven end to end through the checkout — the population to
    judge behaviour on.

    SEEDED is everything else: `/api/simulate` triggers, CSV batch imports,
    fixtures. Useful for showing the system works at volume, useless for
    "what does a recovery actually cost", because a case the agent never
    worked costs nothing and would drag every average toward zero.

    Both populations are Razorpay TEST MODE. This distinguishes how a case
    got here, not whether the money was real — no money in either is.
    """
    pid = str(rec.get("payment_id") or "")
    if pid.startswith("pay_sim_"):
        return ORIGIN_SEEDED
    agent_ran = int((rec.get("llm_usage") or {}).get("calls") or 0) > 0
    has_order = bool(rec.get("razorpay_order_id")
                     or rec.get("original_order_id"))
    return ORIGIN_LIVE if (agent_ran and has_order) else ORIGIN_SEEDED


def case_economics(rec: dict, p: dict[str, float] | None = None,
                   billed_voice: dict[str, float] | None = None) -> dict:
    """The full cost anatomy of one case. Pure — pass the record in.

    `billed_voice` maps payment_id -> rupees SuperU actually charged. Where a
    figure exists it REPLACES the per-call estimate: an invoice line beats our
    own multiplication, and the result says which it used.
    """
    p = p or prices()
    usage = rec.get("llm_usage") or {}
    contacts = rec.get("contacts") or []

    in_tok = int(usage.get("input_tokens") or 0)
    out_tok = int(usage.get("output_tokens") or 0)
    llm_cost = in_tok / 1e6 * p["llm_in_per_mtok"] + out_tok / 1e6 * p["llm_out_per_mtok"]

    voice_calls = sum(1 for c in contacts if c.get("channel") == "voice")
    emails = sum(1 for c in contacts if c.get("channel") == "email")
    sms = sum(1 for c in contacts if c.get("channel") == "sms")

    # A billed figure exists for a call the provider has finished accounting
    # for — which can be true even when this case's `contacts` never recorded
    # the delivery (an older case, a call placed before contact tracking).
    # Trust the invoice either way.
    pid = rec.get("payment_id", "")
    billed = (billed_voice or {}).get(pid)
    if billed is not None:
        voice_cost = float(billed)
        voice_provenance = "BILLED"
    else:
        voice_cost = voice_calls * p["voice_per_call"]
        voice_provenance = "MEASURED" if voice_calls else "NONE"
    comms_cost = (voice_cost
                  + emails * p["email_per_msg"] + sms * p["sms_per_msg"])

    amount = _f(rec.get("amount"))
    recovered = _f(rec.get("recovered_amount"))
    # An accepted discount is money deliberately given away to close the case;
    # it belongs on the cost side or the "recovered" number flatters itself.
    discount = round(max(0.0, amount - recovered), 2) if recovered > 0 else 0.0

    total = round(llm_cost + comms_cost + discount, 2)
    from recovery_agent.agent.classify import failure_kind
    return {
        "payment_id": rec.get("payment_id", ""),
        "origin": case_origin(rec),
        "failure_kind": failure_kind(rec) or "unknown",
        "status": rec.get("status", ""),
        "amount": round(amount, 2),
        "recovered": round(recovered, 2),
        "llm_calls": int(usage.get("calls") or 0),
        "tokens_in": in_tok,
        "tokens_out": out_tok,
        "wall_seconds": _f(usage.get("wall_seconds")),
        "voice_calls": voice_calls,
        "messages": emails + sms,
        "links_minted": len(rec.get("recovery_links") or []),
        "llm_cost": round(llm_cost, 2),
        "voice_cost": round(voice_cost, 2),
        "voice_provenance": voice_provenance,
        "comms_cost": round(comms_cost, 2),
        "discount_given": discount,
        "total_cost": total,
        "net_recovered": round(recovered - total, 2),
    }


def summarise(records: Iterable[dict], scope: str = "all") -> dict:
    """Aggregate economics over the cases the agent has touched.

    `scope` is "live", "seeded" or "all". The two populations answer different
    questions — live shows what recovery costs and how the agent behaves,
    seeded shows the system holding up at volume — and blending them makes
    both misleading, because cases the agent never worked cost nothing.
    """
    p = prices()
    # Invoice lines, where the provider has given us any.
    try:
        from recovery_agent import cost_ledger
        billed_voice = cost_ledger.by_payment(cost_ledger.SURFACE_VOICE)
    except Exception:
        billed_voice = {}
    rows = []
    for rec in records:
        worked = (rec.get("llm_usage") or rec.get("contacts")
                  or rec.get("recovery_links")
                  or rec.get("payment_id", "") in billed_voice
                  or rec.get("status") in ("recovering", "recovered",
                                           "escalated", "unrecoverable"))
        if not worked:
            continue
        row = case_economics(rec, p, billed_voice=billed_voice)
        if scope in (ORIGIN_LIVE, ORIGIN_SEEDED) and row["origin"] != scope:
            continue
        rows.append(row)

    def _bucket(items: list[dict]) -> dict:
        recovered = sum(r["recovered"] for r in items)
        cost = sum(r["total_cost"] for r in items)
        return {
            "cases": len(items),
            "recovered": round(recovered, 2),
            "llm_cost": round(sum(r["llm_cost"] for r in items), 2),
            "voice_cost": round(sum(r["voice_cost"] for r in items), 2),
            "voice_cost_billed": round(
                sum(r["voice_cost"] for r in items
                    if r["voice_provenance"] == "BILLED"), 2),
            "comms_cost": round(sum(r["comms_cost"] for r in items), 2),
            "discount_given": round(sum(r["discount_given"] for r in items), 2),
            "total_cost": round(cost, 2),
            "net_recovered": round(recovered - cost, 2),
            "llm_calls": sum(r["llm_calls"] for r in items),
            "tokens": sum(r["tokens_in"] + r["tokens_out"] for r in items),
            "voice_calls": sum(r["voice_calls"] for r in items),
            "messages": sum(r["messages"] for r in items),
            "links_minted": sum(r["links_minted"] for r in items),
            # Cost to bring back ₹100. The one number a merchant reads first.
            "cost_per_100_recovered": (round(cost / recovered * 100, 2)
                                       if recovered > 0 else None),
        }

    by_kind: dict[str, list[dict]] = {}
    for r in rows:
        by_kind.setdefault(r["failure_kind"], []).append(r)

    recovered_rows = [r for r in rows if r["recovered"] > 0]
    totals = _bucket(rows)
    totals["recovered_cases"] = len(recovered_rows)
    totals["avg_cost_per_recovery"] = (
        round(sum(r["total_cost"] for r in recovered_rows) / len(recovered_rows), 2)
        if recovered_rows else None)

    rows.sort(key=lambda r: r["total_cost"], reverse=True)
    try:
        from recovery_agent import cost_ledger
        ledger = cost_ledger.summarise()
    except Exception:
        ledger = {}

    billed_cases = sum(1 for r in rows if r["voice_provenance"] == "BILLED")
    attributed = sum(r["voice_cost"] for r in rows
                     if r["voice_provenance"] == "BILLED")
    # Money the provider charged for calls whose case record no longer exists
    # (a purged data dir, an older demo). It was still spent. Reporting only
    # what happens to still have a record would understate real spend, which
    # is the opposite of the point of this view.
    ledger_voice = float((ledger.get("surfaces") or {})
                         .get("voice", {}).get("inr") or 0)
    unattributed = round(max(0.0, ledger_voice - attributed), 2)
    return {
        "scope": scope,
        # Both populations are Razorpay TEST MODE — this says how a case got
        # here, not that any of the money moved.
        "test_mode": str(os.getenv("RAZORPAY_KEY_ID", "")).startswith("rzp_test"),
        "prices": p,
        "pricing_note": ("Quantities — tokens, calls, messages, links, "
                         "discounts — are measured, not estimated. Voice is "
                         "priced from SuperU's own per-call billing where it "
                         "has been reconciled; everything else is priced at "
                         "the configured INR rates."),
        "provenance": {
            "voice_billed_cases": billed_cases,
            "voice_billed_inr": round(attributed, 2),
            # Billed voice spend with no surviving case record.
            "voice_billed_unattributed_inr": unattributed,
            "voice_billed_total_inr": round(ledger_voice, 2),
            "ledger": ledger,
        },
        "totals": totals,
        "by_failure_kind": {k: _bucket(v) for k, v in sorted(by_kind.items())},
        "cases": rows[:50],
    }
