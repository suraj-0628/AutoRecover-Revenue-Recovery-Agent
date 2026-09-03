"""ReAct Agent Tools — @tool decorator for LangGraph ToolNode.

Each tool is a real implementation that does actual work:
- Razorpay SDK calls (payment links, status checks, customer history)
- SuperU voice calls
- Knowledge base search (RAG)
- Escalation ticket creation
- Scheduled retry

The LLM calls these tools via tool_calls. ToolNode executes them automatically.
ToolRuntime is injected by ToolNode at runtime (not in function signature).

Architecture: Context7-grounded @tool pattern from langgraph docs.
"""
from __future__ import annotations

import json
import re
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain.tools import tool

from recovery_agent.agent import ladder


def _live_record(payment_id: str) -> dict:
    """The durable case record. A Case object is rebuilt on every hand-off run,
    so anything read from it alone forgets what earlier rungs did."""
    try:
        from recovery_agent.state_store import StateStore
        return StateStore().get_payment(payment_id) or {}
    except Exception:
        return {}


def _still_worth_doing(payment_id: str, what: str) -> str:
    """Refuse an action that costs money or reaches the customer, when the case
    is already settled or the customer is mid-response.

    Perception tells the agent the case is settled; this makes it so even when
    the agent has not looked, and even inside a single turn where the situation
    changed between one tool call and the next. Live, a customer clicked the
    push at 04:16:51 and the agent created a discounted link at 04:17:02 —
    within the same turn, so nothing had a chance to re-brief it.

    Returns a JSON refusal, or "" to proceed.
    """
    def _remember_refusal(why: str) -> None:
        """A refusal the next run can see.

        `tool_call_history` resets each run, so a plan refused in one run looked
        brand new in the next. Live, the agent proposed the same blocked link
        three runs running — and said so itself, "third consecutive mid-payment
        block observed" — because nothing carried the refusal forward.
        """
        try:
            from recovery_agent.state_store import StateStore
            store = StateStore()
            rec = store.get_payment(payment_id)
            if rec is None:
                return
            refusals = dict(rec.get("refusals") or {})
            key = f"{what}: {why}"
            refusals[key] = int(refusals.get(key, 0)) + 1
            store.update_payment(payment_id, refusals=refusals)
            store.flush()
        except Exception:
            pass

    from recovery_agent.agent.perception import ground_truth
    facts = ground_truth(payment_id)
    if facts.get("settled"):
        _remember_refusal("the case is already settled")
        return json.dumps({
            "status": "blocked",
            "reason": f"this case is already settled — INR "
                      f"{facts.get('received', 0):,.2f} has been received, so "
                      f"{what} would reach a customer who has already paid",
            "guidance": "Call close_case with outcome='recovered'.",
        })

    rec = _live_record(payment_id)
    push = rec.get("pending_push") or {}
    outcome = rec.get("push_outcome") or {}

    # "Mid-payment" is a WINDOW, not a state. A click means someone is paying
    # right now; ten minutes later it means nothing, and a click that ended in a
    # decline means less than nothing. Left unbounded this froze a live case:
    # three consecutive runs refused every action because of one click that had
    # already failed at the bank.
    fresh = False
    if outcome.get("action") == "acted" and outcome.get("at"):
        try:
            clicked = datetime.fromisoformat(str(outcome["at"]).replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - clicked).total_seconds() / 60
            fresh = age < float(os.getenv("MID_PAYMENT_WINDOW_MINUTES", "4"))
        except (ValueError, TypeError):
            fresh = False

    if push.get("sent_at") and fresh:
        _remember_refusal("the customer was mid-payment")
        return json.dumps({
            "status": "blocked",
            "reason": "the customer has just acted on your notification and is "
                      "in the middle of paying; interrupting now risks charging "
                      "or discounting someone who is already completing",
            "guidance": "Call wait_for_customer and let them finish. You will be "
                        "started again with the outcome.",
        })
    return ""




# ═══════════════════════════════════════════════════════════════
# CONTEXT SCHEMA — passed to every tool via ToolRuntime
# ═══════════════════════════════════════════════════════════════

from dataclasses import dataclass


@dataclass
class RecoveryContext:
    """Runtime context accessible from tools via runtime.context."""
    guardrail_engine: Any = None
    case: Any = None


# ═══════════════════════════════════════════════════════════════
# TOOLS — @tool decorator. runtime injected by ToolNode at call time.
# ═══════════════════════════════════════════════════════════════

@tool
def diagnose_payment_failure(
    payment_id: str,
    failure_code: str,
    failure_reason: str,
    amount: int = 0,
    customer_id: str = "",
    runtime=None,
) -> str:
    """Diagnose why a payment failed using Razorpay error mapping and LLM analysis.

    Args:
        payment_id: The Razorpay payment ID
        failure_code: The error code from Razorpay (e.g., '51', 'card_expired')
        failure_reason: The human-readable failure reason
        amount: Payment amount in RUPEES (from context, if available). This
            docstring used to say "in paise", so the model dutifully passed paise
            and the diagnosis then reasoned about an order 100x larger than the
            customer's. The unit is rupees, like every other tool.
        customer_id: Customer ID (from context, if available)

    Returns:
        JSON with root_cause, confidence, reasoning, and recommended_action
    """
    from recovery_agent.agent.diagnosis import run_diagnosis
    from recovery_agent.models import Case, PaymentEvent

    # Defensive: normalise against the real case rather than trusting the
    # argument, since a wrong amount here silently skews the diagnosis.
    _case = getattr(getattr(runtime, "context", None), "case", None) if runtime else None
    _owed = float(getattr(getattr(_case, "payment", None), "amount", 0) or 0)
    if _owed and amount and abs(float(amount) - _owed * 100) < 1:
        amount = _owed

    case = Case(
        payment=PaymentEvent(
            event_type="payment_failed",
            payment_id=payment_id,
            amount=amount,
            currency="INR",
            failure_code=failure_code,
            failure_reason=failure_reason,
            customer_id=customer_id,
        ),
    )
    case = run_diagnosis(case)

    if case.diagnosis:
        return json.dumps({
            "root_cause": case.diagnosis.root_cause.value,
            "confidence": case.diagnosis.confidence,
            "reasoning": case.diagnosis.reasoning,
            "category": case.diagnosis.category,
        })
    return json.dumps({"root_cause": "unknown", "confidence": 0.0, "reasoning": "Diagnosis failed"})


@tool
def check_payment_status(
    payment_id: str,
    runtime=None,
) -> str:
    """Check the current status of a payment on Razorpay.

    Args:
        payment_id: The Razorpay payment ID (e.g., pay_abc123)

    Returns:
        JSON with current_status, amount, method, bank, failure_reason
    """
    from recovery_agent.agent.perception import ground_truth, as_briefing
    from recovery_agent.razorpay_client import RazorpayClient

    # This used to answer a narrower question than the one being asked. The
    # agent wants to know whether the CASE is settled; this fetched a Razorpay
    # payment id, and a checkout abandoned before Razorpay saw it has no such
    # id — so it returned "no_data" with the guidance "do not call this tool
    # again for this case".
    #
    # That single line is the reason the agent could not know when to stop. The
    # one instrument it had for looking at the world told it that looking was
    # pointless, so it acted on narration instead, and its tools had to be taken
    # away to keep it safe. Checking must always be worth it.
    facts = ground_truth(payment_id, verify=True)
    if facts["known"]:
        return json.dumps({
            "status": "settled" if facts["settled"] else "open",
            "briefing": as_briefing(facts),
            **{k: v for k, v in facts.items() if k != "known"},
        })

    client = RazorpayClient()
    if not client.is_configured:
        return json.dumps({"status": "unavailable", "message": "Razorpay client not configured"})

    if not str(payment_id or "").startswith("pay_") or len(str(payment_id)) != 18:
        return json.dumps({
            "status": "no_data",
            "payment_id": payment_id,
            "message": "no case record and not a Razorpay payment id — the "
                       "customer never got far enough for Razorpay to record a "
                       "payment",
            "guidance": "Use the failure_code you were given and proceed. This "
                        "tool is still worth calling later, once the case has a "
                        "record: it is how you find out whether the money "
                        "arrived.",
        })

    try:
        payment = client.client.payment.fetch(payment_id)
        return json.dumps({
            "status": "ok",
            "payment_id": payment_id,
            "current_status": payment.get("status", "unknown"),
            "amount": payment.get("amount", 0) / 100,
            "currency": payment.get("currency", "INR"),
            "method": payment.get("method", "unknown"),
            "bank": payment.get("bank", ""),
            "failure_reason": payment.get("failure_reason", ""),
            "error_code": payment.get("error_code", ""),
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool
def get_customer_payment_history(
    customer_id: str,
    runtime=None,
) -> str:
    """Fetch the customer's past payment attempts and outcomes from Razorpay.

    Args:
        customer_id: Customer identifier (email or ID)

    Returns:
        JSON with total_payments, successful_count, method_success_rates, recent_payments
    """
    from recovery_agent.razorpay_client import RazorpayClient

    client = RazorpayClient()
    if not client.is_configured:
        return json.dumps({"status": "unavailable", "message": "Razorpay client not configured"})

    try:
        # Razorpay has no "payments for this customer" endpoint, so we scan
        # recent payments and match locally. The previous version only looked in
        # `notes` — which the checkout does not populate — so every lookup
        # returned no_data. Match the fields Razorpay actually sets (`email`,
        # `contact`) as well, and scan deeper than one page.
        wanted = (customer_id or "").strip().lower()
        if not wanted:
            return json.dumps({"status": "no_data", "message": "no customer_id given"})

        items = []
        for skip in (0, 100):
            page = client.client.payment.fetch_all({"count": 100, "skip": skip})
            batch = page.get("items", [])
            items.extend(batch)
            if len(batch) < 100:
                break

        customer_payments = []
        for p in items:
            notes = p.get("notes") or {}
            candidates = {
                str(notes.get("customer_id", "")).lower(),
                str(notes.get("customer_email", "")).lower(),
                str(p.get("email", "")).lower(),
                str(p.get("contact", "")).lower(),
                str(p.get("customer_id", "")).lower(),
            }
            if wanted in candidates:
                customer_payments.append({
                    "payment_id": p.get("id", ""),
                    "amount": p.get("amount", 0) / 100,
                    "method": p.get("method", "unknown"),
                    "status": p.get("status", "unknown"),
                    "failure_reason": p.get("failure_reason", ""),
                })

        successful = [p for p in customer_payments if p["status"] == "captured"]
        method_success = {}
        for p in customer_payments:
            m = p["method"]
            if m not in method_success:
                method_success[m] = {"success": 0, "failed": 0}
            if p["status"] == "captured":
                method_success[m]["success"] += 1
            elif p["status"] == "failed":
                method_success[m]["failed"] += 1

        return json.dumps({
            "status": "ok" if customer_payments else "no_data",
            "customer_id": customer_id,
            "total_payments": len(customer_payments),
            "successful_count": len(successful),
            "method_success_rates": method_success,
            "recent_payments": customer_payments[:10],
            "last_successful_method": successful[-1]["method"] if successful else None,
            "scanned": len(items),
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool
def generate_recovery_payment_link(
    payment_id: str,
    amount: float,
    customer_name: str = "Customer",
    customer_email: str = "",
    allowed_rails: str = "upi,card,netbanking",
    expire_in_minutes: int = 16,
    runtime=None,
) -> str:
    """Generate a pre-filled Razorpay payment link for the customer to retry.

    Args:
        payment_id: Original payment ID to link recovery to
        amount: Payment amount in RUPEES (e.g. 2499.00 for INR 2,499). Never paise —
            passing 249900 here would bill the customer INR 2,49,900.
        customer_name: Customer's name for the payment link
        customer_email: Customer's email for the payment link
        allowed_rails: Comma-separated payment methods (e.g., 'upi,card,netbanking')
        expire_in_minutes: How long the link stays payable. Short windows create
            urgency, but Razorpay enforces a 15-minute minimum, so anything
            smaller is raised to 16. To promise the customer a shorter window,
            state the deadline in the message — do not expect the link to enforce it.

    Returns:
        JSON with link_url, link_id, allowed_rails
    """
    _stop = _still_worth_doing(payment_id, "creating a payment link")
    if _stop:
        return _stop

    from recovery_agent.razorpay_client import RazorpayClient

    client = RazorpayClient()
    if not client.is_configured:
        return json.dumps({"status": "unavailable", "message": "Razorpay client not configured"})

    # Refuse to bill an amount that does not match the case.
    #
    # The unit is stated in the docstring, but a model that passes paise anyway
    # would charge the customer 100x — which is exactly the bug that reached the
    # live account by a different route. Checking against the case makes the
    # amount unforgeable rather than merely documented.
    case = getattr(getattr(runtime, "context", None), "case", None) if runtime else None
    expected = getattr(getattr(case, "payment", None), "amount", None)
    if expected:
        expected = float(expected)
        amount_f = float(amount)

        # A discount is a legitimate reason for the link to be below the debt, so
        # the check is a RANGE, not equality: never above what is owed (that is
        # the 100x overcharge), never below the deepest discount policy allows
        # (that is giving the store away). An earlier equality check rejected a
        # correctly-authorised 5% offer, and the agent silently fell back to full
        # price — the customer got an email promising a discount it did not carry.
        from recovery_agent.agent.offers import load_policy
        floor = expected * (1 - load_policy().max_discount_pct / 100.0)

        if amount_f > expected + 0.01:
            hint = (" You passed paise; this argument is in rupees."
                    if abs(amount_f - expected * 100) < 1 else "")
            return json.dumps({
                "status": "error",
                "message": f"amount {amount_f} is MORE than the {expected} owed."
                           f"{hint} Never charge above the debt.",
            })
        if amount_f < floor - 0.01:
            return json.dumps({
                "status": "error",
                "message": f"amount {amount_f} is below the policy floor of "
                           f"{floor:.2f} (max {load_policy().max_discount_pct:.0f}% off "
                           f"{expected}). Call get_recovery_offer for an authorised figure.",
            })

    rails = [r.strip() for r in allowed_rails.split(",")]
    expire_by = None
    if expire_in_minutes:
        import time as _time
        # Razorpay rejects "expire_by: timestamp must be atleast 15 minutes in
        # future", and exactly 15 fails once request latency is counted, so the
        # floor is 16. A shorter customer-facing deadline has to be enforced on
        # our side (the offer expires), not by the link.
        expire_by = int(_time.time() + max(16, int(expire_in_minutes)) * 60)
    try:
        # `create_payment_link` takes RUPEES and converts to paise itself.
        # Converting here too made every recovery link 100x the debt — verified
        # live: a Rs 2,999 payment produced a Rs 2,99,900 link. The other two
        # callers (daemon_worker, frontend) already pass rupees.
        link = client.create_payment_link(
            amount=amount,
            expire_by=expire_by,
            customer={"name": customer_name, "email": customer_email},
            notes={
                # Carried so a paid recovery can always be traced back to the
                # checkout it belongs to — the original order keeps a matching
                # pointer the other way.
                "recovery_agent": "ReActAgent",
                "original_payment": payment_id,
                "original_order_id": str(
                    (getattr(getattr(case, "payment", None), "metadata", None) or {})
                    .get("original_order_id", "")),
                "original_amount": str(expected or ""),
                "allowed_rails": ",".join(rails),
            },
        )
        if "error" in link:
            return json.dumps({"status": "error", "message": link["error"]})
        link_url = link.get("short_url", "")
        link_id = link.get("id", "")
        # Stash it so the notification tool can still find it if the model
        # forgets to pass it along.
        if case is not None and link_url:
            case.payment.metadata["recovery_link"] = link_url
            case.payment.metadata["recovery_link_id"] = link_id
            # What the customer will actually be charged. Everything the agent
            # then says about price is checked against this.
            case.payment.metadata["recovery_link_amount"] = float(amount)
            if expire_by:
                case.payment.metadata["recovery_link_expire_by"] = int(expire_by)
        # A link on different rails, or at a different price, is a genuinely
        # different route for the customer — so it counts. The same link again
        # does not.
        ladder.record_action(payment_id,
                             f"link:{'+'.join(sorted(rails))}:{float(amount):.2f}")
        return json.dumps({
            "status": "ok",
            "payment_id": payment_id,
            "link_url": link_url,
            "link_id": link_id,
            "allowed_rails": rails,
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def _last_recovery_link(runtime, payment_id: str) -> str:
    """Fall back to the link already created for this case.

    The model should pass `payment_link`, but if it forgets, the case still
    knows about the link the previous tool call produced — better to recover it
    than to send a dead-end email.
    """
    case = getattr(getattr(runtime, "context", None), "case", None) if runtime else None
    if case is None:
        return ""
    return str((case.payment.metadata or {}).get("recovery_link", "") or "")


@tool
def show_page_offer(
    payment_id: str,
    headline: str,
    body: str,
    payable_amount: float,
    discount_pct: float = 0.0,
    expires_in_minutes: int = 15,
    cta_text: str = "Claim offer & pay",
    payment_link: str = "",
    runtime=None,
) -> str:
    """Put an incentive on the checkout page and notify the customer about it.

    Use after get_recovery_offer has authorised a discount and the payment link
    has been created at that price. The page shows a coupon banner with the
    original price struck through, alongside a notification.

    The figure you pass is checked against what the payment link actually
    charges. A page that advertises one price while the link charges another is
    worse than no offer at all, so a mismatch is refused rather than displayed.

    Args:
        payment_id: The payment this relates to
        headline: Short line, e.g. "Here's 5% off to finish your order"
        body: One or two sentences
        payable_amount: What the customer will pay, in RUPEES. Must equal the
            payment link's amount.
        discount_pct: The percentage being given, for display
        expires_in_minutes: How long the banner says the offer lasts. Capped to
            the payment link's real remaining life — the countdown can never
            outlast the link it points at.
        cta_text: Label for the button
        payment_link: The link the button opens

    Returns:
        JSON with delivery status.
    """
    _stop = _still_worth_doing(payment_id, "showing a discount")
    if _stop:
        return _stop

    case = getattr(getattr(runtime, "context", None), "case", None) if runtime else None
    meta = (case.payment.metadata if case is not None else {}) or {}
    original = float(getattr(getattr(case, "payment", None), "amount", 0) or 0)
    charged = meta.get("recovery_link_amount")

    try:
        payable = float(payable_amount)
    except (TypeError, ValueError):
        return json.dumps({"status": "error", "message": "payable_amount must be a number"})

    if charged is not None and abs(payable - float(charged)) > 0.01:
        return json.dumps({
            "status": "error",
            "message": f"you are about to show INR {payable:,.2f} on the page but the "
                       f"payment link charges INR {float(charged):,.2f}.",
            "guidance": f"Regenerate the link at {payable:.2f}, or show "
                        f"{float(charged):.2f}. They must agree.",
        })

    # The discount must also sit inside policy, not just match the link.
    if original > 0 and payable < original:
        from recovery_agent.agent.offers import load_policy
        floor = original * (1 - load_policy().max_discount_pct / 100.0)
        if payable < floor - 0.01:
            return json.dumps({
                "status": "error",
                "message": f"INR {payable:,.2f} is below the policy floor of "
                           f"INR {floor:,.2f} for a INR {original:,.2f} order.",
            })

    # The countdown must match when the link actually dies. Left to the agent
    # these disagreed live: the banner promised 48h while the link expired in 24,
    # and the spec asked for minutes. A deadline the link does not honour is
    # either a false promise or a stranded customer, so the real expiry wins.
    minutes = max(1, int(expires_in_minutes or 15))
    expire_by = meta.get("recovery_link_expire_by")
    if expire_by:
        import time as _time
        remaining = int((int(expire_by) - _time.time()) // 60)
        if remaining > 0:
            minutes = min(minutes, remaining)

    payload = {
        "payment_id": payment_id,
        "headline": headline[:120],
        "body": body[:400],
        "cta_text": cta_text[:40],
        "payment_link": payment_link or _last_recovery_link(runtime, payment_id),
        "offer_text": (f"{float(discount_pct):.0f}% off — pay INR {payable:,.2f}"
                       if discount_pct else f"Pay INR {payable:,.2f}"),
        "wait_minutes": minutes,
        "offer": {
            "original_rupees": round(original, 2) if original else None,
            "payable_rupees": round(payable, 2),
            "discount_pct": round(float(discount_pct or 0), 1),
            "expires_in_minutes": minutes,
        },
    }

    try:
        from recovery_agent import push_bus
        result = push_bus.deliver(payload)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"offer display failed: {e}"})

    if case is not None:
        case.payment.metadata["page_offer"] = payload["offer"]

    if result.get("status") == "delivered":
        ladder.record_rung(payment_id, "offer", payload["offer_text"])
        ladder.record_action(payment_id, f"page_offer:{payable_amount}", is_rung=True)

    return json.dumps({
        "status": result.get("status", "unknown"),
        "payment_id": payment_id,
        "showing": payload["offer_text"],
        "note": result.get("note", ""),
    })


@tool
def wait_for_customer(
    payment_id: str,
    waiting_for: str,
    expected_within_minutes: int = 5,
    runtime=None,
) -> str:
    """Say that you have done what you can for now and are waiting on someone else.

    Call this when your work this turn is finished and the next thing that
    matters is out of your hands — the customer responding to a notification,
    paying a link, or a scheduled retry coming due.

    It ends your turn cleanly. You are NOT abandoning the case: you will be
    started again with whatever happened, and your session for this payment
    continues from here.

    Args:
        payment_id: The payment you are waiting on
        waiting_for: What you expect to happen, in a few words
            (e.g. "customer to act on the in-page offer")
        expected_within_minutes: Roughly how long that should take

    Returns:
        Confirmation that the wait is recorded and your turn is over.
    """
    case = getattr(getattr(runtime, "context", None), "case", None) if runtime else None
    minutes = max(1, int(expected_within_minutes or 5))
    if case is not None:
        case.payment.metadata["waiting_for"] = {
            "reason": str(waiting_for)[:200],
            "expected_within_minutes": minutes,
        }
    return json.dumps({
        "status": "ok",
        "payment_id": payment_id,
        "waiting_for": str(waiting_for)[:200],
        "note": f"Turn ended. You will be started again when this resolves or "
                f"after about {minutes} minute(s). Do not call any more tools.",
    })


@tool
def close_case(
    payment_id: str,
    outcome: str,
    what_happened: str,
    lesson: str = "",
    runtime=None,
) -> str:
    """Declare this case finished and stop working it. The last thing you call.

    Stopping used to be something the agent did by simply not calling another
    tool — indistinguishable, from the outside, from running out of ideas or
    hitting a limit. Nothing recorded that the agent had *decided* the case was
    over, so nothing could tell a deliberate ending from a stalled one, and the
    case's outcome had to be inferred from which tools happened to run last.

    This is the explicit ending. It writes the outcome to the case record and
    the lesson to long-term memory in one act, and closes the case so no later
    signal reopens it.

    Args:
        payment_id: The payment you are closing
        outcome: One of "recovered" (the money is in), "escalated" (a person has
            it now), or "unrecoverable" (nothing further is worth trying and no
            person can help either)
        what_happened: One or two sentences a colleague could read cold
        lesson: What you would want to know facing this customer or this failure
            again. Stored for future cases; leave empty only if there is nothing
            worth carrying forward.

    Returns:
        Confirmation the case is closed. Your turn is over — call nothing else.
    """
    from recovery_agent.agent.perception import ground_truth

    outcome = str(outcome or "").strip().lower()
    if outcome not in ("recovered", "escalated", "unrecoverable"):
        return json.dumps({
            "status": "error",
            "reason": f"unknown outcome {outcome!r}",
            "guidance": "Use recovered, escalated, or unrecoverable.",
        })

    facts = ground_truth(payment_id, verify=True)

    # A claim of victory is checked against the money, not taken on trust. The
    # agent writes the summary; whether the payment happened is not its to
    # assert.
    if outcome == "recovered" and not facts.get("settled"):
        return json.dumps({
            "status": "blocked",
            "reason": "no payment is recorded for this case, so it cannot be "
                      "closed as recovered",
            "outstanding": facts.get("outstanding"),
            "guidance": "Call check_payment_status to see where this stands. If "
                        "the money genuinely is not back, keep working the "
                        "ladder or close it with a different outcome.",
        })

    # Symmetry with "recovered": the agent may not claim a person has the case
    # unless a person actually does. Escalation used to end implicitly — the
    # ticket was filed and the run simply stopped — which is the same ambiguity
    # a successful recovery had, with the same consequence: no record that the
    # agent decided anything.
    if outcome == "escalated":
        try:
            from recovery_agent.escalation_queue import list_tickets
            has_ticket = any(t.get("payment_id") == payment_id
                             for t in list_tickets(status=None))
        except Exception:
            has_ticket = True           # never block a closure on a queue read
        if not has_ticket:
            return json.dumps({
                "status": "blocked",
                "reason": "no escalation ticket exists for this payment, so it "
                          "is not with a human",
                "guidance": "Call escalate_to_human first — that is what puts it "
                            "in front of a person — then close the case.",
            })

    if outcome == "unrecoverable":
        from recovery_agent.agent import ladder as _lad
        try:
            from recovery_agent.state_store import StateStore
            rec = StateStore().get_payment(payment_id) or {}
        except Exception:
            rec = {}
        if rec and not _lad.exhausted(rec) and not _lad.pursuit_barred(rec):
            nxt = _lad.next_rung(rec)
            return json.dumps({
                "status": "blocked",
                "reason": "the ladder is not exhausted, so this is not "
                          "unrecoverable yet",
                "next_rung": nxt["rung"] if nxt else None,
                "guidance": (f"Try {nxt['what']} first." if nxt else
                             "Escalate to a human instead."),
            })

    closed = {
        "outcome": outcome,
        "what_happened": str(what_happened or "")[:600],
        "at": datetime.now(timezone.utc).isoformat(),
        "amount_recovered": facts.get("received", 0.0),
    }
    try:
        from recovery_agent.state_store import StateStore
        store = StateStore()
        if store.get_payment(payment_id) is not None:
            fields = {"closed": closed, "closing_summary": closed["what_happened"]}
            if outcome == "recovered":
                fields["status"] = "recovered"
            elif outcome == "escalated":
                fields["status"] = "escalated"
            else:
                fields["status"] = "unrecoverable"
            store.update_payment(payment_id, **fields)
            store.flush()
    except Exception as exc:
        return json.dumps({"status": "error",
                           "message": f"could not record the closure: {exc}"})

    # The lesson goes in here rather than being a second call the agent has to
    # remember — closing and learning are one act, and a lesson that depends on
    # a follow-up tool is a lesson that gets lost when the turn ends first.
    stored = ""
    if lesson.strip():
        try:
            from recovery_agent.agent.graph import get_memory_store
            import uuid as _uuid
            cust = (facts.get("customer_email")
                    or (StateStore().get_payment(payment_id) or {}).get("customer_email")
                    or "unknown")
            key = str(_uuid.uuid4())
            get_memory_store().put(("recovery", ns_safe(cust)), key, {
                "content": f"[{outcome}] {lesson.strip()[:800]}",
                "payment_id": payment_id,
                "amount_recovered": facts.get("received", 0.0),
            })
            stored = key
        except Exception as exc:
            stored = f"(not stored: {exc})"

    return json.dumps({
        "status": "closed",
        "payment_id": payment_id,
        "outcome": outcome,
        "amount_recovered": facts.get("received", 0.0),
        "lesson_stored": stored,
        "note": "This case is closed. Your turn is over — call nothing else, "
                "and reply with a short final summary.",
    })


@tool
def send_page_push(
    payment_id: str,
    headline: str,
    body: str,
    cta_text: str = "Complete payment",
    payment_link: str = "",
    offer_text: str = "",
    wait_minutes: int = 5,
    runtime=None,
) -> str:
    """Show an in-page notification to the customer who is still on the checkout.

    This is the SILENT first step: it costs nothing, interrupts nobody, and does
    not put an offer on the table. Use it before email, SMS or a call.

    The customer can act on it, dismiss it, or ignore it. Whatever they do comes
    back to you as a `push_outcome` observation so you can decide what to do
    next — the outcome is a signal about their intent, not just a status.

    Args:
        payment_id: The payment this relates to
        headline: Short line the customer sees first
        body: One or two sentences of context
        cta_text: Label for the action button
        payment_link: Where the button sends them, if you have a link
        offer_text: Optional incentive line. Only include one you obtained from
            get_recovery_offer — never invent a discount.
        wait_minutes: How long to wait for a response before you are called again

    Returns:
        JSON with delivery status. `delivered` means it reached a live page;
        `no_active_session` means the customer already left.
    """
    case = getattr(getattr(runtime, "context", None), "case", None) if runtime else None
    meta = (case.payment.metadata if case is not None else {}) or {}

    # A plain nudge is the SILENT rung. Once the customer has dismissed one, that
    # channel has already failed, and sending another changes nothing — except on
    # screen, where the plain notification appears moments before the offer banner
    # and looks as though the plain one carried the discount.
    # Read the LIVE record, not the snapshot taken when this run started. The
    # metadata copy is written once at case construction, so anything the
    # customer did after that — including the dismissal that triggered this very
    # run — can be missing from it. Live in a real run the agent sent a second
    # plain push to a customer who had already dismissed one, because the
    # snapshot said there had been no outcome.
    prior = meta.get("push_outcome") or {}
    live: dict = {}
    try:
        from recovery_agent.state_store import StateStore
        live = StateStore().get_payment(payment_id) or {}
        prior = live.get("push_outcome") or prior
    except Exception:
        pass

    if prior.get("action") in ("dismissed", "ignored"):
        return json.dumps({
            "status": "blocked",
            "reason": f"the customer already {prior['action']} a plain notification "
                      f"on this page; repeating it will not work",
            "guidance": "Use show_page_offer with an authorised discount, or reach "
                        "them on another channel.",
        })

    # One notification per customer, and this is it.
    #
    # Once the offer rung is reached the on-page channel belongs to the discount
    # banner, which carries its own price, countdown and button. A push as well
    # means two things stacked on one checkout for someone who has already read
    # the first — and the second is a worse version of the banner sitting right
    # above it.
    if ladder.climbed(live, "offer"):
        return json.dumps({
            "status": "blocked",
            "reason": "an offer is already on this page; a second notification "
                      "on top of it is noise, not another chance",
            "guidance": "The banner carries the offer, the price and the button. "
                        "If the customer has not taken it, change channel — do "
                        "not add to the page.",
        })

    link = payment_link or _last_recovery_link(runtime, payment_id)

    payload = {
        "payment_id": payment_id,
        "headline": headline[:120],
        "body": body[:400],
        "cta_text": cta_text[:40],
        "payment_link": link,
        "offer_text": offer_text[:160],
        "wait_minutes": max(1, int(wait_minutes or 5)),
    }

    try:
        from recovery_agent import push_bus
        result = push_bus.deliver(payload)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"push failed: {e}"})

    if case is not None:
        case.payment.metadata["last_push"] = payload

    if result.get("status") == "delivered":
        ladder.record_rung(payment_id, "page_push", payload["headline"])
        ladder.record_action(payment_id, f"page_push:{'offer' if offer_text else 'plain'}", is_rung=True)

    return json.dumps({
        "status": result.get("status", "unknown"),
        "payment_id": payment_id,
        "waiting_minutes": payload["wait_minutes"],
        "note": result.get("note", ""),
    })


@tool
def get_recovery_offer(
    amount: float,
    stage: str = "email_offer",
    payment_id: str = "",
    requested_discount_pct: float = -1.0,
    already_offered_pct: float = 0.0,
    runtime=None,
) -> str:
    """Ask what incentive the store's policy allows you to offer right now.

    Use this BEFORE promising a customer anything. The discount comes from the
    merchant's own dunning policy in the knowledge base — you may not invent one,
    and any figure you request is clamped to the policy ceiling.

    Args:
        amount: The amount owed, in RUPEES.
        stage: Where you are in the recovery ladder — one of
            'silent_push' (no incentive), 'email_offer', 'ui_offer',
            'voice' (negotiation, highest ceiling).
        payment_id: The case. Pass it — the answer depends on WHY the payment
            failed, not only on the stage.
        requested_discount_pct: What you (or the customer) are asking for.
            Omit to be given the maximum the stage allows. A customer claiming a
            better price elsewhere does not raise the ceiling.
        already_offered_pct: Discount already granted on this transaction.

    Returns:
        JSON with allowed, discount_pct, discount_rupees, payable_rupees, and the
        policy caps it was derived from.
    """
    from recovery_agent.agent.offers import quote

    # A discount is a lever for reluctance. It does nothing for a refused
    # instrument, and answering "5% is authorised" to a bank decline invites
    # exactly the wrong move: the agent switched to the customer's working rail
    # — correctly — and then discounted it anyway, giving away INR 124.95 to
    # solve a problem that was never about price.
    #
    # So refuse the offer while a full-price attempt on a different rail is
    # still untried. If that also fails, the customer's reluctance is real and
    # the discount is back on the table.
    if payment_id:
        rec = _live_record(payment_id)
        code = str(rec.get("failure_code") or "").lower()
        reason = str(rec.get("failure_reason") or "").lower()
        blob = f"{code} {reason}"
        method_failure = any(w in blob for w in (
            "declin", "insufficient", "expired", "invalid", "issuer", "bank",
            "authentication", "cvv"))
        owed = float(rec.get("amount") or amount or 0)
        tried_full_price = any(
            a.startswith("link:") and a.endswith(f":{owed:.2f}")
            for a in (rec.get("actions_tried") or []))
        if method_failure and not tried_full_price:
            return json.dumps({
                "status": "not_indicated",
                "allowed": False,
                "reason": f"this payment was refused by the bank, not turned down "
                          f"on price — the customer tried to pay you. A discount "
                          f"does not make a declined instrument work.",
                "do_this_instead": f"Create the link at the FULL INR {owed:,.2f} "
                                   f"on a rail that has worked for this customer "
                                   f"(see get_customer_payment_history). If that "
                                   f"also fails, ask again and an offer is "
                                   f"authorised.",
            })

    try:
        offer = quote(
            amount_rupees=float(amount),
            stage=str(stage),
            requested_pct=None if float(requested_discount_pct) < 0 else float(requested_discount_pct),
            already_offered_pct=float(already_offered_pct or 0),
        )
    except (TypeError, ValueError) as e:
        return json.dumps({"status": "error", "message": f"bad offer request: {e}"})

    body = offer.as_dict()
    body["status"] = "ok" if offer.allowed else "no_data"
    if offer.allowed:
        case = getattr(getattr(runtime, "context", None), "case", None) if runtime else None
        if case is not None:
            case.payment.metadata["offer_pct"] = offer.discount_pct
            case.payment.metadata["offer_payable"] = offer.payable_rupees
    return json.dumps(body)


@tool
def send_recovery_notification(
    payment_id: str,
    customer_email: str,
    customer_phone: str,
    message: str,
    payment_link: str = "",
    amount: float = 0,
    attempt_count: int = 0,
    runtime=None,
) -> str:
    """Send an email/SMS notification to the customer about the failed payment.

    Args:
        payment_id: The payment ID
        customer_email: Customer's email address
        customer_phone: Customer's phone number (with country code)
        message: The message to send
        payment_link: The recovery link URL from generate_recovery_payment_link.
            ALWAYS pass this if you have created one — without it the customer
            gets a message telling them to pay with no way to do so.
        amount: Payment amount in RUPEES, not paise
        attempt_count: Current attempt number

    Returns:
        JSON with channels_dispatched, results
    """
    _stop = _still_worth_doing(payment_id, "sending a message")
    if _stop:
        return _stop

    # The message is the model's to write — tone and wording are exactly what it
    # is good at. What it must not do is promise a price the link will not
    # charge. This happened live: the agent created the link at full price,
    # *then* obtained a 5% offer, and emailed "you only pay INR 2,374.05" against
    # a link charging INR 2,499. The customer would have clicked through to a
    # different number than the one they were promised.
    #
    # So the prose is not templated — it is verified. The claim is checked
    # against what the link actually charges, and the send is refused if they
    # disagree.
    case = getattr(getattr(runtime, "context", None), "case", None) if runtime else None
    meta = (case.payment.metadata if case is not None else {}) or {}
    offered = meta.get("offer_payable")
    charged = meta.get("recovery_link_amount")
    if offered is not None and charged is not None and abs(float(offered) - float(charged)) > 0.01:
        return json.dumps({
            "status": "error",
            "message": f"the payment link charges INR {float(charged):,.2f} but you "
                       f"were authorised to offer INR {float(offered):,.2f}. Do not "
                       f"tell the customer about a discount the link will not honour.",
            "guidance": f"Call generate_recovery_payment_link again with "
                        f"amount={float(offered):.2f}, then send the message.",
        })

    # Refuse to send a message the customer cannot act on.
    #
    # Live: the link tool failed (Razorpay test-mode quota), the agent sent the
    # email anyway, and the customer received a 5% discount offer with no button
    # and no URL. An offer with no way to pay is worse than no offer — it spends
    # the one contact you had and gives the customer nothing to do with it.
    _link = payment_link or _last_recovery_link(runtime, payment_id)
    if not _link:
        return json.dumps({
            "status": "error",
            "message": "there is no payment link for this case, so the customer "
                       "would have no way to pay.",
            "guidance": "Call generate_recovery_payment_link first and pass its "
                        "link_url as payment_link. If that tool is failing, do "
                        "not send a message — escalate instead.",
        })

    from recovery_agent.notifications import NotificationDispatcher

    dispatcher = NotificationDispatcher()
    try:
        result = dispatcher.dispatch(
            payment_id=payment_id,
            customer_email=customer_email,
            customer_phone=customer_phone,
            action="send_notification",
            # Was hardcoded to "" — so the "Complete Payment" button never
            # rendered and the email told the customer to "try again from the
            # checkout page" with no link. A recovery message the customer
            # cannot act on is not a recovery.
            recovery_link=payment_link or _last_recovery_link(runtime, payment_id),
            failure_reason=message,
            amount=amount,
            attempt_count=attempt_count,
        )
        # Which rung an email is depends on what came before it: the first one
        # carries the offer, one sent after a call is the follow-up the call
        # agreed to. Same tool, different place on the ladder.
        rung = "post_call_email" if ladder.climbed(_live_record(payment_id),
                                                   "voice_call") else "offer"
        ladder.record_rung(payment_id, rung, message[:200])
        ladder.record_action(
            payment_id,
            f"notify:{'+'.join(sorted(result.get('channels') or []))}:"
            f"{payment_link or _last_recovery_link(runtime, payment_id)}",
            is_rung=True)
        return json.dumps({
            "status": "ok",
            "rung": rung,
            "channels": result.get("channels", []),
            "results": result.get("results", []),
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool
def schedule_retry(
    payment_id: str,
    target_iso_timestamp: str,
    runtime=None,
) -> str:
    """Schedule a background retry at a specific future timestamp.

    Args:
        payment_id: Payment ID to retry
        target_iso_timestamp: ISO 8601 timestamp for when to retry (e.g., '2026-09-01T12:01:00+05:30')

    Returns:
        JSON with job_id, target_time, delay_hours
    """
    from recovery_agent.state_store import StateStore

    try:
        target_time = datetime.fromisoformat(target_iso_timestamp)
        now = datetime.now(timezone.utc)
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)
        delay_seconds = (target_time - now).total_seconds()

        if delay_seconds <= 0:
            return json.dumps({"status": "error", "message": "Target time is in the past. Use retry_in_hours instead."})

        store = StateStore()
        job_id = f"job_{payment_id}_{int(target_time.timestamp())}"
        store.schedule_job(
            job_id=job_id,
            payment_id=payment_id,
            target_time=target_iso_timestamp,
            action="retry_payment",
        )
        store.flush()

        ladder.record_action(payment_id, f"retry:{round(hours, 1)}h")
        return json.dumps({
            "status": "scheduled",
            "job_id": job_id,
            "delay_hours": round(delay_seconds / 3600, 1),
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool
def retry_in_hours(
    payment_id: str,
    hours: float,
    runtime=None,
) -> str:
    """Schedule a retry after a number of hours from now. Use this instead of schedule_retry when you don't know the exact timestamp.

    Args:
        payment_id: Payment ID to retry
        hours: Number of hours from now to retry (e.g., 2.0 for 2 hours, 24.0 for tomorrow)

    Returns:
        JSON with job_id, target_time, delay_hours
    """
    from recovery_agent.state_store import StateStore

    try:
        now = datetime.now(timezone.utc)
        target_time = now.timestamp() + (hours * 3600)
        target_dt = datetime.fromtimestamp(target_time, tz=timezone.utc)
        target_iso = target_dt.isoformat()

        store = StateStore()
        job_id = f"job_{payment_id}_{int(target_time)}"
        store.schedule_job(
            job_id=job_id,
            payment_id=payment_id,
            target_time=target_iso,
            action="retry_payment",
        )
        store.flush()

        ladder.record_action(payment_id, f"retry:{round(hours, 1)}h")
        return json.dumps({
            "status": "scheduled",
            "job_id": job_id,
            "target_time": target_iso,
            "delay_hours": round(hours, 1),
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool
def escalate_to_human(
    payment_id: str,
    reason: str,
    runtime=None,
) -> str:
    """Create an escalation ticket for human intervention.

    Args:
        payment_id: Payment ID to escalate
        reason: Why this case needs human intervention

    Returns:
        JSON with ticket_id, status
    """
    from recovery_agent.escalation_queue import enqueue

    # A human queue is the LAST resort, not the fallback for a refused tool.
    # Live, the approval gate blocked a discounted link at rung 2 and the agent
    # filed a ticket for a INR 1,19,970 order that had been contacted exactly
    # once — one silent in-page push. The customer had not been emailed, called,
    # or offered anything. This refuses that, and names what to do instead.
    #
    # A rung that CANNOT be climbed here (no phone, voice calling switched off)
    # does not block escalation — it is carried into the ticket instead, so the
    # person picking it up knows what was never tried and why.
    _rec = _live_record(payment_id)
    _ladder = ladder.state(_rec)
    _barred = ladder.pursuit_barred(_rec)
    if _ladder["remaining"] and not _barred:
        nxt = _ladder["remaining"][0]
        return json.dumps({
            "status": "blocked",
            "reason": "the recovery ladder is not exhausted; escalation is the "
                      "final step, not a fallback",
            "already_tried": _ladder["climbed"] or ["nothing yet"],
            "actions_so_far": ladder.actions_tried(_rec) or ["none"],
            "next_rung": nxt["rung"],
            "next_step": nxt["what"],
            "still_available": [r["rung"] for r in _ladder["remaining"]],
            "guidance": f"Do this next: {nxt['what']}. Escalate only once every "
                        f"rung has been tried and the customer still has not paid.",
        })

    # Carry the case with it. A ticket holding only {payment_id, reason} forces
    # whoever picks it up to reconstruct everything from logs before they can say
    # a word to the customer, which is how a queue becomes a place cases go to be
    # forgotten.
    case = getattr(getattr(runtime, "context", None), "case", None) if runtime else None
    payment = getattr(case, "payment", None)
    meta = (getattr(payment, "metadata", None) or {}) if payment else {}

    attempts = [
        {"action": getattr(a.action_type, "value", str(a.action_type)),
         "result": a.result,
         "at": a.timestamp.isoformat() if getattr(a, "timestamp", None) else ""}
        for a in (getattr(case, "attempts", None) or [])
    ]

    signals = []
    outcome = meta.get("push_outcome") or {}
    if outcome.get("action"):
        signals.append(f"in-page notification {outcome['action']} after "
                       f"{outcome.get('seconds_shown', '?')}s")
    if meta.get("page_offer"):
        signals.append(f"was shown a {meta['page_offer'].get('discount_pct')}% offer")

    ticket = enqueue(
        payment_id=payment_id,
        reason=reason,
        amount=float(getattr(payment, "amount", 0) or 0),
        currency=getattr(payment, "currency", "INR") or "INR",
        customer={
            "email": meta.get("customer_email") or getattr(payment, "customer_id", ""),
            "name": meta.get("customer_name", ""),
            "contact": meta.get("customer_phone", ""),
        },
        attempts=attempts,
        customer_signals=signals + [
            f"ladder climbed: {', '.join(_ladder['climbed']) or 'none'}"
        ] + [
            f"never tried — {r['rung']}: {r.get('why_not', 'not applicable')}"
            for r in _ladder["unavailable"]
        ],
        offer=meta.get("page_offer") or ({"payable_rupees": meta.get("offer_payable")}
                                         if meta.get("offer_payable") else {}),
        recovery_link=meta.get("recovery_link", ""),
        failure_code=getattr(payment, "failure_code", "") or "",
        source="agent",
    )

    return json.dumps({
        "status": "escalated",
        "ticket_id": ticket["ticket_id"],
        "payment_id": payment_id,
        "reason": reason,
        "queued_for_human": True,
        "next": "The case is with a person now, but it is not finished until you "
                "say so. Call close_case with outcome='escalated', what happened, "
                "and the lesson worth carrying forward.",
    })


@tool
def initiate_voice_call(
    payment_id: str,
    customer_name: str,
    customer_phone: str,
    amount: float,
    failure_reason: str = "",
    runtime=None,
) -> str:
    """Initiate an AI voice call via SuperU to recover a failed payment.

    Args:
        payment_id: Payment ID
        customer_name: Customer's name
        customer_phone: Customer's phone (with country code)
        amount: Payment amount in INR
        failure_reason: Why the payment failed

    Returns:
        JSON with call status and campaign_id
    """
    _stop = _still_worth_doing(payment_id, "placing a call")
    if _stop:
        return _stop

    if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
        return json.dumps({"status": "skipped", "reason": "test_environment"})

    # Voice calls cost real money and the SuperU allowance is small, so they are
    # OFF unless explicitly switched on. The tool stays registered and choosable
    # either way — the agent should still learn when a call is the right move,
    # and a disabled result tells it so without spending anything.
    if os.getenv("VOICE_CALLS_ENABLED", "").strip().lower() not in ("1", "true", "yes", "on"):
        return json.dumps({
            "status": "disabled",
            "reason": "voice calling is switched off (VOICE_CALLS_ENABLED is not set)",
            "guidance": "Do not retry this tool. Choose another channel or escalate.",
            "would_have_called": customer_phone,
        })

    # The call comes AFTER the customer has had a real chance to use the offer.
    # Ringing someone thirty seconds after emailing them a discount is not a
    # negotiation, it is pestering — and it burns a paid call on a customer who
    # was about to pay anyway.
    _rec = _live_record(payment_id)
    if ladder.climbed(_rec, "offer"):
        _left = ladder.voice_wait_remaining_minutes(_rec)
        if _left > 0:
            return json.dumps({
                "status": "too_soon",
                "reason": f"the offer went out {ladder.VOICE_DELAY_MINUTES - _left:.0f} "
                          f"minute(s) ago; a call is allowed after "
                          f"{ladder.VOICE_DELAY_MINUTES}",
                "minutes_remaining": round(_left, 1),
                "guidance": "Call wait_for_customer and let the offer work. You "
                            "will be started again when the wait is over.",
            })

    # A per-call cost cap belongs in code, not in the model's judgment. Left to
    # the prompt, the agent reached for voice on every follow-up regardless of
    # amount, which would drain a small SuperU allowance in an afternoon.
    try:
        min_amount = float(os.getenv("VOICE_MIN_AMOUNT_RUPEES", "5000"))
    except ValueError:
        min_amount = 5000.0
    if float(amount or 0) < min_amount:
        return json.dumps({
            "status": "blocked",
            "reason": f"INR {float(amount or 0):,.0f} is below the voice-call "
                      f"threshold of INR {min_amount:,.0f}; a call costs more than "
                      f"it can recover here",
            "guidance": "Use a cheaper channel: send_recovery_notification, "
                        "retry_in_hours, or escalate_to_human.",
        })

    if not str(customer_phone or "").strip():
        return json.dumps({
            "status": "blocked",
            "reason": "no phone number on file for this customer",
            "guidance": "Use email or escalate.",
        })

    from recovery_agent.integrations.superu_client import get_superu_client

    client = get_superu_client()
    result = client.initiate_recovery_call(
        payment_id=payment_id,
        customer_name=customer_name,
        customer_phone=customer_phone,
        amount=amount,
        failure_reason=failure_reason,
        # Was hardcoded "" — the same defect the email had. A voice agent that
        # cannot give the customer a link cannot close the recovery.
        recovery_link=_last_recovery_link(runtime, payment_id),
    )

    if result.get("status") in ("ok", "initiated", "queued", "success"):
        ladder.record_rung(payment_id, "voice_call",
                           f"called {customer_phone}")
        ladder.record_action(payment_id, f"voice:{customer_phone}", is_rung=True)

    return json.dumps({
        "status": result.get("status", "unknown"),
        "campaign_id": result.get("campaign_id", ""),
        "phone": result.get("phone", customer_phone),
        "next": "After the call, send the agreed pay link by email "
                "(send_recovery_notification).",
    })


@tool
def query_knowledge_base(
    query: str,
    runtime=None,
) -> str:
    """Search the Razorpay knowledge base for error documentation and recovery best practices.

    Args:
        query: Search query (e.g., 'error code 51 insufficient funds retry policy')

    Returns:
        JSON with answer, groundedness_score, sources
    """
    from recovery_agent.agent.agentic_rag import LlamaIndexAgenticRAG

    try:
        rag = LlamaIndexAgenticRAG()
    except Exception as e:
        return json.dumps({"status": "error", "message": f"RAG unavailable: {e}"})

    payload = {
        "failure_code": query,
        "failure_reason": query,
        "error_description": query,
        "method": "unknown",
        "provider": "unknown",
        "amount": 0,
    }

    response = rag.query(payload)

    # The RAG engine's `answer` is the sub-questions it generated with the
    # retrieved text appended, so returning it verbatim handed the LLM its own
    # question back as if it were an answer. Return the retrieved passages only,
    # deduplicated and in relevance order.
    seen: set[str] = set()
    passages: list[str] = []
    for chunk in response.retrieved_chunks:
        text = " ".join(chunk.text.split())
        if not text or text in seen:
            continue
        seen.add(text)
        passages.append(text)
        if len(passages) >= 6:
            break

    if not passages:
        return json.dumps({
            "status": "no_data",
            "message": f"nothing in the knowledge base matches {query!r}",
        })

    return json.dumps({
        "status": "ok",
        "query": query,
        "passages": passages,
        "groundedness_score": round(response.groundedness_score, 3),
        "sources": sorted({c.source_file for c in response.retrieved_chunks}),
    })


# ═══════════════════════════════════════════════════════════════
# PLANNER TOOL — calls pydantic-ai Agent for structured planning
# ═══════════════════════════════════════════════════════════════

@tool
def get_recovery_plan(
    payment_id: str,
    failure_code: str,
    failure_reason: str,
    customer_id: str,
    amount: float,
    attempt: int,
    max_attempts: int,
    runtime=None,
) -> str:
    """Get a structured recovery plan from the planning agent.

    Uses pydantic-ai Agent to generate a typed RecoveryPlan with steps.
    The LLM reasons about the failure and creates an ordered action plan.

    Args:
        payment_id: The Razorpay payment ID
        failure_code: The error code from Razorpay
        failure_reason: The human-readable failure reason
        customer_id: Customer identifier
        amount: Payment amount in INR
        attempt: Current attempt number
        max_attempts: Maximum allowed attempts

    Returns:
        JSON with plan (failure_type, root_cause, steps, reasoning)
    """
    from recovery_agent.agent.planner import generate_plan
    from recovery_agent.models import Case, PaymentEvent, RecoveryTier

    case = Case(
        payment=PaymentEvent(
            event_type="payment_failed",
            payment_id=payment_id,
            amount=amount,
            currency="INR",
            failure_code=failure_code,
            failure_reason=failure_reason,
            customer_id=customer_id,
        ),
        attempt_count=attempt,
    )

    # Pass guardrails from runtime context if available
    guardrail_engine = None
    if runtime and hasattr(runtime, 'context') and runtime.context:
        guardrail_engine = getattr(runtime.context, 'guardrail_engine', None)

    try:
        plan = generate_plan(case, guardrail_engine=guardrail_engine)
        return json.dumps({
            "status": "ok",
            "plan": {
                "failure_type": plan.failure_type,
                "root_cause": plan.root_cause,
                "confidence": plan.confidence,
                "steps": [
                    {
                        "step_number": s.step_number,
                        "action": s.action,
                        "description": s.description,
                        "reasoning": s.reasoning,
                        "expected_outcome": s.expected_outcome,
                    }
                    for s in plan.steps
                ],
                "reasoning": plan.reasoning,
                "estimated_recoverability": plan.estimated_recoverability,
            },
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Planning failed: {e}"})


# ═══════════════════════════════════════════════════════════════
# MEMORY TOOLS — langmem-based agent memory
# ═══════════════════════════════════════════════════════════════

from langmem import create_manage_memory_tool, create_search_memory_tool

_manage_memory_raw = create_manage_memory_tool(namespace=("recovery", "{customer_id}"))
_search_memory_raw = create_search_memory_tool(namespace=("recovery", "{customer_id}"))


def ns_safe(value: str) -> str:
    """Make a value usable as a langmem namespace label.

    The memory namespace is ("recovery", "{customer_id}") and customer_id is an
    email, so every memory write died with:

        InvalidNamespaceError: Namespace labels cannot contain periods ('.')

    langmem resolves the namespace from LangGraph's *ambient* runtime config, not
    from the config passed to `.invoke()`, so this must be applied where the graph
    config is built — not inside the tool. Every caller must use this helper or
    memories are written under one key and searched under another.
    """
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value or "")) or "unknown"


@tool
def manage_memory(action: str, content: str = "", id: str = "", config=None) -> str:
    """Store, update, or delete customer memory. Use to remember facts, preferences, or lessons learned.

    Args:
        action: One of 'create', 'update', 'delete'
        content: The memory content to store (required for create/update, ignored for delete)
        id: Memory ID for update/delete operations (leave empty for create)
    """
    kwargs = {"action": action}
    if content:
        kwargs["content"] = content
    if id:
        kwargs["id"] = id
    return _manage_memory_raw.invoke(kwargs, config=config)


@tool
def search_memory(query: str, limit: int = 10, config=None) -> str:
    """Search customer memory for past facts, preferences, or recovery attempts.

    Args:
        query: What to search for
        limit: Maximum results to return (default 10)
    """
    try:
        raw = _search_memory_raw.invoke({"query": query, "limit": limit}, config=config)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"memory search failed: {e}"})

    # langmem returns a bare list, so an empty store came back as "[]" — which
    # the LLM read as an unclear result and searched again, burning a turn. Speak
    # the same status vocabulary as every other tool so "nothing here" is
    # unambiguous and terminal.
    items = raw if isinstance(raw, list) else None
    if items is None:
        text = str(raw).strip()
        if text in ("", "[]", "None"):
            items = []
    if items is not None and len(items) == 0:
        return json.dumps({
            "status": "no_data",
            "message": "No memories stored for this customer yet.",
            "guidance": "This is normal on a first recovery. Do NOT search memory "
                        "again — continue without it.",
        })
    return json.dumps({"status": "ok", "query": query,
                       "memories": [str(m)[:500] for m in items]}) if items is not None \
        else json.dumps({"status": "ok", "query": query, "memories": str(raw)[:2000]})


@tool
def search_similar_episodes(
    failure_code: str,
    failure_reason: str,
    customer_id: str,
) -> str:
    """Search for past recovery episodes with similar failure patterns.

    Use this before taking action to see what worked (or didn't) for similar failures.
    The LLM can learn from past episodes to avoid repeating mistakes.

    Args:
        failure_code: The error code from Razorpay
        failure_reason: The human-readable failure reason
        customer_id: Customer identifier (used to scope episode search)

    Returns:
        JSON with list of similar episodes and their outcomes
    """
    try:
        from recovery_agent.agent.graph import get_memory_store
        store = get_memory_store()

        # Search for episodes from this customer
        results = store.search(
            ("episodes", customer_id),
            query=f"{failure_code} {failure_reason}",
            limit=5,
        )

        episodes = []
        for item in results:
            ep = item.value
            episodes.append({
                "payment_id": ep.get("payment_id"),
                "failure_code": ep.get("failure_code"),
                "failure_reason": ep.get("failure_reason"),
                "tool_calls": ep.get("tool_calls", []),
                "summary": ep.get("summary", "")[:200],
                "status": ep.get("status"),
            })

        return json.dumps({
            "status": "ok",
            "episode_count": len(episodes),
            "episodes": episodes,
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Episode search failed: {e}"})


# ═══════════════════════════════════════════════════════════════
# DISCOVERY TOOL — two-phase KG discovery (semantic + process)
# ═══════════════════════════════════════════════════════════════

@tool
def discover_recovery_rail(
    failure_code: str,
    failure_reason: str = "",
    amount: float = 0.0,
    customer_id: str = "",
    preferred_channel: str = "",
    runtime=None,
) -> str:
    """Discover the best recovery payment rails using knowledge graph + semantic search.

    Two-phase discovery:
    1. Semantic retrieval: embed failure context → find top-k similar rails
    2. Process enrichment: add rails connected via business process edges

    This reduces the agent's search space from all 6 rails to a small
    relevant subset with process ordering.

    Args:
        failure_code: The Razorpay failure code (e.g., 'card_expired', 'insufficient_funds')
        failure_reason: Human-readable failure reason
        amount: Payment amount in RUPEES, not paise
        customer_id: Customer identifier
        preferred_channel: Preferred communication channel (sms, email, whatsapp, push)

    Returns:
        JSON with discovered rails, recommended rail, and process order
    """
    from recovery_agent.agent.kg_router import RazorpayKnowledgeGraph

    try:
        kg = RazorpayKnowledgeGraph()
        result = kg.discover_recovery_rails(
            failure_code=failure_code,
            failure_reason=failure_reason,
            amount=amount,
            customer_id=customer_id,
            preferred_channel=preferred_channel,
        )
        return json.dumps({
            "status": "ok",
            "recommended": result["recommended"],
            "process_order": result["process_order"],
            "rail_count": len(result["rails"]),
            "rails": result["rails"],
            "discovery_method": result["discovery_method"],
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Discovery failed: {e}"})


# ═══════════════════════════════════════════════════════════════
# FEEDBACK TOOLS — training with human feedback (CrewAI pattern)
# ═══════════════════════════════════════════════════════════════

@tool
def store_feedback(
    customer_id: str,
    outcome: str,
    feedback: str,
    recovery_strategy: str,
) -> str:
    """Store human feedback about a recovery attempt for future learning.

    CrewAI: "train with human feedback" — distill lessons into memory.
    MANDATE 1: Uses langmem create_manage_memory_tool with explicit store (real SDK).
    MANDATE 4: Verified API — action='create', store passed explicitly.

    Args:
        customer_id: The customer identifier.
        outcome: Recovery outcome (success, partial, failed, escalated).
        feedback: Human feedback notes about what worked or didn't.
        recovery_strategy: The strategy that was used (e.g., retry, notify, escalate).
    """
    from langmem import create_manage_memory_tool
    from recovery_agent.agent.graph import get_memory_store

    store = get_memory_store()
    # MANDATE 1: langmem create_manage_memory_tool with explicit store (real SDK)
    # MANDATE 4: action='create' (verified API — 'store' is not valid)
    feedback_tool = create_manage_memory_tool(
        namespace=("recovery", "{customer_id}"),
        store=store,
        actions_permitted=("create",),
    )
    content = (
        f"Recovery feedback for customer {customer_id}:\n"
        f"Strategy: {recovery_strategy}\n"
        f"Outcome: {outcome}\n"
        f"Notes: {feedback}\n"
        f"Store this as a lesson for future recovery attempts."
    )
    result = feedback_tool.invoke({
        "content": content,
        "action": "create",
    })
    return f"Feedback stored for {customer_id}: {outcome}. {result}"


# ═══════════════════════════════════════════════════════════════
# TOOL LIST — used by bind_tools() and ToolNode
# ═══════════════════════════════════════════════════════════════

RECOVERY_TOOLS = [
    diagnose_payment_failure,
    check_payment_status,
    get_customer_payment_history,
    generate_recovery_payment_link,
    get_recovery_offer,
    send_page_push,
    show_page_offer,
    wait_for_customer,
    close_case,
    send_recovery_notification,
    retry_in_hours,
    escalate_to_human,
    query_knowledge_base,
    search_memory,
    discover_recovery_rail,
    manage_memory,
    initiate_voice_call,
]

TOOLS_BY_NAME = {t.name: t for t in RECOVERY_TOOLS}


# ═══════════════════════════════════════════════════════════════
# TOOL SUBSETS — specialized tool groups per recovery phase (P1)
# ═══════════════════════════════════════════════════════════════

PLANNING_TOOLS = [
    search_memory,
    discover_recovery_rail,
    get_recovery_plan,
    search_similar_episodes,
    query_knowledge_base,
]

DIAGNOSIS_TOOLS = [
    diagnose_payment_failure,
    check_payment_status,
    get_customer_payment_history,
    query_knowledge_base,
]

EXECUTION_TOOLS = [
    generate_recovery_payment_link,
    get_recovery_offer,
    send_page_push,
    show_page_offer,
    wait_for_customer,
    send_recovery_notification,
    schedule_retry,
    escalate_to_human,
    initiate_voice_call,
    check_payment_status,
]


MEMORY_TOOLS = [
    manage_memory,
    search_memory,
    store_feedback,
]
