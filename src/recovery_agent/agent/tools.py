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


def _record_contact(payment_id: str, channels: list[str]) -> None:
    """A delivery actually happened — write it where the controls can see it.

    Two ledgers, one event: the durable record's `contacts` (per-case, feeds
    unit economics), and the customer profile's history (cross-case, feeds the
    frequency-cap guardrail). Recording only claimed deliveries keeps both
    honest — this is called from the paths that verified delivery, never from
    the attempt. Never raises.
    """
    if not channels:
        return
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        from recovery_agent.state_store import StateStore
        store = StateStore()
        rec = store.get_payment(payment_id)
        if rec is not None:
            contacts = list(rec.get("contacts") or [])
            contacts.extend({"channel": ch, "at": now_iso} for ch in channels)
            store.update_payment(payment_id, contacts=contacts)
            store.flush()
            customer = (rec.get("customer") or {}).get("email") \
                or rec.get("customer_email") or "unknown"
        else:
            customer = "unknown"
    except Exception:
        customer = "unknown"
    try:
        from recovery_agent.agent.memory import CustomerMemoryStore
        profile_store = CustomerMemoryStore.live()
        for ch in channels:
            profile_store.record_contact(customer, ch, payment_id=payment_id)
    except Exception:
        pass




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
    customer_phone: str = "",
    runtime=None,
) -> str:
    """Fetch the customer's past payment attempts and outcomes from Razorpay.

    Includes FAILED attempts, which are the more useful half: a rail that has
    just declined someone is not the rail to route them back into.

    Args:
        customer_id: Customer identifier (email or ID)
        customer_phone: Their phone. Pass it — Razorpay anonymises the email on
            every payment made through a payment link, so the phone is the only
            join that sees the recoveries this agent itself created.

    Returns:
        JSON with total_payments, successful_count, method_success_rates, recent_payments
    """
    from recovery_agent.razorpay_client import RazorpayClient

    # Fall back to the case if the model did not pass a phone, rather than
    # silently searching on the email alone.
    if not customer_phone:
        case = getattr(getattr(runtime, "context", None), "case", None) if runtime else None
        meta = (case.payment.metadata if case is not None else {}) or {}
        customer_phone = meta.get("customer_phone", "") or ""

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

        # Match on the phone as well as the email.
        #
        # Razorpay anonymises the email on any payment made through a payment
        # LINK — every one of them comes back as `void@razorpay.com`. Matching on
        # email alone therefore hides exactly the payments this agent creates:
        # every recovery link it has ever sent, paid or failed.
        #
        # Live, that produced `netbanking {success: 1, failed: 0}` for a customer
        # whose account held three netbanking failures — so the agent read the
        # rail that had just declined them as their most reliable one, and routed
        # the recovery straight back into it.
        #
        # The phone survives anonymisation, so it is the join that works.
        def _digits(v: str) -> str:
            return "".join(ch for ch in str(v or "") if ch.isdigit())[-10:]

        wanted_phone = _digits(customer_phone) or _digits(wanted)

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
            phones = {_digits(p.get("contact")), _digits(notes.get("customer_phone"))}
            phones.discard("")
            if wanted in candidates or (wanted_phone and wanted_phone in phones):
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
            msg = str(link["error"])
            # A test account allows 30 payment links, ever — cancelling one does
            # not give it back. Once that is spent, every case jams at rung 2:
            # no link can be made, no email may be sent without one, and
            # escalation is refused because the ladder is not exhausted.
            #
            # So say plainly that this is the environment, not the request, and
            # mark the case so the ladder treats the offer rung as impossible
            # here — exactly as it treats a voice call when calling is switched
            # off. A blocked channel must not become a stuck case.
            if "limit" in msg.lower() and "payment_link" in msg.lower():
                try:
                    from recovery_agent.state_store import StateStore
                    st = StateStore()
                    if st.get_payment(payment_id) is not None:
                        st.update_payment(payment_id, links_unavailable=True)
                        st.flush()
                except Exception:
                    pass
                return json.dumps({
                    "status": "unavailable",
                    "reason": "this Razorpay test account has spent its lifetime "
                              "allowance of 30 payment links, so no link can be "
                              "created for any case here",
                    "detail": msg,
                    "guidance": "Nothing you do differently will fix this — do "
                                "not retry, and do not send a message promising "
                                "a link that does not exist. The offer rung is "
                                "unavailable in this environment; take a route "
                                "that needs no link, or escalate.",
                })
            return json.dumps({"status": "error", "message": msg})
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
            # What the customer can actually pay WITH. The message may tell
            # them to "try UPI"; that has to be true of the link they are
            # being sent, not a rail the agent liked the sound of.
            case.payment.metadata["recovery_link_rails"] = list(rails)
            if expire_by:
                case.payment.metadata["recovery_link_expire_by"] = int(expire_by)
        # A link on different rails, or at a different price, is a genuinely
        # different route for the customer — so it counts. The same link again
        # does not.
        ladder.record_action(payment_id,
                             f"link:{'+'.join(sorted(rails))}:{float(amount):.2f}")
        # Durable ledger of minted links. Each one spends a unit of a
        # 30-per-lifetime account quota, so a case must be able to say how many
        # it consumed after the run's metadata is gone.
        try:
            from recovery_agent.state_store import StateStore
            _st = StateStore()
            _rec = _st.get_payment(payment_id)
            if _rec is not None:
                _links = list(_rec.get("recovery_links") or [])
                _links.append({"link_id": link_id, "amount": float(amount),
                               "at": datetime.now(timezone.utc).isoformat()})
                _st.update_payment(payment_id, recovery_links=_links)
                _st.flush()
        except Exception:
            pass
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

    # A zero-discount "offer" on a method failure is a RAIL SWITCH, not a
    # promotion: the customer tried to pay and the instrument was refused, so
    # the banner's job is another route at the same price. "% OFF" dressing on
    # a bank problem reads as a bribe for our own decline — the surface must
    # say what the situation is.
    from recovery_agent.agent.classify import failure_kind
    rail_switch = (not discount_pct
                   and failure_kind(_live_record(payment_id)) == "method")

    payload = {
        "payment_id": payment_id,
        "headline": headline[:120],
        "body": body[:400],
        "cta_text": cta_text[:40],
        "payment_link": payment_link or _last_recovery_link(runtime, payment_id),
        "mode": "rail_switch" if rail_switch else "offer",
        "offer_text": (f"{float(discount_pct):.0f}% off — pay INR {payable:,.2f}"
                       if discount_pct else
                       (f"Pay INR {payable:,.2f} — try another payment method"
                        if rail_switch else f"Pay INR {payable:,.2f}")),
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
        # On the durable record too, not only the Case. A Case is rebuilt each
        # run, so the discount that was shown vanished the moment the run ended
        # — which is why the recovery observation's "after a 5% offer" note
        # could never fire.
        try:
            from recovery_agent.state_store import StateStore
            st = StateStore()
            if st.get_payment(payment_id) is not None:
                st.update_payment(payment_id, page_offer=payload["offer"])
                st.flush()
        except Exception:
            pass
        # A banner carrying no discount is not the offer rung — on a bank
        # decline it is the rail switch (same price, another way to pay), and
        # recording it as "offer" spent a rung that gave nothing away.
        _rung = ("rail_switch"
                 if rail_switch and ladder.has_rung(_live_record(payment_id),
                                                    "rail_switch")
                 else "offer")
        ladder.record_rung(payment_id, _rung, payload["offer_text"])
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

    # THE WAIT IS A PROMISE, NOT A NOTE.
    #
    # This used to write the wait into Case metadata and nothing anywhere read
    # it, which broke in both directions. Live 2026-09-04 (pay_woo85c9gh): the
    # agent was refused a link during quiet hours, said "wake me in 60
    # minutes", and nothing ever did — no timer, no job, no watcher — so the
    # case sat at awaiting_customer for good. And on the C1 funds case the
    # opposite: the run was classified `failed`, the ladder hand-off fired ten
    # seconds later, and the agent was marched up rungs it had explicitly
    # asked to wait through.
    #
    # Registering a real job fixes both. The daemon already polls for due jobs
    # and calls the frontend back; "wake_agent" is a job that reaches nobody
    # and moves no money — it just brings the agent back when it asked.
    woken_at = ""
    try:
        from datetime import timedelta
        from recovery_agent.state_store import StateStore
        store = StateStore()
        if store.get_payment(payment_id) is not None:
            due = datetime.now(timezone.utc) + timedelta(minutes=minutes)
            woken_at = due.isoformat()
            store.schedule_job(
                job_id=f"wake_{payment_id}_{int(due.timestamp())}",
                payment_id=payment_id,
                target_time=woken_at,
                action="wake_agent",
                metadata={"reason": str(waiting_for)[:200]},
            )
            store.update_payment(payment_id, waiting_for={
                "reason": str(waiting_for)[:200],
                "until": woken_at,
            })
            store.flush()
    except Exception:
        pass                    # a bookkeeping failure must not break the turn

    return json.dumps({
        "status": "ok",
        "payment_id": payment_id,
        "waiting_for": str(waiting_for)[:200],
        "wake_at": woken_at,
        "note": (f"Turn ended. You will be started again when this resolves or "
                 f"at {woken_at or f'about {minutes} minute(s) from now'}. "
                 f"Do not call any more tools."),
    })


def _write_episode(payment_id: str, outcome: str, facts: dict,
                   lesson: str = "") -> None:
    """One closed case → one structured episode, plus a profile update.

    Namespaced by failure kind so retrieval at case start is "cases like this
    one", not "everything ever". Text fields are PII-masked before storage —
    memory outlives cases, and a store full of card numbers and phone numbers
    is a liability, not a memory. Never raises.
    """
    try:
        from recovery_agent.state_store import StateStore
        from recovery_agent.agent.classify import failure_kind
        from recovery_agent.agent.governance import mask_pii
        from recovery_agent.agent.graph import get_memory_store
        from recovery_agent.agent import ladder as _lad
        import uuid as _uuid

        rec = StateStore().get_payment(payment_id) or {}
        kind = failure_kind(rec) or "unknown"
        owed = float(facts.get("owed") or 0)
        received = float(facts.get("received") or 0)
        discount_pct = (round((owed - received) / owed * 100, 1)
                        if outcome == "recovered" and owed > 0 and received < owed
                        else 0.0)
        climbed = list((_lad.state(rec) or {}).get("climbed") or [])
        contacts = [c.get("channel", "") for c in (rec.get("contacts") or [])]

        get_memory_store().put(
            ("recovery", "episodes", ns_safe(kind)), str(_uuid.uuid4()), {
                "failure_kind": kind,
                "failure_code": str(rec.get("failure_code") or ""),
                "outcome": outcome,
                "amount": owed,
                "recovered": received,
                "discount_pct": discount_pct,
                "climbed": climbed,
                "contacts": contacts,
                "payment_id": payment_id,
                "lesson": mask_pii(str(lesson or ""))[:400],
                "at": datetime.now(timezone.utc).isoformat(),
            })
    except Exception:
        return

    # The profile learns the outcome too: which channel closed it (or failed
    # to), so "this customer pays when you email them" is a number, not a hunch.
    try:
        from recovery_agent.agent.memory import CustomerMemoryStore
        customer = (rec.get("customer") or {}).get("email") \
            or rec.get("customer_email") or "unknown"
        channel = contacts[-1] if contacts else \
            ("voice" if "voice_call" in climbed else
             "email" if ("offer" in climbed or "post_call_email" in climbed) else
             "page" if "page_push" in climbed else "")
        CustomerMemoryStore.live().update_profile_after_attempt(
            customer,
            attempt={"payment_id": payment_id, "amount": received,
                     "failure_type": kind},
            success=(outcome == "recovered"),
            channel=channel,
        )
    except Exception:
        pass


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
                # Write the figure, not just the verdict.
                #
                # This wrote `status` and dropped `closed["amount_recovered"]`,
                # which is gateway-verified and sitting right there. A case then
                # read `recovered` with no `recovered_amount`, and /api/payments
                # papered over it with `recovered_amount or amount` — silently
                # crediting the FULL owed figure and handing the agent credit
                # for the discount it gave away.
                received = float(facts.get("received") or 0)
                if received > 0:
                    fields["recovered_amount"] = received
            elif outcome == "escalated":
                fields["status"] = "escalated"
            else:
                fields["status"] = "unrecoverable"
            # An ended case keeps no timers. `wait_for_customer` registers a
            # real wake-up now, so closing without clearing it leaves an alarm
            # set on a finished case — live on pay_4tnzl57fu, a 16-minute wake
            # outlived a recovery that completed in 15 seconds.
            fields["waiting_for"] = None
            store.update_payment(payment_id, **fields)
            store.cancel_jobs_for(payment_id, reason=f"case closed: {outcome}")
            store.flush()
            # The stopping rule, in the record. An auditor asking "why did this
            # stop?" gets the agent's own stated reason and the outcome it
            # declared, at the moment it declared them.
            from recovery_agent import audit
            rec = store.get_payment(payment_id) or {}
            audit.record(audit.CASE_CLOSED, payment_id=payment_id,
                         batch_run_id=rec.get("batch_run_id") or "",
                         result=outcome, reason=closed["what_happened"],
                         amount_rupees=closed["amount_recovered"],
                         climbed=sorted(rec.get("ladder") or {}))
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

    # The structured episode is NOT optional and NOT the agent's to remember:
    # every closed case teaches the next one, whether or not the model thought
    # to write a lesson. What failed, what was climbed, what ended it and at
    # what price — keyed by failure kind so the next bank-decline can be told
    # "in six cases like this, the method switch worked four times". Failure to
    # store must never block a closure.
    _write_episode(payment_id, outcome, facts, lesson)

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
    wait_minutes: int = 5,
    runtime=None,
) -> str:
    """Show the ONE in-page notification, to a customer still on the checkout.

    The SILENT first rung: it costs nothing, interrupts nobody, and puts no
    offer on the table. It cannot carry a link or a discount — those belong to
    show_page_offer, which draws a banner instead. This is the plain nudge, and
    there is only ever one of it.

    It used to accept `payment_link` and `offer_text`, and the agent used them:
    at the offer rung it sent a second notification carrying the link, which
    landed on top of the banner already showing the same price. Two things
    saying one thing, seconds apart, to a customer who had read the first. The
    parameters are gone, so that is no longer something to remember not to do.

    The customer can act on it, dismiss it, or ignore it. Whatever they do comes
    back to you as a `push_outcome` observation so you can decide what to do
    next — the outcome is a signal about their intent, not just a status.

    Args:
        payment_id: The payment this relates to
        headline: Short line the customer sees first
        body: One or two sentences of context
        cta_text: Label for the action button. It reopens the checkout they are
            already on; there is no link to send them to.
        wait_minutes: How long to wait for a response before you are called again

    Returns:
        JSON with delivery status. `delivered` means it reached a live page;
        `no_active_session` means the customer already left.
    """
    case = getattr(getattr(runtime, "context", None), "case", None) if runtime else None

    # ONE push per case, ever.
    #
    # This used to be two guards — one for "the customer already dismissed a
    # plain notification", one for "an offer is already on the page" — and both
    # existed to stop the same thing: a second notification. The rung itself is
    # the simpler statement of it. If a push has been delivered for this case,
    # the silent rung is spent, whatever the customer did with it.
    #
    # Read the LIVE record, not the snapshot taken when this run started: the
    # metadata copy is written once at case construction, so anything the
    # customer did after that — including the dismissal that triggered this very
    # run — can be missing from it.
    live: dict = {}
    try:
        from recovery_agent.state_store import StateStore
        live = StateStore().get_payment(payment_id) or {}
    except Exception:
        pass

    if ladder.climbed(live, "page_push"):
        outcome = (live.get("push_outcome") or {}).get("action") or "no response"
        return json.dumps({
            "status": "blocked",
            "reason": f"this customer has already had the one in-page "
                      f"notification ({outcome}); there is no second one",
            "guidance": "The page has had its turn. Put an authorised discount "
                        "on it with show_page_offer, or reach them off-page.",
        })

    # No link and no offer line. The button reopens the checkout the customer is
    # already looking at, which is the whole idea of the silent rung.
    payload = {
        "payment_id": payment_id,
        "headline": headline[:120],
        "body": body[:400],
        "cta_text": cta_text[:40],
        "payment_link": "",
        "offer_text": "",
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
        ladder.record_action(payment_id, "page_push:plain", is_rung=True)

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
        # One taxonomy, shared with the batch view and the briefing. This used
        # to re-derive the answer with its own word list, which is how the same
        # case could be a method failure to the offer policy and a drop-off to
        # everything else.
        from recovery_agent.agent.classify import failure_kind
        kind = failure_kind(rec)
        owed = float(rec.get("amount") or amount or 0)
        tried_full_price = any(
            a.startswith("link:") and a.endswith(f":{owed:.2f}")
            for a in (rec.get("actions_tried") or []))
        # INCENTIVES ARE ALLOWLISTED, not blocklisted. Only a confirmed
        # drop-off — the customer choosing not to pay — is a price problem,
        # so only "dropoff" unlocks a discount outright. Everything else must
        # try the full amount on another route first. The old shape (refuse
        # method and transient, allow the rest) meant an UNCLASSIFIED failure
        # defaulted to "discount legal": live, a bank decline whose code got
        # blurred at ingress was offered 5% for a payment the customer had
        # been trying to MAKE at full price. Fail closed: when we do not know
        # why it failed, we do not pay the customer to tell us.

        # A transient failure is never a price objection, at any stage. The
        # gateway timed out; the customer's intent was fine and their card was
        # fine. Discounting here pays someone to forgive our own outage.
        if kind == "transient":
            return json.dumps({
                "status": "not_indicated",
                "allowed": False,
                "reason": "this payment failed in transit — the gateway or the "
                          "network dropped it, not the customer and not their "
                          "card. There is nothing here that money fixes.",
                "do_this_instead": f"Let them try again at the full INR "
                                   f"{owed:,.2f}, or schedule a quiet retry. If a "
                                   f"clean retry ALSO fails, come back and an "
                                   f"offer is authorised.",
            })

        if kind == "risk":
            return json.dumps({
                "status": "not_indicated",
                "allowed": False,
                "reason": "this case looks like risk or fraud; it is not to be "
                          "pursued, let alone incentivised.",
                "do_this_instead": "escalate_to_human.",
            })

        if kind == "method" and not tried_full_price:
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

        if kind == "funds" and not tried_full_price:
            return json.dumps({
                "status": "not_indicated",
                "allowed": False,
                "reason": "the account was short at the time — a timing problem, "
                          "not a price one. A discount does not put money in "
                          "their account.",
                "do_this_instead": f"Schedule a retry aimed at when they are "
                                   f"likely to be paid, or offer the full INR "
                                   f"{owed:,.2f} on another rail. If a full-price "
                                   f"attempt fails again, ask here once more.",
            })

        if kind not in ("dropoff", "method", "funds") and not tried_full_price:
            return json.dumps({
                "status": "not_indicated",
                "allowed": False,
                "reason": f"this failure is unclassified ({kind}), so there is no "
                          f"evidence price was the problem. Incentives are "
                          f"reserved for confirmed drop-offs.",
                "do_this_instead": f"Offer the full INR {owed:,.2f} again — a "
                                   f"page push or a full-price link on another "
                                   f"rail. If that is refused or ignored, ask "
                                   f"here again and an offer is authorised.",
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


#: Rails a message might tell the customer to use, and the words people write
#: them with. Only checked where the text RECOMMENDS one — "your card was
#: declined, try UPI" names two rails and only the second is a promise.
_RAIL_WORDS = {
    "upi": ("upi",),
    "netbanking": ("netbanking", "net banking", "net-banking", "internet banking"),
    "card": ("card", "credit card", "debit card"),
    "wallet": ("wallet",),
    "emi": ("emi",),
}
_RECOMMENDS = re.compile(
    r"(?:try|use|pay(?:ing)?\s+(?:with|via|using|by)|switch(?:ing)?\s+to|"
    r"choose|select|opt\s+for)\b[^.;!?]{0,40}?"
    r"\b(upi|net\s?-?banking|internet banking|card|wallet|emi)\b", re.I)
#: A money figure only counts as a claim when it is marked as money. Bare
#: numbers are "16 minutes" and "3 attempts" far more often than prices.
_MONEY = re.compile(r"(?:₹|INR|Rs\.?)\s*([\d,]+(?:\.\d{1,2})?)", re.I)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
#: Saying "discount" when none was authorised is a promise the link will not keep.
_DISCOUNT_WORDS = re.compile(
    r"\b(discount|% off|percent off|reduced price|special price|"
    r"cheaper|save(?:\s+₹|\s+INR)?)\b", re.I)


def _ungrounded_claims(text: str, facts: dict) -> str:
    """What this message asserts that the case cannot back up. '' if honest.

    The agent writes the words — tone and wording are exactly what it is good
    at, and a templated "Reason: <prose>" email was both generic AND wrong.
    What it must not do is promise something the customer will not find when
    they click: a price the link does not charge, a discount nobody authorised,
    or a payment method the link does not accept.

    So the prose is free and the FACTS in it are checked. Same principle as
    show_page_offer refusing a banner that disagrees with its link.
    """
    owed = float(facts.get("owed") or 0)
    charged = facts.get("charged")
    pct = float(facts.get("discount_pct") or 0)
    rails = [str(r).lower() for r in (facts.get("rails") or [])]

    allowed_amounts = {round(owed, 2)} if owed else set()
    if charged is not None:
        allowed_amounts.add(round(float(charged), 2))
        if owed:
            allowed_amounts.add(round(owed - float(charged), 2))   # "save ₹124.95"

    for raw in _MONEY.findall(text):
        try:
            figure = round(float(raw.replace(",", "")), 2)
        except ValueError:
            continue
        if allowed_amounts and not any(abs(figure - a) < 0.01 for a in allowed_amounts):
            return (f"it quotes ₹{figure:,.2f}, which is neither the amount owed "
                    f"nor what the link charges "
                    f"({', '.join(f'₹{a:,.2f}' for a in sorted(allowed_amounts))})")

    for raw in _PERCENT.findall(text):
        try:
            claimed = float(raw)
        except ValueError:
            continue
        if pct <= 0:
            return (f"it claims {claimed:g}% off, but no discount is authorised "
                    f"on this case")
        if abs(claimed - pct) > 0.01:
            return (f"it claims {claimed:g}% off but the authorised discount is "
                    f"{pct:g}%")

    if pct <= 0 and _DISCOUNT_WORDS.search(text):
        return ("it promises a discount, but none is authorised on this case — "
                "the customer would arrive at the full price")

    for match in _RECOMMENDS.finditer(text):
        word = match.group(1).lower().replace("-", " ").replace("  ", " ")
        rail = next((k for k, words in _RAIL_WORDS.items()
                     if any(w == word or w.replace(" ", "") == word.replace(" ", "")
                            for w in words)), None)
        if rail and rails and rail not in rails:
            return (f"it tells the customer to pay by {rail}, which this payment "
                    f"link does not accept (it takes {', '.join(rails)})")
    return ""


@tool
def send_recovery_notification(
    payment_id: str,
    customer_email: str,
    customer_phone: str,
    message: str,
    subject: str = "",
    payment_link: str = "",
    amount: float = 0,
    attempt_count: int = 0,
    runtime=None,
) -> str:
    """Email/SMS the customer. YOU write it — subject and body both.

    Write for THIS customer and THIS failure. There is no template any more:
    a bank decline and an abandoned cart used to receive the identical
    "Payment Recovery: Complete Your Pending Payment", which is the least
    openable line available and tells the customer nothing they did not know.

    Say what actually happened and what to do about it — "your bank declined
    the card, UPI works on this link", "here is 5% off to finish your order".

    EVERY FACT YOU STATE IS CHECKED against the case before it is sent:
      - any ₹ figure must be the amount owed, what the link charges, or the
        saving between them
      - any % must be the discount actually authorised, and you may not use
        the language of a discount when none is authorised
      - any payment method you tell them to USE must be one the link accepts
    A message that fails these is refused with the reason, not quietly fixed.
    This is not about tone — write freely — it is about not promising the
    customer something they will not find when they click.

    Args:
        payment_id: The payment ID
        customer_email: Customer's email address
        customer_phone: Customer's phone number (with country code)
        message: The body, in your own words. No greeting or sign-off needed.
        subject: The email subject line. Keep it under 60 characters so it is
            not truncated on a phone, and put no prices or percentages in it —
            money belongs in the body where the link is. Leave empty only if
            you have nothing better than the generic line.
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

    # Every FACT the message states, checked against the case. The prose is the
    # agent's; the promises in it are the system's to keep.
    _rec_now = _live_record(payment_id)
    _claims = _ungrounded_claims(f"{subject}\n{message}", {
        "owed": float(_rec_now.get("amount") or amount or 0),
        "charged": charged,
        "discount_pct": meta.get("offer_pct") or 0,
        "rails": meta.get("recovery_link_rails") or [],
    })
    if _claims:
        return json.dumps({
            "status": "error",
            "message": f"this message promises something the case cannot back "
                       f"up: {_claims}.",
            "guidance": "Rewrite it so every figure, percentage and payment "
                        "method matches what the link actually does, then send "
                        "again. Do not remove the link or the offer — correct "
                        "the words.",
        })

    if len(subject) > 60:
        return json.dumps({
            "status": "error",
            "message": f"the subject is {len(subject)} characters; anything past "
                       f"60 is cut off on a phone, which is where most of these "
                       f"are read.",
            "guidance": "Shorten the subject and send again.",
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
            # The agent's own words, as the subject and the body — not stuffed
            # into a "Reason:" line inside a template, which is what used to
            # happen to every message it wrote.
            subject=subject,
            body=message,
            failure_reason=message,
            amount=amount,
            attempt_count=attempt_count,
        )
        # A rung is a claim that the customer was reached. dispatch() reports
        # only channels that genuinely delivered; when none did, nothing was
        # tried as far as the ladder is concerned — recording it anyway turns
        # "already tried: the offer email" into a false memory, and the next
        # run skips a rung the customer never saw.
        if result.get("status") != "dispatched":
            why = "; ".join(f"{ch}: {reason}" for ch, reason in
                            (result.get("undelivered") or {}).items()) \
                  or "no channel was available"
            return json.dumps({
                "status": "error",
                "message": f"the message could not be delivered ({why})",
                "guidance": "The customer has NOT been contacted and this rung "
                            "is not climbed. Put the offer on the page with "
                            "show_page_offer, or take another route — do not "
                            "assume they saw anything.",
            })
        # Which rung an email is depends on what came before it AND on what
        # broke. One sent after a call is the follow-up the call agreed to. On
        # a bank decline, a message carrying a FULL-PRICE link is the rail
        # switch — the customer is being offered another way to pay, not a
        # cheaper price — and calling that "the offer" made the ladder think
        # the discount rung was spent when nothing had been discounted.
        _rec = _live_record(payment_id)
        if ladder.climbed(_rec, "voice_call"):
            rung = "post_call_email"
        else:
            # Only when we KNOW what the link charges. Defaulting the unknown
            # case to "the full amount" claimed a rail switch for any message
            # whose link price was not recorded — the cheap rung would be
            # marked climbed while the expensive one stayed open, which is the
            # wrong way round. Unknown price falls back to the offer rung.
            owed = float(_rec.get("amount") or 0)
            charged = meta.get("recovery_link_amount")
            full_price = (owed > 0 and charged is not None
                          and abs(float(charged) - owed) < 0.01)
            rung = ("rail_switch"
                    if full_price and ladder.has_rung(_rec, "rail_switch")
                    else "offer")
        ladder.record_rung(payment_id, rung, message[:200])
        ladder.record_action(
            payment_id,
            f"notify:{'+'.join(sorted(result.get('channels') or []))}:"
            f"{payment_link or _last_recovery_link(runtime, payment_id)}",
            is_rung=True)
        _record_contact(payment_id, sorted(result.get("channels") or []))
        return json.dumps({
            "status": "ok",
            "rung": rung,
            "channels": result.get("channels", []),
            "results": result.get("results", []),
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool
def retry_in_hours(
    payment_id: str,
    hours: float,
    runtime=None,
) -> str:
    """Schedule a silent background retry after a number of hours from now.

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

        # A quiet retry IS a rung — the FIRST one for a funds or transient
        # failure, where contacting the customer cannot fix anything. It used
        # to record only a free-text action, so the ladder never saw it and a
        # case that had correctly scheduled a payday retry still read as
        # "nothing tried yet" and got marched into contact rungs.
        ladder.record_rung(payment_id, "silent_retry",
                           f"retry scheduled in {round(hours, 1)}h")
        ladder.record_action(payment_id, f"retry:{round(hours, 1)}h", is_rung=True)
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

    # A retry still on the clock is not a stuck case — it is a working one.
    # Live (C1, 2026-09-03): an insufficient-funds case was handed to a human
    # while retries were scheduled for the next day AND three days out, and
    # the closing note claimed both had "failed" when neither had run. For a
    # short account the pending retry is the single most likely thing to
    # recover the money; spending a person on it first is pure waste.
    if ladder.retry_pending(_rec) and not _barred:
        _job = _rec.get("scheduled_job") or {}
        return json.dumps({
            "status": "blocked",
            "reason": "a retry for this case is already scheduled and has not "
                      "fired yet, so this is not a case a person needs",
            "retry_at": _job.get("target_timestamp") or _job.get("target_time"),
            "guidance": "Call wait_for_customer. You will be started again when "
                        "the retry runs. Do NOT report the retry as failed — it "
                        "has not happened yet.",
        })

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


#: Statuses that mean a voice call genuinely went out. `superu_client` returns
#: "call_initiated" on success — a status this check did not accept, so a
#: successful PAID call recorded nothing: the voice rung stayed unclimbed and
#: the same customer could be rung again. The tool-status vocabulary is an
#: implicit protocol; keep the one success status the client actually speaks
#: in a named set the tests can pin against the client itself.
VOICE_CALL_OK_STATUSES = frozenset(
    {"call_initiated", "ok", "initiated", "queued", "success"})


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
    # Through `ladder.voice_available()`, which reads the operator setting and
    # falls back to the env var. Reading the env directly here made the
    # dashboard's voice switch decorative — and worse, it could disagree with
    # the ladder, so a rung shown as available would refuse when called.
    if not ladder.voice_available():
        return json.dumps({
            "status": "disabled",
            "reason": "voice calling is switched off for this deployment",
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
        from recovery_agent import guardrail_config
        min_amount = float(guardrail_config.get("voice_min_amount"))
    except Exception:
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

    if result.get("status") in VOICE_CALL_OK_STATUSES:
        ladder.record_rung(payment_id, "voice_call",
                           f"called {customer_phone}")
        ladder.record_action(payment_id, f"voice:{customer_phone}", is_rung=True)
        _record_contact(payment_id, ["voice"])

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
