"""Full-stack frontend — Observable agent flow.

Agent acts → customer sees what happened → customer responds → agent observes → decides next step.

Usage:
    python -m recovery_agent.frontend
    # Customer: http://localhost:5002/pay
    # Merchant: http://localhost:5002/merchant
"""
from __future__ import annotations

import os
import time
import traceback
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

import threading
from flask import Flask, render_template, render_template_string, request, jsonify
from flask_socketio import SocketIO

from recovery_agent import audit
from recovery_agent.razorpay_client import RazorpayClient
from recovery_agent.state_store import StateStore
import json
from langchain_core.messages import AIMessage, ToolMessage

# --- OpenTelemetry: manual spans for the parent→child hierarchy ---
#
# This used to build its OWN TracerProvider with a bare OTLP exporter and set
# it as the global — racing llm_client's lazy Phoenix register for the one
# global slot OTel allows. Whoever lost, silently: spans without OpenInference
# conventions, no project, no LangChain instrumentation, no cost attributes.
# All tracing now goes through the single shared init in observability.py.

def _get_tracer():
    from recovery_agent.observability import get_tracer
    return get_tracer("recovery-agent")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24).hex()
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

razorpay_client = RazorpayClient()
store = StateStore()

def _do_flush():
    """Persist current state to disk."""
    try:
        store.flush()
    except Exception:
        pass


def _emit_status_summary():
    """Emit current payment summary to connected clients."""
    try:
        all_p = store.all_payments()
        total = len(all_p)
        recovered = sum(1 for p in all_p.values() if p.get("status") == "recovered")
        socketio.emit("status_summary", {
            "total": total,
            "recovered": recovered,
            "recovery_rate": f"{(recovered/total*100):.1f}%" if total else "0%",
        })
    except Exception:
        pass


def _watch_for_recovery(payment_id: str, max_wait_seconds: int = 300):
    """Poll Razorpay for payment capture. Belt to webhook's suspenders.

    Two-pronged polling:
    1. Check if webhook already updated the store (fast path)
    2. Fetch the payment link from Razorpay to see if it's been paid
    3. Fetch recent payments to find one with original_payment note match
    """
    import time
    from recovery_agent.razorpay_client import RazorpayClient

    client = RazorpayClient()
    if not client.is_configured:
        return

    start_ts = time.time()
    link_created_at = int(start_ts)      # anything earlier predates this watch
    while time.time() - start_ts < max_wait_seconds:
        time.sleep(15)
        try:
            # Fast path: check if webhook already handled it
            if store.has_payment(payment_id):
                p = store.get_payment(payment_id)
                if p.get("status") == "recovered":
                    return

            # Strategy 1: fetch EVERY link this case has minted.
            #
            # This used to read a single id out of `order_id`, which holds only
            # the most recently minted link. A case that climbs the ladder mints
            # more than one -- the offer rung mints a discounted link, then a
            # rail switch mints another -- and the second overwrote the first.
            #
            # Live (pay_p9oxiasll): the customer paid the link that was EMAILED
            # to them, plink_TXwDValFzzEGgV, INR 1,709.05 captured. By then a
            # rail-switch link had replaced it in `order_id`, so the watcher
            # polled a link nobody would ever pay, saw nothing for the whole
            # window, and the case was escalated to a human at zero recovered
            # while the money was already in the account. A customer who has
            # paid must never be asked to pay again.
            p = store.get_payment(payment_id) if store.has_payment(payment_id) else {}
            link_ids, seen = [], set()
            for cand in ([l.get("link_id") for l in (p.get("recovery_links") or [])]
                         + [p.get("order_id", ""), p.get("recovery_link_id", "")]):
                if cand and cand.startswith("plink_") and cand not in seen:
                    seen.add(cand)
                    link_ids.append(cand)
            for link_id in link_ids:
                try:
                    link_data = client.client.payment_link.fetch(link_id)
                    link_status = link_data.get("status", "")
                    if link_status == "paid":
                        # Extract payment details from the link
                        payments = link_data.get("payments", [])
                        for pay in payments:
                            if pay.get("status") == "captured":
                                # Razorpay returns `payment_id` on a link's
                                # payments, not `id`. Reading the wrong key left
                                # the recovery payment unrecorded — the trail
                                # literally logged "Payment: ." — so nothing
                                # linked the money that came in to the case it
                                # came in for.
                                _mark_recovered(
                                    payment_id, pay.get("amount", 0) / 100,
                                    pay.get("payment_id") or pay.get("id", ""),
                                    int(time.time() - start_ts),
                                    f"link {link_id} paid (poll)",
                                )
                                return
                except Exception:
                    continue  # this link is unreadable; try the next one

            # A FAILED attempt is a signal, and a strong one.
            #
            # The watcher only ever looked for a capture, so a customer who
            # clicked the recovery link and was declined on it produced nothing
            # at all. Live: the agent routed a INR 59,985 case to netbanking,
            # the customer tried it on the link at 10:40:57 and was declined,
            # and the agent learned nothing — it would have sat out its
            # sixteen-minute window while the customer was actively struggling.
            # They happened to retry on their own 58 seconds later.
            #
            # Someone who clicks and tries is the most engaged a customer ever
            # gets. If the rail we chose fails them, that is the moment to
            # widen it — not a quarter of an hour later.
            try:
                phone = "".join(ch for ch in str(
                    (p.get("customer") or {}).get("contact")
                    or p.get("customer_phone") or "") if ch.isdigit())[-10:]
                owed = round(float(p.get("amount") or 0), 2)
                seen_failures = set(p.get("seen_failed_attempts") or [])
                if phone and owed:
                    for pay in client.client.payment.all(
                            {"count": 20}).get("items", []):
                        if pay.get("status") != "failed":
                            continue
                        if pay.get("id") in seen_failures:
                            continue
                        if round(pay.get("amount", 0) / 100, 2) != owed:
                            continue
                        digits = "".join(c for c in str(pay.get("contact") or "")
                                         if c.isdigit())[-10:]
                        if digits != phone:
                            continue
                        if pay.get("created_at", 0) < link_created_at:
                            continue          # the failure that started the case

                        seen_failures.add(pay["id"])
                        store.update_payment(
                            payment_id,
                            seen_failed_attempts=sorted(seen_failures))
                        store.flush()
                        method = pay.get("method") or "that method"
                        push_event(payment_id, "attempt_failed", {
                            "detail": f"Customer tried {method} and was declined "
                                      f"({pay.get('error_description') or 'no reason given'})",
                        })
                        _handoff_to_agent(
                            payment_id,
                            f"THE CUSTOMER JUST TRIED AND FAILED AGAIN. They "
                            f"acted on your recovery link and were declined on "
                            f"{method}: "
                            f"{pay.get('error_description') or 'no reason given'}. "
                            f"They are engaged and trying right now — this is the "
                            f"best moment you will get. The rail you chose is not "
                            f"working for them, so widen it or offer a different "
                            f"one. Do not wait out the window.",
                            scenario=f"attempt_failed:{pay['id']}",
                        )
                        break
            except Exception:
                pass

            # Strategy 2: the ORIGINAL checkout order. When the agent's push has
            # no link, its CTA reopens Razorpay on the page against this order —
            # so this is the object that actually gets paid on the commonest
            # recovery path of all, and nothing here used to look at it.
            order_id = p.get("original_order_id") or p.get("order_id") or ""
            if order_id.startswith("order_"):
                try:
                    od = client.client.order.fetch(order_id)
                    if od.get("status") == "paid":
                        for pay in client.client.order.payments(order_id).get("items", []):
                            if pay.get("status") == "captured":
                                _mark_recovered(
                                    payment_id, pay.get("amount", 0) / 100,
                                    pay.get("id", ""), int(time.time() - start_ts),
                                    f"order {order_id} paid (poll)",
                                )
                                return
                except Exception:
                    pass

            # Strategy 3: Fetch the original payment — maybe it was captured directly
            try:
                payment = client.client.payment.fetch(payment_id)
                if payment.get("status") == "captured":
                    # This branch used to update the store and return WITHOUT
                    # calling _notify_agent_of_recovery, so a case recovered this
                    # way went quiet: recorded as paid, agent never told, nothing
                    # learned from it.
                    _mark_recovered(payment_id, payment.get("amount", 0) / 100,
                                    payment_id, int(time.time() - start_ts),
                                    "original payment captured (poll)")
                    return
            except Exception:
                pass

        except Exception:
            pass

    # The customer has not paid. Deciding what happens next is the agent's job,
    # not this poller's — it used to emit "customer may need follow-up" and stop,
    # which quietly ended every unrecovered case. Hand the situation back to the
    # agent with what actually happened and let it choose: another channel, a
    # voice call, a scheduled retry, or escalation.
    push_event(payment_id, "recovery_timeout", {
        "detail": f"Payment not captured after {max_wait_seconds}s. Handing back to the agent.",
    })

    p = store.get_payment(payment_id) if store.has_payment(payment_id) else {}
    prior = p.get("last_action", "a recovery message")
    minutes = max(1, max_wait_seconds // 60)
    # Tell it where it stands. Without this the agent reasons about "the next
    # channel" with no idea whether one is left, and a case whose ladder is
    # finished can look to it like a case with more to try.
    from recovery_agent.agent import ladder as _lad
    st = _lad.state(p)
    nxt = st["remaining"][0] if st["remaining"] else None
    where = (f"Next rung: {nxt['rung']} — {nxt['what']}." if nxt else
             "Every rung of the ladder has now been tried and the money is "
             "still out. This is the point escalate_to_human exists for; it "
             "will accept the case now. Escalate and stop.")
    _handoff_to_agent(
        payment_id,
        f"No payment after {minutes} minute(s). A previous recovery attempt "
        f"({prior}) was delivered and the customer has not paid. "
        f"Already climbed: {', '.join(st['climbed']) or 'nothing'}. "
        f"Already tried: {', '.join(_lad.actions_tried(p)) or 'nothing'}. "
        f"Reason about why, then act. Do not repeat anything above. {where}",
        scenario="followup",
    )


# ═══════════════════════════════════════════════════════════════
# IN-PAGE PUSH — the silent first rung of the recovery ladder
# ═══════════════════════════════════════════════════════════════
#
# Cheapest possible nudge: the customer is still on the checkout, so talk to
# them there. No email, no SMS, no incentive. Whatever they do with it — act,
# dismiss, or ignore — comes back to the agent as a signal about their intent,
# and the agent decides the next channel from that.


def deliver_page_push(payload: dict) -> dict:
    """Emit an agent-authored push to the customer's open checkout page."""
    payment_id = payload.get("payment_id", "")
    if not payment_id:
        return {"status": "error", "note": "no payment_id"}

    if store.has_payment(payment_id):
        store.update_payment(
            payment_id,
            pending_push={**payload, "sent_at": datetime.now(timezone.utc).isoformat()},
            push_outcome=None,
        )
        store.flush()

    if not presence.is_live(payment_id):
        return {"status": "no_active_session",
                "note": "no checkout page is listening for this payment — "
                        "nothing was shown, so no rung was climbed"}

    socketio.emit("agent_push", payload)
    push_event(payment_id, "page_push", {
        "detail": payload.get("headline", ""), "body": payload.get("body", ""),
    })

    socketio.start_background_task(
        _watch_push_response, payment_id, int(payload.get("wait_minutes", 5)) * 60
    )
    return {"status": "delivered",
            "note": f"waiting up to {payload.get('wait_minutes', 5)} min for a response"}


def _watch_push_response(payment_id: str, timeout_seconds: int) -> None:
    """If the customer neither acts nor dismisses, that silence is itself a signal."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        socketio.sleep(3)
        p = store.get_payment(payment_id) if store.has_payment(payment_id) else {}
        if p.get("push_outcome"):
            return                      # /api/push-response already handled it
        if p.get("status") in ("recovered", "escalated"):
            return

    _record_push_outcome(payment_id, "ignored", seconds_shown=timeout_seconds,
                         detail="Customer left the notification on screen without acting.")


def _record_push_outcome(payment_id: str, action: str, seconds_shown: float = 0.0,
                         detail: str = "") -> None:
    """Store what the customer did, then let the agent interpret it."""
    if not store.has_payment(payment_id):
        return
    p = store.get_payment(payment_id)
    prior = p.get("push_outcome") or {}
    if prior:
        return                          # first real outcome wins; never double-handle

    push = p.get("pending_push") or {}
    outcome = {
        "action": action,                       # acted | dismissed | ignored
        "seconds_shown": round(float(seconds_shown or 0), 1),
        "headline": push.get("headline", ""),
        "detail": detail,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    store.update_payment(payment_id, push_outcome=outcome)
    store.flush()
    push_event(payment_id, "push_outcome", outcome)

    if action == "acted":
        # This used to just return, on the belief that "the recovery watcher has
        # it". For a plain push nothing was watching: `_watch_for_recovery` is
        # only started when a payment LINK is sent. So the one path the customer
        # is most likely to take — tap the first notification, pay inline — had
        # no observer at all. Start one now. The browser also reports success
        # directly; this is the belt for when the tab is closed first.
        socketio.start_background_task(_watch_for_recovery, payment_id, 420)
        return

    # Hand the observation to the agent. Deliberately NOT interpreted here — the
    # agent is told what the customer did and reasons about why, and about which
    # channel that implies. Encoding "dismissed => send email" in this function
    # would be putting the judgement back in the plumbing.
    _handoff_to_agent(
        payment_id,
        f"In-page notification outcome: the customer {action.upper()} it after "
        f"{outcome['seconds_shown']:.0f}s. Shown: {push.get('headline','(none)')!r}"
        + f". {detail} Work out what that behaviour implies about their intent, "
          f"then choose the channel most likely to recover this payment. Do not "
          f"repeat a channel that has already failed.",
        scenario="push_followup",
    )


def _link_original_order_to_recovery(payment_id: str, recovery_payment_id: str) -> None:
    """Annotate the original checkout order with where the money actually arrived.

    A discounted recovery MUST be a separate object — Razorpay refuses to change
    an order's amount ("amount is/are not required and should not be sent"), and
    the same holds for payment links. So two objects always exist, and the
    original will read unpaid in the dashboard forever, because it genuinely was.

    Notes ARE editable on an order, so the two are cross-referenced: the recovery
    link carries the original payment id, and the original order now carries the
    recovery payment id. Anyone opening either one in Razorpay can see the other.
    """
    p = store.get_payment(payment_id) if store.has_payment(payment_id) else {}
    original_order = p.get("original_order_id") or ""
    if not original_order.startswith("order_") or not razorpay_client.is_configured:
        return
    try:
        razorpay_client.client.order.edit(original_order, {"notes": {
            "recovered_by_payment": recovery_payment_id or "",
            "recovered_amount": str(p.get("recovered_amount") or ""),
            "recovery_case": payment_id,
            "note": "not paid directly; recovered via the recovery agent",
        }})
        print(f"[Frontend] linked {original_order} -> {recovery_payment_id}", flush=True)
    except Exception as exc:
        print(f"[Frontend] could not annotate {original_order}: {exc}", flush=True)


#: How the money actually arrived, said plainly. The agent writes the lesson
#: that goes into permanent memory, so it has to be told the truth about what
#: worked — not left to infer it from whatever action happened to be last.
_ARRIVAL_PHRASES = (
    ("link", "the customer paid the recovery payment link you sent"),
    ("checkout page", "the customer paid on the checkout page itself"),
    ("order", "the customer paid the original checkout order"),
    ("original payment", "the original payment was captured after all"),
)


def _how_it_arrived(how: str) -> str:
    low = (how or "").lower()
    for needle, phrase in _ARRIVAL_PHRASES:
        if needle in low:
            return phrase
    return "the payment was captured"


def _notify_agent_of_recovery(payment_id: str, amount: float, recovery_payment_id: str,
                              seconds: int, how: str = "") -> None:
    """Tell the agent it worked.

    The poller was updating the store and returning, so the case read
    `recovered` while the agent never learned the outcome. It could not confirm
    to the customer, and — worse — it never recorded *what worked*. The offer
    that actually recovered the money is the single most valuable thing a
    recovery agent can learn, and it was being thrown away on every success.
    """
    p = store.get_payment(payment_id) if store.has_payment(payment_id) else {}
    # "195s after none" — `last_action` is "none" whenever the last run took no
    # recovery action, and it was pasted straight into the sentence.
    what = p.get("last_action") or ""
    if what in ("", "none"):
        what = "your last recovery attempt"
    # `ui_spec` never had an `offer` field, so this read was always empty and
    # the note it built never appeared. The offer tool records the real one.
    offer = p.get("page_offer") or {}
    offer_note = (f" after a {offer.get('discount_pct')}% offer"
                  if offer.get("discount_pct") else "")

    _link_original_order_to_recovery(payment_id, recovery_payment_id)

    # Say WHICH channel the money came through, because the agent writes a
    # lesson from this and that lesson is permanent.
    #
    # This used to say "paid {seconds}s after {last_action}", and last_action is
    # merely the last thing recorded. Live: the customer paid the recovery LINK,
    # while a 24-hour retry sat scheduled for the following day and had not
    # fired. The agent read "65s after wait_and_retry", concluded the retry had
    # worked, and stored "CONFIRMED WINNING STRATEGY: silent 24h background
    # retry" — a false lesson, in memory that now survives restarts, that would
    # push it toward waiting a day instead of sending the link that actually
    # worked.
    arrival = _how_it_arrived(how)

    # Name the surfaces the link went out on, and say we cannot tell them apart.
    #
    # The same URL is put on the page banner, in the email and in the SMS.
    # Razorpay reports that the LINK was paid; it does not report which surface
    # the click came from. Told only "they paid the link", the agent filled the
    # gap and wrote "full recovery via email/SMS channel" into a permanent
    # lesson — while a discount banner with that same link sat on the page in
    # front of the customer. A guess is not a finding, and this one would push
    # the next recovery toward email when the page may have done the work.
    surfaces = []
    for a in (p.get("actions_tried") or []):
        if a.startswith("page_offer:"):
            surfaces.append("the offer banner on the page")
        elif a.startswith("notify:"):
            channels = a.split(":", 2)[1].replace("+", " and ") if ":" in a else "email"
            surfaces.append(f"{channels}")
    ambiguous = "link" in (how or "") and len(surfaces) > 1
    if ambiguous:
        arrival += (f" — and that link was on {len(surfaces)} surfaces at once "
                    f"({', '.join(surfaces)}), so which one they clicked is not "
                    f"recorded anywhere")

    pending_retry = ""
    if (p.get("scheduled_job") or {}).get("target_timestamp") and "link" in (how or ""):
        pending_retry = (" A background retry was scheduled and has NOT fired; it "
                         "is not what recovered this.")

    _handoff_to_agent(
        payment_id,
        f"RECOVERED. INR {amount:,.2f} is in "
        f"({recovery_payment_id or 'payment id unavailable'}), {seconds}s after "
        f"{what}{offer_note}. HOW IT ARRIVED: {arrival}.{pending_retry} "
        + (f"Credit the OFFER, which is what you can see worked — naming a "
           f"channel here would be a guess. "
           if ambiguous else
           f"Attribute the win to that channel and nothing else. ")
        + f"The lesson you store is permanent, and a wrong one will send the "
        f"next recovery down the wrong path. Close the case with close_case, "
        f"outcome 'recovered'. Do NOT contact the customer again.",
        scenario="recovered",
    )


def _never_entered_recovery(rec: dict) -> bool:
    """True for a record that exists only for order reconciliation.

    /api/create-order opens a `pending` record for every checkout, because the
    order id is the key that ties a later recovery back to the sale. When no
    failure signal ever arrived — no failure code, no ladder, no trail — the
    payment simply succeeded first try, and the recovery system has no
    business claiming it: no agent events, no "recovered" status, no revenue
    credited to a rescue that never happened.
    """
    return (str(rec.get("status") or "") == "pending"
            and not rec.get("failure_code")
            and not rec.get("ladder")
            and not (rec.get("trail") or []))


def _mark_recovered(payment_id: str, amount: float, rzp_payment_id: str,
                    seconds: int, how: str) -> bool:
    """Single place where a case becomes RECOVERED.

    Every route that can learn the money arrived — the link poller, the original
    order poller, the browser reporting its own checkout success — has to do the
    same four things: write the store, tell the dashboard, cross-reference the
    order in Razorpay, and tell the agent. They were each doing a different
    subset of that, which is how a captured payment could sit in Razorpay while
    the case still read `awaiting_customer`.

    Returns False if the case was already recovered, so a poller and a browser
    callback racing on the same payment cannot double-count it.
    """
    if not store.has_payment(payment_id):
        return False
    sp = store.get_payment(payment_id)
    if sp.get("status") == "recovered":
        return False
    sp["status"] = "recovered"
    sp["recovered_amount"] = amount
    sp["recovered_payment_id"] = rzp_payment_id
    # The money is in, so any wake-up this case set for itself is moot. Left
    # pending it fires later and is refused by the hand-off guard — harmless,
    # but a settled case should not keep its own alarms set.
    store.cancel_jobs_for(payment_id, reason="recovered")
    sp["waiting_for"] = None
    sp.setdefault("trail", []).append({
        "step": "recovery_confirmed",
        "msg": f"Payment captured: INR {amount:,.2f}",
        "detail": f"{how}. Payment {rzp_payment_id}. After {seconds}s",
        "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
    })
    store.flush()
    # The join that makes money measurable per batch. The run id was stamped on
    # the record by the executor when it acted; carrying it onto the event is
    # what lets a run's total be a sum over the log rather than a guess from
    # timestamps. A case recovered outside any batch carries an empty id and is
    # simply not counted against one.
    audit.record(audit.MONEY_RECOVERED, payment_id=payment_id,
                 batch_run_id=sp.get("batch_run_id") or "",
                 actor="observer", result=how, amount_rupees=amount,
                 rzp_payment_id=rzp_payment_id, seconds=seconds)
    push_event(payment_id, "recovery_confirmed", {
        "status": "recovered", "payment_id": payment_id, "amount": amount,
        "captured_payment_id": rzp_payment_id, "detail": f"{how} after {seconds}s",
    })
    socketio.emit("agent_event", {"payment_id": payment_id, "event": "complete",
                                  "status": "recovered"})
    _notify_agent_of_recovery(payment_id, amount, rzp_payment_id, seconds, how)
    return True


def _handoff_to_agent(payment_id: str, observation: str, scenario: str) -> None:
    """Re-invoke the agent on an existing case with something new that happened."""
    p = store.get_payment(payment_id) if store.has_payment(payment_id) else {}
    # A recovered case still gets exactly one closing run — that is the whole
    # point of telling the agent it worked.
    if scenario != "recovered" and p.get("status") in ("recovered", "escalated"):
        return
    # The agent said this case was over. A closure it declared explicitly is a
    # decision, not a side effect of running out of tools, and a late timer or a
    # stray push outcome must not restart it.
    if p.get("closed"):
        push_event(payment_id, "handoff_blocked", {
            "detail": f"case closed as {(p['closed'] or {}).get('outcome')} — "
                      f"not reopening",
        })
        return
    # One hand-off per OCCURRENCE, not per kind.
    #
    # Keyed on the scenario alone, a customer could only ever be listened to
    # once per kind of event. Live: they dismissed the first push (hand-off,
    # rung 2 sent), then dismissed the discount banner too — and that second
    # dismissal produced nothing, because `handoff_push_followup` was already
    # set. The ladder stalled at rung 2 with the agent waiting for a customer
    # who had already answered. The loop guard is still there; it is the
    # occurrence key that stops a single event firing twice.
    key = f"handoff_{scenario}"
    occurrence = ""
    if scenario == "push_followup":
        occurrence = str((p.get("push_outcome") or {}).get("at") or "")
    if occurrence:
        key = f"{key}:{occurrence}"
    if p.get(key):
        return
    store.update_payment(payment_id, **{key: True})
    store.flush()

    customer = {k: v for k, v in (p.get("customer") or {}).items() if v} or {
        k: v for k, v in {
            "email": p.get("customer_email", ""),
            "name": p.get("customer_name", ""),
            "contact": p.get("customer_phone", ""),
        }.items() if v
    }
    if not customer.get("email") and not customer.get("contact"):
        # Better to stop than to let the agent invent a recipient.
        push_event(payment_id, "handoff_blocked", {
            "detail": "no customer contact on file — cannot follow up",
        })
        return

    socketio.start_background_task(
        run_agent_for_payment, payment_id, float(p.get("amount", 0) or 0),
        observation, customer, scenario,
        p.get("decline_strategy", "") or "no_response", "", "",
        True,                           # queue rather than drop — see the guard
    )


# Register with the neutral bus so tools reach *this* Socket.IO server. Importing
# this module by name from a tool would load a second copy (see push_bus).
from recovery_agent import push_bus as _push_bus
_push_bus.register_delivery(deliver_page_push)


@app.route("/api/batches", methods=["GET"])
def api_batches():
    """Revenue still at risk, sorted into batches that share a fix.

    Every bin the classifier produces is shown, the un-runnable ones included.
    `unclassified` and `risk` are load-bearing classifications — one keeps a
    mystery failure out of the discount path, the other keeps fraud unchased —
    and a control that matters is a control the operator should be able to SEE
    working. Hiding them would also make the at-risk headline quietly smaller
    than the store it claims to describe.
    """
    from recovery_agent.agent.classify import summarise
    batches = summarise(list(store.payment_values()))
    return jsonify({
        "batches": batches,
        "total_at_risk": round(sum(b["value"] for b in batches), 2),
        "total_cases": sum(b["count"] for b in batches),
        "running": sorted(active_agent_payments),
    })


@app.route("/api/drop-reasons", methods=["GET"])
def api_drop_reasons():
    """What the checkout offers when it asks the customer why they stopped."""
    from recovery_agent import drop_reasons
    return jsonify({"choices": drop_reasons.choices()})


@app.route("/api/drop-reason/skip", methods=["POST"])
def api_drop_reason_skip():
    """The customer declined to say, or said nothing at all.

    A question the agent waits on has to have a way of ending, or a closed tab
    strands the case forever. Silence is not an answer, so nothing is recorded
    as one: the agent simply starts on the path it would have taken from the
    failure code alone.
    """
    data = request.get_json(silent=True) or {}
    payment_id = str(data.get("payment_id") or "")
    rec = store.get_payment(payment_id) if payment_id else None
    if rec is None:
        return jsonify({"error": "unknown case"}), 404
    if not rec.get("drop_reason_pending"):
        return jsonify({"status": "not_waiting", "payment_id": payment_id})

    store.update_payment(payment_id, drop_reason_pending=False)
    _do_flush()
    push_event(payment_id, "reason_skipped", {
        "step": "awaiting_reason",
        "msg": "No answer — proceeding on the failure code alone",
        "detail": "The customer skipped or did not reply. Silence is not "
                  "recorded as a reason; the agent starts on what it can infer.",
    })
    socketio.start_background_task(
        run_agent_for_payment, payment_id, float(rec.get("amount") or 0),
        rec.get("failure_reason") or "Payment failed",
        rec.get("customer") or {}, "standard",
        rec.get("failure_code") or "", rec.get("error_source") or "",
        rec.get("error_step") or "")
    return jsonify({"status": "released", "payment_id": payment_id})


@app.route("/api/drop-reason", methods=["POST"])
def api_drop_reason():
    """The customer's own answer for why they abandoned.

    Two customers who abandon after a bank decline look identical here: one had
    no balance (a discount is useless — they need time) and one found it
    cheaper elsewhere (a discount is exactly the lever). No error code
    separates them, so guessing loses money in both directions. This is the
    answer, in their words, and it outranks every inference.
    """
    from recovery_agent import drop_reasons

    data = request.get_json(silent=True) or {}
    payment_id = str(data.get("payment_id") or "")
    code = str(data.get("code") or "")
    text = str(data.get("text") or "").strip()[:400]

    spec = drop_reasons.get(code)
    if not payment_id or spec is None:
        return jsonify({"error": "payment_id and a known reason code required"}), 400
    if not store.has_payment(payment_id):
        return jsonify({"error": "unknown case"}), 404

    reason = {"code": code, "label": spec["label"], "text": text,
              "at": datetime.now(timezone.utc).isoformat()}
    store.update_payment(payment_id, drop_reason=reason,
                         drop_reason_pending=False)

    entry = {
        "step": "customer_said_why",
        "msg": f"Customer told us why: {spec['label']}",
        "detail": text or spec["means"],
        "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
    }
    rec = store.get_payment(payment_id) or {}
    rec.setdefault("trail", []).append(entry)
    push_event(payment_id, "customer_said_why", entry)

    # An answer we did not anticipate is read by a person, not keyword-matched
    # into the nearest bucket. Flagging it is the honest response to "other".
    flagged = False
    if spec.get("free_text") and text:
        try:
            from recovery_agent.escalation_queue import enqueue
            customer = rec.get("customer") or {}
            enqueue(
                payment_id=payment_id,
                reason=f"Customer gave their own reason for abandoning: {text}",
                amount=float(rec.get("amount") or 0),
                customer={"email": customer.get("email") or rec.get("customer_email", ""),
                          "name": customer.get("name", ""),
                          "contact": customer.get("contact") or rec.get("customer_phone", "")},
                customer_signals=[f"stated reason (free text): {text}"],
                failure_code=str(rec.get("failure_code") or ""),
                source="customer_stated_reason",
            )
            flagged = True
        except Exception as exc:
            print(f"[frontend] could not flag stated reason: {exc}", flush=True)
    store.flush()

    # Their answer decides the next move, so hand it to the agent now rather
    # than letting it act on whatever it had already inferred.
    _handoff_to_agent(payment_id, drop_reasons.briefing_line(reason),
                      scenario=f"stated_reason:{code}")

    return jsonify({"status": "recorded", "code": code,
                    "flagged_for_review": flagged})


@app.route("/api/guardrails", methods=["GET", "POST"])
def api_guardrails():
    """Read or change the guardrail policy at runtime.

    These were env-only, so the only way to move a boundary was shell access
    and a restart of four services — which is the wrong audience and the wrong
    moment. Live on pay_ls1dep23k a real bank-decline recovery was refused at
    "4/3 contacts in 24h" and there was nothing the operator could do about it
    from the dashboard they were watching it fail in.
    """
    from recovery_agent import guardrail_config as gc

    if request.method == "GET":
        return jsonify({"settings": gc.describe()})

    data = request.get_json(silent=True) or {}
    if data.get("reset"):
        return jsonify({"values": gc.reset(), "rejected": [], "reset": True})

    values, rejected = gc.update(data.get("changes") or {})
    # Say plainly what is now in force. The engine is rebuilt per run, so a
    # change applies to the next agent turn without a restart.
    push_event("guardrails", "policy_changed", {
        "detail": ", ".join(f"{k}={v}" for k, v in
                            (data.get("changes") or {}).items())[:300],
        "rejected": rejected,
    })
    return jsonify({"values": values, "rejected": rejected})


@app.route("/api/evals", methods=["GET"])
def api_evals():
    """Everything the EVALS view shows.

    Deliberately tolerant of a missing or stale scorecard: a fresh checkout has
    never run the evals, and a demo must not 500 because of that. Absent data
    is reported as absent (`ok: false` plus a reason the UI can render), never
    as zeros -- a zero would read as "nothing held" on the red-team bar, which
    is the exact opposite of "we have not measured yet".
    """
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[2]
    card_path = root / "evals" / "results" / "scorecard.json"
    if not card_path.exists():
        return jsonify({"ok": False,
                        "reason": "The evals have never been run here. "
                                  "Run `make evals` to produce a scorecard."})
    try:
        card = json.loads(card_path.read_text())
    except Exception as exc:
        return jsonify({"ok": False, "reason": f"Scorecard unreadable: {exc}"})

    from recovery_agent.evals import quality
    modes_in = card.get("modes") or {}
    modes, gated = {}, 0
    for name, d in modes_in.items():
        if not isinstance(d, dict):
            continue
        g = quality.gateability(name, d)
        gated += 1 if g.get("gateable") else 0
        modes[name] = {
            "gateable": bool(g.get("gateable")),
            "reasons": list(g.get("reasons") or []),
            "ran_at": d.get("ran_at"),
            "age_hours": quality.age_hours(d),
            "rate": d.get("conformance_rate"),
            "rate_with_ci": d.get("rate_with_ci"),
            "decisions": d.get("decisions"),
            "agreement_rate": d.get("agreement_rate"),
            "held_rate": d.get("held_rate"),
            "baits": d.get("baits"), "held": d.get("held"),
            "caught_by_gate": d.get("caught_by_gate"),
            "leaked": d.get("leaked"),
            "rows": d.get("rows") or [],
            "violations_by_rule": d.get("violations_by_rule") or {},
            "examples": d.get("examples") or [],
        }

    out = {"ok": True, "modes": modes,
           "verdict": {"verified": gated > 0, "gated": gated,
                       "of": len(modes)}}

    # Corpus health -- the "is the alarm still plugged in" block. Turns and
    # cases are reported separately on purpose: 76 turns are not 76
    # independent samples, and collapsing them is how an underpowered corpus
    # passes for a strong one.
    try:
        corpus = root / "evals" / "corpus" / "decisions.jsonl"
        rows = [json.loads(l) for l in corpus.read_text().splitlines() if l.strip()]
        unit, cov = quality.unit_of_analysis(rows), quality.coverage(rows)
        out["corpus"] = {**unit, **cov, "min_cases_to_gate": quality.MIN_CASES_TO_GATE}
    except Exception as exc:
        out["corpus"] = {"error": str(exc)}

    # The counterfactual is the money exhibit. It is pure replay over the
    # corpus -- no LLM, no network -- so it is safe to compute per request.
    try:
        from recovery_agent.evals import counterfactual
        out["counterfactual"] = counterfactual.compare()
    except Exception as exc:
        out["counterfactual"] = {"error": str(exc)}

    try:
        base = root / "evals" / "baseline.json"
        out["baseline"] = {"exists": base.exists()}
        if base.exists():
            b = json.loads(base.read_text()).get("modes", {}).get("recorded", {})
            out["baseline"].update({
                "conformance_rate": b.get("conformance_rate"),
                "decisions": b.get("decisions"),
                "cases": (b.get("unit") or {}).get("cases"),
                "ran_at": b.get("ran_at"),
            })
    except Exception:
        out["baseline"] = {"exists": False}

    return jsonify(out)


@app.route("/api/economics", methods=["GET"])
def api_economics():
    """Everything the ops view shows: what recovery costs, what policy refused,
    what the agent remembers, and what the last eval run scored."""
    from recovery_agent import economics
    from recovery_agent.agent.policy_gate import read_verdicts
    from recovery_agent.agent.memory import CustomerMemoryStore

    # ?scope=live shows only cases the agent worked end to end against a real
    # gateway order; ?scope=seeded shows the demo/batch volume. Blended by
    # default would flatter the cost-per-recovery, because a seeded case the
    # agent never touched costs nothing.
    scope = (request.args.get("scope") or "live").strip().lower()
    if scope not in ("live", "seeded", "all"):
        scope = "live"
    records = list(store.payment_values())
    out = {"economics": economics.summarise(records, scope=scope)}
    # The LLM gate's ledger: how many calls this process has made and how long
    # they queued behind the quota — the free tier's cost in seconds, shown
    # rather than suffered silently.
    from recovery_agent.ratelimit import llm_gate
    out["llm_gate"] = llm_gate().stats()
    counts = {"live": 0, "seeded": 0}
    for r in records:
        counts[economics.case_origin(r)] += 1
    out["scopes"] = {"selected": scope, "counts": counts}

    verdicts = read_verdicts(300)
    tallies: dict[str, dict[str, int]] = {}
    for v in verdicts:
        if v.get("outcome") == "blocked":
            name = next((c["guardrail"] for c in (v.get("checks") or [])
                         if c.get("verdict") in ("blocked", "modified")), "policy")
        else:
            name = "all_passed"
        t = tallies.setdefault(name, {"pass": 0, "blocked": 0})
        t[v.get("outcome", "pass")] = t.get(v.get("outcome", "pass"), 0) + 1
    out["guardrails"] = {
        "tallies": tallies,
        "recent_blocks": [v for v in verdicts if v.get("outcome") == "blocked"][:12],
        "evaluations": len(verdicts),
    }

    try:
        mem = CustomerMemoryStore.live().get_stats()
    except Exception:
        mem = {}
    episodes: dict[str, int] = {}
    try:
        from recovery_agent.agent.graph import get_memory_store
        for item in get_memory_store().search(("recovery", "episodes"), limit=500):
            kind = (item.value or {}).get("failure_kind", "unknown")
            episodes[kind] = episodes.get(kind, 0) + 1
    except Exception:
        pass
    out["memory"] = {"profiles": mem, "episodes_by_kind": episodes,
                     "episodes_total": sum(episodes.values())}

    try:
        from pathlib import Path as _P
        scorecard = _P(__file__).resolve().parents[2] / "evals" / "results" / "scorecard.json"
        out["evals"] = json.loads(scorecard.read_text()) if scorecard.exists() else None
    except Exception:
        out["evals"] = None

    # Deep links into Phoenix: every case is a session there, with tokens AND
    # dollar cost rolled up by its price table. None while Phoenix is down.
    try:
        from recovery_agent.observability import phoenix_project_info
        out["phoenix"] = phoenix_project_info()
    except Exception:
        out["phoenix"] = None

    # What the agent has left to spend today. The email allowance is the one
    # that runs out quietly — a send over the cap simply never arrives.
    try:
        from recovery_agent import email_quota
        out["budgets"] = {"email": email_quota.status()}
    except Exception:
        out["budgets"] = {}

    # Payment links are an ACCOUNT-wide lifetime quota, so it is counted over
    # every record regardless of the scope being viewed — a link spent by a
    # seeded case is just as gone.
    try:
        # Counted from the case records PLUS a baseline, because the records
        # are resettable and the quota is not. Clearing data/ for a clean demo
        # took the count back to zero while the account had already spent ~21
        # of its 30 lifetime links — a gauge reading "30 left" when nine
        # remain is worse than no gauge at all.
        #
        # Razorpay cannot supply the true figure: payment_link.all() returns
        # an empty list on this account even for links that payment_link.fetch()
        # confirms exist. So the baseline is operator-set and deliberately
        # labelled as a floor rather than a fact.
        from_records = sum(len(r.get("recovery_links") or []) for r in records)
        baseline = int(os.getenv("RAZORPAY_LINKS_ALREADY_SPENT", "0"))
        minted = from_records + baseline
        limit = int(os.getenv("RAZORPAY_LINK_LIFETIME_LIMIT", "30"))
        out["budgets"]["links"] = {
            "minted": minted, "limit": limit,
            "from_records": from_records, "baseline": baseline,
            "remaining": max(0, limit - minted),
            "exhausted": minted >= limit,
        }
    except Exception:
        pass

    return jsonify(out)


def _batch_candidates(key: str) -> list[dict]:
    """The cases in a batch, as the classifier sorts them right now.

    Read fresh on every call rather than carried from the plan: a case can
    settle, be reclassified or climb a rung between planning and its turn, and
    the whole point of re-checking per case is that the batch is not frozen.
    """
    from recovery_agent.agent.classify import classify
    return [r for r in store.payment_values()
            if classify(r) == key and r.get("payment_id")]


@app.route("/api/batches/<key>/plan", methods=["GET", "POST"])
def api_plan_batch(key: str):
    """What this batch would do, and to whom — without doing any of it.

    Costs nothing and spends no payment-link quota, because it runs the same
    twelve prechecks the live run does and stops before the first side effect.
    That is what makes it possible to show a two-hundred-case batch's plan on a
    test account that owns five links.
    """
    from recovery_agent.agent.classify import BATCH_BY_KEY
    from recovery_agent.batch import run as batch_run
    from recovery_agent.batch.plan import BatchBudget
    from recovery_agent.batch.planner import plans_for

    meta = BATCH_BY_KEY.get(key)
    if not meta:
        return jsonify({"error": f"unknown batch {key!r}"}), 404
    if not meta.get("runnable"):
        return jsonify({"error": f"{key} is not worked as a batch",
                        "why": meta.get("what", "")}), 400

    data = request.get_json(silent=True) or {}
    records = _batch_candidates(key)
    plans, rejected = plans_for(key, records)
    for entry in rejected:
        audit.record(audit.BATCH_PLAN_REJECTED, subject_type=audit.BATCH_RUN,
                     subject_id=f"plan:{key}", actor="planner",
                     reason=entry["why"], batch_key=key, tier=entry["tier"])

    run = batch_run.BatchRun(batch_key=key, plans=plans, dry_run=True,
                             budget=BatchBudget.from_request(data.get("budget")),
                             started_by="dry_run")
    report = run.execute(records)
    return jsonify({"batch": key, "title": meta["title"], "report": report,
                    "plans": {t: p.as_dict() for t, p in plans.items()},
                    "rejected": rejected})


@app.route("/api/batches/<key>/run", methods=["POST"])
def api_run_batch(key: str):
    """Work a batch: one decision per band, applied to every case in it.

    Returns as soon as the run is registered. The report is resolved on read
    from the audit log, so `GET /api/batch-runs/<id>` is correct while the run
    is still going and keeps climbing after it finishes as customers pay.
    """
    from recovery_agent.agent.classify import BATCH_BY_KEY
    from recovery_agent.batch import run as batch_run
    from recovery_agent.batch.plan import BatchBudget
    from recovery_agent.batch.planner import plans_for

    meta = BATCH_BY_KEY.get(key)
    if not meta:
        return jsonify({"error": f"unknown batch {key!r}"}), 404
    if not meta.get("runnable"):
        return jsonify({"error": f"{key} is not worked as a batch",
                        "why": meta.get("what", "")}), 400

    data = request.get_json(silent=True) or {}
    if data.get("dry_run"):
        return api_plan_batch(key)

    # One live run per batch. A second click while the first is going would
    # spend a second budget against the same cases.
    for existing in batch_run.live():
        if existing.batch_key == key:
            return jsonify({"error": "that batch is already running",
                            "batch_run_id": existing.run_id}), 409

    records = _batch_candidates(key)
    plans, rejected = plans_for(key, records)
    if not plans:
        return jsonify({"error": "no plan could be made for this batch",
                        "rejected": rejected}), 422

    run = batch_run.register(batch_run.BatchRun(
        batch_key=key, plans=plans,
        budget=BatchBudget.from_request(data.get("budget")),
        started_by=str(data.get("started_by") or "dashboard")))

    def _on_decision(decision):
        """Watch every link this run creates, the way the agent path does.

        Without this the batch is a send-only system: links go out, customers
        pay them, and nothing ever notices — so "measured money recovered"
        reports zero however much actually came back. The watcher keys off
        `order_id`, which is where the agent path also puts the link id.
        """
        link_id = decision.detail.get("link_id")
        if decision.outcome != "acted" or not link_id:
            return
        try:
            store.update_payment(decision.payment_id, order_id=link_id,
                                 recovery_link=decision.detail.get("link_url", ""))
            store.flush()
            socketio.start_background_task(_watch_for_recovery,
                                           decision.payment_id, 900)
        except Exception:
            pass

    def _work():
        try:
            report = run.execute(records, on_decision=_on_decision)
        except Exception as exc:                      # pragma: no cover
            run.stop_reason = f"{type(exc).__name__}: {exc}"
            report = run.report()
        push_event("batch", "batch_finished",
                   {"batch": key, "batch_run_id": run.run_id, **report})

    socketio.start_background_task(_work)
    push_event("batch", "batch_started",
               {"batch": key, "batch_run_id": run.run_id,
                "candidates": len(records), "tiers": sorted(plans)})
    return jsonify({"batch": key, "title": meta["title"],
                    "batch_run_id": run.run_id, "candidates": len(records),
                    "plans": {t: p.as_dict() for t, p in plans.items()},
                    "rejected": rejected,
                    "budget": run.budget.as_dict()}), 202


@app.route("/api/batch-runs", methods=["GET"])
def api_batch_runs():
    """Every run, newest first, each with its money resolved as of now."""
    from recovery_agent.batch import run as batch_run
    opened = audit.log().of_kind(audit.BATCH_OPENED, limit=200)
    runs = [batch_run.projection(e["batch_run_id"]) for e in reversed(opened)
            if e.get("batch_run_id")]
    return jsonify({"runs": runs, "live": [r.run_id for r in batch_run.live()]})


@app.route("/api/batch-runs/<run_id>", methods=["GET"])
def api_batch_run(run_id: str):
    """One run's report, recomputed from the log.

    A batch finishes sending in seconds and customers pay over the following
    minutes, so this number climbs after the run has closed. That is correct: a
    total frozen at `finished_at` would be wrong in the flattering direction.
    """
    from recovery_agent.batch import run as batch_run
    report = batch_run.projection(run_id)
    live = batch_run.get(run_id)
    # A run is answered from the moment it is registered, not from its first
    # event: `POST /run` returns before the background task has opened the log,
    # and a caller that polls immediately — which is every caller — would
    # otherwise be told its own run does not exist.
    if not report["events"] and live is None:
        return jsonify({"error": f"unknown run {run_id!r}"}), 404
    if live is not None:
        report.update(status=live.status, stop_reason=live.stop_reason,
                      spend=live.spend.as_dict(),
                      decisions=[d.as_dict() for d in live.decisions])
    if request.args.get("events"):
        report["trail"] = audit.log().for_run(run_id)
    return jsonify(report)


def _queued_session_runner(referral):
    """One full agent session for a case the batch could not decide."""
    rec = store.get_payment(referral.payment_id) or {}
    customer = {k: v for k, v in (rec.get("customer") or {}).items() if v}
    run_agent_for_payment(
        referral.payment_id, float(rec.get("amount") or 0),
        f"BATCH EXCEPTION: {referral.why}. The batch machinery judged this "
        f"case to need a person's kind of attention rather than a shared "
        f"plan — its history is in the briefing.",
        customer, "batch_exception", rec.get("failure_code", "") or "")


@app.route("/api/batches/waves", methods=["POST"])
def api_start_waves():
    """Start a wave cycle: bin, decide once per bin, apply, watch, re-bin,
    repeat — until everyone paid, a wave changes nothing, or a human stops it.

    `champion: true` puts the live agent in the loop: one representative case
    per bin is worked as a full session and its decisions become the bin's
    plan. Without it the policy defaults carry every bin — the mode for a room
    where the LLM proxy cannot be trusted to show up.
    """
    payload, status = _launch_wave_cycle(request.get_json(silent=True) or {})
    return jsonify(payload), status


def _launch_wave_cycle(data: dict) -> tuple[dict, int]:
    """One door for starting a cycle, shared by the route and the autopilot."""
    from recovery_agent.batch import waves
    from recovery_agent.batch.plan import BatchBudget

    ids = [str(x) for x in (data.get("payment_ids") or []) if x]
    if not ids:
        from recovery_agent.agent.classify import BATCH_BY_KEY, classify
        wanted = str(data.get("batch_key") or "")
        ids = [r["payment_id"] for r in store.payment_values()
               if r.get("payment_id")
               and (classify(r) == wanted if wanted
                    else (BATCH_BY_KEY.get(classify(r) or "") or {}).get("runnable"))]
    if not ids:
        return {"error": "no open cases to work"}, 422

    for existing in waves.all_cycles():
        if existing.status == waves.RUNNING:
            return {"error": "a wave cycle is already running",
                    "cycle_id": existing.cycle_id}, 409

    config = waves.WaveConfig(
        max_waves=max(1, min(int(data.get("max_waves") or 3), 6)),
        settle_seconds=max(5.0, min(float(data.get("settle_seconds") or 60), 900)),
        champion_mode="live" if data.get("champion") else "off",
        budget=BatchBudget.from_request(data.get("budget")))

    def _champion_runner(record, context):
        """One real agent session, run to completion, for the bin's champion."""
        customer = {k: v for k, v in (record.get("customer") or {}).items() if v}
        run_agent_for_payment(
            record["payment_id"], float(record.get("amount") or 0),
            context, customer, "batch_champion",
            record.get("failure_code", "") or "")

    def _watch(decision):
        # Same wiring as a plain batch run: every link the cycle creates gets
        # a watcher, or the money that comes back is never noticed.
        link_id = decision.detail.get("link_id")
        if decision.outcome != "acted" or not link_id:
            return
        try:
            store.update_payment(decision.payment_id, order_id=link_id,
                                 recovery_link=decision.detail.get("link_url", ""))
            store.flush()
            socketio.start_background_task(_watch_for_recovery,
                                           decision.payment_id, 900)
        except Exception:
            pass

    cycle = waves.register(waves.WaveCycle(
        payment_ids=ids, config=config,
        champion_runner=_champion_runner if config.champion_mode == "live" else None,
        on_decision=_watch,
        started_by=str(data.get("started_by") or "dashboard")))

    def _work():
        try:
            report = cycle.execute()
        except Exception as exc:                       # pragma: no cover
            cycle.stop_reason = f"{type(exc).__name__}: {exc}"
            cycle.status = waves.ABORTED
            report = cycle.report()
        if data.get("dispatch_exceptions"):
            # "To the agent" becomes literal: each leftover gets a full
            # session, one at a time, off the queue — paced by the same LLM
            # gate as everything else. Off by default because each session
            # costs ~5 model calls, and a room without the proxy should get
            # a report, not a hang.
            from recovery_agent.batch import agent_queue
            q = agent_queue.get(_queued_session_runner)
            for entry in report.get("exceptions", []):
                q.submit(agent_queue.Referral(
                    entry["payment_id"], entry.get("why", ""),
                    batch_run_id=entry.get("batch_run_id", ""),
                    cycle_id=cycle.cycle_id))
            report["dispatched_to_agent"] = q.depth() + q.stats()["worked"]
        push_event("batch", "wave_cycle_finished", report)

    socketio.start_background_task(_work)
    push_event("batch", "wave_cycle_started",
               {"cycle_id": cycle.cycle_id, "cases": len(ids),
                "config": config.as_dict()})
    return {"cycle_id": cycle.cycle_id, "cases": len(ids),
            "config": config.as_dict()}, 202


def _wave_autopilot_blocked(now=None) -> str:
    """'' when a scheduled pass may start; otherwise the reason it must not.

    Split from the loop so the decision is testable without a running server:
    the loop is a sleep and a call, and everything worth arguing about is here.
    """
    from recovery_agent.batch import waves
    if any(c.status == waves.RUNNING for c in waves.all_cycles()):
        return "a cycle is already running"
    try:
        from recovery_agent.agent.guardrails import in_quiet_hours
        # Deliberately stricter than the per-case policy. Quiet hours restrict
        # only contact that interrupts, so a live agent may legitimately email
        # a case overnight — it is responding to something. An unattended bulk
        # pass is not responding to anything; nobody is watching it, and a
        # merchant who finds two hundred 02:00 timestamps in their outbox was
        # surprised by their own system. Bulk waits for morning.
        if (os.getenv("GUARDRAIL_QUIET_DISABLED", "").strip().lower()
                not in ("1", "true", "yes") and in_quiet_hours(now)):
            return "quiet hours"
    except Exception:
        return "guardrails unavailable"      # fail closed: no unattended sends
    return ""


def _wave_autopilot(minutes: float) -> None:
    """The unattended pass that gives 'deferred' its meaning.

    A case deferred for quiet hours is a promise — "not now" — and a promise
    needs someone to come back. This loop is that someone: on a timer, outside
    quiet hours, when nothing else is running, it starts an ordinary wave
    cycle over whatever is open. Every safeguard is downstream and unchanged —
    the twelve prechecks skip what should not be touched, the ladder forbids
    repeats, the budget bounds the spend — so the only thing scheduled here is
    attention.

    Off unless WAVE_AUTOPILOT_MINUTES is set: an unattended sender is a thing
    a merchant turns on, never a thing that turns itself on.
    """
    while True:
        socketio.sleep(max(60.0, minutes * 60))
        why_not = _wave_autopilot_blocked()
        if why_not:
            continue
        payload, status = _launch_wave_cycle({"started_by": "autopilot"})
        if status == 202:
            print(f"[Autopilot] wave cycle {payload['cycle_id']} over "
                  f"{payload['cases']} open case(s)", flush=True)


@app.route("/api/labels", methods=["GET"])
def api_labels():
    """The operator's verdict vocabulary, in render order."""
    from recovery_agent import labels
    return jsonify({"labels": labels.choices()})


@app.route("/api/cases/<payment_id>/label", methods=["POST"])
def api_label_case(payment_id: str):
    """A human's verdict on why an attempt did not land.

    The one verdict deliberately missing is "it succeeded" — money is marked
    recovered only by the gateway. `paid_outside` closes the case in its own
    column and adds nothing to recovered totals, so no amount of clicking can
    inflate the number the batch report calls measured.
    """
    from recovery_agent import labels
    from recovery_agent.agent.classify import classify

    rec = store.get_payment(payment_id)
    if rec is None:
        return jsonify({"error": f"unknown case {payment_id!r}"}), 404

    data = request.get_json(silent=True) or {}
    spec = labels.get(data.get("code"))
    if spec is None:
        return jsonify({"error": f"unknown label {data.get('code')!r}",
                        "labels": [l["code"] for l in labels.choices()]}), 400
    note = str(data.get("note") or "").strip()[:300]
    if spec.get("wants_note") and not note:
        return jsonify({"error": "this label needs a note — it exists for "
                                 "the answers the list did not anticipate"}), 400

    fields: dict = {"operator_label": {
        "code": spec["code"], "note": note, "by": "operator",
        "at": datetime.now(timezone.utc).isoformat(),
    }}
    if spec.get("opts_out"):
        fields["opted_out"] = True
    if spec.get("settles"):
        fields["status"] = "settled_outside"
        fields["closed"] = {
            "outcome": "settled_outside",
            "what_happened": (f"operator reports payment arrived outside the "
                              f"recovery rail. {note}").strip(),
            "at": fields["operator_label"]["at"],
            # Deliberately no amount: this money is claimed, not verified.
        }
        try:
            store.cancel_jobs_for(payment_id, "settled outside the rail")
        except Exception:
            pass
    store.update_payment(payment_id, **fields)
    store.flush()

    fresh = store.get_payment(payment_id) or {}
    audit.record(audit.CASE_LABELED, payment_id=payment_id,
                 batch_run_id=fresh.get("batch_run_id") or "",
                 actor="operator", result=spec["code"], reason=spec["label"],
                 note=note, rebins_to=classify(fresh),
                 opted_out=bool(spec.get("opts_out")),
                 settled_outside=bool(spec.get("settles")))
    push_event(payment_id, "labeled", {"label": spec["label"],
                                       "code": spec["code"]})
    return jsonify({"payment_id": payment_id, "label": spec["code"],
                    "rebins_to": classify(fresh),
                    "next": spec.get("next", "")})


@app.route("/api/batch-verdicts", methods=["GET"])
def api_batch_verdicts():
    """Cases a batch acted on that still need a human's why.

    Acted-but-unpaid is the whole list: paid cases verdict themselves through
    the gateway, and untouched cases have nothing to explain yet. A case
    already labeled since its last action drops off — the verdict is in.
    """
    out = []
    for rec in store.payment_values():
        run_id = rec.get("batch_run_id") or ""
        if not run_id:
            continue
        if str(rec.get("status") or "") in ("recovered", "settled_outside",
                                            "escalated"):
            continue
        acted_at = str(rec.get("batch_attributed_at") or "")
        label = rec.get("operator_label") or {}
        if label and str(label.get("at") or "") >= acted_at:
            continue                      # verdict already in for this attempt
        last = ""
        try:
            for e in reversed(audit.log().for_payment(rec["payment_id"])):
                if e["kind"] == audit.ACTION_RESULT:
                    last = f"{e.get('action') or 'acted'} · {e['created_at'][11:16]}"
                    break
        except Exception:
            pass
        customer = rec.get("customer") or {}
        out.append({"payment_id": rec["payment_id"],
                    "amount": float(rec.get("amount") or 0),
                    "name": customer.get("name") or "",
                    "batch_run_id": run_id, "acted_at": acted_at,
                    "last_action": last})
    out.sort(key=lambda r: r["acted_at"], reverse=True)
    return jsonify({"verdicts": out[:50]})


@app.route("/api/batch-activity", methods=["GET"])
def api_batch_activity():
    """The batch engine's feed — a straight read of the append-only log."""
    try:
        since = int(request.args.get("since") or 0)
    except (TypeError, ValueError):
        since = 0
    events = audit.log().batch_activity(since_event_id=since, limit=200)
    from recovery_agent.batch import waves as _waves
    return jsonify({
        "events": events,
        "cursor": events[-1]["event_id"] if events else since,
        "live": bool([c for c in _waves.all_cycles()
                      if c.status == _waves.RUNNING]),
    })


@app.route("/api/batch-cycles", methods=["GET"])
def api_batch_cycles():
    """Every cycle — live ones from the registry, finished ones rebuilt from
    the audit log, so a restart loses the object but never the report."""
    from recovery_agent.batch import waves
    live = {c.cycle_id: c.report() for c in waves.all_cycles()}
    cycles = []
    for cid in waves.known_cycle_ids():
        report = live.pop(cid, None) or waves.cycle_projection(cid)
        if report:
            cycles.append(report)
    cycles.extend(live.values())          # registered but not yet in the log
    return jsonify({"cycles": cycles})


@app.route("/api/batch-cycles/<cycle_id>", methods=["GET"])
def api_batch_cycle(cycle_id: str):
    from recovery_agent.batch import agent_queue, waves
    cycle = waves.get(cycle_id)
    report = cycle.report() if cycle is not None \
        else waves.cycle_projection(cycle_id)
    if report is None:
        return jsonify({"error": f"unknown cycle {cycle_id!r}"}), 404
    q = agent_queue.get()
    if q is not None:
        report["agent_queue"] = q.stats()
    return jsonify(report)


@app.route("/api/batch-cycles/<cycle_id>/abort", methods=["POST"])
def api_abort_batch_cycle(cycle_id: str):
    from recovery_agent.batch import waves
    cycle = waves.get(cycle_id)
    if cycle is None or cycle.status != waves.RUNNING:
        return jsonify({"error": f"no live cycle {cycle_id!r}"}), 404
    data = request.get_json(silent=True) or {}
    cycle.abort(str(data.get("reason") or "aborted from the dashboard"))
    return jsonify({"cycle_id": cycle_id, "status": "aborting"})


@app.route("/api/batch-runs/<run_id>/abort", methods=["POST"])
def api_abort_batch_run(run_id: str):
    """Stop a run. An action already in flight finishes — a payment link created
    and never mentioned to the customer is worse than one extra email."""
    from recovery_agent.batch import run as batch_run
    live = batch_run.get(run_id)
    if live is None or live.status != batch_run.OPEN:
        return jsonify({"error": f"no live run {run_id!r}"}), 404
    data = request.get_json(silent=True) or {}
    live.abort(str(data.get("reason") or "aborted from the dashboard"))
    return jsonify({"batch_run_id": run_id, "status": "aborting",
                    "stop_reason": live.stop_reason})


@app.route("/api/escalations", methods=["GET"])
def api_escalations():
    """The batch a human works. Open by default."""
    from recovery_agent.escalation_queue import list_tickets, summary
    status = request.args.get("status", "open") or None
    return jsonify({"summary": summary(), "tickets": list_tickets(status=status)})


@app.route("/api/escalations/<ticket_id>/resolve", methods=["POST"])
def api_resolve_escalation(ticket_id: str):
    from recovery_agent.escalation_queue import resolve
    data = request.get_json(silent=True) or {}
    closed = resolve(ticket_id, outcome=str(data.get("outcome") or ""),
                     by=str(data.get("by") or "human"))
    if closed is None:
        return jsonify({"error": "no such ticket"}), 404
    return jsonify(closed)


@app.route("/api/push-response", methods=["POST"])
def push_response():
    """The checkout page reports what the customer did with the notification."""
    data = request.get_json(silent=True) or {}
    payment_id = str(data.get("payment_id") or "")
    action = str(data.get("action") or "").lower()
    if not payment_id or action not in ("acted", "dismissed"):
        return jsonify({
            "error": "payment_id and action (acted|dismissed) required"
        }), 400
    _record_push_outcome(payment_id, action,
                         seconds_shown=data.get("seconds_shown", 0),
                         detail=str(data.get("detail") or ""))
    return jsonify({"status": "recorded", "action": action})


# Transient — only tracks in-flight agent threads, not persisted
active_agent_payments: set[str] = set()


# What each case has already shown on the HUD, keyed by tool_call id.
#
# `mask_tool_outputs_node` returns the WHOLE message list as its update, and the
# checkpointer restores the previous runs' messages into it — so a hand-off run
# streamed every earlier tool call again. Live, run 2 of a case re-printed run
# 1's push, diagnosis and history verbatim, and re-printed
# "created memory 22463f71-..." with the identical id, which is what gave it
# away: the agent had not called anything twice, the HUD had shown it twice.
# It made a correct agent look like it was looping, and made a guard that had
# refused a repeated push look like it had never fired.
#
# Per-run dedup cannot fix this — the duplicates arrive in a later run. Keyed by
# tool_call id, which is unique per real call, so a genuine repeat still shows.
_emitted_tool_events: dict[str, set[str]] = {}
_EMITTED_CASES_MAX = 500


from recovery_agent import presence


@socketio.on("watch_payment")
def _on_watch_payment(data):
    """A checkout page announcing itself. See `presence` for why this matters."""
    presence.watch(request.sid, str((data or {}).get("payment_id") or ""))


@socketio.on("disconnect")
def _on_disconnect(*_a, **_k):
    presence.forget(request.sid)


def push_event(payment_id: str, event_type: str, data: dict):
    payload = {"payment_id": payment_id, "event": event_type, "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"), **data}
    socketio.emit("agent_event", payload)
    socketio.emit("agent_stream", payload)


TIER_COLORS = {
    "silent": "#3b82f6",
    "active": "#f59e0b",
    "hard_decline_blocked": "#ef4444",
}
TIER_BADGES = {
    "silent": "SILENT RECOVERY",
    "active": "ACTIVE RECOVERY",
    "hard_decline_blocked": "HARD DECLINE BLOCKED",
}

# Decline code → strategy description mapping (for telemetry card)
DECLINE_STRATEGY_DISPLAY = {
    "insufficient_funds": "Payday Timing Scheduler",
    "card_expired": "Card Update Flow",
    "network_timeout": "Metadata Enrichment + Retry",
    "bank_declined": "Multi-Rail Failover",
    "mandate_revoked": "Re-Auth Notification",
    "risk_block": "Human Escalation",
    "card_declined": "Network Penalty Prevention",
    "unknown": "LLM Diagnostic Routing",
}

# Network fine rates (Visa/MC per-attempt penalty)
NETWORK_FINE_PER_ATTEMPT = 0.10  # USD


def push_tier_event(
    payment_id: str,
    tier: str,
    penalties_prevented: int = 0,
    decline_strategy: str = "",
    payday_target_date: str = "",
):
    """Broadcast tier badge, penalty counter, and strategy info to merchant dashboard."""
    tier_badge = TIER_BADGES.get(tier, "ACTIVE RECOVERY")
    tier_color = TIER_COLORS.get(tier, "#f59e0b")
    socketio.emit("tier_update", {
        "payment_id": payment_id,
        "tier": tier,
        "tier_badge": tier_badge,
        "tier_color": tier_color,
        "penalties_prevented": penalties_prevented,
        "penalties_value": f"${penalties_prevented * NETWORK_FINE_PER_ATTEMPT:.2f}",
        "penalties_value_inr": f"INR {penalties_prevented * 8.30:.2f}",
        "decline_strategy": decline_strategy,
        "payday_target_date": payday_target_date,
        "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
    })


def push_decline_strategy_event(payment_id: str, failure_code: str, strategy: str, tier: str):
    """Emit decline-code strategy routing update for the telemetry card."""
    socketio.emit("decline_strategy", {
        "payment_id": payment_id,
        "failure_code": failure_code,
        "strategy": strategy,
        "tier": tier,
        "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
    })


def format_reasoning_newlines(text: str) -> str:
    """Format diagnostic & strategic reasoning strings into clean multiline strings."""
    if not text:
        return ""
    formatted = str(text).strip()

    # Remove any existing ↳ symbols or awkward extra whitespace
    formatted = formatted.replace("↳", "").strip()

    # Format Diagnostic Reflection Steps (Step 1, Step 2, Step 3, Step 4)
    for i in range(1, 5):
        formatted = formatted.replace(f" Step {i} — ", f"\n  • Step {i} — ")
        formatted = formatted.replace(f"Step {i} — ", f"\n  • Step {i} — ")
        formatted = formatted.replace(f" Step {i}: ", f"\n  • Step {i}: ")
        formatted = formatted.replace(f"Step {i}: ", f"\n  • Step {i}: ")

    # Format Strategy Planner Points (1., 2., 3., 4., 5.)
    for i in range(1, 6):
        formatted = formatted.replace(f" {i}. ", f"\n  • {i}. ")
        formatted = formatted.replace(f"{i}. ", f"\n  • {i}. ")

    # Clean up any empty or whitespace-only lines
    lines = [line.rstrip() for line in formatted.splitlines() if line.strip()]
    return "\n".join(lines)


def run_agent_for_payment(payment_id: str, amount: float, failure_reason: str, customer: dict, scenario_type: str = "standard", failure_code: str = "", error_source: str = "", error_step: str = "", queue_if_busy: bool = False):
    """Run agent step by step with live streaming thoughts, tool cards, guardrails, and LLM-generated UI morphing."""
    # Two agent runs on one case at the same time would interleave on the same
    # thread_id, so only one may hold a payment. But *why* a second run arrived
    # matters, and this used to ignore the difference.
    #
    # A repeat of the same trigger is noise — drop it. A hand-off is a new fact
    # about the case, and dropping it loses the customer's action outright. A
    # customer who dismissed the first notification 5.7s after it appeared —
    # while the first run was still writing its summary — got nothing at all,
    # because the hand-off was discarded and its `handoff_` flag stayed set, so
    # nothing would ever retry it. Dismiss the same notification two minutes
    # later and the ladder worked. The bug was a race, not a rung.
    if payment_id in active_agent_payments:
        if not queue_if_busy:
            print(f"[Frontend] Agent thread already active for {payment_id}. Skipping duplicate trigger.")
            return
        waited = 0
        while payment_id in active_agent_payments and waited < 180:
            socketio.sleep(2)
            waited += 2
        if payment_id in active_agent_payments:
            # Never silently swallow it: clear the flag so a later signal — the
            # push timeout, a payment, the next hand-off — can still act.
            for k in [k for k in (store.get_payment(payment_id) or {})
                      if k.startswith(f"handoff_{scenario_type}")]:
                store.update_payment(payment_id, **{k: False})
            store.flush()
            push_event(payment_id, "handoff_dropped", {
                "detail": f"Previous run still active after {waited}s; "
                          f"'{scenario_type}' hand-off released for retry.",
            })
            return
        print(f"[Frontend] Queued '{scenario_type}' hand-off for {payment_id} ran after {waited}s.")
    active_agent_payments.add(payment_id)

    try:
        _run_agent_for_payment_inner(payment_id, amount, failure_reason, customer, scenario_type, failure_code, error_source, error_step)
    except Exception as e:
        # Bug #3: Surface errors to the WebSocket trail instead of silently crashing
        print(f"[Frontend] Agent execution error for {payment_id}: {e}")
        try:
            push_event(payment_id, "error", {
                "step": "error",
                "msg": f"Agent Execution Error: {str(e)}",
                "detail": traceback.format_exc(),
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            })
        except Exception:
            pass
        if store.has_payment(payment_id):
            p = store.get_payment(payment_id)
            p["status"] = "failed"
            p.setdefault("trail", []).append({
                "step": "error",
                "msg": f"Agent Execution Error: {str(e)}",
                "detail": traceback.format_exc(),
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            })
            store.flush()
    finally:
        active_agent_payments.discard(payment_id)


def _parse_tool_result(content) -> dict:
    """Parse a ToolMessage body into a dict.

    `mask_tool_outputs_node` prepends a "[TOOL ERROR] ..." line to failures, so a
    plain `startswith("{")` check treated every failed tool result as unparseable
    raw text and the reason was lost.
    """
    text = str(content or "")
    start = text.find("{")
    if start >= 0:
        try:
            return json.loads(text[start:])
        except (ValueError, TypeError):
            pass
    return {"raw": text}


def _select_primary_action(tool_calls_made: dict) -> tuple[str, dict, bool]:
    """Pick the run's primary action, its receipt, and whether a ticket exists.

    Choose by meaning, not by call order: this used to walk the tool names in
    reverse and take the first match, so a run that created a link and *then*
    sent a notification picked the notification's result — which has no
    link_url — and a customer who paid was never noticed.

    Every result is read through the same success filter. `escalate_res` used to
    be taken raw, so a *blocked* escalation — `{"status": "blocked"}`, the ladder
    refusing a premature ticket — was truthy, and the run was recorded as
    escalated: no ticket existed, `agent_already_escalated` suppressed the
    safety net, and `_handoff_to_agent` refused to reopen an "escalated" case.
    The case died silently, which is the one outcome the ladder exists to
    prevent. A refusal is not an action.

    Returns (action_val, sdk_res, agent_escalated) where `agent_escalated` is
    True only when escalate_to_human actually filed a ticket — the flag the
    safety-net enqueue uses to avoid double-ticketing.
    """
    def _ok(name, statuses=("ok", "scheduled", "delivered")):
        r = (tool_calls_made.get(name) or {}).get("result") or {}
        return r if isinstance(r, dict) and r.get("status") in statuses else {}

    push_res = _ok("send_page_push")
    link_res = _ok("generate_recovery_payment_link")
    retry_res = _ok("retry_in_hours")
    notify_res = _ok("send_recovery_notification")
    escalate_res = _ok("escalate_to_human", statuses=("escalated",))
    close_res = _ok("close_case", statuses=("closed",))
    agent_escalated = bool(escalate_res)

    if close_res:
        # The run that closes a case reported "Primary action: none", because
        # closing was not an action anything recognised. It outranks the rest.
        return "close_case", close_res, agent_escalated
    if escalate_res:
        return "escalate_to_human", escalate_res, agent_escalated
    if push_res and not (link_res or notify_res):
        # A push on its own is the silent first rung. The case is waiting on
        # the customer's response, not finished — treating it as "no action"
        # marked the case escalated, which then blocked the follow-up hand-off
        # when they dismissed it.
        return "page_push", push_res, agent_escalated
    if link_res or notify_res:
        # The link is what the customer acts on, so it is the receipt even
        # when a notification was the last thing sent.
        return "send_notification", (link_res or notify_res), agent_escalated
    if retry_res:
        return "wait_and_retry", retry_res, agent_escalated
    # No recovery action was taken this turn. That is NOT the same as giving
    # up: the closing run after a successful payment calls only manage_memory,
    # and this branch used to read that as "escalate_to_human" — so a case that
    # had just recovered INR 47,481 was filed as a ticket asking a human to
    # chase the customer for non-payment. A fallback should be the least
    # consequential option available, never the most severe one.
    return "none", {}, agent_escalated


def _run_agent_for_payment_inner(payment_id: str, amount: float, failure_reason: str, customer: dict, scenario_type: str = "standard", failure_code: str = "", error_source: str = "", error_step: str = ""):
    """Inner agent execution — wires real LangGraph ReAct agent, Razorpay SDK, and retry scheduler.

    The agent uses LLM reasoning to decide which tools to call at each step.
    """
    import json
    import uuid
    from recovery_agent.agent.guardrails import GuardrailEngine
    from recovery_agent.agent.memory import CustomerMemoryStore


    from recovery_agent.models import Case, CaseStatus, PaymentEvent
    from recovery_agent.retry_scheduler import get_retry_windows, get_next_retry_time

    tracer = _get_tracer()
    parent_attrs = {
        "payment_id": payment_id,
        "amount": amount,
        "failure_reason": failure_reason,
        "failure_code": failure_code,
        "error_source": error_source,
        "error_step": error_step,
        "customer_email": customer.get("email", ""),
        "customer_id": customer.get("id", ""),
        "customer_name": customer.get("name", ""),
        "scenario_type": scenario_type,
        "currency": "INR",
    }

    # One case = one Phoenix session. The context manager stamps
    # session.id onto every instrumented child span (each LLM turn, each
    # tool call), and the root span carries it too — so Phoenix groups all
    # of this case's runs together and rolls tokens and cost up per case.
    from recovery_agent.observability import case_session, session_id_for
    parent_attrs["session.id"] = session_id_for(payment_id)

    from recovery_agent.observability import (KIND_AGENT, KIND_CHAIN,
                                              traced_span)

    with case_session(payment_id, failure_code=failure_code,
                      scenario=scenario_type, amount=amount), \
         traced_span("agent_recovery", kind=KIND_AGENT, tracer=tracer,
                     attributes=parent_attrs) as parent_span:
      # The shared persistent store — a profile learned on one run must exist
      # on the next, or channel win-rates and contact caps are theatre.
      memory_store = CustomerMemoryStore.live()
      guardrail_engine = GuardrailEngine()

      customer_email = customer.get("email", "")
      cust_profile = memory_store.get_or_create_profile(customer_email)

      # Persist who the customer is. The dict is handed to this run as an
      # argument and was never stored, so any later hand-off (push dismissed,
      # payment timeout) rebuilt an empty one and the agent invented a
      # placeholder — the follow-up email with the discount was addressed to
      # "customer@email.com" and reached nobody.
      if customer and store.has_payment(payment_id):
          store.update_payment(
              payment_id,
              customer={k: v for k, v in customer.items() if v},
              customer_email=customer_email,
              customer_name=customer.get("name", ""),
              customer_phone=customer.get("contact") or customer.get("phone", ""),
          )
          store.flush()

      # A case is a story told across several runs — push, then email, then the
      # confirmation that the money arrived. Starting this empty made every
      # hand-off erase everything before it: the run that closes a recovery
      # deleted the `recovery_confirmed` entry that started it, and the HUD
      # showed only the last rung of a ladder the agent had climbed.
      trail = list((store.get_payment(payment_id) or {}).get("trail") or []) \
          if store.has_payment(payment_id) else []

      def emit_thought(step: str, thought: str, detail: str = "", tool_call: dict = None, guardrail: dict = None, memory: dict = None, case_state: dict = None):
          entry = {
              "step": step,
              "msg": thought,
              "detail": detail,
              "tool_call": tool_call,
              "guardrail": guardrail,
              "memory": memory,
              "case_state": case_state,
              "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
          }
          trail.append(entry)
          store.set_trail(payment_id, trail)
          if store.has_payment(payment_id):
              store.get_payment(payment_id)["trail"] = list(trail)
          push_event(payment_id, step, entry)

      # ── INITIALIZE CASE ──
      with traced_span("init_case", kind=KIND_CHAIN, tracer=tracer) as init_span:
        raw_reason = failure_reason or "Payment failed during checkout"
        norm = {}
        try:
            from recovery_agent.razorpay_knowledge_base import normalize_razorpay_failure
            norm = normalize_razorpay_failure(raw_reason)
        except Exception:
            pass

        event = PaymentEvent(
            payment_id=payment_id,
            customer_id=customer_email,
            amount=amount,
            currency="INR",
            failure_code=failure_code or norm.get("failure_code", "payment_failed"),
            failure_reason=raw_reason,
            metadata={
                "customer_name": customer.get("name", ""),
                "scenario": scenario_type,
                "error_code": failure_code or norm.get("error_code", "BAD_REQUEST_ERROR"),
                "error_source": error_source or norm.get("error_source", "gateway"),
                "error_step": error_step or norm.get("error_step", "payment_authorization"),
                "error_description": raw_reason,
                "recommended_rail": norm.get("recommended_rail", "payment_link"),
                "customer_profile": cust_profile.model_dump(),
                # What the customer already did, so the agent reasons with it and
                # any escalation ticket carries it.
                **({"push_outcome": (store.get_payment(payment_id) or {}).get("push_outcome")}
                   if store.has_payment(payment_id)
                   and (store.get_payment(payment_id) or {}).get("push_outcome") else {}),
                "customer_email": customer.get("email", ""),
                "customer_name": customer.get("name", ""),
                "customer_phone": customer.get("contact") or customer.get("phone", ""),
            },
        )
        case = Case(payment=event, max_attempts=3)

        # Carry the case's real state into the fresh Case object. A new Case is
        # built on every run and defaults to OPEN, so a recovered case arrived at
        # the agent looking brand new — and the terminal-tool guard, which is what
        # stops it re-contacting a customer who has already paid, never fired.
        _prior = store.get_payment(payment_id) if store.has_payment(payment_id) else {}
        _status_map = {
            "recovered": CaseStatus.RECOVERED,
            "escalated": CaseStatus.ESCALATED,
            "failed": CaseStatus.STOPPED,
            "stopped": CaseStatus.STOPPED,
        }
        if _prior.get("status") in _status_map:
            case.status = _status_map[_prior["status"]]
        if _prior.get("status") == "recovered" or float(_prior.get("recovered_amount") or 0) > 0:
            case.recovered = True
            case.recovered_amount = float(_prior.get("recovered_amount") or 0)

        if scenario_type == "followup":
            # The customer has already been contacted, so this case is past the
            # silent tier by definition — and the active tier is what unlocks the
            # customer-facing channels a follow-up needs.
            from recovery_agent.models import RecoveryTier
            case.recovery_tier = RecoveryTier.ACTIVE
        init_span.set_attribute("failure_code", case.payment.failure_code)
        init_span.set_attribute("amount", amount)

        emit_thought(
            step="init",
            thought=f"Case Initialized: {payment_id}",
            detail=f"Amount: INR {amount:,.2f} | Reason: {raw_reason} | Customer: {customer_email}",
            memory={
                "customer_id": customer_email,
                "payday_window": cust_profile.salary_window.is_salary_due,
                "promise_to_pay": cust_profile.promises[0].promised_date if cust_profile.promises else None,
            },
        )

      # ── RUN LANGGRAPH REACT AGENT ──
      with traced_span("graph_execution", kind=KIND_CHAIN,
                       tracer=tracer) as graph_span:
        from recovery_agent.agent import RecoveryAgent
        from recovery_agent.agent.graph import build_initial_state, RecoveryContext
        
        agent = RecoveryAgent(guardrail_engine=guardrail_engine)
        
        initial_state = build_initial_state(case)

        # Pass emit_thought as callback for real-time graph progress
        def graph_emit(step, msg, detail=""):
            emit_thought(step=step, thought=msg, detail=detail)

        context = RecoveryContext(
            guardrail_engine=agent.guardrails,
            case=case,
        )

        from recovery_agent.agent.tools import ns_safe
        config = {
            "configurable": {
                # One session per PAYMENT, not per run.
                #
                # This was `case.id`, and a fresh Case (new uuid) is built on
                # every invocation — so a single payment ran as several unrelated
                # sessions. pay_fw1l0oppo had three: the first attempt, the
                # push-dismissed follow-up, and the closing run after payment.
                # Each began with an empty thread and knew only what the frontend
                # re-injected, like reopening the same file as a blank buffer.
                #
                # Keying on payment_id gives each payment one continuous session
                # and guarantees two payments can never share one.
                "thread_id": f"case:{payment_id}",
                "payment_id": payment_id,
                # Namespace label for langmem — periods are rejected.
                "customer_id": ns_safe(customer_email),
            },
        }

        emit_thought(
            step="agent_start",
            thought="Launching ReAct Agent — LLM will reason, call tools, and adapt",
            detail=f"Loop: agent → tools → agent → ... → END",
        )

        _run_started = time.monotonic()

        try:
            # Only THIS run's messages count.
            #
            # Nodes return the whole message list and the checkpointer restores
            # earlier runs into it, so `all_messages` was the entire case
            # history. Two things came out wrong on every closing run: the
            # summary shown was the previous run's ("Waiting on the customer"
            # printed after the money had already arrived), and the primary
            # action was the previous run's too — a run that called only
            # manage_memory reported "Agent executed: wait_for_customer,
            # Primary action: page_push".
            prior_ids: set[str] = set()
            try:
                snap = agent.graph.get_state(config)
                prior_ids = {m.id for m in (snap.values or {}).get("messages", [])
                             if getattr(m, "id", None)}
            except Exception:
                pass

            all_messages = []
            seen_events = set()
            if len(_emitted_tool_events) > _EMITTED_CASES_MAX:
                _emitted_tool_events.clear()
            shown = _emitted_tool_events.setdefault(payment_id, set())
            current_phase = "initializing"
            for s in agent.graph.stream(initial_state, config=config, context=context, stream_mode="updates"):
                if not s:
                    continue
                for node_name, state_update in s.items():
                    if not state_update:
                        continue
                    # Track phase from graph state updates
                    new_phase = state_update.get("phase")
                    if new_phase and new_phase != current_phase:
                        current_phase = new_phase
                        push_event(payment_id, "phase_update", {
                            "phase": current_phase,
                            "node": node_name,
                        })
                    msgs = [m for m in state_update.get("messages", [])
                            if getattr(m, "id", None) not in prior_ids]
                    all_messages.extend(msgs)
                    for msg in msgs:
                        # Only emit for AIMessage and ToolMessage — skip Human/System
                        if not isinstance(msg, (AIMessage, ToolMessage)):
                            continue
                        # Dedup by content signature to handle copies from mask_outputs
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                sig = f"thinking:{tc.get('id') or tc['name']}"
                                if sig in shown:
                                    continue
                                shown.add(sig)
                                emit_thought(
                                    step="agent_thinking",
                                    thought=f"[AGENT] Calling tool: {tc['name']}",
                                    detail=f"Args: {json.dumps(tc['args'], default=str)[:200]}",
                                )
                        elif isinstance(msg, ToolMessage):
                            sig = f"result:{msg.tool_call_id}"
                            if sig in shown:
                                continue
                            shown.add(sig)
                            content = msg.content[:300] if msg.content else "(empty)"
                            emit_thought(
                                step="tool_result",
                                thought=f"[TOOL] {msg.name}: {content}",
                                detail="",
                            )
                        # LLM final response (AIMessage with content but NO tool_calls)
                        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                            # One summary per case. The graph produces a final
                            # response and then the self-critique node produces
                            # another, and both arrived as "agent_summary" — so
                            # every case showed two conclusions, often phrased
                            # differently, with no way to tell which was current.
                            sig = f"summary:{msg.content[:100]}"
                            if seen_events.intersection({"summary_emitted"}):
                                continue
                            seen_events.add("summary_emitted")
                            if sig not in seen_events:
                                seen_events.add(sig)
                                emit_thought(
                                    step="agent_summary",
                                    thought=f"[AGENT] {msg.content[:300]}",
                                    detail="",
                                )
            from recovery_agent.economics import record_run_wall
            record_run_wall(payment_id, time.monotonic() - _run_started)

            # If the model that answered was not the model we asked for, say
            # so on the trail. 18 of 18 calls across four live cases were
            # served by a flash-LITE model after the primary hit its quota,
            # and nothing anywhere said so — the operator was left judging the
            # agent's competence without knowing what was doing the thinking.
            for note in ((store.get_payment(payment_id) or {}).get(
                    "model_degraded") or []):
                key = f"degraded:{note.get('served')}"
                if key in shown:
                    continue
                shown.add(key)
                emit_thought(
                    step="model_degraded",
                    thought=f"[MODEL] Reasoning on {note.get('served')} — "
                            f"{note.get('wanted')} was unavailable",
                    detail="The primary model refused this turn (quota or "
                           "outage) and the run continued on a fallback. "
                           "Decisions below were made by the smaller model.",
                )
        except Exception as e:
            # Graph error or LLM unavailable — surface the real error
            try:
                from recovery_agent.economics import record_run_wall
                record_run_wall(payment_id, time.monotonic() - _run_started)
            except Exception:
                pass
            print(f"[Frontend] Graph stream error for {payment_id}: {type(e).__name__}: {e}", flush=True)
            import traceback as _tb; _tb.print_exc()
            # Say what actually happened. There is no bandit fallback here and
            # never was — the run is simply over. Claiming "using empirical data
            # for decisions" described a recovery that was not being attempted.
            settled = (store.get_payment(payment_id) or {}).get("status") \
                if store.has_payment(payment_id) else None
            terminal = settled in ("recovered", "escalated")
            emit_thought(
                step="llm_unavailable",
                thought=("AGENT RUN FAILED — the case keeps its settled outcome"
                         if terminal else
                         "AGENT RUN FAILED — no recovery action was taken"),
                detail=f"Error: {e}",
            )
            # A failed run must NOT overwrite a settled case. `pay_f6qrbnlez` was
            # recovered for real — INR 2,374.05 captured — and the CLOSING run,
            # whose only job was to write a memory, hit an LLM 404 and rewrote
            # the case to `failed`. A recovery became a recorded loss because
            # the agent could not talk about it afterwards.
            fields = dict(strategy_reasoning=str(e), strategy_source="graph_error",
                          last_error=str(e)[:300],
                          ts=datetime.now(timezone.utc).strftime("%H:%M:%S"))
            if not terminal:
                fields.update(status="failed", tier="ERROR")
            store.update_payment(payment_id, **fields)
            _do_flush()
            _emit_status_summary()
            return {
                "status": settled if terminal else "failed",
                "strategy_reasoning": str(e),
                "tier": "SETTLED" if terminal else "ERROR",
            }

      # ── EXTRACT RESULTS FROM REACT AGENT ──
      with traced_span("act", kind=KIND_CHAIN, tracer=tracer) as act_span:
        cause = case.payment.failure_code or "unknown"
        strategy_source = "react_agent"
        recovery_tier = case.payment.metadata.get("recovery_tier", "active")

        # Parse tool calls and results from the agent's message history.
        #
        # Results are matched by tool_call_id. The previous version assigned each
        # result to the first tool that still had `result is None`, in dict
        # insertion order — so with the 3-4 parallel tool calls this agent makes
        # every turn, results landed on the wrong tools. The practical effect was
        # that `generate_recovery_payment_link` carried some other tool's output,
        # `sdk_res["link_url"]` was empty, `_watch_for_recovery` never started,
        # and a customer who paid was never noticed.
        calls_by_id: dict[str, dict] = {}
        order: list[str] = []
        for msg in all_messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["id"] not in calls_by_id:
                        calls_by_id[tc["id"]] = {"name": tc["name"], "args": tc["args"],
                                                 "result": None}
                        order.append(tc["id"])
            elif isinstance(msg, ToolMessage):
                entry = calls_by_id.get(getattr(msg, "tool_call_id", None))
                if entry is not None:
                    entry["result"] = _parse_tool_result(msg.content)

        # Downstream wants name -> {args, result}; keep the latest call per name,
        # preferring one that actually produced a result.
        tool_calls_made = {}
        for cid in order:
            entry = calls_by_id[cid]
            prev = tool_calls_made.get(entry["name"])
            if prev is None or entry["result"] is not None:
                tool_calls_made[entry["name"]] = {"args": entry["args"],
                                                  "result": entry["result"]}

        # Determine the primary action, and the receipt that represents it.
        # The selection rules (and the incidents behind them) live in
        # _select_primary_action, which is pure so the tests can exercise it.
        action_val, sdk_res, agent_already_escalated = \
            _select_primary_action(tool_calls_made)

        # Name the tool that DID the thing, not whichever ran last.
        #
        # This took the last key in call order, so a run whose agent closed the
        # case reported "Agent executed: manage_memory" — the self-critique
        # node's own reflection write, which is not something the agent did at
        # all. The line sat directly above "Primary action: close_case" and
        # contradicted it.
        _ACTION_TOOL = {
            "close_case": "close_case",
            "escalate_to_human": "escalate_to_human",
            "page_push": "send_page_push",
            "send_notification": "send_recovery_notification",
            "wait_and_retry": "retry_in_hours",
        }
        tool_name_str = (_ACTION_TOOL.get(action_val)
                         or (list(tool_calls_made.keys())[-1] if tool_calls_made
                             else "agent"))
        act_span.set_attribute("action", action_val)
        act_span.set_attribute("strategy_source", strategy_source)

        # ── WAIT_AND_RETRY: Register background job ──
        if action_val == "wait_and_retry" and sdk_res.get("status") == "scheduled":
            from recovery_agent.models import FailureType
            from recovery_agent.daemon_worker import register_retry_job
            try:
                ft = FailureType(cause)
            except ValueError:
                ft = FailureType.NETWORK_TIMEOUT
            windows = get_retry_windows(ft, case.attempt_count, amount, customer_email)
            next_time = get_next_retry_time(ft, case.attempt_count)
            best_window = windows[0] if windows else None

            # `retry_in_hours` returns `target_time`. Reading only
            # `target_timestamp` meant that key was always missing, so the
            # agent's decision was silently replaced by the scheduler's own
            # guess: it asked for 24 hours and the daemon registered a job three
            # minutes out. Honour what the agent chose; fall back only if it
            # gave no time at all.
            target_ts = (sdk_res.get("target_timestamp")
                         or sdk_res.get("target_time")
                         or (next_time or datetime.now(timezone.utc)).isoformat())
            registered_job = register_retry_job(
                payment_id=payment_id,
                amount=amount,
                target_timestamp=target_ts,
                # The tool already scheduled it; enrich that job, do not add one.
                job_id=sdk_res.get("job_id", ""),
                action="retry_payment",
                method=case.payment.metadata.get("method", "card"),
                customer={"name": customer.get("name", ""), "email": customer_email},
                # The scheduler's canned reason ("Immediate retry — network
                # issues are transient") was attached to a 24-hour job the agent
                # had chosen deliberately, so the record contradicted itself.
                reason=(f"agent scheduled a retry in "
                        f"{sdk_res.get('delay_hours', '?')}h"
                        if sdk_res.get("delay_hours") is not None
                        else (best_window.reason if best_window else "Scheduled retry")),
                confidence=best_window.confidence if best_window else 0.5,
            )
            sdk_res = registered_job
            tool_name_str = "DaemonWorker.register_retry_job"

            if not store.has_payment(payment_id):
                store.save_payment(payment_id, {"payment_id": payment_id, "amount": amount, "status": "scheduled", "trail": [], "attempts": 0})
            p = store.get_payment(payment_id)
            p["scheduled_job"] = registered_job
            p["status"] = "scheduled"

            socketio.emit("scheduled_job", {
                "payment_id": payment_id,
                **registered_job,
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            })

        # ── GENERATE UI SPEC FROM AGENT'S FINAL RESPONSE ──
        # MANDATE 0: No separate LLM call. Parse the agent's own response for UI context.
        agent_final_text = ""
        for msg in reversed(all_messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                agent_final_text = msg.content
                break

        emit_thought(
            step="acting",
            thought=f"Agent executed: {tool_name_str}",
            detail=f"Primary action: {action_val}",
            tool_call={
                "tool": tool_name_str,
                "args": {"payment_id": payment_id, "amount_in_paise": int(amount * 100), "currency": "INR", "customer": customer_email},
                "raw_razorpay_response": sdk_res,
            },
            # The facts about the case, stated as facts. These used to travel
            # inside a "GenerativeUISpec" — a dataclass whose docstring promised
            # LLM-generated UI and delivered a dict lookup, a canned Hinglish
            # line nothing read, and `tone="supportive"` on every case.
            case_state={
                "recovery_tier": recovery_tier,
                "decline_strategy": cause,
                "penalties_prevented": case.penalties_prevented,
                "scheduled_job": sdk_res if action_val == "wait_and_retry" else None,
            },
        )


        # ── OBSERVE & RECOVER ──
        if action_val == "wait_and_retry":
            emit_thought(
                step="stopping",
                thought=f"Background Retry Scheduled: {sdk_res.get('job_id', 'N/A')}",
                detail=f"Target: {sdk_res.get('target_timestamp', 'N/A')} | Confidence: {sdk_res.get('confidence', 0):.0%} | Reason: {sdk_res.get('reason', '')}",
            )
        elif action_val == "close_case":
            emit_thought(
                step="stopping",
                thought=f"Case closed: {sdk_res.get('outcome', 'unknown').upper()}"
                        + (f" — INR {float(sdk_res.get('amount_recovered') or 0):,.2f} recovered"
                           if sdk_res.get("outcome") == "recovered" else ""),
                detail=(tool_calls_made.get("close_case", {}).get("args", {})
                        .get("what_happened", "")),
            )
        elif action_val == "escalate_to_human":
            emit_thought(
                step="stopping",
                thought=f"Escalated to Human: {sdk_res.get('ticket_id', 'N/A')}",
                detail=f"Reason: {sdk_res.get('reason', failure_reason)}",
            )
        elif action_val == "send_notification" and sdk_res.get("link_url"):
            store.save_pending(payment_id, {
                "action": action_val,
                "execution": sdk_res,
                "attempt": 0,
                "trail": trail,
                "amount": amount,
            })
            push_event(payment_id, "waiting_for_customer", {"action": action_val, "detail": f"Payment link sent: {sdk_res['link_url']}"})
            socketio.start_background_task(_watch_for_recovery, payment_id, 300)

        # Determine final status from Case.status (graph is source of truth)
        # Fallback to tool-name mapping if Case.status not yet set
        from recovery_agent.models import CaseStatus
        case_status = case.status
        if case_status == CaseStatus.RECOVERED:
            final_status = "recovered"
        elif case_status == CaseStatus.ESCALATED:
            final_status = "escalated"
        elif case_status == CaseStatus.STOPPED:
            # STOPPED means this RUN is over, not that the case is lost. When the
            # agent's last act was to schedule a retry, the money is still being
            # worked on — reading that as `failed` wrote off a live case, and
            # made the dashboard count a pending recovery as a loss.
            final_status = "scheduled" if action_val == "wait_and_retry" else "failed"
        elif case_status == CaseStatus.AWAITING_CUSTOMER:
            final_status = "awaiting_customer"
        else:
            # Fallback: derive from action_val for granular statuses
            if action_val == "none":
                # Nothing happened to the case this turn; keep what it already had.
                final_status = (store.get_payment(payment_id) or {}).get(
                    "status", "recovering") if store.has_payment(payment_id) else "recovering"
            elif action_val == "page_push":
                final_status = "awaiting_customer"
            elif action_val == "wait_and_retry":
                final_status = "scheduled"
            elif action_val == "send_notification":
                final_status = "awaiting_customer"
            elif action_val == "escalate_to_human":
                final_status = "escalated"
            else:
                final_status = "completed"

        if not store.has_payment(payment_id):
            store.save_payment(payment_id, {"payment_id": payment_id, "amount": amount, "status": final_status, "trail": [], "attempts": 0})
        p = store.get_payment(payment_id)
        p["attempts"] = p.get("attempts", 0) + 1
        p["last_action"] = action_val
        p["last_detail"] = sdk_res.get("message", "")
        p["trail"] = trail
        if sdk_res.get("link_id"):
            p["recovery_link_id"] = sdk_res["link_id"]
        p["order_id"] = sdk_res.get("link_id", "") or p.get("order_id", "")
        # Never downgrade a recovered case. The closing run calls only
        # manage_memory, so the action selector would fall through to its
        # escalate default and rewrite a paid case as "escalated" — losing a
        # recovery that had already been confirmed against Razorpay.
        if p.get("status") == "recovered" and final_status != "recovered":
            print(f"[Frontend] keeping {payment_id} as recovered "
                  f"(closing run reported {final_status})", flush=True)
        else:
            p["status"] = final_status
        p["recovery_tier"] = recovery_tier
        p["decline_strategy"] = cause
        p["penalties_prevented"] = case.penalties_prevented

        # Anything that finishes without the money coming back is a person's
        # problem now. Relying on the agent to always call escalate_to_human
        # would let a case that simply ran out of attempts end in silence.
        # Gate on what actually happened, not on a status string. `final_status`
        # is derived from which tools ran, so a bookkeeping turn could label a
        # paid case "escalated" and put it in front of a human. Money in the bank
        # ends the case, whatever the label says.
        already_recovered = (
            (store.get_payment(payment_id) or {}).get("status") == "recovered"
            or float((store.get_payment(payment_id) or {}).get("recovered_amount") or 0) > 0
            or case.recovered
        ) if store.has_payment(payment_id) else case.recovered

        # `agent_already_escalated` (set by _select_primary_action) is True only
        # when escalate_to_human actually filed a ticket with the full case
        # context. This safety net exists for cases that simply ran out of road,
        # not to second-guess the agent — firing both produced two tickets for
        # one payment 29 seconds apart. A BLOCKED escalation does not count: no
        # ticket exists, so suppressing the net here would let the case die
        # silently.

        # The net must obey the same rule the agent does: escalation is the LAST
        # rung. This fired on any run that ended `failed` or `stopped`,
        # whichever rung the case was on — so it filed a ticket for a case whose
        # agent had just scheduled a 24h retry, walking straight around the gate
        # that exists to stop exactly that.
        #
        # When rungs remain, the case has not run out of road; it has run out of
        # THIS RUN. Hand it back to the agent to climb the next one, because
        # nothing else will restart it and the alternative is silence.
        from recovery_agent.agent import ladder as _ladder_mod
        _rec = store.get_payment(payment_id) or {}
        _rungs_left = (not _ladder_mod.exhausted(_rec)
                       and not _ladder_mod.pursuit_barred(_rec))

        if (final_status in ("escalated", "failed", "stopped")
                and not already_recovered and not agent_already_escalated
                and _rungs_left):
            nxt = _ladder_mod.next_rung(_rec)
            _handoff_to_agent(
                payment_id,
                f"That run ended without the money coming back, and the recovery "
                f"ladder is not finished — {nxt['rung']} has not been tried. "
                f"Already climbed: {', '.join(_ladder_mod.state(_rec)['climbed']) or 'nothing'}. "
                f"Already tried: {', '.join(_ladder_mod.actions_tried(_rec)) or 'nothing'}. "
                f"Do this next: {nxt['what']}. Do not repeat anything above.",
                scenario=f"ladder_{nxt['rung']}",
            )

        elif (final_status in ("escalated", "failed", "stopped")
                and not already_recovered and not agent_already_escalated):
            try:
                from recovery_agent.escalation_queue import enqueue
                enqueue(
                    payment_id=payment_id,
                    reason=(case.payment.metadata.get("agent_summary")
                            or f"recovery ended as {final_status} without payment")[:400],
                    amount=float(amount or 0),
                    customer={"email": customer_email,
                              "name": customer.get("name", ""),
                              "contact": customer.get("contact") or customer.get("phone", "")},
                    attempts=[{"action": getattr(a.action_type, "value", ""),
                               "result": a.result} for a in case.attempts],
                    customer_signals=[
                        f"in-page notification {o['action']} after {o.get('seconds_shown','?')}s"
                        for o in [(store.get_payment(payment_id) or {}).get("push_outcome") or {}]
                        if o.get("action")
                    ] + [
                        f"ladder climbed: "
                        f"{', '.join(_ladder_mod.state(_rec)['climbed']) or 'none'}"
                    ] + [
                        f"never tried — {r['rung']}: {r.get('why_not', 'n/a')}"
                        for r in _ladder_mod.state(_rec)["unavailable"]
                    ],
                    offer=case.payment.metadata.get("page_offer") or {},
                    recovery_link=case.payment.metadata.get("recovery_link", ""),
                    failure_code=case.payment.failure_code or "",
                    source="ladder_exhausted",
                )
            except Exception as exc:
                print(f"[Frontend] escalation enqueue failed for {payment_id}: {exc}", flush=True)

        push_tier_event(
            payment_id,
            tier=recovery_tier,
            penalties_prevented=case.penalties_prevented,
            decline_strategy=cause,
            payday_target_date="",
        )

        push_event(payment_id, "complete", {
            "status": final_status,
            "attempts": p.get("attempts", 1),
            "trail": trail,
            "amount": amount,
            "recovery_tier": recovery_tier,
            "decline_strategy": cause,
            "penalties_prevented": case.penalties_prevented,
            "order_id": sdk_res.get("link_id", ""),
            "scheduled_job": sdk_res if action_val == "wait_and_retry" else None,
        })
        parent_span.set_attribute("final_status", final_status)
        parent_span.set_attribute("recovery_tier", recovery_tier)
        parent_span.set_attribute("strategy_source", strategy_source)
        parent_span.set_attribute("root_cause", cause)
        parent_span.set_attribute("decided_action", action_val)
        parent_span.set_attribute("silent_attempts", case.silent_attempts)
        parent_span.set_attribute("attempt_count", case.attempt_count)
        store.flush()



# ─── Customer Payment Page ────────────────────────────────────
PAY_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SoulStreet — Online Store</title>
<script src="https://checkout.razorpay.com/v1/checkout.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#f5f5f0;--card:#fff;--text:#1a1a1a;--muted:#6b7280;--accent:#2563eb;--accent-hover:#1d4ed8;--green:#16a34a;--red:#dc2626;--amber:#f59e0b;--border:#e5e5e5;--radius:12px}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg);min-height:100vh;color:var(--text)}
@keyframes fadeIn{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideRight{from{transform:translateX(100%)}to{transform:translateX(0)}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
@keyframes pulseGlow{0%,100%{box-shadow:0 0 0 0 rgba(37,99,235,0)}50%{box-shadow:0 0 20px 4px rgba(37,99,235,0.15)}}

/* ── Header ── */
.header{background:var(--card);border-bottom:1px solid var(--border);padding:0 24px;height:60px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}
.header-left{display:flex;align-items:center;gap:24px}
.logo{font-size:22px;font-weight:800;letter-spacing:-0.03em;color:var(--text)}
.logo span{color:var(--accent)}
.nav-links{display:flex;gap:20px}
.nav-links a{font-size:13px;font-weight:600;color:var(--muted);text-decoration:none;text-transform:uppercase;letter-spacing:0.05em;transition:color .15s}
.nav-links a:hover,.nav-links a.active{color:var(--text)}
.header-right{display:flex;align-items:center;gap:16px}
.search-box{display:flex;align-items:center;gap:8px;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:8px 14px;width:220px}
.search-box input{border:none;background:none;outline:none;font-size:13px;color:var(--text);width:100%;font-family:inherit}
.search-box input::placeholder{color:#9ca3af}
.cart-btn{position:relative;background:none;border:none;cursor:pointer;padding:8px;border-radius:8px;transition:background .15s}
.cart-btn:hover{background:var(--bg)}
.cart-btn svg{width:22px;height:22px;color:var(--text)}
.cart-count{position:absolute;top:2px;right:2px;background:var(--accent);color:#fff;font-size:10px;font-weight:700;width:18px;height:18px;border-radius:50%;display:flex;align-items:center;justify-content:center;display:none}

/* ── Store View ── */
.store-view{max-width:1100px;margin:0 auto;padding:24px 20px}
.store-banner{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);border-radius:16px;padding:40px 36px;margin-bottom:32px;color:#fff;position:relative;overflow:hidden}
.store-banner::after{content:'';position:absolute;top:-50%;right:-20%;width:400px;height:400px;background:radial-gradient(circle,rgba(37,99,235,0.15),transparent 70%);pointer-events:none}
.store-banner h1{font-size:28px;font-weight:800;margin-bottom:6px;letter-spacing:-0.03em}
.store-banner p{font-size:14px;color:rgba(255,255,255,0.7);max-width:400px;line-height:1.5}
.section-title{font-size:18px;font-weight:700;margin-bottom:20px;letter-spacing:-0.02em}
.product-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:20px}
.product-card{background:var(--card);border-radius:var(--radius);overflow:hidden;border:1px solid var(--border);transition:all .2s;cursor:pointer}
.product-card:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,0.08);border-color:#d1d5db}
.product-img{width:100%;aspect-ratio:3/4;overflow:hidden;background:#f3f4f6}
.product-img img{width:100%;height:100%;object-fit:cover;transition:transform .3s}
.product-card:hover .product-img img{transform:scale(1.03)}
.product-tag{position:absolute;top:10px;left:10px;background:var(--accent);color:#fff;font-size:10px;font-weight:700;padding:3px 8px;border-radius:4px;text-transform:uppercase;letter-spacing:0.05em}
.product-body{padding:14px 16px}
.product-brand{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px}
.product-name{font-size:14px;font-weight:600;margin-bottom:6px;line-height:1.3}
.product-price{display:flex;align-items:baseline;gap:6px}
.product-price .current{font-size:16px;font-weight:700;color:var(--text)}
.product-price .original{font-size:12px;color:var(--muted);text-decoration:line-through}
.product-price .discount{font-size:11px;color:var(--green);font-weight:600}
.add-btn{width:100%;margin-top:10px;padding:10px;border:1.5px solid var(--text);background:transparent;color:var(--text);border-radius:8px;font-size:12px;font-weight:600;cursor:pointer;transition:all .15s;text-transform:uppercase;letter-spacing:0.04em;font-family:inherit}
.add-btn:hover{background:var(--text);color:#fff}
.add-btn.added{background:var(--green);border-color:var(--green);color:#fff;pointer-events:none}

/* ── Cart Sidebar ── */
.cart-overlay{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.4);z-index:200;opacity:0;pointer-events:none;transition:opacity .25s}
.cart-overlay.open{opacity:1;pointer-events:auto}
.cart-sidebar{position:fixed;top:0;right:0;width:400px;max-width:90vw;height:100vh;background:var(--card);z-index:201;transform:translateX(100%);transition:transform .3s cubic-bezier(.4,0,.2,1);display:flex;flex-direction:column;box-shadow:-8px 0 30px rgba(0,0,0,0.1)}
.cart-sidebar.open{transform:translateX(0)}
.cart-header{padding:20px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.cart-header h2{font-size:18px;font-weight:700}
.cart-close{background:none;border:none;cursor:pointer;padding:6px;border-radius:6px;transition:background .15s;font-size:20px;color:var(--muted)}
.cart-close:hover{background:var(--bg)}
.cart-items{flex:1;overflow-y:auto;padding:16px 24px}
.cart-empty{text-align:center;padding:40px 0;color:var(--muted);font-size:14px}
.cart-item{display:flex;gap:12px;padding:14px 0;border-bottom:1px solid var(--border)}
.cart-item-img{width:64px;height:80px;border-radius:8px;overflow:hidden;background:#f3f4f6;flex-shrink:0}
.cart-item-img img{width:100%;height:100%;object-fit:cover}
.cart-item-info{flex:1;min-width:0}
.cart-item-brand{font-size:10px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em}
.cart-item-name{font-size:13px;font-weight:600;margin:2px 0 4px}
.cart-item-price{font-size:14px;font-weight:700}
.cart-item-qty{display:flex;align-items:center;gap:8px;margin-top:6px}
.qty-btn{width:26px;height:26px;border-radius:6px;border:1px solid var(--border);background:var(--card);cursor:pointer;font-size:14px;display:flex;align-items:center;justify-content:center;transition:all .15s;font-family:inherit}
.qty-btn:hover{background:var(--bg);border-color:#d1d5db}
.qty-val{font-size:13px;font-weight:600;min-width:20px;text-align:center}
.cart-item-remove{background:none;border:none;color:var(--muted);cursor:pointer;font-size:11px;margin-top:4px;padding:2px 0;transition:color .15s}
.cart-item-remove:hover{color:var(--red)}
.cart-footer{padding:20px 24px;border-top:1px solid var(--border);background:var(--card)}
.cart-total{display:flex;justify-content:space-between;margin-bottom:14px}
.cart-total-label{font-size:14px;color:var(--muted)}
.cart-total-val{font-size:20px;font-weight:700}
.checkout-btn{width:100%;padding:14px;background:var(--text);color:#fff;border:none;border-radius:var(--radius);font-size:14px;font-weight:600;cursor:pointer;transition:all .2s;font-family:inherit;letter-spacing:-0.01em}
.checkout-btn:hover{background:#333;transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,0.15)}
.checkout-btn:disabled{background:#d1d5db;cursor:not-allowed;transform:none;box-shadow:none}

/* ── Checkout View ── */
.checkout-view{display:none;max-width:600px;margin:0 auto;padding:24px 20px}
.checkout-view.active{display:block}
.store-view.hidden{display:none}
.back-btn{background:none;border:none;cursor:pointer;font-size:13px;font-weight:600;color:var(--muted);margin-bottom:20px;display:flex;align-items:center;gap:6px;transition:color .15s;font-family:inherit}
.back-btn:hover{color:var(--text)}
.checkout-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:32px;box-shadow:0 4px 16px rgba(0,0,0,0.04)}
.checkout-title{font-size:22px;font-weight:700;margin-bottom:4px;letter-spacing:-0.02em}
.checkout-subtitle{font-size:13px;color:var(--muted);margin-bottom:20px}
.cust-form{display:grid;gap:12px;margin-bottom:20px}
.cust-field{display:flex;flex-direction:column;gap:4px}
.cust-field label{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.04em}
.cust-field input{padding:10px 12px;border:1.5px solid var(--border);border-radius:8px;font-size:14px;font-family:inherit;color:var(--text);background:var(--bg);transition:border-color .15s}
.cust-field input:focus{outline:none;border-color:var(--accent);background:var(--card)}
.cust-field input::placeholder{color:#9ca3af}
.cust-field input.invalid{border-color:var(--red)}
.checkout-items{margin-bottom:20px}
.checkout-item{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #f3f4f6;font-size:13px}
.checkout-item:last-child{border-bottom:none}
.checkout-item-name{color:var(--text);font-weight:500}
.checkout-item-price{font-weight:600}
.checkout-divider{height:1px;background:var(--border);margin:16px 0}
.checkout-row{display:flex;justify-content:space-between;font-size:13px;padding:4px 0}
.checkout-row.total{font-size:18px;font-weight:700;padding-top:12px;border-top:2px solid var(--text);margin-top:8px}
.pay-btn{width:100%;padding:16px;background:linear-gradient(135deg,var(--accent),var(--accent-hover));color:#fff;border:none;border-radius:var(--radius);font-size:15px;font-weight:700;cursor:pointer;transition:all .2s;margin-top:24px;font-family:inherit;letter-spacing:-0.01em}
.pay-btn:hover{transform:translateY(-1px);box-shadow:0 8px 20px rgba(37,99,235,0.3)}
.pay-btn:disabled{background:linear-gradient(135deg,#94a3b8,#64748b);cursor:not-allowed;transform:none;box-shadow:none}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid currentColor;border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite;flex-shrink:0;vertical-align:middle;margin-right:6px}

/* ── Status, Decline ── */
.status-bar{margin-top:16px;padding:14px 16px;border-radius:12px;display:none;font-size:13px;font-weight:500;animation:fadeIn .3s}
.status-bar.active{display:flex;align-items:center;gap:8px}
.s-processing{background:rgba(59,130,246,0.08);color:#2563eb;border:1px solid rgba(59,130,246,0.15)}
.s-waiting{background:rgba(245,158,11,0.08);color:#92400e;border:1px solid rgba(245,158,11,0.15)}
.s-success{background:rgba(16,185,129,0.08);color:#059669;border:1px solid rgba(16,185,129,0.15)}
.s-failed{background:rgba(239,68,68,0.08);color:#dc2626;border:1px solid rgba(239,68,68,0.15)}
.s-recovering{background:rgba(245,158,11,0.08);color:#92400e;border:1px solid rgba(245,158,11,0.15)}

.decline-wall{background:linear-gradient(135deg,rgba(245,158,11,0.06),rgba(245,158,11,0.02));border:1px solid rgba(245,158,11,0.15);border-radius:16px;padding:24px;margin-top:18px;display:none;backdrop-filter:blur(10px)}
.decline-wall.visible{display:block;animation:slideUp .35s ease-out}
.decline-wall h3{color:#92400e;font-size:16px;margin-bottom:8px;font-weight:600;letter-spacing:-0.01em}
.decline-wall p{color:#78716c;font-size:13px;margin-bottom:12px;line-height:1.6}
.decline-wall .reason{background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.12);border-radius:10px;padding:12px;font-size:12px;color:#92400e;margin-bottom:14px;line-height:1.5}
.alt-rails{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap}
.alt-rail-btn{background:rgba(255,255,255,0.8);backdrop-filter:blur(8px);border:1.5px solid #e2e8f0;border-radius:10px;padding:10px 16px;font-size:13px;font-weight:500;color:#475569;cursor:pointer;transition:all .2s;display:flex;align-items:center;gap:6px;font-family:inherit}
.alt-rail-btn:hover{border-color:var(--accent);color:var(--accent);background:rgba(59,130,246,0.06);transform:translateY(-1px)}
.alt-rail-btn.active{border-color:var(--accent);color:var(--accent);background:rgba(59,130,246,0.08)}
.action-box{background:linear-gradient(135deg,rgba(59,130,246,0.06),rgba(59,130,246,0.02));border:1px solid rgba(59,130,246,0.12);border-radius:14px;padding:20px;margin-top:16px;display:none}
.action-box.visible{display:block;animation:slideUp .3s ease-out}
.action-box h4{color:var(--accent);font-size:14px;margin-bottom:6px;font-weight:600}
.action-box p{color:var(--muted);font-size:13px;margin-bottom:14px;line-height:1.5}
.btn-respond{background:linear-gradient(135deg,var(--amber),#d97706);color:#fff;border:none;padding:12px 20px;border-radius:10px;font-size:13px;font-weight:600;cursor:pointer;transition:all .2s;width:100%;font-family:inherit}
.btn-respond:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(245,158,11,0.3)}


@media(max-width:768px){.product-grid{grid-template-columns:repeat(2,1fr);gap:12px}.header{padding:0 16px}.nav-links{display:none}.search-box{width:160px}.store-banner{padding:28px 24px}.store-banner h1{font-size:22px}}
@media(max-width:480px){.product-grid{grid-template-columns:1fr}.cart-sidebar{width:100%}}
</style>
</head>
<body>

</div>

<!-- Header -->
<div class="header">
  <div class="header-left">
    <div class="logo">Soul<span>Street</span></div>
    <div class="nav-links">
      <a href="#" class="active">Men</a>
      <a href="#">Women</a>
      <a href="#">New Arrivals</a>
      <a href="#">Sale</a>
    </div>
  </div>
  <div class="header-right">
    <div class="search-box">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
      <input placeholder="Search products..." />
    </div>
    <button class="cart-btn" onclick="toggleCart()">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 2L3 6v14a2 2 0 002 2h14a2 2 0 002-2V6l-3-4z"/><line x1="3" y1="6" x2="21" y2="6"/><path d="M16 10a4 4 0 01-8 0"/></svg>
      <span class="cart-count" id="cart-count">0</span>
    </button>
  </div>
</div>

<!-- Store View -->
<div class="store-view" id="store-view">
  <div class="store-banner">
    <h1>New Season Collection</h1>
    <p>Premium streetwear crafted for the bold. Free shipping on orders above INR 1,999.</p>
  </div>
  <div class="section-title">Trending Now</div>
  <div class="product-grid" id="product-grid"></div>
</div>

<!-- Checkout View -->
<div class="checkout-view" id="checkout-view">
  <button class="back-btn" onclick="showStore()">&#8592; Back to Shop</button>
  <div class="checkout-card">
    <div class="checkout-title">Checkout</div>
    <div class="checkout-subtitle">Review your order and complete payment</div>
    <div class="cust-form" id="cust-form">
      <div class="cust-field">
        <label for="cust-name">Full Name</label>
        <input type="text" id="cust-name" placeholder="Rahul Sharma" autocomplete="name" />
      </div>
      <div class="cust-field">
        <label for="cust-email">Email</label>
        <input type="email" id="cust-email" placeholder="rahul@example.com" autocomplete="email" />
      </div>
      <div class="cust-field">
        <label for="cust-phone">Phone Number</label>
        <input type="tel" id="cust-phone" placeholder="+91 98765 43210" autocomplete="tel" />
      </div>
    </div>
    <div class="checkout-items" id="checkout-items"></div>
    <div class="checkout-divider"></div>
    <div class="checkout-row"><span>Subtotal</span><span id="co-subtotal">INR 0</span></div>
    <div class="checkout-row"><span>Shipping</span><span style="color:var(--green);font-weight:600">FREE</span></div>
    <div class="checkout-row total"><span>Total</span><span id="co-total">INR 0</span></div>
    <button class="pay-btn" id="pay-btn" onclick="startPayment()">Pay Now</button>
    <div class="status-bar" id="status"></div>
    <div class="decline-wall" id="decline-wall">
      <h3 id="decline-headline">Let's try a different way to complete your payment</h3>
      <p id="decline-subtext">Sometimes one payment method doesn't work — we've found alternatives that often succeed.</p>
      <div class="reason" id="decline-reason"></div>
      <p style="font-size:12px;color:#78716c;margin-bottom:8px;">Switch to an alternative payment method:</p>
      <div class="alt-rails" id="alt-rails">
        <button class="alt-rail-btn" onclick="switchRail('upi')"><span style="font-size:16px">&#128241;</span> UPI Autopay</button>
        <button class="alt-rail-btn" onclick="switchRail('netbanking')"><span style="font-size:16px">&#127974;</span> Netbanking</button>
        <button class="alt-rail-btn" onclick="switchRail('new_card')"><span style="font-size:16px">&#128179;</span> New Card</button>
        <button class="alt-rail-btn" onclick="switchRail('wallet')"><span style="font-size:16px">&#128092;</span> Wallet</button>
      </div>
    </div>
    <div class="action-box" id="action-box">
      <h4 id="action-title">Agent took an action</h4>
      <p id="action-detail"></p>
      <button class="btn-respond" id="respond-btn" onclick="respondToAgent()">I'll complete the payment now</button>
    </div>
  </div>
</div>

<!-- Cart Sidebar -->
<div class="cart-overlay" id="cart-overlay" onclick="toggleCart()"></div>
<div class="cart-sidebar" id="cart-sidebar">
  <div class="cart-header">
    <h2>Your Cart (<span id="cart-count-text">0</span>)</h2>
    <button class="cart-close" onclick="toggleCart()">&times;</button>
  </div>
  <div class="cart-items" id="cart-items">
    <div class="cart-empty">Your cart is empty</div>
  </div>
  <div class="cart-footer" id="cart-footer" style="display:none">
    <div class="cart-total">
      <span class="cart-total-label">Total</span>
      <span class="cart-total-val" id="cart-total">INR 0</span>
    </div>
    <button class="checkout-btn" onclick="goToCheckout()">Proceed to Checkout</button>
  </div>
</div>

<script>
/* ── Product Catalog ── */
const PRODUCTS = [
  {id:1, name:"Oversized Graphic Tee", brand:"SoulStreet Originals", price:1299, originalPrice:1799, category:"tops", img:"https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&h=500&fit=crop&auto=format", tag:"Trending"},
  {id:2, name:"Premium Cotton Hoodie", brand:"Street Essentials", price:2499, originalPrice:3299, category:"outerwear", img:"https://images.unsplash.com/photo-1618354691373-d851c5c3a990?w=400&h=500&fit=crop&auto=format", tag:"New"},
  {id:3, name:"Washed Denim Jacket", brand:"Heritage Denim", price:3499, originalPrice:4499, category:"outerwear", img:"https://images.unsplash.com/photo-1576995853123-5a10305d93c0?w=400&h=500&fit=crop&auto=format", tag:null},
  {id:4, name:"Slim Fit Joggers", brand:"Athleisure Co.", price:1799, originalPrice:2299, category:"bottoms", img:"https://images.unsplash.com/photo-1552902865-b72c031ac5ea?w=400&h=500&fit=crop&auto=format", tag:null},
  {id:5, name:"Polo Classic Tee", brand:"SoulStreet Originals", price:1499, originalPrice:1999, category:"tops", img:"https://images.unsplash.com/photo-1586363104862-3a5e2ab60d99?w=400&h=500&fit=crop&auto=format", tag:null},
  {id:6, name:"Urban Crop Top", brand:"Street Essentials", price:999, originalPrice:1499, category:"tops", img:"https://images.unsplash.com/photo-1503342217505-b0a15ec3261c?w=400&h=500&fit=crop&auto=format", tag:"-33%"},
  {id:7, name:"Relaxed Cargo Pants", brand:"Heritage Denim", price:2199, originalPrice:2999, category:"bottoms", img:"https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=400&h=500&fit=crop&auto=format", tag:"Popular"},
  {id:8, name:"Classic Bomber Jacket", brand:"SoulStreet Originals", price:3999, originalPrice:5499, category:"outerwear", img:"https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400&h=500&fit=crop&auto=format", tag:"-27%"}
];

/* ── Cart State ── */
let cart = {};

function renderProducts() {
  const grid = document.getElementById("product-grid");
  grid.innerHTML = PRODUCTS.map(p => {
    const discount = Math.round((1 - p.price / p.originalPrice) * 100);
    const inCart = cart[p.id];
    return '<div class="product-card">' +
      '<div class="product-img" style="position:relative">' +
        (p.tag ? '<span class="product-tag">' + p.tag + '</span>' : '') +
        '<img src="' + p.img + '" alt="' + p.name + '" loading="lazy" />' +
      '</div>' +
      '<div class="product-body">' +
        '<div class="product-brand">' + p.brand + '</div>' +
        '<div class="product-name">' + p.name + '</div>' +
        '<div class="product-price">' +
          '<span class="current">INR ' + p.price.toLocaleString() + '</span>' +
          '<span class="original">INR ' + p.originalPrice.toLocaleString() + '</span>' +
          '<span class="discount">' + discount + '% off</span>' +
        '</div>' +
        '<button class="add-btn' + (inCart ? ' added' : '') + '" onclick="addToCart(' + p.id + ')" id="add-btn-' + p.id + '">' +
          (inCart ? '&#10003; Added' : 'Add to Cart') +
        '</button>' +
      '</div>' +
    '</div>';
  }).join("");
}

function addToCart(id) {
  if (cart[id]) { cart[id].qty++; }
  else { cart[id] = { qty: 1 }; }
  updateCartUI();
  const btn = document.getElementById("add-btn-" + id);
  if (btn) { btn.classList.add("added"); btn.innerHTML = "&#10003; Added"; }
}

function removeFromCart(id) {
  delete cart[id];
  updateCartUI();
  const btn = document.getElementById("add-btn-" + id);
  if (btn) { btn.classList.remove("added"); btn.innerHTML = "Add to Cart"; }
}

function changeQty(id, delta) {
  if (!cart[id]) return;
  cart[id].qty += delta;
  if (cart[id].qty <= 0) { removeFromCart(id); return; }
  updateCartUI();
}

function getCartTotal() {
  let total = 0;
  for (const id in cart) {
    const p = PRODUCTS.find(x => x.id === parseInt(id));
    if (p) total += p.price * cart[id].qty;
  }
  return total;
}

function getCartCount() {
  let c = 0; for (const id in cart) c += cart[id].qty; return c;
}

function updateCartUI() {
  const count = getCartCount();
  const total = getCartTotal();
  document.getElementById("cart-count").textContent = count;
  document.getElementById("cart-count").style.display = count > 0 ? "flex" : "none";
  document.getElementById("cart-count-text").textContent = count;
  const footer = document.getElementById("cart-footer");
  footer.style.display = count > 0 ? "block" : "none";
  document.getElementById("cart-total").textContent = "INR " + total.toLocaleString();
  const itemsEl = document.getElementById("cart-items");
  if (count === 0) {
    itemsEl.innerHTML = '<div class="cart-empty">Your cart is empty</div>';
    return;
  }
  itemsEl.innerHTML = "";
  for (const id in cart) {
    const p = PRODUCTS.find(x => x.id === parseInt(id));
    if (!p) continue;
    itemsEl.innerHTML += '<div class="cart-item">' +
      '<div class="cart-item-img"><img src="' + p.img + '" alt="' + p.name + '" /></div>' +
      '<div class="cart-item-info">' +
        '<div class="cart-item-brand">' + p.brand + '</div>' +
        '<div class="cart-item-name">' + p.name + '</div>' +
        '<div class="cart-item-price">INR ' + (p.price * cart[id].qty).toLocaleString() + '</div>' +
        '<div class="cart-item-qty">' +
          '<button class="qty-btn" onclick="changeQty(' + id + ',-1)">-</button>' +
          '<span class="qty-val">' + cart[id].qty + '</span>' +
          '<button class="qty-btn" onclick="changeQty(' + id + ',1)">+</button>' +
        '</div>' +
        '<button class="cart-item-remove" onclick="removeFromCart(' + id + ')">Remove</button>' +
      '</div>' +
    '</div>';
  }
  /* update add buttons */
  PRODUCTS.forEach(p => {
    const btn = document.getElementById("add-btn-" + p.id);
    if (!btn) return;
    if (cart[p.id]) { btn.classList.add("added"); btn.innerHTML = "&#10003; Added"; }
    else { btn.classList.remove("added"); btn.innerHTML = "Add to Cart"; }
  });
}

function toggleCart() {
  const overlay = document.getElementById("cart-overlay");
  const sidebar = document.getElementById("cart-sidebar");
  const isOpen = sidebar.classList.contains("open");
  if (isOpen) { overlay.classList.remove("open"); sidebar.classList.remove("open"); }
  else { overlay.classList.add("open"); sidebar.classList.add("open"); }
}

function goToCheckout() {
  const count = getCartCount();
  if (count === 0) return;
  toggleCart();
  document.getElementById("store-view").classList.add("hidden");
  const cv = document.getElementById("checkout-view");
  cv.classList.add("active");
  const items = document.getElementById("checkout-items");
  items.innerHTML = "";
  for (const id in cart) {
    const p = PRODUCTS.find(x => x.id === parseInt(id));
    if (!p) continue;
    items.innerHTML += '<div class="checkout-item"><span class="checkout-item-name">' + p.name + ' &times; ' + cart[id].qty + '</span><span class="checkout-item-price">INR ' + (p.price * cart[id].qty).toLocaleString() + '</span></div>';
  }
  const total = getCartTotal();
  document.getElementById("co-subtotal").textContent = "INR " + total.toLocaleString();
  document.getElementById("co-total").textContent = "INR " + total.toLocaleString();
  window.scrollTo(0, 0);
}

function showStore() {
  document.getElementById("checkout-view").classList.remove("active");
  document.getElementById("store-view").classList.remove("hidden");
  window.scrollTo(0, 0);
}

/* ── Payment Logic (preserved) ── */
const paymentId = "pay_" + Math.random().toString(36).substr(2,9);
const socket = io();

/* Tell the server this page exists. Without it the server cannot know whether
   anyone is on the checkout, and an in-page push would be recorded as a
   contact that never happened. Re-sent on every reconnect. */
socket.on("connect", function () {
  socket.emit("watch_payment", {payment_id: paymentId});
});

function getCustomerData() {
  return {
    name: (document.getElementById("cust-name").value || "").trim(),
    email: (document.getElementById("cust-email").value || "").trim(),
    contact: (document.getElementById("cust-phone").value || "").replace(/\s/g, "").trim(),
  };
}

function validateCustomerForm() {
  const c = getCustomerData();
  let valid = true;
  ["cust-name", "cust-email", "cust-phone"].forEach(id => {
    document.getElementById(id).classList.remove("invalid");
  });
  if (!c.name) { document.getElementById("cust-name").classList.add("invalid"); valid = false; }
  if (!c.email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(c.email)) { document.getElementById("cust-email").classList.add("invalid"); valid = false; }
  if (!c.contact || c.contact.replace(/\D/g, "").length < 10) { document.getElementById("cust-phone").classList.add("invalid"); valid = false; }
  return valid;
}

function startPayment() {
  const btn = document.getElementById("pay-btn");
  const total = getCartTotal();
  if (total === 0) return;
  if (!validateCustomerForm()) { showStatus("failed", "Please fill in all fields with valid information."); return; }
  const cust = getCustomerData();
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Processing...';
  document.getElementById("decline-wall").classList.remove("visible");
  fetch("/api/create-order", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({amount:total, payment_id:paymentId})})
  .then(r => r.json()).then(data => {
    if (data.error) { showStatus("failed", data.error); btn.disabled = false; btn.innerHTML = "Pay Now"; return; }
    const rzp = new Razorpay({key:data.key_id, amount:data.amount, currency:data.currency, name:"SoulStreet", description:"Order #" + paymentId.slice(-6).toUpperCase(), order_id:data.order_id,
    handler: function(r) {
      showStatus("success", "Payment successful! ID: " + r.razorpay_payment_id);
      btn.innerHTML = "&#10003; Paid";
      btn.style.background = "var(--green)";
      document.getElementById("action-box").classList.remove("visible");
      document.getElementById("decline-wall").classList.remove("visible");
      const ap = document.getElementById("agent-push"); if (ap) ap.remove();
      _rzpModalOpen = false; _heldAgentPush = null;
      const ab = document.getElementById("agent-offer-banner");
      if (ab) { ab.remove(); document.body.style.paddingTop = ""; }
      /* Tell the server. Without this the money reaches Razorpay and the case
         never learns, so the agent keeps chasing a customer who has paid. The
         server verifies with Razorpay before believing any of it. */
      fetch("/api/payment-succeeded", {method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({payment_id:paymentId, razorpay_payment_id:r.razorpay_payment_id,
                             razorpay_order_id:r.razorpay_order_id})}).catch(function(){});
    },
    prefill: {name:cust.name, email:cust.email, contact:cust.contact},
    theme: {color:"#2563eb"},
    modal: {ondismiss: function() {
      _rzpModalOpen = false;
      /* Closing AFTER a failure is its own thing — neither a fresh drop-off
         nor nothing at all.

         It is not a drop-off: reporting it as one let a synthetic
         customer_cancelled outrank a real bank decline and authorise 5% off a
         payment the customer had been TRYING to make (pay_qssihjc5z). The
         server keeps the gateway's diagnosis for exactly that reason.

         But it is not nothing either: they were declined, Razorpay showed
         them its other rails, and they closed anyway. That is reluctance
         demonstrated, and the server records it as `abandoned_after_failure`
         — which is what moves the offer ahead of another rail switch. So the
         signal is still sent; the server decides what it means. */
      if (_closedAfterFailure) {
        _closedAfterFailure = false;
        showStatus("failed", "Payment didn't go through. Our agent is on it.");
        btn.disabled = false; btn.innerHTML = "Retry Payment";
        triggerRecovery({code:"customer_cancelled",
                         reason:"Closed the checkout after a failed attempt",
                         source:"customer", step:"payment_processing"}, true);
        /* Ask WHY before pushing anything at them. "No balance" and "cheaper
           elsewhere" produce the identical bank decline and need opposite
           responses — one needs time and no discount, the other a discount and
           no waiting. Nothing in an error code separates them, so we ask, and
           the held notification waits until they answer or skip. */
        askWhyTheyStopped();
        return;
      }
      /* The OTHER way out of a checkout: dismissed without ever attempting a
         payment. This path did not ask, which had it exactly backwards — a
         plain walk-away carries NO error code at all, so there is nothing to
         infer from and the customer's own answer is the only evidence there
         will ever be. Live (pay_kaagj53tv), someone closed the modal and got
         a "Complete your order" push before being asked anything.

         Same contract as the branch above: hold the agent, ask, and let the
         answer — or Skip, or the timeout — decide what it does. */
      showStatus("failed", "Payment cancelled. Our agent will help you recover it.");
      btn.disabled = false; btn.innerHTML = "Retry Payment";
      triggerRecovery({code:"customer_cancelled",
                       reason:"Payment cancelled by customer",
                       source:"customer", step:"payment_processing"}, true);
      askWhyTheyStopped();
    }}});
    rzp.on("payment.failed", function(r) {
      showStatus("failed", "Payment couldn't be processed: " + r.error.description);
      btn.disabled = false; btn.innerHTML = "Retry Payment";
      showDeclineWall(r.error.code || "failed", r.error.description);
      /* Record the decline, but DO NOT start the agent yet. The customer is
         still standing in the Razorpay modal with its other rails in front of
         them, and the agent starting here reasoned to completion and committed
         to a page push before anyone had asked why they stopped — a push that
         renders under the iframe they are still using (pay_glpfpyq90). Holding
         the notification was never enough: the DECISION is what has to wait,
         because a decision taken before the reason is known cannot use it.
         The close is the signal that they are done trying; `ondismiss` then
         asks why and releases the agent with the answer in hand. */
      triggerRecovery({code:r.error.code || "technical_error", reason:r.error.description || r.error.reason || "Payment failed", source:r.error.source || "gateway", step:r.error.step || "payment_processing"}, true);
      /* LET THEM TRY. Razorpay's own screen offers other rails after a
         decline, and a customer reaching for UPI themselves is the cheapest
         recovery there is — it costs us no link, no discount and no message.
         Closing the window for them (which this did) interrupted exactly that.
         Our notification stays HELD while the modal is open, because it would
         render under the iframe anyway; the customer's own close is the signal
         that they are done trying, and that is when it is delivered.
         `_closedAfterFailure` still marks that close as ours-after-a-failure,
         so it is recorded as giving up on a declined payment rather than as an
         unprompted change of mind. */
      _closedAfterFailure = true;
    });
    _rzpModalOpen = true;
    rzp.open();
  }).catch(() => { showStatus("failed", "Connection error"); btn.disabled = false; btn.innerHTML = "Pay Now"; });
}

function showDeclineWall(code, description) {
  document.getElementById("decline-wall").classList.add("visible");
  document.getElementById("decline-reason").textContent = "Reason: " + (description || code);
  const reasons = {card_expired:"Your card has expired. Let's try UPI or update your card.", insufficient_funds:"Insufficient funds. Try a different payment method.", network_timeout:"Connection issue. We'll retry automatically.", bank_declined:"Bank couldn't process this. Try another method."};
  document.getElementById("decline-subtext").textContent = reasons[code] || "We found alternative payment methods that often succeed.";
}

function switchRail(rail) {
  document.querySelectorAll(".alt-rail-btn").forEach(b => b.classList.remove("active"));
  event.currentTarget.classList.add("active");
  showStatus("processing", "Switching to " + rail.charAt(0).toUpperCase() + rail.slice(1) + "...");
  triggerRecovery({code:"method_switch", reason:"Customer switched to " + rail, source:"customer", step:"payment_processing"});
}

function triggerRecovery(err, deferAgent) {
  const total = getCartTotal();
  const cust = getCustomerData();
  /* deferAgent records the case but holds the agent until the customer has
     answered "why did you stop?" — see api_payment_failed. Without it the
     question and the agent race, and the agent wins by seconds. */
  fetch("/api/payment-failed", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({payment_id:paymentId, amount:total, failure_code:err.code||"technical_error", failure_reason:err.reason||"Payment failed", error_source:err.source||"gateway", error_step:err.step||"payment_processing", customer:cust, defer_agent: !!deferAgent})});
  showStatus("recovering", "Payment failed — our agent is working on it...");
}

function respondToAgent() {
  fetch("/api/customer-responded", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({payment_id:paymentId})});
  document.getElementById("action-box").classList.remove("visible");
  document.getElementById("decline-wall").classList.remove("visible");
  showStatus("processing", "Thank you! Processing your payment...");
  const btn = document.getElementById("pay-btn");
  btn.disabled = false;
  btn.innerHTML = "Complete Payment Now";
  btn.className = "pay-btn";
  btn.onclick = function() { startPayment(); };
}

function showStatus(type, msg) {
  const el = document.getElementById("status");
  el.className = "status-bar active s-" + type;
  if (type === "processing" || type === "recovering") el.innerHTML = '<span class="spinner"></span> ' + msg;
  else el.innerHTML = msg;
}

/* ── Agent push: the silent first rung of the recovery ladder ──
   The agent authors this; the page only renders it and reports back what the
   customer did. Acting, dismissing and ignoring are three different signals and
   all three are sent, because the agent reasons about intent from them. */
let _pushShownAt = 0, _pushTimer = null, _pushReported = false;

function reportPush(action, detail) {
  if (_pushReported) return;
  _pushReported = true;
  if (_pushTimer) { clearInterval(_pushTimer); _pushTimer = null; }
  fetch("/api/push-response", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      payment_id: paymentId, action: action,
      seconds_shown: (Date.now() - _pushShownAt) / 1000, detail: detail || ""
    })
  }).catch(function(){});
}

function closePush(action, detail) {
  const el = document.getElementById("agent-push");
  if (el) el.remove();
  reportPush(action || "dismissed", detail);
}

/* SURFACE-AWARE DELIVERY: Razorpay's checkout iframe owns the top z-index, so
   anything the agent draws while the modal is open renders UNDER it —
   unclickable, invisible, and the agent then waits minutes for a response to
   a notification the customer physically cannot see. Hold the latest push
   while the modal is open and deliver it the moment the modal closes. A push
   held when the payment SUCCEEDS is dropped: it belongs to a case that just
   ended. */
let _rzpModalOpen = false;
//: Set when WE close the checkout because the payment failed, so the dismiss
//: that follows is not misread as the customer walking away.
let _closedAfterFailure = false;
let _heldAgentPush = null;

function _flushHeldPush() {
  if (!_heldAgentPush) return;
  const held = _heldAgentPush; _heldAgentPush = null;
  setTimeout(function () { renderAgentPush(held); }, 400);
}

socket.on("agent_push", function(d) {
  if (d.payment_id !== paymentId) return;
  if (_rzpModalOpen) { _heldAgentPush = d; return; }
  renderAgentPush(d);
});

/* ── "Why did you stop?" ───────────────────────────────────────────────────
   One question, on the page they are already looking at, before any recovery
   surface. Their answer is testimony about their own intent and outranks
   every inference we would otherwise draw from the failure code. Skipping is
   always allowed: a customer who does not want to answer must not be trapped,
   and silence simply leaves the agent on its inferred path. */
function askWhyTheyStopped() {
  if (document.getElementById("why-card")) return;
  /* Defined out here, not inside the render, because the paths that never
     show the card at all must release the agent too — a question we could
     not ask must not hold a case shut. */
  let _released = false;
  function release() {
    if (_released) return;
    _released = true;
    fetch("/api/drop-reason/skip", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({payment_id: paymentId})}).catch(function () {});
  }
  fetch("/api/drop-reasons").then(r => r.json()).then(function (d) {
    const choices = (d && d.choices) || [];
    if (!choices.length) { release(); _flushHeldPush(); return; }
    const wrap = document.createElement("div");
    wrap.id = "why-card";
    wrap.style.cssText = "position:fixed;right:20px;bottom:20px;z-index:9999;" +
      "max-width:380px;background:#fff;border:1px solid #e5e7eb;border-radius:14px;" +
      "padding:18px;box-shadow:0 12px 40px rgba(0,0,0,.16);" +
      "font:14px/1.5 ui-sans-serif,system-ui,sans-serif;animation:slideUp .3s ease-out";
    let opts = "";
    choices.forEach(function (c) {
      opts += '<button class="why-opt" data-code="' + c.code + '" data-free="' +
        (c.free_text ? "1" : "") + '" style="display:block;width:100%;text-align:left;' +
        'margin-top:7px;padding:9px 12px;border:1px solid #e5e7eb;border-radius:9px;' +
        'background:#fff;font-size:13px;cursor:pointer;color:#111827">' +
        String(c.label).replace(/[&<>"]/g, "") + '</button>';
    });
    wrap.innerHTML =
      '<div style="font-weight:650;color:#111827">Before you go — what happened?</div>' +
      '<div style="color:#4b5563;margin-top:5px;font-size:12.5px">One tap. It tells us ' +
      'whether to hold off or help — we will not guess.</div>' + opts +
      '<div id="why-free" style="display:none;margin-top:9px">' +
      '<input id="why-text" maxlength="200" placeholder="In your own words…" ' +
      'style="width:100%;padding:9px 11px;border:1px solid #e5e7eb;border-radius:9px;' +
      'font-size:13px;font-family:inherit">' +
      '<button id="why-send" style="margin-top:7px;width:100%;background:#2563eb;' +
      'color:#fff;border:0;border-radius:9px;padding:9px;font-size:13px;' +
      'font-weight:600;cursor:pointer">Send</button></div>' +
      '<button id="why-skip" style="margin-top:10px;width:100%;background:none;' +
      'border:0;color:#9ca3af;font-size:12px;cursor:pointer">Skip</button>';
    document.body.appendChild(wrap);

    /* Either branch must release the agent, or a deferred case waits for an
       answer that is never coming. `answered` distinguishes them: a reply
       starts the agent through /api/drop-reason with the reason attached;
       a skip, or the timeout below, starts it with nothing added. */
    function done(answered) {
      clearTimeout(_whyTimer);
      const el = document.getElementById("why-card");
      if (el) el.remove();
      if (!answered) release();
      _flushHeldPush();
    }
    /* A customer who walks away must not hold the case open. Two minutes is
       longer than anyone spends on one question and short enough that the
       recovery window is barely touched. */
    const _whyTimer = setTimeout(function () { done(false); }, 120000);
    function send(code, text) {
      fetch("/api/drop-reason", {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({payment_id: paymentId, code: code, text: text || ""})
      }).catch(function () {}).then(function () { done(true); });
    }
    wrap.querySelectorAll(".why-opt").forEach(function (b) {
      b.onclick = function () {
        if (b.dataset.free) {
          wrap.querySelectorAll(".why-opt").forEach(function (x) { x.style.display = "none"; });
          document.getElementById("why-free").style.display = "block";
          document.getElementById("why-text").focus();
          document.getElementById("why-send").onclick = function () {
            send("other", document.getElementById("why-text").value);
          };
          return;
        }
        send(b.dataset.code, "");
      };
    });
    document.getElementById("why-skip").onclick = function () { done(false); };
  }).catch(function () { release(); _flushHeldPush(); });
}

function renderAgentPush(d) {
  var esc = function (t) {
    return String(t == null ? "" : t).replace(/[&<>"']/g, function (c) {
      return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c];
    });
  };
  const old = document.getElementById("agent-push");
  /* Nothing replaces a notification any more: there is one push, and an offer
     arrives as a banner. If the plain card is still up when the banner lands,
     it goes quietly — the banner's own outcome becomes the case's outcome, so
     nothing is lost by not reporting a handover that no longer happens. */
  if (old) old.remove();
  _pushShownAt = Date.now(); _pushReported = false;

  /* Coupon banner: the offer is not just described in the notification, it is
     applied to the page the customer is looking at. Original struck through,
     new total beside it — the same figure the payment link charges, because the
     tool refuses to display one that disagrees. */
  const oldBanner = document.getElementById("agent-offer-banner");
  if (oldBanner) oldBanner.remove();
  if (d.offer && d.offer.payable_rupees) {
    const o = d.offer;
    const money = function (n) {
      return "\u20B9" + Number(n).toLocaleString("en-IN", {minimumFractionDigits: 2});
    };
    const bar = document.createElement("div");
    bar.id = "agent-offer-banner";
    bar.style.cssText = "position:fixed;top:0;left:0;right:0;z-index:9998;" +
      (d.mode === "rail_switch"
        ? "background:linear-gradient(90deg,#1e3a8a,#1d4ed8);"
        : "background:linear-gradient(90deg,#065f46,#047857);") +
      "color:#fff;padding:11px 16px;" +
      "display:flex;align-items:center;justify-content:center;gap:14px;flex-wrap:wrap;" +
      "font:14px/1.4 ui-sans-serif,system-ui,sans-serif;box-shadow:0 2px 12px rgba(0,0,0,.18)";
    bar.innerHTML =
      (o.discount_pct ? '<span style="background:#fff;color:#065f46;border-radius:999px;' +
        'padding:3px 11px;font-weight:700;font-size:12.5px">' +
        Number(o.discount_pct).toFixed(0) + '% OFF</span>' : '') +
      '<span style="font-weight:600">' + esc(d.headline) + '</span>' +
      (o.original_rupees ? '<span style="opacity:.75;text-decoration:line-through">' +
        money(o.original_rupees) + '</span>' : '') +
      '<span style="font-weight:700;font-size:16px">' + money(o.payable_rupees) + '</span>' +
      '<span id="offer-countdown" style="opacity:.85;font-size:12px"></span>' +
      '<button id="offer-cta" style="margin-left:6px;background:#fff;color:#065f46;' +
        'border:0;border-radius:7px;padding:6px 14px;font-size:13px;font-weight:700;' +
        'cursor:pointer">' + esc(d.cta_text || "Pay now") + '</button>' +
      '<button id="offer-x" aria-label="Dismiss" style="background:none;border:0;' +
        'color:rgba(255,255,255,.75);font-size:18px;cursor:pointer;line-height:1;' +
        'padding:0 2px">&times;</button>';
    document.body.appendChild(bar);
    document.body.style.paddingTop = "44px";

    /* The banner carries the offer on its own now, so it needs its own way to
       act on it and its own way to be turned down — both are signals the agent
       reads. */
    _pushShownAt = Date.now(); _pushReported = false;
    document.getElementById("offer-cta").onclick = function () {
      reportPush("acted", "Customer clicked the offer banner.");
      if (d.payment_link) window.open(d.payment_link, "_blank");
      else startPayment();
    };
    document.getElementById("offer-x").onclick = function () {
      reportPush("dismissed", "Customer closed the offer banner.");
      bar.remove(); document.body.style.paddingTop = "";
    };

    let osec = Math.max(1, o.expires_in_minutes || 15) * 60;
    const oc = document.getElementById("offer-countdown");
    const otimer = setInterval(function () {
      osec -= 1;
      if (osec <= 0) {
        clearInterval(otimer);
        const b = document.getElementById("agent-offer-banner");
        if (b) b.remove();
        document.body.style.paddingTop = "";
        return;
      }
      if (oc) oc.textContent = "expires in " + Math.floor(osec / 60) + ":" +
        String(osec % 60).padStart(2, "0");
    }, 1000);
  }

  /* An offer is a banner, not a second notification.
     Both were drawn from this one event, so the discount arrived as a bar
     across the top AND another card in the corner, moments after the plain
     one — two notifications for a customer who had already seen the first. */
  if (d.offer && d.offer.payable_rupees) return;

  const wrap = document.createElement("div");
  wrap.id = "agent-push";
  wrap.style.cssText = "position:fixed;right:20px;bottom:20px;z-index:9999;max-width:360px;" +
    "background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:18px 18px 16px;" +
    "box-shadow:0 12px 40px rgba(0,0,0,.16);font:14px/1.5 ui-sans-serif,system-ui,sans-serif;" +
    "animation:slideUp .3s ease-out";
  wrap.innerHTML =
    '<button id="push-x" aria-label="Dismiss" style="position:absolute;top:10px;right:12px;' +
      'border:0;background:none;font-size:19px;color:#9ca3af;cursor:pointer;line-height:1">&times;</button>' +
    '<div style="font-weight:650;color:#111827;padding-right:22px">' + esc(d.headline) + '</div>' +
    '<div style="color:#4b5563;margin-top:6px;font-size:13px">' + esc(d.body) + '</div>' +

    '<button id="push-cta" style="margin-top:13px;width:100%;background:#2563eb;color:#fff;' +
      'border:0;border-radius:9px;padding:11px;font-size:14px;font-weight:600;cursor:pointer">' +
      esc(d.cta_text || "Complete payment") + '</button>' +
    '<div id="push-countdown" style="margin-top:8px;font-size:11.5px;color:#9ca3af;text-align:center"></div>';
  document.body.appendChild(wrap);

  document.getElementById("push-x").onclick = function () {
    closePush("dismissed", "Customer closed the notification.");
  };
  document.getElementById("push-cta").onclick = function () {
    /* The plain nudge never carries a link — it reopens the checkout the
       customer is already on. Offers travel by banner. */
    reportPush("acted", "Customer clicked the call to action.");
    startPayment();
    const el = document.getElementById("agent-push"); if (el) el.remove();
  };

  let left = Math.max(1, (d.wait_minutes || 5)) * 60;
  const cd = document.getElementById("push-countdown");
  _pushTimer = setInterval(function () {
    left -= 1;
    if (left <= 0) { clearInterval(_pushTimer); _pushTimer = null; if (cd) cd.textContent = ""; return; }
    if (cd) cd.textContent = "Offer expires in " + Math.floor(left / 60) + ":" +
      String(left % 60).padStart(2, "0");
  }, 1000);
}

socket.on("agent_event", function(data) {
  if (data.payment_id !== paymentId) return;
  if (data.event === "waiting_for_customer") {
    const box = document.getElementById("action-box");
    box.classList.add("visible");
    document.getElementById("action-title").textContent = "Agent: " + (data.action || "took action");
    document.getElementById("action-detail").textContent = data.detail || "Please respond to continue recovery.";
    showStatus("waiting", "Agent is waiting for your response...");
  }
  if (data.event === "complete") {
    const s = document.getElementById("status");
    /* A run ending is not the case ending. The agent finishes a turn and waits
       — for this notification, for an email, for a retry — and the customer was
       being told "Could not recover automatically" while a live offer sat on
       screen above it. Only a settled case gets a verdict. */
    const settled = ["recovered", "escalated", "unrecoverable", "failed"];
    if (settled.indexOf(data.status) === -1) return;
    document.getElementById("action-box").classList.remove("visible");
    document.getElementById("decline-wall").classList.remove("visible");
    if (data.status === "recovered") { s.className = "status-bar active s-success"; s.innerHTML = "Payment recovered! Thank you."; }
    else { s.className = "status-bar active s-failed"; s.innerHTML = "Could not recover automatically. Please try again or update your payment method."; }
  }
});



/* The agent no longer rewrites this page.
   `applyGenerativeUISpec` pasted the agent's own spec into the checkout —
   `spec.headline` into the title (which has held the agent's markdown summary
   verbatim), a second discount banner, and a status line reading
   "Background Retry Active — customer_cancelled": internal labels, shown to
   the customer. Together with the `ui_spec_overlay` panel that is now gone,
   the agent had three ways to write on a page it should only ever speak to
   through its one notification and the offer banner.
   The spec itself is unchanged and still drives the merchant HUD. */

/* ── Init ── */
renderProducts();
</script>
</body></html>"""


# ─── Merchant Dashboard ───────────────────────────────────────


# ─── Routes ───────────────────────────────────────────────────
@app.route("/")
@app.route("/merchant")
def merchant_page():
    return render_template("index.html")

@app.route("/pay")
def pay_page():
    return render_template_string(PAY_PAGE)

@app.route("/graph")
def graph_page():
    from recovery_agent.dashboard import GRAPH_TEMPLATE
    return render_template_string(GRAPH_TEMPLATE)

@app.route("/api/simulate/<scenario>", methods=["POST", "GET"])
def simulate_scenario(scenario: str):
    import random
    payment_id = f"pay_sim_{random.randint(1000, 9999)}"

    # Full scenario context — no hardcoded frontend values
    scenarios = {
        "degradation": {
            "amount": 4999.0,
            "reason": "Gateway timeout during payment processing",
            "scenario_type": "degradation",
            "failure_code": "gateway_timeout",
            "error_source": "gateway",
            "error_step": "payment_authorization",
            "customer": {"name": "Rahul Kumar", "email": "rahul@example.com", "contact": "+919876543210"},
            "biz_name": "SaaS Subscription Platform",
            "biz_detail": "Annual Pro Plan",
            "badge": "Gateway 504 Degradation",
            "item_name": "Pro Subscription Plan (Annual)",
            "card_last4": "4242",
            "card_expiry": "08/29",
            "card_holder": "RAHUL KUMAR",
        },
        "abandonment": {
            "amount": 2999.0,
            "reason": "Customer closed tab during checkout",
            "scenario_type": "abandonment",
            "failure_code": "customer_cancelled",
            "error_source": "customer",
            "error_step": "payment_initiation",
            "customer": {"name": "Priya Sharma", "email": "priya@example.com", "contact": "+919876543211"},
            "biz_name": "E-Commerce Magic Checkout",
            "biz_detail": "Premium Audio Gear",
            "badge": "Cart Abandonment",
            "item_name": "Noise-Cancelling Headphones",
            "card_last4": "8901",
            "card_expiry": "12/26",
            "card_holder": "PRIYA SHARMA",
        },
        "card_expiry": {
            "amount": 12999.0,
            "reason": "Card expiry date is in the past",
            "scenario_type": "card_expiry",
            "failure_code": "card_expired",
            "error_source": "customer",
            "error_step": "payment_authentication",
            "customer": {"name": "Amit Patel", "email": "amit@example.com", "contact": "+919876543212"},
            "biz_name": "Recurring Auto-Debit Membership",
            "biz_detail": "Enterprise Cloud Infrastructure",
            "badge": "Card Expiry Failure",
            "item_name": "Enterprise Cloud Infrastructure",
            "card_last4": "5678",
            "card_expiry": "03/24",
            "card_holder": "AMIT PATEL",
        },
        "voice_call": {
            "amount": 8500.0,
            "reason": "High-value mandate failure requiring voice intervention",
            "scenario_type": "voice_call",
            "failure_code": "mandate_revoked",
            "error_source": "bank",
            "error_step": "payment_authorization",
            "customer": {"name": "Neha Gupta", "email": "neha@example.com", "contact": "+919876543213"},
            "biz_name": "B2B Enterprise Invoice Gateway",
            "biz_detail": "Quarterly API License",
            "badge": "High-Value Voice AI",
            "item_name": "Quarterly API Gateway License",
            "card_last4": "3456",
            "card_expiry": "11/27",
            "card_holder": "NEHA GUPTA",
        },
    }

    # The "Run 30-Case Batch" button is gone, and so is what it did: a synthetic
    # CSV evaluation of 10 fabricated cases, labelled 30. It measured nothing
    # about this system and shared no code with real recovery. Batches are now a
    # real view over real cases — see /api/batches.
    if scenario not in scenarios:
        scenario = "degradation"

    ctx = scenarios[scenario]
    customer = ctx["customer"]
    store.save_payment(payment_id, {
        "payment_id": payment_id,
        "amount": ctx["amount"],
        "status": "recovering",
        "attempts": 0,
        "last_action": "",
        "last_detail": "",
        "trail": [],
        
        "recovery_tier": "silent",
        "decline_strategy": "",
        "penalties_prevented": 0,
    })

    socketio.start_background_task(run_agent_for_payment, payment_id, ctx["amount"], ctx["reason"], customer, ctx["scenario_type"], ctx["failure_code"], ctx["error_source"], ctx["error_step"])
    return jsonify({
        "status": "simulating",
        "scenario": scenario,
        "payment_id": payment_id,
        "amount": ctx["amount"],
        "reason": ctx["reason"],
        "customer": customer,
        "biz_name": ctx["biz_name"],
        "biz_detail": ctx["biz_detail"],
        "badge": ctx["badge"],
        "item_name": ctx["item_name"],
        "card_last4": ctx["card_last4"],
        "card_expiry": ctx["card_expiry"],
        "card_holder": ctx["card_holder"],
    })

@app.route("/api/create-order", methods=["POST"])
def create_order():
    data = request.json
    amount = data.get("amount", 2999)
    payment_id = data.get("payment_id", "pay_unknown")
    def _record_order(order_id_value: str) -> None:
        """Attach the new order WITHOUT destroying an in-flight recovery.

        This used to call `store.save_payment`, which REPLACES the whole record —
        so a second "Pay" click on the same page wiped the case: trail, customer,
        status, recovered amount, all of it. A recovery that had already been
        confirmed against Razorpay (INR 37,990.50 on pay_k3jofbz4j) was reset to
        `pending` with an empty trail, which is why it looked as though the agent
        had not caught the payment.
        """
        if store.has_payment(payment_id):
            # `order_id` is later reused to hold the RECOVERY link, so the
            # original checkout order has to live under its own key or it is lost
            # — and with it any way to reconcile the two sides in Razorpay.
            store.update_payment(payment_id, order_id=order_id_value, amount=amount,
                                 original_order_id=order_id_value)
        else:
            store.save_payment(payment_id, {
                "payment_id": payment_id, "amount": amount, "status": "pending",
                "order_id": order_id_value, "original_order_id": order_id_value,
                "attempts": 0, "last_action": "", "last_detail": "", "trail": [],
            })
        store.flush()

    if razorpay_client.is_configured:
        order = razorpay_client.create_order(amount=amount, notes={"payment_id": payment_id})
        if "error" not in order:
            _record_order(order["id"])
            return jsonify({"order_id": order["id"], "amount": order["amount"], "currency": order["currency"], "key_id": razorpay_client.key_id})
    order_id = f"order_sim_{payment_id}"
    _record_order(order_id)
    return jsonify({"order_id": order_id, "amount": int(amount * 100), "currency": "INR", "key_id": razorpay_client.key_id or "rzp_test_demo"})

@app.route("/api/payment-succeeded", methods=["POST"])
def payment_succeeded():
    """The checkout page reports that Razorpay's inline handler fired.

    This route did not exist. The `handler` callback in startPayment() only
    painted the button green, so a customer who paid through the agent's own
    push notification — the first and cheapest rung of the ladder, and the one
    with no payment link, so its CTA reopens the inline checkout — was invisible
    to the backend. The money landed in Razorpay and the case sat in
    `awaiting_customer` until it timed out.

    The browser's word is NOT taken for it. A POST here is only a hint that
    something happened; the payment is fetched from Razorpay and must come back
    captured, for this case's own order, before a rupee is recorded.
    """
    data = request.get_json(silent=True) or {}
    payment_id = str(data.get("payment_id") or "")
    rzp_payment_id = str(data.get("razorpay_payment_id") or "")
    rzp_order_id = str(data.get("razorpay_order_id") or "")
    if not payment_id or not rzp_payment_id:
        return jsonify({"error": "payment_id and razorpay_payment_id required"}), 400
    if not store.has_payment(payment_id):
        return jsonify({"error": "unknown case"}), 404

    if not razorpay_client.is_configured:
        return jsonify({"status": "unverified", "note": "razorpay not configured"}), 503

    try:
        pay = razorpay_client.client.payment.fetch(rzp_payment_id)
    except Exception as exc:
        push_event(payment_id, "verification_failed", {
            "detail": f"Could not verify {rzp_payment_id}: {exc}"})
        return jsonify({"status": "unverified", "note": str(exc)}), 502

    if pay.get("status") != "captured":
        return jsonify({"status": "not_captured", "payment_status": pay.get("status")}), 409

    known = {store.get_payment(payment_id).get(k, "") for k in
             ("original_order_id", "order_id")}
    claimed_order = pay.get("order_id") or rzp_order_id
    if claimed_order and known - {""} and claimed_order not in known:
        # A real capture, but not for this case. Recording it here would credit
        # one case with another's money.
        return jsonify({"status": "order_mismatch", "order_id": claimed_order}), 409

    amount = pay.get("amount", 0) / 100

    # A payment that never failed is a SALE, not a recovery. /api/create-order
    # opens a `pending` record for every checkout (it has to — the order id is
    # the reconciliation key if a failure comes later), so a clean first-try
    # success used to flow into _mark_recovered: agent events fired for a case
    # that never existed, a phantom card flashed on the HUD and vanished on
    # refresh, and "recovered revenue" was credited with money that was never
    # at risk. Record the capture for reconciliation and stay silent.
    if _never_entered_recovery(store.get_payment(payment_id) or {}):
        store.update_payment(payment_id, status="paid",
                             paid_payment_id=rzp_payment_id,
                             paid_amount=amount)
        store.flush()
        return jsonify({"status": "paid_clean", "amount": amount,
                        "payment_id": rzp_payment_id})

    recorded = _mark_recovered(payment_id, amount, rzp_payment_id, 0,
                               "customer paid on the checkout page")
    return jsonify({"status": "recovered" if recorded else "already_recorded",
                    "amount": amount, "payment_id": rzp_payment_id})


#: Failure codes the checkout PAGE invents, as opposed to codes the gateway
#: returned. `customer_cancelled` means "the Razorpay modal closed" — after a
#: real decline that is a CONSEQUENCE of the failure, not a new diagnosis.
_SYNTHETIC_FAILURE_CODES = {"customer_cancelled", "technical_error", "method_switch"}


@app.route("/api/payment-failed", methods=["POST"])
def payment_failed():
    data = request.json
    payment_id = data.get("payment_id", "")
    amount = data.get("amount", 0)
    failure_reason = data.get("failure_reason", "payment_failed")
    failure_code = data.get("failure_code", "")
    error_source = data.get("error_source", "")
    error_step = data.get("error_step", "")
    customer = data.get("customer", {})
    if store.has_payment(payment_id):
        if store.get_payment(payment_id).get("status") == "recovering":
            return jsonify({"status": "already_recovering", "payment_id": payment_id})
    if not store.has_payment(payment_id):
        store.save_payment(payment_id, {"payment_id": payment_id, "amount": amount, "status": "recovering", "attempts": 0, "last_action": "", "last_detail": "", "trail": [], "customer": customer or {}})

    # SIGNAL PRECEDENCE: a synthetic page event never overwrites a gateway
    # diagnosis. Live (pay_fotw1e7b6, ₹59,976): netbanking failed at the BANK,
    # the customer sat in the Razorpay modal, and when they finally closed it
    # `ondismiss` posted `customer_cancelled` — which overwrote the decline.
    # From then on the case classified as a drop-off, so the offer policy
    # (which refuses discounts for method failures) legally authorised 5% off
    # a payment the customer had been trying to MAKE. The dismissal is kept,
    # but as a journey signal on the trail — not as the diagnosis.
    prior = store.get_payment(payment_id) or {}
    prior_code = str(prior.get("failure_code") or "")
    if (failure_code in _SYNTHETIC_FAILURE_CODES
            and prior_code and prior_code not in _SYNTHETIC_FAILURE_CODES):
        note = {
            "step": "signal_precedence",
            "msg": f"Checkout closed after a real failure ({prior_code}) — same "
                   f"failure, not a new drop-off",
            "detail": f"page reported {failure_code!r}; keeping the gateway's "
                      f"{prior_code!r} as the diagnosis",
            "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        }
        prior.setdefault("trail", []).append(note)
        push_event(payment_id, "signal_precedence", note)
        failure_code = prior_code
        failure_reason = str(prior.get("failure_reason") or failure_reason)
        # ...but the cancellation is not NOTHING. Walking away after a failure
        # is different from walking away with nothing wrong: the customer saw
        # the decline, had Razorpay's other rails in front of them, and chose
        # to stop. That is reluctance demonstrated, which is exactly the
        # evidence a method failure otherwise has to earn by trying full price
        # first. The diagnosis stays "the bank declined"; what changes is that
        # price is now a defensible lever.
        _abandoned_after_failure = True
    else:
        _abandoned_after_failure = bool(prior.get("abandoned_after_failure"))

    # A NEW failure means the previous attempt is over.
    #
    # `push_outcome` was left as "acted" from whenever the customer last clicked
    # a notification, and the mid-payment guard reads it. Live: the customer
    # clicked, their bank declined, and for the rest of the case every attempt
    # to act was refused with "the customer is in the middle of paying" — three
    # runs in a row, each one blocked on a click that had already failed.
    #
    # The failure code is persisted too. It was not stored at all, so the record
    # said `failure_code: None` and the only trace of a bank decline was prose
    # in a message — which is why the agent reached for a discount on what was
    # a payment-method problem.
    # `decline_strategy` is refreshed with the code, not left from the previous
    # attempt. It is a DERIVED field that `failure_kind` consults, so a stale
    # one silently re-diagnoses the new failure as the old one.
    # The customer is persisted, not merely passed along. The non-deferred path
    # hands `customer` straight to run_agent_for_payment and so never noticed it
    # was missing from the record; the deferred path restarts the agent by
    # reading the record back, got {}, and the agent — correctly, given what it
    # could see — concluded there was "no contact info available" and escalated
    # to a human on its first move. Every drop-off that stopped to ask the
    # customer why lost that customer's email and phone in the process.
    # Merged, never blanked: a later failure that omits the contact must not
    # erase the details an earlier one supplied.
    _known = dict((store.get_payment(payment_id) or {}).get("customer") or {})
    _known.update({k: v for k, v in (customer or {}).items() if v})
    store.update_payment(payment_id, status="recovering", push_outcome=None,
                         failure_code=failure_code or "",
                         failure_reason=failure_reason or "",
                         decline_strategy=failure_code or "",
                         customer=_known,
                         abandoned_after_failure=_abandoned_after_failure)


    # The checkout asks the customer WHY they stopped, and that answer
    # outranks anything inferable from an error code. Starting the agent here
    # anyway meant the question and the agent raced, and the agent won: on
    # pay_96fxy62mc it minted a link, gave 5% away and emailed the offer, and
    # only THEN did the customer say "I didn't have enough balance" — the one
    # answer for which a discount is useless and a link is a link nobody can
    # pay. The reply arrives seconds later; the case can wait that long.
    if bool(data.get("defer_agent")):
        # Stamped, because the only thing that released this hold was a
        # setTimeout in the customer's browser. Close the tab and the case
        # waited for an answer that could no longer arrive — forever, with no
        # agent and no recovery. The daemon sweeps stale holds using this.
        store.update_payment(payment_id, drop_reason_pending=True,
                             drop_reason_pending_at=datetime.now(timezone.utc)
                             .isoformat())
        _do_flush()
        # Say so, out loud. Holding was the right call but it emitted nothing,
        # so the HUD opened a session tab and then showed an empty monologue —
        # which looks exactly like an agent that has crashed. Correct behaviour
        # that cannot be seen is indistinguishable from broken behaviour.
        push_event(payment_id, "awaiting_reason", {
            "step": "awaiting_reason",
            "msg": "Holding — asked the customer why they stopped",
            "detail": ("No error code separates 'no balance' from 'found it "
                       "cheaper elsewhere', and they need opposite responses. "
                       "Nothing is sent until they answer, skip, or 2 minutes "
                       "pass."),
            "amount": amount,
        })
        return jsonify({"status": "awaiting_reason", "payment_id": payment_id})

    socketio.start_background_task(run_agent_for_payment, payment_id, amount, failure_reason, customer, "standard", failure_code, error_source, error_step)
    return jsonify({"status": "recovery_started"})

@app.route("/api/customer-responded", methods=["POST"])
def customer_responded():
    data = request.json or {}
    payment_id = data.get("payment_id", "")
    updated_expiry = data.get("updated_expiry", "08/29")

    if store.has_pending(payment_id):
        store.remove_pending(payment_id)

    if store.has_payment(payment_id):
        p = store.get_payment(payment_id)
        amount = p.get("amount", 0)
        order_id = p.get("order_id", "")

        is_simulated = order_id.startswith("order_rzp_") or order_id.startswith("order_sim_")
        order_paid = False

        if order_id and not is_simulated and razorpay_client.is_configured:
            order_data = razorpay_client.fetch_order(order_id)
            order_status = order_data.get("status", "")
            order_paid = order_status == "paid"
        elif is_simulated or not razorpay_client.is_configured:
            order_paid = True

        if order_paid:
            p["status"] = "recovered"
            trail_entry = {
                "step": "stopping",
                "msg": f"Card Expiry Updated ({updated_expiry}) & Payment Recovered!",
                "detail": f"Updated expiry: {updated_expiry}. Charge verified via Razorpay API Capture.",
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            }
            p.setdefault("trail", []).append(trail_entry)
            push_event(payment_id, "stopping", trail_entry)
            push_event(payment_id, "complete", {"status": "recovered", "attempts": 1, "amount": amount})
            store.flush()
            return jsonify({"status": "recovered", "payment_id": payment_id, "amount": amount})
        else:
            trail_entry = {
                "step": "stopping",
                "msg": "Customer clicked complete, but Razorpay capture not found",
                "detail": f"Order {order_id} status: not paid. Awaiting Razorpay capture webhook.",
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            }
            p.setdefault("trail", []).append(trail_entry)
            push_event(payment_id, "stopping", trail_entry)
            store.flush()
            return jsonify({"status": "recovering", "payment_id": payment_id, "detail": "Order not yet paid"})

    return jsonify({"status": "no_pending_action"})


@app.route("/api/webhook-forward", methods=["POST"])
def webhook_forward():
    data = request.json or {}
    event = data.get("event", "")
    payload = data.get("payload", {})

    print(f"[frontend] Webhook forwarded: {event}")

    if event == "payment.failed":
        payment_entity = payload.get("payment", {}).get("entity", {})
        payment_id = payment_entity.get("id", "")
        amount = payment_entity.get("amount", 0) / 100
        error_code = payment_entity.get("error_code", "")
        error_reason = payment_entity.get("error_reason", "")
        error_step = payment_entity.get("error_step", "")
        contact = payment_entity.get("contact", "")
        email = payment_entity.get("email", "")
        notes = payment_entity.get("notes", {})
        customer_id = notes.get("customer_id", payment_entity.get("customer_id", f"cust_{payment_id}"))

        if store.has_payment(payment_id):
            if store.get_payment(payment_id).get("status") == "recovering":
                return jsonify({"status": "already_recovering", "payment_id": payment_id})

        if not store.has_payment(payment_id):
            store.save_payment(payment_id, {
                "payment_id": payment_id,
                "amount": amount,
                "status": "recovering",
                "attempts": 0,
                "last_action": "",
                "last_detail": "",
                "trail": [],
            })
        store.update_payment(payment_id, status="recovering")

        socketio.start_background_task(
            run_agent_for_payment,
            payment_id,
            amount,
            f"{error_code}: {error_reason}",
            {"id": customer_id, "name": contact, "email": email},
            "standard",
            error_code,
            payment_entity.get("error_source", ""),
            error_step,
        )
        store.flush()

        return jsonify({"status": "recovery_started", "payment_id": payment_id})

    elif event == "payment.captured":
        payment_entity = payload.get("payment", {}).get("entity", {})
        payment_id = payment_entity.get("id", "")
        amount = payment_entity.get("amount", 0) / 100
        notes = payment_entity.get("notes", {})

        # Resolve payment link → original payment ID.
        # When customer pays via a recovery link, Razorpay creates a NEW payment.
        # The original_payment note on the link ties the new payment back to our case.
        original_payment_id = notes.get("original_payment", "")
        if original_payment_id and store.has_payment(original_payment_id):
            payment_id = original_payment_id

        if store.has_pending(payment_id):
            store.remove_pending(payment_id)

        # Through the one recorder, exactly like the pollers and the browser
        # callback. This branch used to write status/recovered_amount by hand,
        # which skipped the already-recovered guard, skipped the order
        # cross-reference, and — worst — never called _notify_agent_of_recovery:
        # on the real webhook path the agent was never told it worked, so the
        # one lesson worth keeping (what actually recovered the money) was
        # thrown away on every webhook-confirmed success.
        # A clean first-try sale stays out of the recovery books here too.
        rec0 = store.get_payment(payment_id) or {}
        if rec0 and _never_entered_recovery(rec0):
            store.update_payment(payment_id, status="paid",
                                 paid_payment_id=payment_entity.get("id", ""),
                                 paid_amount=amount)
            store.flush()
            return jsonify({"status": "paid_clean", "payment_id": payment_id})

        how = ("recovery link paid (webhook payment.captured)"
               if original_payment_id else
               "original payment captured (webhook payment.captured)")
        recorded = _mark_recovered(payment_id, amount,
                                   payment_entity.get("id", ""), 0, how)

        # No case, no card: an event for an unknown payment id draws a phantom
        # on the HUD that vanishes at the next refresh.
        if store.has_payment(payment_id):
            push_event(payment_id, "webhook_captured", {
                "status": "recovered",
                "amount": amount,
                "payment_id": payment_id,
                "original_payment_id": original_payment_id,
                "captured_payment_id": payment_entity.get("id", ""),
            })
        store.flush()

        if recorded:
            status = "captured"
        else:
            status = ("already_recovered" if store.has_payment(payment_id)
                      else "unknown_case")
        return jsonify({"status": status, "payment_id": payment_id})

    return jsonify({"status": "ignored", "event": event})


@app.route("/api/daemon-retry-complete", methods=["POST"])
def daemon_retry_complete():
    data = request.json or {}
    job_id = data.get("job_id", "")
    payment_id = data.get("payment_id", "")
    action = data.get("action", "")
    result = data.get("result", {})
    source = data.get("source", "daemon_worker")

    print(f"[frontend] Daemon retry complete: job={job_id} payment={payment_id} status={result.get('status')}")

    if store.has_payment(payment_id):
        p = store.get_payment(payment_id)
        p["last_action"] = action
        p["last_detail"] = result.get("message", "")
        if result.get("order_id"):
            p["order_id"] = result["order_id"]
        if result.get("link_url"):
            p["payment_link"] = result["link_url"]

    push_event(payment_id, "daemon_retry_executed", {
        "job_id": job_id,
        "action": action,
        "result_status": result.get("status", "unknown"),
        "message": result.get("message", ""),
        "order_id": result.get("order_id", ""),
        "link_url": result.get("link_url", ""),
        "source": source,
    })
    store.flush()

    # The retry is the LAST rung most cases reach, and this was where they
    # stopped. The daemon created the order, said so, and nothing watched it:
    # a customer who paid the retry order was never noticed, and one who did not
    # left the case sitting at `scheduled` for good — never recovered, never
    # escalated, never seen by anyone.
    #
    # Watching it closes both ends. If the money arrives, `_watch_for_recovery`
    # records it and tells the agent to stop. If it does not, the same watcher
    # hands the case back — and by this point the ladder is exhausted, so the
    # agent's escalate_to_human is finally permitted and the case reaches a
    # person instead of silence.
    # The agent asked to be woken and the wait has elapsed. Hand it back with
    # what it said it was waiting for, so it resumes its own plan instead of
    # re-deriving one — and so a case that deferred (quiet hours, a pending
    # response) cannot sit at `awaiting_customer` for ever.
    if result.get("status") == "woken":
        rec = store.get_payment(payment_id) or {}
        why = (result.get("reason")
               or (rec.get("waiting_for") or {}).get("reason")
               or "the wait you asked for")
        store.update_payment(payment_id, waiting_for=None)
        store.flush()
        _handoff_to_agent(
            payment_id,
            f"THE WAIT YOU ASKED FOR IS OVER. You said you were waiting for: "
            f"{why}. Check where the case stands now, then either take the next "
            f"rung or wait again with a fresh reason — do not repeat anything "
            f"already tried.",
            scenario=f"wait_elapsed:{job_id}",
        )

    if result.get("status") in ("retry_created", "link_created"):
        window = int(os.getenv("RETRY_WATCH_MINUTES", "30")) * 60
        socketio.start_background_task(_watch_for_recovery, payment_id, window)

    # The agent asked to be woken and the wait has elapsed. Hand it back with
    # what it said it was waiting for, so it resumes its own plan instead of
    # re-deriving one — and so a case that deferred (quiet hours, a pending
    # response) cannot sit at `awaiting_customer` for ever.
    if result.get("status") == "woken":
        rec = store.get_payment(payment_id) or {}
        why = (result.get("reason")
               or (rec.get("waiting_for") or {}).get("reason")
               or "the wait you asked for")
        store.update_payment(payment_id, waiting_for=None)
        store.flush()
        _handoff_to_agent(
            payment_id,
            f"THE WAIT YOU ASKED FOR IS OVER. You said you were waiting for: "
            f"{why}. Check where the case stands now, then either take the next "
            f"rung or wait again with a fresh reason — do not repeat anything "
            f"already tried.",
            scenario=f"wait_elapsed:{job_id}",
        )

    return jsonify({"status": "received", "payment_id": payment_id})


@app.route("/api/payments")
def api_payments():
    p_list = store.payment_values()

    # Money that came back is not at risk.
    #
    # This summed EVERY payment, recovered ones included, so the header read
    # INR 9,23,031 at risk while the batch view — which counts only what is
    # still out — read INR 5,36,960. Two numbers for the same thing on the same
    # screen, and the larger one was wrong.
    #
    # `recovered` is also the amount that actually arrived, not the amount that
    # was owed: a INR 2,499 order recovered at a 5% discount brought back INR
    # 2,374.05, and counting the full 2,499 quietly credits the agent with the
    # discount it gave away.
    def _settled(p):
        return p.get("status") == "recovered" or float(p.get("recovered_amount") or 0) > 0

    total_at_risk = sum(float(p.get("amount") or 0) for p in p_list if not _settled(p))
    # No fallback to the owed amount. A settled case without a recovered figure
    # is a data defect, and crediting it at face value hides both the defect and
    # any discount that was given away. Count it as zero and report it.
    total_recovered = sum(float(p.get("recovered_amount") or 0)
                          for p in p_list if _settled(p))
    unattributed = [p.get("payment_id") for p in p_list
                    if _settled(p) and not p.get("recovered_amount")]
    rec_count = sum(1 for p in p_list if _settled(p))
    rate = (rec_count / len(p_list) * 100) if p_list else 0.0
    total_penalties = sum(p.get("penalties_prevented", 0) for p in p_list)
    return jsonify({
        "payments": p_list,
        "total_at_risk": total_at_risk,
        "total_recovered": total_recovered,
        "unattributed_settled": unattributed,
        "recovery_rate": round(rate, 1),
        "total_penalties_prevented": total_penalties,
        "penalties_saved_usd": round(total_penalties * NETWORK_FINE_PER_ATTEMPT, 2),
        "penalties_saved_inr": round(total_penalties * 8.30, 2),
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "payments": len(store.all_payments())})

def main():
    port = int(os.getenv("FRONTEND_PORT", "5002"))
    autopilot_minutes = float(os.getenv("WAVE_AUTOPILOT_MINUTES", "0") or 0)
    if autopilot_minutes > 0:
        socketio.start_background_task(_wave_autopilot, autopilot_minutes)
        print(f"  Wave autopilot:     every {autopilot_minutes:g} min, "
              f"outside quiet hours")
    print(f"\n  Customer Checkout:  http://localhost:{port}/pay")
    print(f"  Merchant Dashboard: http://localhost:{port}/merchant")
    print(f"  Agent Flow:         http://localhost:{port}/graph\n")
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)

if __name__ == "__main__":
    main()
