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

from recovery_agent.razorpay_client import RazorpayClient
from recovery_agent.state_store import StateStore
import json
from langchain_core.messages import AIMessage, ToolMessage

# ── HITL Approval Gate for Voice Calls ──
# When agent decides VOICE_CALL, it blocks here until merchant approves or 60s timeout.
_pending_voice_approvals: dict[str, threading.Event] = {}  # payment_id → Event
_voice_call_approved: dict[str, bool] = {}                 # payment_id → True/False

# --- OpenTelemetry: Manual spans for coherent parent→child trace hierarchy ---
_otel_tracer = None

def _get_tracer():
    """Lazy-init a tracer for frontend agent spans (parent→child)."""
    global _otel_tracer
    if _otel_tracer is not None:
        return _otel_tracer
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        provider = TracerProvider()
        exporter = OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _otel_tracer = trace.get_tracer("recovery-agent")
    except Exception:
        # Graceful degradation — spans become no-ops
        from opentelemetry import trace
        _otel_tracer = trace.get_tracer("recovery-agent")
    return _otel_tracer

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
    while time.time() - start_ts < max_wait_seconds:
        time.sleep(15)
        try:
            # Fast path: check if webhook already handled it
            if store.has_payment(payment_id):
                p = store.get_payment(payment_id)
                if p.get("status") == "recovered":
                    return

            # Strategy 1: Fetch payment link by link_id stored in order_id
            p = store.get_payment(payment_id) if store.has_payment(payment_id) else {}
            link_id = p.get("order_id", "")
            if link_id and link_id.startswith("plink_"):
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
                except Exception as e:
                    pass  # Payment link fetch failed, try next strategy

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
    if prior and prior.get("action") != "superseded":
        return                          # first real outcome wins; never double-handle

    push = p.get("pending_push") or {}
    outcome = {
        "action": action,                       # acted | dismissed | ignored
        "seconds_shown": round(float(seconds_shown or 0), 1),
        "headline": push.get("headline", ""),
        "offer_text": push.get("offer_text", ""),
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

    if action == "superseded":
        # The agent replaced its own message. That is not the customer telling us
        # anything, so it is recorded but must not trigger another hand-off —
        # doing so would have the agent react to itself.
        return

    # Hand the observation to the agent. Deliberately NOT interpreted here — the
    # agent is told what the customer did and reasons about why, and about which
    # channel that implies. Encoding "dismissed => send email" in this function
    # would be putting the judgement back in the plumbing.
    _handoff_to_agent(
        payment_id,
        f"In-page notification outcome: the customer {action.upper()} it after "
        f"{outcome['seconds_shown']:.0f}s. Shown: {push.get('headline','(none)')!r}"
        + (f" with offer {push.get('offer_text')!r}" if push.get("offer_text") else " with no offer")
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
    offer = (p.get("ui_spec") or {}).get("offer") or {}
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
    pending_retry = ""
    if (p.get("scheduled_job") or {}).get("target_timestamp") and "link" in (how or ""):
        pending_retry = (" A background retry was scheduled and has NOT fired; it "
                         "is not what recovered this.")

    _handoff_to_agent(
        payment_id,
        f"RECOVERED. INR {amount:,.2f} is in "
        f"({recovery_payment_id or 'payment id unavailable'}), {seconds}s after "
        f"{what}{offer_note}. HOW IT ARRIVED: {arrival}.{pending_retry} "
        f"Attribute the win to that channel and nothing else — the lesson you "
        f"store here is permanent, and a wrong one will send the next recovery "
        f"down the wrong path. Close the case with close_case, outcome "
        f"'recovered'. Do NOT contact the customer again.",
        scenario="recovered",
    )


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
    sp.setdefault("trail", []).append({
        "step": "recovery_confirmed",
        "msg": f"Payment captured: INR {amount:,.2f}",
        "detail": f"{how}. Payment {rzp_payment_id}. After {seconds}s",
        "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
    })
    store.flush()
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
    """Revenue still at risk, sorted into batches that share a fix."""
    from recovery_agent.agent.classify import summarise
    batches = summarise(list(store.payment_values()))
    return jsonify({
        "batches": batches,
        "total_at_risk": round(sum(b["value"] for b in batches), 2),
        "total_cases": sum(b["count"] for b in batches),
        "running": sorted(active_agent_payments),
    })


#: How many cases one batch run will work. A batch can hold hundreds; a run that
#: quietly worked all of them would send hundreds of emails and burn the payment
#: link quota in one click. The cap is visible in the UI and in the response, so
#: a partial run never reads as a complete one.
BATCH_RUN_LIMIT = int(os.getenv("BATCH_RUN_LIMIT", "5"))


@app.route("/api/batches/<key>/run", methods=["POST"])
def api_run_batch(key: str):
    """Work a batch: one agent session per payment, never two on one case."""
    from recovery_agent.agent.classify import BATCH_BY_KEY, classify
    meta = BATCH_BY_KEY.get(key)
    if not meta:
        return jsonify({"error": f"unknown batch {key!r}"}), 404

    data = request.get_json(silent=True) or {}
    limit = max(1, min(int(data.get("limit") or BATCH_RUN_LIMIT), 50))

    candidates = [r for r in store.payment_values() if classify(r) == key]
    started, skipped = [], []
    for rec in candidates:
        pid = rec.get("payment_id") or ""
        if not pid:
            continue
        if len(started) >= limit:
            skipped.append({"payment_id": pid, "why": "over this run's limit"})
            continue
        # Sessions are per payment. A case the real-time agent is already
        # working must not get a second run — that is the one rule the whole
        # session model rests on.
        if pid in active_agent_payments:
            skipped.append({"payment_id": pid, "why": "already being worked"})
            continue
        customer = {k: v for k, v in (rec.get("customer") or {}).items() if v} or {
            k: v for k, v in {"email": rec.get("customer_email", ""),
                              "name": rec.get("customer_name", ""),
                              "contact": rec.get("customer_phone", "")}.items() if v}
        if not (customer.get("email") or customer.get("contact")):
            skipped.append({"payment_id": pid, "why": "no way to contact them"})
            continue

        # The batch is the context. The agent is told what this group has in
        # common and what that implies, so it does not re-derive the category
        # for every case — which is the whole reason for sorting them first.
        socketio.start_background_task(
            run_agent_for_payment, pid, float(rec.get("amount") or 0),
            f"BATCH: {meta['title']}. {meta['what']} "
            f"This payment is one of {len(candidates)} in that batch. Work it on "
            f"its own merits — the batch says what kind of problem it is, not "
            f"what to do about this particular customer.",
            customer, f"batch_{key}",
            rec.get("failure_code", "") or "",
        )
        started.append(pid)

    push_event("batch", "batch_started", {"batch": key, "started": len(started),
                                          "skipped": len(skipped)})
    return jsonify({"batch": key, "title": meta["title"],
                    "started": started, "skipped": skipped,
                    "limit": limit, "total_in_batch": len(candidates)})


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
    if not payment_id or action not in ("acted", "dismissed", "superseded"):
        return jsonify({
            "error": "payment_id and action (acted|dismissed|superseded) required"
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


def _run_agent_for_payment_inner(payment_id: str, amount: float, failure_reason: str, customer: dict, scenario_type: str = "standard", failure_code: str = "", error_source: str = "", error_step: str = ""):
    """Inner agent execution — wires real LangGraph ReAct agent, Razorpay SDK, and retry scheduler.

    The agent uses LLM reasoning to decide which tools to call at each step.
    """
    import json
    import uuid
    from recovery_agent.agent.guardrails import GuardrailEngine
    from recovery_agent.agent.memory import CustomerMemoryStore


    from recovery_agent.models import Case, CaseStatus, PaymentEvent, GenerativeUISpec
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

    with tracer.start_as_current_span("agent_recovery", attributes=parent_attrs) as parent_span:
      memory_store = CustomerMemoryStore()
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

      def emit_thought(step: str, thought: str, detail: str = "", tool_call: dict = None, guardrail: dict = None, memory: dict = None, ui_morph: str = None, ui_spec: dict = None):
          entry = {
              "step": step,
              "msg": thought,
              "detail": detail,
              "tool_call": tool_call,
              "guardrail": guardrail,
              "memory": memory,
              "ui_morph": ui_morph,
              "ui_spec": ui_spec,
              "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
          }
          trail.append(entry)
          store.set_trail(payment_id, trail)
          if store.has_payment(payment_id):
              store.get_payment(payment_id)["trail"] = list(trail)
          push_event(payment_id, step, entry)

      # ── INITIALIZE CASE ──
      with tracer.start_as_current_span("init_case") as init_span:
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
      with tracer.start_as_current_span("graph_execution") as graph_span:
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
        except Exception as e:
            # Graph error or LLM unavailable — surface the real error
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
      with tracer.start_as_current_span("act") as act_span:
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
        #
        # This used to walk the tool names in reverse and take the first match,
        # so a run that created a link and *then* sent a notification picked the
        # notification's result — which has no link_url. `order_id` came out
        # empty and `_watch_for_recovery` never started, so a customer who paid
        # was never noticed. Choose by meaning, not by call order, and always
        # prefer the payment link as the receipt when one was created.
        def _ok(name):
            r = (tool_calls_made.get(name) or {}).get("result") or {}
            return r if isinstance(r, dict) and r.get("status") in (
                "ok", "scheduled", "delivered") else {}

        push_res = _ok("send_page_push")
        link_res = _ok("generate_recovery_payment_link")
        retry_res = _ok("retry_in_hours") or _ok("schedule_retry")
        notify_res = _ok("send_recovery_notification")
        escalate_res = (tool_calls_made.get("escalate_to_human") or {}).get("result") or {}
        close_res = (tool_calls_made.get("close_case") or {}).get("result") or {}
        if close_res.get("status") == "closed":
            # The run that closes a case reported "Primary action: none",
            # because closing was not an action anything recognised.
            action_val, sdk_res = "close_case", close_res

        if close_res.get("status") == "closed":
            pass                        # already chosen above; it outranks the rest
        elif escalate_res:
            action_val, sdk_res = "escalate_to_human", escalate_res
        elif push_res and not (link_res or notify_res):
            # A push on its own is the silent first rung. The case is waiting on
            # the customer's response, not finished — treating it as "no action"
            # marked the case escalated, which then blocked the follow-up
            # hand-off when they dismissed it.
            action_val, sdk_res = "page_push", push_res
        elif link_res or notify_res:
            # The link is what the customer acts on, so it is the receipt even
            # when a notification was the last thing sent.
            action_val, sdk_res = "send_notification", (link_res or notify_res)
        elif retry_res:
            action_val, sdk_res = "wait_and_retry", retry_res
        else:
            # No recovery action was taken this turn. That is NOT the same as
            # giving up: the closing run after a successful payment calls only
            # manage_memory, and this branch used to read that as
            # "escalate_to_human" — so a case that had just recovered INR 47,481
            # was filed as a ticket asking a human to chase the customer for
            # non-payment. A fallback should be the least consequential option
            # available, never the most severe one.
            action_val, sdk_res = "none", {}

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

        from recovery_agent.models import GenerativeUISpec
        ui_type_map = {
            "card_expired": "CARD_EXPIRY_FIXER",
            "network_timeout": "SMART_FAILOVER_BANNER",
            "bank_declined": "BANK_DECLINED_RECOVERY",
            "insufficient_funds": "INSUFFICIENT_FUNDS_SCHEDULER",
            "mandate_revoked": "MANDATE_REAUTH_MODAL",
            "risk_block": "RISK_VERIFICATION_FLOW",
        }
        ui_spec_dict = GenerativeUISpec(
            ui_type=ui_type_map.get(cause, "PAYMENT_LINK_MODAL"),
            headline=agent_final_text[:120] if agent_final_text else f"Payment of INR {amount:,.2f} needs attention",
            subtext=f"Recovery strategy: {action_val.replace('_', ' ').title()}",
            primary_cta_text="Complete Payment" if action_val != "escalate_to_human" else "Contact Support",
            discount_incentive="",
            target_rail=case.payment.metadata.get("recommended_rail", "payment_link"),
            hinglish_voice_script=f"Namaste {customer.get('name', '')} ji! Aapka payment fail ho gaya hai. Hum aapki help karenge.",
            tone="supportive",
        )
        emit_thought(
            step="ui_spec",
            thought=f"UI Spec: {ui_spec_dict.ui_type} ({ui_spec_dict.tone} tone)",
            detail=f"Headline: {ui_spec_dict.headline}",
            ui_morph=ui_spec_dict.ui_type,
            ui_spec=ui_spec_dict.model_dump(),
        )

        emit_thought(
            step="acting",
            thought=f"Agent executed: {tool_name_str}",
            detail=f"Primary action: {action_val}",
            tool_call={
                "tool": tool_name_str,
                "args": {"payment_id": payment_id, "amount_in_paise": int(amount * 100), "currency": "INR", "customer": customer_email},
                "raw_razorpay_response": sdk_res,
            },
            ui_morph=ui_spec_dict.ui_type,
            ui_spec={
                **ui_spec_dict.model_dump(),
                "recovery_tier": recovery_tier,
                "decline_strategy": cause,
                "penalties_prevented": case.penalties_prevented,
                "scheduled_job": sdk_res if action_val == "wait_and_retry" else None,
            },
        )

        socketio.emit("ui_spec_overlay", {
            "payment_id": payment_id,
            **ui_spec_dict.model_dump(),
            "recovery_tier": recovery_tier,
            "decline_strategy": cause,
        })

        # ── OBSERVE & RECOVER ──
        if action_val == "wait_and_retry":
            emit_thought(
                step="stopping",
                thought=f"Background Retry Scheduled: {sdk_res.get('job_id', 'N/A')}",
                detail=f"Target: {sdk_res.get('target_timestamp', 'N/A')} | Confidence: {sdk_res.get('confidence', 0):.0%} | Reason: {sdk_res.get('reason', '')}",
                ui_morph="SCHEDULED_RETRY",
            )
        elif action_val == "close_case":
            emit_thought(
                step="stopping",
                thought=f"Case closed: {sdk_res.get('outcome', 'unknown').upper()}"
                        + (f" — INR {float(sdk_res.get('amount_recovered') or 0):,.2f} recovered"
                           if sdk_res.get("outcome") == "recovered" else ""),
                detail=(tool_calls_made.get("close_case", {}).get("args", {})
                        .get("what_happened", "")),
                ui_morph="RECOVERED" if sdk_res.get("outcome") == "recovered" else "ESCALATED",
            )
        elif action_val == "escalate_to_human":
            emit_thought(
                step="stopping",
                thought=f"Escalated to Human: {sdk_res.get('ticket_id', 'N/A')}",
                detail=f"Reason: {sdk_res.get('reason', failure_reason)}",
                ui_morph="ESCALATED",
            )
        elif action_val == "send_notification" and sdk_res.get("link_url"):
            store.save_pending(payment_id, {
                "action": action_val,
                "execution": sdk_res,
                "attempt": 0,
                "trail": trail,
                "amount": amount,
            })
            push_event(payment_id, "waiting_for_customer", {"action": action_val, "detail": f"Payment link sent: {sdk_res['link_url']}", "ui_morph": ui_spec_dict.ui_type})
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
        p["ui_spec"] = ui_spec_dict.model_dump()
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

        # Skip when the agent already escalated through its own tool — that call
        # queued a ticket with the full case context. This safety net exists for
        # cases that simply ran out of road, not to second-guess the agent, and
        # firing both produced two tickets for one payment 29 seconds apart.
        agent_already_escalated = bool(escalate_res)

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
.discount-banner{background:linear-gradient(135deg,rgba(16,185,129,0.08),rgba(16,185,129,0.04));border:1px solid rgba(16,185,129,0.15);border-radius:10px;padding:12px;margin-bottom:16px;font-size:13px;color:#059669;font-weight:600;text-align:center;display:none;animation:fadeIn .3s}
.discount-banner.visible{display:block}
.action-box{background:linear-gradient(135deg,rgba(59,130,246,0.06),rgba(59,130,246,0.02));border:1px solid rgba(59,130,246,0.12);border-radius:14px;padding:20px;margin-top:16px;display:none}
.action-box.visible{display:block;animation:slideUp .3s ease-out}
.action-box h4{color:var(--accent);font-size:14px;margin-bottom:6px;font-weight:600}
.action-box p{color:var(--muted);font-size:13px;margin-bottom:14px;line-height:1.5}
.btn-respond{background:linear-gradient(135deg,var(--amber),#d97706);color:#fff;border:none;padding:12px 20px;border-radius:10px;font-size:13px;font-weight:600;cursor:pointer;transition:all .2s;width:100%;font-family:inherit}
.btn-respond:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(245,158,11,0.3)}

.ui-overlay{position:fixed;top:20px;right:20px;max-width:380px;background:rgba(255,255,255,0.95);backdrop-filter:blur(20px);border:1px solid rgba(0,0,0,0.08);border-radius:16px;padding:24px;box-shadow:0 20px 60px rgba(0,0,0,0.15);z-index:2147483647;display:none;animation:slideUp .4s ease-out;font-family:'Inter',sans-serif}
.ui-overlay.visible{display:block}
.ui-overlay-headline{font-size:18px;font-weight:700;color:var(--text);margin-bottom:8px;letter-spacing:-0.02em}
.ui-overlay-subtext{font-size:13px;color:var(--muted);line-height:1.6;margin-bottom:16px}
.ui-overlay-cta{display:inline-block;background:linear-gradient(135deg,var(--accent),var(--accent-hover));color:#fff;padding:10px 20px;border-radius:10px;font-size:14px;font-weight:600;text-decoration:none;transition:all .2s}
.ui-overlay-cta:hover{transform:translateY(-1px);box-shadow:0 6px 16px rgba(37,99,235,0.3)}
.ui-overlay-discount{background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.15);border-radius:8px;padding:8px 12px;margin-top:12px;font-size:12px;color:#059669;font-weight:600}
.ui-overlay-tier{display:inline-block;padding:3px 8px;border-radius:6px;font-size:10px;font-weight:600;margin-top:8px}
.ui-overlay-tier.silent{background:rgba(59,130,246,0.1);color:#2563eb}
.ui-overlay-tier.active{background:rgba(245,158,11,0.1);color:#d97706}

@media(max-width:768px){.product-grid{grid-template-columns:repeat(2,1fr);gap:12px}.header{padding:0 16px}.nav-links{display:none}.search-box{width:160px}.store-banner{padding:28px 24px}.store-banner h1{font-size:22px}}
@media(max-width:480px){.product-grid{grid-template-columns:1fr}.cart-sidebar{width:100%}}
</style>
</head>
<body>

<div class="ui-overlay" id="ui-overlay">
  <div class="ui-overlay-headline" id="overlay-headline"></div>
  <div class="ui-overlay-subtext" id="overlay-subtext"></div>
  <div id="overlay-cta-wrap"><a class="ui-overlay-cta" id="overlay-cta" href="#">Complete Payment</a></div>
  <div class="ui-overlay-discount" id="overlay-discount" style="display:none"></div>
  <div class="ui-overlay-tier" id="overlay-tier" style="display:none"></div>
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
    <div class="discount-banner" id="discount-banner"></div>
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
      showStatus("failed", "Payment cancelled. Our agent will help you recover it.");
      btn.disabled = false; btn.innerHTML = "Retry Payment";
      triggerRecovery({code:"customer_cancelled", reason:"Payment cancelled by customer", source:"customer", step:"payment_processing"});
    }}});
    rzp.on("payment.failed", function(r) {
      showStatus("failed", "Payment couldn't be processed: " + r.error.description);
      btn.disabled = false; btn.innerHTML = "Retry Payment";
      showDeclineWall(r.error.code || "failed", r.error.description);
      triggerRecovery({code:r.error.code || "technical_error", reason:r.error.description || r.error.reason || "Payment failed", source:r.error.source || "gateway", step:r.error.step || "payment_processing"});
    });
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

function triggerRecovery(err) {
  const total = getCartTotal();
  const cust = getCustomerData();
  fetch("/api/payment-failed", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({payment_id:paymentId, amount:total, failure_code:err.code||"technical_error", failure_reason:err.reason||"Payment failed", error_source:err.source||"gateway", error_step:err.step||"payment_processing", customer:cust})});
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

socket.on("agent_push", function(d) {
  if (d.payment_id !== paymentId) return;
  var esc = function (t) {
    return String(t == null ? "" : t).replace(/[&<>"']/g, function (c) {
      return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c];
    });
  };
  const old = document.getElementById("agent-push");
  if (old) {
    /* A newer notification is taking its place. Removing it silently made it
       look like the message disappeared on its own after a few seconds, and the
       customer's non-response to it was lost — so report it honestly first. */
    reportPush("superseded", "Replaced by a newer notification.");
    old.remove();
  }
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
      "background:linear-gradient(90deg,#065f46,#047857);color:#fff;padding:11px 16px;" +
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
      '<span id="offer-countdown" style="opacity:.85;font-size:12px"></span>';
    document.body.appendChild(bar);
    document.body.style.paddingTop = "44px";

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
    (d.offer_text ? '<div style="margin-top:10px;background:#ecfdf5;border:1px solid #a7f3d0;' +
       'color:#065f46;border-radius:8px;padding:8px 10px;font-size:12.5px;font-weight:600">' +
       esc(d.offer_text) + '</div>' : '') +
    '<button id="push-cta" style="margin-top:13px;width:100%;background:#2563eb;color:#fff;' +
      'border:0;border-radius:9px;padding:11px;font-size:14px;font-weight:600;cursor:pointer">' +
      esc(d.cta_text || "Complete payment") + '</button>' +
    '<div id="push-countdown" style="margin-top:8px;font-size:11.5px;color:#9ca3af;text-align:center"></div>';
  document.body.appendChild(wrap);

  document.getElementById("push-x").onclick = function () {
    closePush("dismissed", "Customer closed the notification.");
  };
  document.getElementById("push-cta").onclick = function () {
    reportPush("acted", "Customer clicked the call to action.");
    if (d.payment_link) window.open(d.payment_link, "_blank");
    else startPayment();
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
});

socket.on("agent_event", function(data) {
  if (data.payment_id !== paymentId) return;
  if (data.event === "waiting_for_customer") {
    const box = document.getElementById("action-box");
    box.classList.add("visible");
    document.getElementById("action-title").textContent = "Agent: " + (data.action || "took action");
    document.getElementById("action-detail").textContent = data.detail || "Please respond to continue recovery.";
    showStatus("waiting", "Agent is waiting for your response...");
  }
  if (data.event === "acting" && data.ui_spec) { applyGenerativeUISpec(data.ui_spec); }
  if (data.event === "complete") {
    const s = document.getElementById("status");
    document.getElementById("action-box").classList.remove("visible");
    document.getElementById("decline-wall").classList.remove("visible");
    if (data.status === "recovered") { s.className = "status-bar active s-success"; s.innerHTML = "Payment recovered! Thank you."; }
    else { s.className = "status-bar active s-failed"; s.innerHTML = "Could not recover automatically. Please try again or update your payment method."; }
  }
});

socket.on("ui_spec_overlay", function(data) {
  if (data.payment_id !== paymentId) return;
  const overlay = document.getElementById("ui-overlay");
  if (!overlay) return;
  if (data.headline) document.getElementById("overlay-headline").textContent = data.headline;
  if (data.subtext) document.getElementById("overlay-subtext").textContent = data.subtext;
  if (data.primary_cta_text) {
    const cta = document.getElementById("overlay-cta");
    cta.textContent = data.primary_cta_text;
    cta.onclick = function(e) { e.preventDefault(); overlay.classList.remove("visible"); startPayment(); };
  }
  if (data.discount_incentive) { const d = document.getElementById("overlay-discount"); d.textContent = data.discount_incentive; d.style.display = "block"; }
  if (data.recovery_tier) { const t = document.getElementById("overlay-tier"); t.textContent = data.recovery_tier.toUpperCase(); t.className = "ui-overlay-tier " + data.recovery_tier; t.style.display = "inline-block"; }
  overlay.classList.add("visible");
  setTimeout(function() { overlay.classList.remove("visible"); }, 8000);
});

function applyGenerativeUISpec(spec) {
  if (!spec) return;
  if (spec.headline) { document.querySelector(".checkout-title").textContent = spec.headline; }
  if (spec.subtext) { document.querySelector(".checkout-subtitle").textContent = spec.subtext; }
  if (spec.primary_cta_text) { const btn = document.getElementById("pay-btn"); if (btn) { btn.textContent = spec.primary_cta_text; } }
  if (spec.discount_incentive) { const inc = document.getElementById("discount-banner"); inc.textContent = spec.discount_incentive; inc.classList.add("visible"); }
  if (spec.recovery_tier) {
    const tierLabel = spec.recovery_tier === "silent" ? "Background Retry Active" : "Recovery in Progress";
    showStatus("recovering", tierLabel + " — " + (spec.decline_strategy || "Agent working on it"));
  }
}

/* ── Init ── */
renderProducts();
</script>
</body></html>"""


# ─── Merchant Dashboard ───────────────────────────────────────
MERCHANT_PAGE = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Revenue Recovery — Agent Studio</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<style>
:root{
  --bg-canvas:#FFFFFF;--bg-surface:#F7F8FA;--bg-card:#FFFFFF;
  --border-subtle:#E5E7EB;--border-hover:#D1D5DB;
  --text-primary:#1A1A2E;--text-secondary:#6B7280;--text-muted:#9CA3AF;
  --brand-blue:#2563EB;--brand-blue-light:#EFF6FF;
  --success:#16A34A;--success-light:#F0FDF4;--success-border:#BBF7D0;
  --error:#DC2626;--error-light:#FEF2F2;--error-border:#FECACA;
  --warning:#F59E0B;--warning-light:#FFFBEB;--warning-border:#FDE68A;
  --accent-teal:#0EA5E9;
  --sidebar-bg:#F7F8FA;--sidebar-active:#EFF6FF;--sidebar-active-text:#2563EB;
  --card-shadow:0 1px 3px rgba(0,0,0,0.04),0 1px 2px rgba(0,0,0,0.02);
  --card-shadow-hover:0 4px 12px rgba(0,0,0,0.06);
}
[data-theme="dark"]{
  --bg-canvas:#020617;--bg-surface:#0F172A;--bg-card:rgba(15,23,42,0.7);
  --border-subtle:rgba(255,255,255,0.06);--border-hover:rgba(255,255,255,0.1);
  --text-primary:#F8FAFC;--text-secondary:#94A3B8;--text-muted:#475569;
  --brand-blue:#3B82F6;--brand-blue-light:rgba(59,130,246,0.1);
  --success:#10B981;--success-light:rgba(16,185,129,0.1);--success-border:rgba(16,185,129,0.2);
  --error:#EF4444;--error-light:rgba(239,68,68,0.1);--error-border:rgba(239,68,68,0.2);
  --warning:#F59E0B;--warning-light:rgba(245,158,11,0.1);--warning-border:rgba(245,158,11,0.2);
  --sidebar-bg:#0F172A;--sidebar-active:rgba(59,130,246,0.1);--sidebar-active-text:#60A5FA;
  --card-shadow:0 1px 3px rgba(0,0,0,0.2);--card-shadow-hover:0 4px 12px rgba(0,0,0,0.3);
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg-canvas);color:var(--text-primary);letter-spacing:-0.011em;min-height:100vh;display:flex;transition:background .3s,color .3s}

/* Sidebar */
.sidebar{width:220px;background:var(--sidebar-bg);border-right:1px solid var(--border-subtle);display:flex;flex-direction:column;position:fixed;top:0;left:0;bottom:0;z-index:50;transition:background .3s}
.sidebar-logo{padding:20px 20px 16px;border-bottom:1px solid var(--border-subtle);display:flex;align-items:center;gap:10px}
.sidebar-logo-icon{width:32px;height:32px;background:var(--brand-blue);border-radius:6px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:14px;font-weight:700}
.sidebar-logo-text{font-size:14px;color:var(--text-primary)}
.sidebar-logo-sub{font-size:10px;color:var(--text-muted);font-weight:400;margin-top:1px}
.sidebar-nav{flex:1;padding:12px 8px;overflow-y:auto}
.nav-section{font-size:10px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.06em;padding:8px 12px 4px;margin-top:8px}
.nav-item{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:8px;font-size:13px;font-weight:500;color:var(--text-secondary);text-decoration:none;transition:all .15s;cursor:pointer;border-bottom:2px solid transparent}
.nav-item:hover{background:var(--brand-blue-light);color:var(--text-primary);border-bottom-color:var(--brand-blue)}
.nav-item.active{background:var(--sidebar-active);color:var(--sidebar-active-text);font-weight:600}
.nav-item .nav-icon{width:18px;text-align:center;font-size:14px;flex-shrink:0}
.nav-badge{font-size:9px;padding:2px 6px;border-radius:4px;font-weight:600;background:var(--brand-blue);color:#fff;margin-left:auto}
.sidebar-footer{padding:12px 16px;border-top:1px solid var(--border-subtle);font-size:11px;color:var(--text-muted)}

/* Main Area */
.main{margin-left:220px;flex:1;display:flex;flex-direction:column;min-height:100vh;position:relative}
.main::before{content:'';position:fixed;top:0;left:220px;right:0;bottom:0;background:radial-gradient(circle at 20% 50%,rgba(59,130,246,0.03) 0%,transparent 50%),radial-gradient(circle at 80% 20%,rgba(16,185,129,0.03) 0%,transparent 50%);pointer-events:none;z-index:0}
.topbar{height:56px;background:var(--bg-canvas);border-bottom:1px solid var(--border-subtle);display:flex;justify-content:space-between;align-items:center;padding:0 28px;position:sticky;top:0;z-index:40;transition:background .3s}
.topbar-left{display:flex;align-items:center;gap:16px}
.breadcrumb{font-size:13px;color:var(--text-secondary)}
.breadcrumb span{color:var(--text-primary);font-weight:600}
.topbar-right{display:flex;align-items:center;gap:12px}
.topbar-search{background:var(--bg-surface);border:1px solid var(--border-subtle);border-radius:8px;padding:7px 14px 7px 32px;font-size:13px;color:var(--text-primary);width:220px;outline:none;transition:border-color .15s}
.topbar-search:focus{border-color:var(--brand-blue)}
.search-wrap{position:relative}
.search-wrap::before{content:"\\1F50D";position:absolute;left:10px;top:50%;transform:translateY(-50%);font-size:12px;z-index:1}
.theme-toggle{width:36px;height:36px;border-radius:8px;border:1px solid var(--border-subtle);background:var(--bg-surface);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:16px;transition:all .15s}
.theme-toggle:hover{border-color:var(--brand-blue);background:var(--brand-blue-light)}

/* Content */
.content{padding:24px 28px;flex:1}
.agent-hero{background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:14px;padding:24px;margin-bottom:24px;box-shadow:var(--card-shadow);display:flex;justify-content:space-between;align-items:center;transition:all .2s}
.agent-hero:hover{box-shadow:var(--card-shadow-hover)}
.agent-identity{display:flex;align-items:center;gap:14px}
.agent-icon{width:44px;height:44px;background:var(--brand-blue);border-radius:12px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:22px}
.agent-name{font-size:18px;font-weight:700;color:var(--text-primary)}
.agent-subtitle{font-size:13px;color:var(--text-secondary);margin-top:2px}
.health-badge{display:flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:var(--success);padding:6px 14px;border-radius:20px;background:var(--success-light);border:1px solid var(--success-border)}
.health-dot{width:7px;height:7px;background:var(--success);border-radius:50%;animation:blink 1.5s infinite}
.scenario-triggers{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.scenario-triggers button{padding:6px 14px;border-radius:8px;font-size:12px;font-weight:500;cursor:pointer;transition:all .15s;font-family:inherit}
.scenario-trigger{background:transparent;border:1px solid var(--border-subtle);color:var(--text-secondary)}
.scenario-trigger:hover{border-color:var(--brand-blue);color:var(--brand-blue);background:var(--brand-blue-light)}
.scenario-trigger:active{transform:scale(0.97)}
.scenario-batch{background:var(--brand-blue);border:1px solid var(--brand-blue);color:#fff}
.scenario-batch:hover{background:#1d4ed8}

/* Tabs */
.tabs{display:flex;gap:0;border-bottom:2px solid var(--border-subtle);margin-bottom:24px}
.tab{padding:10px 20px;font-size:13px;font-weight:500;color:var(--text-secondary);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-2px;transition:all .15s}
.tab:hover{color:var(--text-primary)}
.tab.active{color:var(--brand-blue);border-bottom-color:var(--brand-blue);font-weight:600}

/* Metrics Row */
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}
.metric{background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:12px;padding:20px 16px;box-shadow:var(--card-shadow);transition:all .2s}
.metric:hover{box-shadow:var(--card-shadow-hover);border-color:var(--border-hover)}
.metric-value{font-size:2em;font-weight:700;color:var(--brand-blue);letter-spacing:-0.02em;line-height:1}
.metric-value.sv{color:var(--success)}.metric-value.wv{color:var(--warning)}.metric-value.rv{color:var(--error)}
.metric-label{color:var(--text-muted);font-size:11px;margin-top:4px;font-weight:500;text-transform:uppercase;letter-spacing:0.04em}
.metric:nth-child(1){border-left:3px solid var(--brand-blue)}
.metric:nth-child(2){border-left:3px solid var(--success)}
.metric:nth-child(3){border-left:3px solid var(--error)}
.metric:nth-child(4){border-left:3px solid var(--warning)}

/* Grid layouts */
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px}

/* Card */
.card{background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:14px;padding:20px;box-shadow:var(--card-shadow);transition:all .2s}
.card:hover{box-shadow:var(--card-shadow-hover)}
.card h2{font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:14px;font-weight:600}

/* Activity Feed */
.activity-feed{max-height:600px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border-subtle) transparent;scroll-behavior:smooth}
.activity-item{display:flex;gap:12px;padding:14px 8px;border-bottom:1px solid var(--border-subtle);font-size:13px;transition:all .2s;cursor:pointer;position:relative}
.activity-item::before{content:'';position:absolute;left:24px;top:46px;bottom:-1px;width:2px;background:var(--border-subtle)}
.activity-item:last-child::before{display:none}
.activity-item:hover{background:var(--bg-surface)}
.activity-icon{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0}
.activity-icon.recovering{background:var(--warning-light);color:var(--warning)}
.activity-icon.recovered{background:var(--success-light);color:var(--success)}
.activity-icon.failed{background:var(--error-light);color:var(--error)}
.activity-icon.escalated{background:var(--brand-blue-light);color:var(--brand-blue)}
.activity-icon.scheduled{background:var(--brand-blue-light);color:var(--brand-blue)}
.activity-body{flex:1;min-width:0}
.activity-title{font-weight:600;color:var(--text-primary);display:flex;align-items:center;gap:8px}
.activity-title .live{color:var(--success);font-size:10px;font-weight:600}
.activity-meta{color:var(--text-secondary);font-size:12px;margin-top:3px;display:flex;gap:12px;align-items:center}
.activity-amount{font-weight:600;color:var(--text-primary)}

/* Table */
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:10px 14px;border-bottom:1px solid var(--border-subtle);font-size:12px}
th{color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:0.05em;font-weight:600}
tr{transition:background .15s}
tr:hover{background:var(--bg-surface)}

/* Badges */
.badge{display:inline-block;padding:3px 10px;border-radius:8px;font-size:10px;font-weight:600;letter-spacing:0.02em}
.bs{background:var(--success-light);color:var(--success);border:1px solid var(--success-border)}
.bf{background:var(--error-light);color:var(--error);border:1px solid var(--error-border)}
.bp{background:var(--brand-blue-light);color:var(--brand-blue);border:1px solid rgba(59,130,246,0.2)}
.bw{background:var(--warning-light);color:var(--warning);border:1px solid var(--warning-border)}
.btier-silent{background:var(--brand-blue-light);color:var(--brand-blue);border:1px solid rgba(59,130,246,0.2)}
.btier-active{background:var(--warning-light);color:var(--warning);border:1px solid var(--warning-border)}
.btier-hard{background:var(--error-light);color:var(--error);border:1px solid var(--error-border)}

/* Strategy items */
.strategy-item{display:flex;justify-content:space-between;align-items:center;padding:10px 12px;border-bottom:1px solid var(--border-subtle);font-size:12px}
.strategy-code{color:var(--brand-blue);font-weight:600;min-width:60px;font-family:'SF Mono',SFMono-Regular,monospace;font-size:11px}
.strategy-name{color:var(--text-primary);flex:1}
.strategy-tier{font-size:10px;padding:3px 8px;border-radius:6px;font-weight:600}

/* Penalty counter */
.penalty-counter{text-align:center;padding:20px}
.penalty-big{font-size:2.2em;font-weight:700;color:var(--success);letter-spacing:-0.02em}
.penalty-sub{color:var(--text-muted);font-size:12px;margin-top:6px;font-weight:500}
.penalty-usd{color:var(--brand-blue);font-size:14px;margin-top:10px;font-weight:600}

/* Trail */
.trail{max-height:500px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border-subtle) transparent}
.trail-item{padding:12px;border-left:3px solid var(--border-subtle);margin-bottom:4px;border-radius:0 10px 10px 0;background:var(--bg-surface);font-size:12px;transition:background .15s}
.trail-item:hover{background:var(--brand-blue-light)}
.trail-item.t-detecting{border-left-color:var(--brand-blue)}.trail-item.t-diagnosing,.trail-item.t-diagnosed{border-left-color:#8B5CF6}.trail-item.t-deciding{border-left-color:var(--warning)}.trail-item.t-acting,.trail-item.t-acted{border-left-color:var(--success)}.trail-item.t-waiting{border-left-color:var(--warning);background:var(--warning-light)}.trail-item.t-observed{border-left-color:#6366F1}.trail-item.t-stopping{border-left-color:var(--error)}.trail-item.t-init{border-left-color:var(--brand-blue)}.trail-item.t-harness_start{border-left-color:#8B5CF6}.trail-item.t-harness_turn{border-left-color:#6366F1}.trail-item.t-llm_unavailable{border-left-color:var(--error);background:rgba(220,38,38,0.05)}.trail-item.t-stopped{border-left-color:var(--error)}.trail-item.t-ui_spec{border-left-color:#10B981}.trail-item.t-approval_needed{border-left-color:var(--warning);background:var(--warning-light)}.trail-item.t-approval_denied{border-left-color:var(--error)}.trail-item.t-waiting_for_customer{border-left-color:var(--warning)}.trail-item.t-complete{border-left-color:var(--success)}
.trail-time{color:var(--text-muted);font-size:10px;font-weight:500}.trail-msg{color:var(--text-primary);margin-top:3px;font-weight:500}.trail-detail{color:var(--text-secondary);margin-top:3px;font-size:11px;line-height:1.5}

/* Toast */
.toast{position:fixed;top:20px;right:20px;background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:12px;padding:12px 18px;font-size:12px;z-index:1000;transform:translateX(120%);transition:transform .3s cubic-bezier(.4,0,.2,1);font-weight:500;box-shadow:var(--card-shadow-hover)}
.toast.show{transform:translateX(0)}.toast.success{border-left:3px solid var(--success)}.toast.info{border-left:3px solid var(--brand-blue)}

/* Empty state */
.empty{text-align:center;padding:40px;color:var(--text-muted);font-size:13px}

/* Case detail drawer */
.drawer-overlay{position:fixed;inset:0;background:rgba(0,0,0,0.3);z-index:200;opacity:0;pointer-events:none;transition:opacity .25s}
.drawer-overlay.open{opacity:1;pointer-events:all}
.drawer{position:fixed;top:0;right:0;width:600px;height:100vh;background:var(--bg-canvas);border-left:1px solid var(--border-subtle);z-index:201;transform:translateX(100%);transition:transform .3s cubic-bezier(.4,0,.2,1);overflow-y:auto;display:flex;flex-direction:column;scroll-behavior:smooth}
.drawer.open{transform:translateX(0)}
.drawer-header{padding:20px 24px;border-bottom:1px solid var(--border-subtle);display:flex;justify-content:space-between;align-items:center}
.drawer-title{font-size:16px;font-weight:700;color:var(--text-primary)}
.drawer-close{width:32px;height:32px;border-radius:8px;border:1px solid var(--border-subtle);background:var(--bg-surface);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:16px;transition:all .15s}
.drawer-close:hover{background:var(--error-light);border-color:var(--error);color:var(--error)}
.drawer-body{padding:20px 24px;flex:1}
.drawer-section{margin-bottom:20px}
.drawer-section h3{font-size:12px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.04em;margin-bottom:10px}
.drawer-trail{max-height:400px;overflow-y:auto}
.drawer-trail .trail-item{margin-bottom:6px}

@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes fadeIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:translateY(0)}}
</style>
</head>
<body>

<!-- Sidebar Navigation -->
<aside class="sidebar">
  <div class="sidebar-logo">
    <div class="sidebar-logo-icon">R</div>
    <div>
      <div class="sidebar-logo-text">AutoRecover</div>
      <div class="sidebar-logo-sub">Agent Studio</div>
    </div>
  </div>
  <nav class="sidebar-nav">
    <div class="nav-section">Payment Products</div>
    <a class="nav-item" href="/merchant"><span class="nav-icon">&#9673;</span> Dashboard<span class="nav-badge">Live</span></a>
    <a class="nav-item" href="/pay" target="_blank"><span class="nav-icon">&#9671;</span> Store</a>
    <a class="nav-item" href="/graph" target="_blank"><span class="nav-icon">&#10230;</span> Agent Flow</a>
    <div class="nav-section">Agent Studio</div>
    <a class="nav-item active" href="/merchant"><span class="nav-icon" style="width:18px;height:18px;background:var(--brand-blue);border-radius:50%;display:inline-flex;align-items:center;justify-content:center;color:#fff;font-size:10px">&#10003;</span> Recovery Agent<span class="nav-badge" style="background:var(--success)">Active</span></a>
    <a class="nav-item" href="#" onclick="return false"><span class="nav-icon">&#8801;</span> Transactions</a>
    <a class="nav-item" href="#" onclick="return false"><span class="nav-icon">$</span> Settlements</a>
    <a class="nav-item" href="#" onclick="return false"><span class="nav-icon">&#128279;</span> Payment Links</a>
    <div class="nav-section">Settings</div>
    <a class="nav-item" href="#" onclick="return false"><span class="nav-icon">&#9881;</span> Configuration</a>
    <a class="nav-item" href="#" onclick="return false"><span class="nav-icon">&#8984;</span> API Keys</a>
  </nav>
  <div class="sidebar-footer">AutoRecover v2.0 — Buildathon</div>
</aside>

<!-- Main Content -->
<div class="main">
  <header class="topbar">
    <div class="topbar-left">
      <div class="breadcrumb">Agent Studio / <span>Recovery Agent</span></div>
    </div>
    <div class="topbar-right">
      <div class="search-wrap"><span style="position:absolute;left:10px;top:50%;transform:translateY(-50%);font-size:12px;color:var(--text-muted);z-index:1">&#128269;</span><input class="topbar-search" placeholder="Search payments..." /></div>
      <button class="theme-toggle" onclick="toggleTheme()" title="Toggle theme">🌓</button>
      <a href="/pay" target="_blank" style="text-decoration:none"><button style="padding:7px 16px;border-radius:8px;border:1px solid var(--brand-blue);background:var(--brand-blue);color:#fff;font-size:12px;font-weight:600;cursor:pointer">Open Store</button></a>
    </div>
  </header>

  <div class="content">
    <!-- Agent Hero Card -->
    <div class="agent-hero">
      <div class="agent-identity">
        <div class="agent-icon">🤖</div>
        <div>
          <div class="agent-name">Payment Recovery Agent</div>
          <div class="agent-subtitle">Recovering failed payments with AI-powered multi-turn reasoning</div>
        </div>
      </div>
      <div class="health-badge"><span class="health-dot"></span> Healthy &amp; Active</div>
    </div>
    <div class="scenario-triggers">
      <button class="scenario-trigger" onclick="simulateScenario('degradation')">504 Degradation</button>
      <button class="scenario-trigger" onclick="simulateScenario('abandonment')">Cart Abandonment</button>
      <button class="scenario-trigger" onclick="simulateScenario('card_expiry')">Expired Card</button>
      <button class="scenario-trigger" onclick="simulateScenario('bank_decline')">Bank Decline</button>
      <button class="scenario-trigger" onclick="simulateScenario('voice_call')">Voice Call</button>
      <button class="scenario-trigger scenario-batch" onclick="simulateScenario('batch')">Run 30-Case Batch</button>
    </div>

    <!-- Tabs -->
    <div class="tabs">
      <div class="tab active" onclick="showTab('activity')">Activity</div>
      <div class="tab" onclick="showTab('settings')">Settings</div>
    </div>

    <!-- Tab: Activity -->
    <div id="tab-activity">
      <!-- Metrics -->
      <div class="metrics">
        <div class="metric"><div class="metric-value" id="m-total">0</div><div class="metric-label">Total Payments</div></div>
        <div class="metric"><div class="metric-value sv" id="m-recovered">0</div><div class="metric-label">Recovered</div></div>
        <div class="metric"><div class="metric-value rv" id="m-failed">0</div><div class="metric-label">Failed</div></div>
        <div class="metric"><div class="metric-value" id="m-rate">0%</div><div class="metric-label">Recovery Rate</div></div>
      </div>
      <!-- Hidden metrics (kept for JS updateMetrics) -->
      <div style="display:none"><div id="m-waiting">0</div><div id="m-penalties">0</div><div id="m-saved">$0.00</div></div>

      <!-- Activity + Sidebar Cards -->
      <div class="grid">
        <!-- Activity Feed (left) -->
        <div class="card">
          <h2>Activity Feed <span class="live"><span class="health-dot"></span> Live</span></h2>
          <div class="activity-feed" id="payments">
            <div class="empty" id="empty-msg">No payments yet. Open the <a href="/pay" target="_blank" style="color:var(--brand-blue)">store</a> to start recovery.</div>
          </div>
        </div>

        <!-- Right column: Agent Trail -->
        <div style="display:flex;flex-direction:column;gap:16px">
          <div class="card" style="flex:1">
            <h2>Agent Trail</h2>
            <div class="trail" id="trail"><div class="empty">Waiting for agent activity...</div></div>
          </div>
        </div>
      </div>

      <!-- Hidden metric IDs for JS updateMetrics -->
      <div style="display:none">
        <span id="tier-silent-count">SILENT: 0</span>
        <span id="tier-active-count">ACTIVE: 0</span>
        <span id="tier-hard-count">HARD BLOCKED: 0</span>
        <div id="decline-strategies"></div>
        <div id="penalty-count">0</div>
        <div id="penalty-value">$0.00 saved</div>
      </div>

      <div style="text-align:center;padding:12px 0;color:var(--text-muted);font-size:11px;position:sticky;bottom:0;background:var(--bg-canvas);border-top:1px solid var(--border-subtle)">
        AI agents can make mistakes. Verify important decisions independently.
      </div>
    </div>

    <!-- Tab: Settings (placeholder) -->
    <div id="tab-settings" style="display:none">
      <div class="card">
        <h2>Agent Configuration</h2>
        <div style="padding:20px;color:var(--text-secondary);font-size:13px">
          <p>Configure recovery agent parameters, guardrail policies, and notification templates.</p>
          <p style="margin-top:12px;color:var(--text-muted);font-size:12px">Settings panel coming soon. Currently managed via configuration files.</p>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Case Detail Drawer -->
<div class="drawer-overlay" id="drawer-overlay" onclick="closeDrawer()"></div>
<div class="drawer" id="drawer">
  <div class="drawer-header">
    <div class="drawer-title" id="drawer-title">Payment Details</div>
    <button class="drawer-close" onclick="closeDrawer()">&#10005;</button>
  </div>
  <div id="drawer-status-bar" style="height:4px;border-radius:0"></div>
  <div class="drawer-body" id="drawer-body">
    <div class="empty">Select a payment to view details</div>
  </div>
</div>

<!-- Toast -->
<div class="toast" id="toast"></div>

<script>
const socket=io();
let paymentsData=[];
let tierStats={silent:0,active:0,hard_decline_blocked:0};
let totalPenalties=0;
let declineStrategies={};

// Theme toggle
function toggleTheme(){
  const html=document.documentElement;
  const current=html.getAttribute("data-theme");
  html.setAttribute("data-theme",current==="light"?"dark":"light");
  localStorage.setItem("theme",html.getAttribute("data-theme"));
}
(function(){const saved=localStorage.getItem("theme");if(saved)document.documentElement.setAttribute("data-theme",saved)})();

// Tab switching
function showTab(name){
  document.querySelectorAll(".tab").forEach(t=>t.classList.remove("active"));
  document.querySelectorAll("[id^=tab-]").forEach(p=>p.style.display="none");
  event.target.classList.add("active");
  document.getElementById("tab-"+name).style.display="block";
}

// Drawer
function openDrawer(paymentId){
  const p=paymentsData.find(x=>x.payment_id===paymentId);
  if(!p)return;
  document.getElementById("drawer-title").textContent=p.payment_id;
  const bar=document.getElementById("drawer-status-bar");
  const statusColors={recovering:"var(--brand-blue)",awaiting_customer:"var(--brand-blue)",recovered:"var(--success)",failed:"var(--error)",escalated:"var(--brand-blue)",scheduled:"var(--warning)"};
  bar.style.background=statusColors[p.status]||"var(--brand-blue)";
  const body=document.getElementById("drawer-body");
  const tierLabel=(p.recovery_tier||"active").toUpperCase();
  const tierCls=p.recovery_tier==="silent"?"btier-silent":p.recovery_tier==="hard_decline_blocked"?"btier-hard":"btier-active";
  const statusBadge=p.status==="recovered"?"bs":p.status==="failed"?"bf":p.status==="escalated"?"bp":"bw";
  let html=`<div class="drawer-section"><h3>Payment Info</h3><div style="font-size:13px;color:var(--text-secondary);display:flex;flex-wrap:wrap;gap:12px"><span><b>Amount:</b> INR ${p.amount.toLocaleString()}</span><span><b>Status:</b> <span class="badge ${statusBadge}">${p.status}</span></span><span class="badge ${tierCls}" style="font-size:11px">${tierLabel}</span><span><b>Attempts:</b> ${p.attempts||0}</span>${p.decline_strategy==='voice_call'?'<br><b>Channel:</b> <span class="drawer-badge blue">AI Voice Call (SuperU)</span>':''}</div></div>`;
  if(p.decline_strategy){html+=`<div class="drawer-section"><h3>Decline Strategy</h3><div style="font-size:13px;color:var(--text-secondary)">${p.decline_strategy}</div></div>`}
  if(p.penalties_prevented){html+=`<div class="drawer-section"><h3>Penalties Prevented</h3><div style="font-size:13px;color:var(--success);font-weight:600">${p.penalties_prevented} blocked ($${(p.penalties_prevented*0.10).toFixed(2)} saved)</div></div>`}
  if(p.trail&&p.trail.length){
    const initStep=p.trail.find(e=>e.step==="init");
    const investigationSteps=p.trail.filter(e=>["investigating","checking_history","checking_status","harness_thinking","harness_start"].includes(e.step));
    const actionSteps=p.trail.filter(e=>["generating_link","scheduling_retry","escalating","calling","acting"].includes(e.step));
    if(initStep){html+=`<div class="drawer-section"><h3>Case Init</h3><div style="font-size:13px;color:var(--text-secondary)">${initStep.msg}${initStep.detail?'<br>'+initStep.detail:''}</div></div>`}
    if(investigationSteps.length){
      html+=`<div class="drawer-section"><h3>Investigation (${investigationSteps.length} steps)</h3><div style="font-size:13px;color:var(--text-secondary)">`;
      investigationSteps.forEach(t=>{html+=`<div style="margin-bottom:6px;padding:6px;border-left:3px solid #8b5cf6;background:var(--bg-surface);border-radius:0 6px 6px 0"><b style="color:#8b5cf6">${t.ts}</b> <span style="font-weight:500">${t.msg}</span>${t.detail?'<br><span style="color:var(--text-muted);font-size:12px">'+t.detail+'</span>':''}</div>`});
      html+=`</div></div>`;
    }
    if(actionSteps.length){
      html+=`<div class="drawer-section"><h3>Actions (${actionSteps.length} steps)</h3><div style="font-size:13px;color:var(--text-secondary)">`;
      actionSteps.forEach(t=>{html+=`<div style="margin-bottom:6px;padding:6px;border-left:3px solid #22c55e;background:var(--bg-surface);border-radius:0 6px 6px 0"><b style="color:#22c55e">${t.ts}</b> <span style="font-weight:500">${t.msg}</span>${t.detail?'<br><span style="color:var(--text-muted);font-size:12px">'+t.detail+'</span>':''}</div>`});
      html+=`</div></div>`;
    }
    html+=`<div class="drawer-section"><h3>Full Trail (${p.trail.length} steps)</h3><div class="drawer-trail">`;
    p.trail.forEach(e=>{
      const isInvestigation=["investigating","checking_history","checking_status","harness_thinking","harness_start","init"].includes(e.step);
      const borderClr=isInvestigation?"#8b5cf6":"#22c55e";
      html+=`<div class="trail-item t-${e.step}" style="cursor:pointer;border-left-color:${borderClr}" onclick="var d=this.querySelector('.trail-detail');if(d)d.style.display=d.style.display==='none'?'block':'none'"><div class="trail-time">${e.ts}</div><div class="trail-msg">${e.msg}</div>${e.detail?'<div class="trail-detail" style="display:block;margin-top:4px">'+e.detail+'</div>':''}</div>`;
    });
    html+=`</div></div>`;
  }
  body.innerHTML=html;
  document.getElementById("drawer-overlay").classList.add("open");
  document.getElementById("drawer").classList.add("open");
}
function closeDrawer(){
  document.getElementById("drawer-overlay").classList.remove("open");
  document.getElementById("drawer").classList.remove("open");
}

function updateMetrics(){
  const t=paymentsData.length,r=paymentsData.filter(p=>p.status==="recovered").length,w=paymentsData.filter(p=>p.status==="recovering"||p.status==="awaiting_customer").length,f=paymentsData.filter(p=>p.status==="failed").length;
  document.getElementById("m-total").textContent=t;document.getElementById("m-recovered").textContent=r;
  document.getElementById("m-waiting").textContent=w;document.getElementById("m-failed").textContent=f;
  document.getElementById("m-rate").textContent=t>0?Math.round(r/t*100)+"%":"0%";
  document.getElementById("m-penalties").textContent=totalPenalties;
  document.getElementById("m-saved").textContent="$"+(totalPenalties*0.10).toFixed(2);
  document.getElementById("penalty-count").textContent=totalPenalties;
  document.getElementById("penalty-value").textContent="$"+(totalPenalties*0.10).toFixed(2)+" saved";
  document.getElementById("tier-silent-count").textContent="SILENT: "+tierStats.silent;
  document.getElementById("tier-active-count").textContent="ACTIVE: "+tierStats.active;
  document.getElementById("tier-hard-count").textContent="HARD BLOCKED: "+tierStats.hard_decline_blocked;
}

function renderDeclineStrategies(){
  const el=document.getElementById("decline-strategies");
  const keys=Object.keys(declineStrategies);
  if(keys.length===0){el.innerHTML='<div class="empty" style="padding:8px">Waiting...</div>';return}
  el.innerHTML=keys.map(code=>{
    const s=declineStrategies[code];
    const tierCls=s.tier==="silent"?"btier-silent":s.tier==="hard_decline_blocked"?"btier-hard":"btier-active";
    const tierLabel=s.tier==="silent"?"SILENT":s.tier==="hard_decline_blocked"?"BLOCKED":"ACTIVE";
    return `<div class="strategy-item"><span class="strategy-code">${code}</span><span class="strategy-name">${s.strategy}</span><span class="strategy-tier ${tierCls}">${tierLabel}</span></div>`;
  }).join("");
}

function renderPayments(){
  const el=document.getElementById("payments"),empty=document.getElementById("empty-msg");
  const times=["Just now","2 min ago","5 min ago","12 min ago","18 min ago","25 min ago","38 min ago","52 min ago","1 hr ago","2 hr ago"];
  if(paymentsData.length===0){empty.style.display="block";el.innerHTML="";el.appendChild(empty);return}
  empty.style.display="none";
  el.innerHTML=paymentsData.map((p,i)=>{
    const iconCls=p.status==="recovered"?"recovered":p.status==="recovering"||p.status==="awaiting_customer"?"recovering":p.status==="failed"?"failed":p.status==="escalated"?"escalated":"scheduled";
    const icon=p.status==="recovered"?"✓":p.status==="failed"?"✕":p.status==="escalated"?"↗":p.status==="scheduled"?"⏱":"●";
    const live=p.status==="recovering"||p.status==="awaiting_customer"?'<span class="live"><span class="health-dot"></span> Live</span>':'';
    const tierCls=p.recovery_tier==="silent"?"btier-silent":p.recovery_tier==="hard_decline_blocked"?"btier-hard":"btier-active";
    const tierLabel=(p.recovery_tier||"active").toUpperCase();
    const action=p.status==="recovered"?"Recovery confirmed":p.status==="escalated"?"Escalated to human":p.status==="scheduled"?"Retry scheduled":p.status==="awaiting_customer"?"Awaiting response":p.status==="voice_call"?"Voice Call Active":"Recovering";
    const ts=times[i]||(i*7+3)+" min ago";
    return `<div class="activity-item" onclick="openDrawer('${p.payment_id}')">
      <div class="activity-icon ${iconCls}">${icon}</div>
      <div class="activity-body">
        <div class="activity-title"><span>${action} ${p.payment_id.slice(0,16)}</span><span style="margin-left:auto;display:flex;align-items:center;gap:8px;flex-shrink:0"><span class="activity-amount">INR ${p.amount.toLocaleString()}</span><span class="badge ${tierCls}">${tierLabel}</span>${live}</span></div>
        <div class="activity-meta"><span>${ts}</span></div>
      </div>
    </div>`;
  }).join("");
}

function renderTrail(trail){
  const el=document.getElementById("trail");
  if(!el)return;
  if(!trail||trail.length===0){el.innerHTML='<div class="empty">Waiting for agent activity...</div>';return}
  el.innerHTML=trail.map(e=>{
    let specHtml='';
    if(e.ui_spec){
      specHtml=`<div class="trail-detail" style="margin-top:6px;padding:8px;border-radius:6px;border-left:3px solid #8B5CF6;background:var(--bg-surface)">
        <strong style="color:#8B5CF6">UI Spec:</strong> ${e.ui_spec.headline||''} — ${e.ui_spec.primary_cta_text||''}
      </div>`;
    }
    return `<div class="trail-item t-${e.step}"><div class="trail-time">${e.ts}</div><div class="trail-msg">${e.msg}</div>${e.detail?'<div class="trail-detail">'+e.detail+'</div>':''}${specHtml}</div>`;
  }).join('');
}
</script></body></html>"""


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

    if scenario not in scenarios and scenario != "batch":
        scenario = "degradation"

    if scenario == "batch":
        from recovery_agent.agent.evaluation import run_batch_evaluation
        result = run_batch_evaluation(num_cases=10, seed=42)
        return jsonify({
            "status": "batch_completed",
            "summary": result.summary(),
            "total_cases": result.total_cases,
            "recovered": result.recovered,
            "yield_pct": result.recovery_rate * 100,
            "recovered_amount": result.recovered_amount,
        })

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
    recorded = _mark_recovered(payment_id, amount, rzp_payment_id, 0,
                               "customer paid on the checkout page")
    return jsonify({"status": "recovered" if recorded else "already_recorded",
                    "amount": amount, "payment_id": rzp_payment_id})


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
        store.save_payment(payment_id, {"payment_id": payment_id, "amount": amount, "status": "recovering", "attempts": 0, "last_action": "", "last_detail": "", "trail": []})

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
    store.update_payment(payment_id, status="recovering", push_outcome=None,
                         failure_code=failure_code or "",
                         failure_reason=failure_reason or "")

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
                "ui_morph": "RECOVERY_SUCCESS",
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
                "ui_morph": "AWAITING_CAPTURE",
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

        if store.has_payment(payment_id):
            p = store.get_payment(payment_id)
            p["status"] = "recovered"
            p["recovered_amount"] = amount
            p["recovered_payment_id"] = payment_entity.get("id", "")
            p.setdefault("trail", []).append({
                "step": "recovery_confirmed",
                "msg": f"Payment captured: INR {amount:,.2f}",
                "detail": f"Captured via payment link. Original: {original_payment_id or payment_id} | Actual: {payment_entity.get('id', '')}",
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            })

        push_event(payment_id, "webhook_captured", {
            "status": "recovered",
            "amount": amount,
            "payment_id": payment_id,
            "original_payment_id": original_payment_id,
            "captured_payment_id": payment_entity.get("id", ""),
        })
        store.flush()

        return jsonify({"status": "captured", "payment_id": payment_id})

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
    if result.get("status") in ("retry_created", "link_created"):
        window = int(os.getenv("RETRY_WATCH_MINUTES", "30")) * 60
        socketio.start_background_task(_watch_for_recovery, payment_id, window)

    return jsonify({"status": "received", "payment_id": payment_id})


@app.route("/api/payments")
def api_payments():
    p_list = store.payment_values()
    total_at_risk = sum(p.get("amount", 0) for p in p_list)
    total_recovered = sum(p.get("amount", 0) for p in p_list if p.get("status") == "recovered")
    rec_count = sum(1 for p in p_list if p.get("status") == "recovered")
    rate = (rec_count / len(p_list) * 100) if p_list else 0.0
    total_penalties = sum(p.get("penalties_prevented", 0) for p in p_list)
    return jsonify({
        "payments": p_list,
        "total_at_risk": total_at_risk,
        "total_recovered": total_recovered,
        "recovery_rate": round(rate, 1),
        "total_penalties_prevented": total_penalties,
        "penalties_saved_usd": round(total_penalties * NETWORK_FINE_PER_ATTEMPT, 2),
        "penalties_saved_inr": round(total_penalties * 8.30, 2),
    })


@app.route("/api/benchmark", methods=["POST"])
def run_benchmark():
    from recovery_agent.eval.chaos_gym import run_before_after_benchmark
    data = request.json or {}
    seed = data.get("seed", 42)
    count = data.get("count", 50)
    result = run_before_after_benchmark(seed=seed, count=count)
    return jsonify(result)

@app.route("/api/voice-call-approval", methods=["POST"])
def voice_call_approval():
    """Merchant approves or denies a pending voice call."""
    data = request.json or {}
    payment_id = data.get("payment_id", "").strip()
    approved = data.get("approved", False)
    if not payment_id:
        return jsonify({"error": "payment_id required"}), 400
    event = _pending_voice_approvals.get(payment_id)
    if not event:
        return jsonify({"error": "No pending approval for this payment"}), 404
    _voice_call_approved[payment_id] = approved
    event.set()  # unblocks the agent thread
    return jsonify({"status": "approved" if approved else "denied", "payment_id": payment_id})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "payments": len(store.all_payments())})

def main():
    port = int(os.getenv("FRONTEND_PORT", "5002"))
    print(f"\n  Customer Checkout:  http://localhost:{port}/pay")
    print(f"  Merchant Dashboard: http://localhost:{port}/merchant")
    print(f"  Agent Flow:         http://localhost:{port}/graph\n")
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)

if __name__ == "__main__":
    main()
