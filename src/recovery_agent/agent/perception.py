"""What is actually true about a case, right now.

The agent had no perception — only narration. Everything it "knew" arrived as
prose the orchestrator chose to write into a message, so it stopped when it was
told "RECOVERED" and kept chasing a customer who had already paid whenever it
was not told. That is not forgetting. It had no way to look.

The one tool that could have answered "has this been paid?" ended with
`"do not call this tool again for this case"`. We taught it that checking was
pointless, then withheld its tools to stop it acting on what it therefore could
not know. That is a cage, and a caged agent is not a careful one — it behaves
correctly only where the bars happen to be.

This module is the other half: one function that reads the durable record and
Razorpay and states plainly whether the goal has been met. The graph puts it in
front of the model every turn, so perceiving is not something the agent has to
remember to do; and `check_payment_status` returns the same thing on demand, so
asking is always worth it.

The interlock stays, but as a backstop for when reasoning fails — not as the
reason the agent behaves.
"""
from __future__ import annotations

from typing import Any


def _f(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def ground_truth(payment_id: str, verify: bool = False) -> dict:
    """Where this case actually stands. Never raises.

    `verify=True` confirms the capture against Razorpay rather than trusting the
    record. Off by default because this runs every turn; on when the agent asks
    directly, because that is the moment accuracy is worth a network call.
    """
    facts: dict[str, Any] = {
        "payment_id": payment_id,
        "known": False,
        "settled": False,
        "owed": 0.0,
        "received": 0.0,
        "outstanding": 0.0,
    }
    try:
        from recovery_agent.state_store import StateStore
        rec = StateStore().get_payment(payment_id) or {}
    except Exception:
        return facts
    if not rec:
        return facts

    facts["known"] = True
    owed = _f(rec.get("amount"))
    received = _f(rec.get("recovered_amount"))
    status = str(rec.get("status") or "")
    captured_id = str(rec.get("recovered_payment_id") or "")

    # Money received settles the case even at a discount: an authorised offer
    # means the reduced figure IS what is owed now. Status alone is not trusted —
    # a status is something a writer chose, an amount is something that happened.
    settled = received > 0 or status == "recovered"

    if verify and captured_id.startswith("pay_"):
        try:
            from recovery_agent.razorpay_client import RazorpayClient
            client = RazorpayClient()
            if client.is_configured:
                pay = client.client.payment.fetch(captured_id)
                facts["razorpay_status"] = pay.get("status")
                if pay.get("status") == "captured":
                    received = max(received, pay.get("amount", 0) / 100)
                    settled = True
                else:
                    facts["warning"] = (
                        f"the record claims {captured_id} but Razorpay reports "
                        f"{pay.get('status')!r}, not captured")
                    settled = received > 0 and status == "recovered"
        except Exception as exc:
            facts["verification_error"] = str(exc)[:120]

    facts.update({
        "owed": round(owed, 2),
        "received": round(received, 2),
        "outstanding": 0.0 if settled else round(max(0.0, owed - received), 2),
        "settled": settled,
        "captured_payment_id": captured_id,
        "case_status": status,
        "escalated": status == "escalated",
    })

    # What KIND of failure this is decides which lever is even relevant. A bank
    # decline is a plumbing problem; offering money off does not fix plumbing.
    # Live, the agent reached for a 5% discount on a `bank_declined` while the
    # customer's own history showed netbanking succeeding 2 times out of 2.
    # One definition of what kind of failure this is, shared with the batch
    # view. Two copies would drift, and a case would be a bank decline in one
    # place and a drop-off in the other.
    from recovery_agent.agent.classify import failure_kind
    kind = failure_kind(rec)
    if kind in ("method", "funds", "transient", "dropoff", "risk"):
        facts["failure_kind"] = kind
    facts["failure_code"] = rec.get("failure_code") or ""
    facts["refusals"] = rec.get("refusals") or {}

    try:
        from recovery_agent.agent import ladder
        st = ladder.state(rec)
        facts["climbed"] = st["climbed"]
        facts["actions_tried"] = ladder.actions_tried(rec)
        nxt = st["remaining"][0] if st["remaining"] else None
        facts["next_rung"] = nxt["rung"] if nxt else None
        # WHICH ladder this case is on. There is one per failure kind, so the
        # agent has to be told the sequence it is climbing — otherwise it
        # reasons about "the ladder" from the prompt and reaches for a rung
        # that belongs to a different failure.
        facts["ladder_rungs"] = [k for k, _ in ladder.rungs_for(rec)]
        facts["retry_pending"] = ladder.retry_pending(rec)
        facts["ladder_exhausted"] = ladder.exhausted(rec)
        facts["unavailable"] = [f"{u['rung']} ({u['why_not']})"
                                for u in st["unavailable"]]
    except Exception:
        pass

    # Memory, perceived rather than remembered-to-ask-for. search_memory
    # existed and the model could call it — which means on most cases it did
    # not, and every case started from zero. The digest is deterministic, cheap
    # (two local store reads), and lands in the same briefing as the money.
    try:
        history = _history_digest(rec, facts.get("failure_kind") or "unknown")
        if history:
            facts["history"] = history
    except Exception:
        pass
    return facts


def _history_digest(rec: dict, kind: str) -> dict:
    """What this customer and cases like this one have actually done.

    Two sources: the customer profile (channel win-rates learned across their
    own cases) and the episode store (every closed case of the same failure
    kind). Both computed to short factual lines — the model gets numbers, not a
    lecture. Returns {} when there is nothing real to say; a briefing line
    built on zero data is noise wearing a memory's clothes.
    """
    out: dict[str, str] = {}

    customer = (rec.get("customer") or {}).get("email") \
        or rec.get("customer_email") or ""
    if customer:
        try:
            from recovery_agent.agent.memory import CustomerMemoryStore
            profile = CustomerMemoryStore.live().get_or_create_profile(customer)
            by_channel: dict[str, list[int]] = {}
            for p in profile.payment_history:
                if p.status in ("success", "failed") and p.channel_used:
                    stats = by_channel.setdefault(p.channel_used, [0, 0])
                    stats[1] += 1
                    if p.status == "success":
                        stats[0] += 1
            parts = [f"{ch} recovered {w}/{t}"
                     for ch, (w, t) in sorted(by_channel.items()) if t]
            if parts:
                out["customer_line"] = "this customer before now: " + ", ".join(parts)
        except Exception:
            pass

    try:
        from recovery_agent.agent.graph import get_memory_store
        from recovery_agent.agent.tools import ns_safe
        episodes = [item.value or {} for item in
                    get_memory_store().search(("recovery", "episodes",
                                               ns_safe(kind)), limit=50)]
        # This case may already have an episode mid-write; exclude itself.
        pid = str(rec.get("payment_id") or "")
        episodes = [e for e in episodes if e.get("payment_id") != pid]
        if episodes:
            recovered = [e for e in episodes if e.get("outcome") == "recovered"]
            line = (f"of {len(episodes)} past {kind} case(s), "
                    f"{len(recovered)} recovered")
            if recovered:
                full = sum(1 for e in recovered
                           if float(e.get("discount_pct") or 0) == 0)
                line += f" — {full} at full price"
                discounted = len(recovered) - full
                if discounted:
                    line += f", {discounted} needed a discount"
            out["similar_line"] = line
    except Exception:
        pass

    return out


def as_briefing(facts: dict) -> str:
    """The same facts as something a model reads before it decides anything."""
    if not facts.get("known"):
        return ("WHAT IS TRUE RIGHT NOW: no record for this payment yet. "
                "Treat the case facts below as the only information you have.")

    money = (f"Owed INR {facts['owed']:,.2f} | received INR {facts['received']:,.2f} "
             f"| outstanding INR {facts['outstanding']:,.2f}")

    if facts["settled"]:
        head = (
            "WHAT IS TRUE RIGHT NOW — read this before you decide anything.\n"
            f"  {money}\n"
            f"  THE MONEY IS IN. Captured as "
            f"{facts.get('captured_payment_id') or 'a confirmed payment'}.\n"
            "  This case is SETTLED. There is nothing left to recover. Any further "
            "message, offer, link or call would reach a customer who has already "
            "paid you. Record what worked with manage_memory, then stop."
        )
    elif facts.get("escalated"):
        head = (
            "WHAT IS TRUE RIGHT NOW — read this before you decide anything.\n"
            f"  {money}\n"
            "  This case is already with a human. Do not work it further."
        )
    else:
        nxt = facts.get("next_rung")
        head = (
            "WHAT IS TRUE RIGHT NOW — read this before you decide anything.\n"
            f"  {money}\n"
            "  The money is NOT back. This case is still open."
        )
        if facts.get("ladder_rungs"):
            head += (f"\n  The ladder for THIS failure: "
                     f"{' -> '.join(facts['ladder_rungs'])}")
        if facts.get("climbed"):
            head += f"\n  Already climbed: {', '.join(facts['climbed'])}"
        if facts.get("actions_tried"):
            head += (f"\n  Already tried: {', '.join(facts['actions_tried'])}"
                     f"\n  Do not repeat any of those — they did not work.")
        if facts.get("retry_pending"):
            head += ("\n  A retry is ALREADY SCHEDULED and has not fired yet. "
                     "That is the most likely thing to recover this, so the "
                     "case is waiting, not stuck: do not escalate, do not "
                     "claim the retry failed, and do not start another rung "
                     "on top of it. Call wait_for_customer.")
        head += (f"\n  Next rung: {nxt}" if nxt else
                 "\n  Every rung has been tried. escalate_to_human will accept "
                 "this case now.")

    kind = facts.get("failure_kind")
    if kind == "transient":
        head += (
            "\n  This did NOT fail on the customer's side. The gateway or the "
            "network dropped it — the card was fine and so was their intent. "
            "There is nothing to apologise for and nothing to discount: the "
            "right move is to let them try again, or to retry it quietly. "
            "Offering money off here pays a customer to forgive our own outage."
        )
    elif kind == "funds":
        head += ("\n  The account was short at the time. That is a timing "
                 "problem, not a price one — a discount does not put money in "
                 "their account. A retry aimed at when they are likely to be "
                 "paid is the move.")
    elif kind == "method":
        head += (
            "\n  This failed at the BANK, not on price. The customer tried to "
            "pay and the instrument was refused, so a discount does not address "
            "what went wrong — a different payment method does. Check "
            "get_customer_payment_history for a rail that has worked for this "
            "customer before, and offer that."
        )
    elif kind == "dropoff":
        head += ("\n  The customer chose not to complete. Nothing is broken, so "
                 "the question is what would make them come back.")

    history = facts.get("history") or {}
    if history and not facts.get("settled") and not facts.get("escalated"):
        head += "\n  WHAT HAS WORKED BEFORE:"
        if history.get("customer_line"):
            head += f"\n    - {history['customer_line']}"
        if history.get("similar_line"):
            head += f"\n    - {history['similar_line']}"
        head += ("\n  Weigh this before choosing a channel or an offer — it is "
                 "measured, not guessed.")

    repeated = {k: n for k, n in (facts.get("refusals") or {}).items() if n >= 2}
    if repeated:
        head += "\n  You have already been refused these, more than once:"
        for k, n in repeated.items():
            head += f"\n    - {k} ({n}x)"
        head += ("\n  Proposing them again will be refused again. Something about "
                 "your plan has to change, not just your wording.")

    if facts.get("warning"):
        head += f"\n  CAUTION: {facts['warning']}"
    if facts.get("unavailable"):
        head += f"\n  Not possible here: {'; '.join(facts['unavailable'])}"
    return head
