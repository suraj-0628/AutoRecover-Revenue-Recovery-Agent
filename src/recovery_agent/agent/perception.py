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

    try:
        from recovery_agent.agent import ladder
        st = ladder.state(rec)
        facts["climbed"] = st["climbed"]
        facts["actions_tried"] = ladder.actions_tried(rec)
        nxt = st["remaining"][0] if st["remaining"] else None
        facts["next_rung"] = nxt["rung"] if nxt else None
        facts["ladder_exhausted"] = not st["remaining"]
        facts["unavailable"] = [f"{u['rung']} ({u['why_not']})"
                                for u in st["unavailable"]]
    except Exception:
        pass
    return facts


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
        if facts.get("climbed"):
            head += f"\n  Already climbed: {', '.join(facts['climbed'])}"
        if facts.get("actions_tried"):
            head += (f"\n  Already tried: {', '.join(facts['actions_tried'])}"
                     f"\n  Do not repeat any of those — they did not work.")
        head += (f"\n  Next rung: {nxt}" if nxt else
                 "\n  Every rung has been tried. escalate_to_human will accept "
                 "this case now.")

    if facts.get("warning"):
        head += f"\n  CAUTION: {facts['warning']}"
    if facts.get("unavailable"):
        head += f"\n  Not possible here: {'; '.join(facts['unavailable'])}"
    return head
