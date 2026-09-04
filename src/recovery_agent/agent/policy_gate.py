"""Policy gate — the GuardrailEngine, on the live tool path at last.

The engine (quiet hours, frequency cap, double-debit lock, opt-out, monetary
cap, hard declines) existed for months with zero callers of validate_action.
It was constructed, passed into the graph context, displayed in the HUD — and
never once consulted before a tool ran. Six guardrails and thirty-one tests
guarding a door nobody walked through.

This node sits between the repetition guard and the approval gate. Every tool
call that contacts a customer or moves toward money is translated into the
engine's vocabulary and validated; a refusal comes back to the model as a
ToolMessage with the reason and a workable alternative, and is recorded on the
durable record's `refusals` — which perception already renders into the next
briefing ("Proposing them again will be refused again"). The guardrail is not
a cage the agent bounces off blindly; it is a fact the agent perceives.

Every evaluation — pass or refuse — is appended to
STATE_DIR/audit_logs/guardrail_verdicts.jsonl with the agent version, so
"which policy refused this contact, and when?" has an answer.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from recovery_agent.models import ActionType

# Live tool name → the engine's action vocabulary. A tool absent here is not a
# customer contact and not a debit, so the engine has nothing to say about it.
#
# Deliberately absent: send_page_push and show_page_offer. They render inside a
# checkout page the customer already has open — nothing lands in an inbox and
# no phone rings, so quiet hours and the frequency cap do not apply. The
# contact ladder's own rules still govern them.
TOOL_ACTION: dict[str, ActionType] = {
    "send_recovery_notification": ActionType.SEND_NOTIFICATION,
    "initiate_voice_call": ActionType.VOICE_CALL,
    # A recovery link exists to be put in front of the customer, and creating
    # one spends a unit of a 30-per-account-lifetime quota. If contacting the
    # customer would be refused, minting the link first just strands it.
    "generate_recovery_payment_link": ActionType.UPDATE_PAYMENT_METHOD,
    "retry_in_hours": ActionType.RETRY_PAYMENT,
}

#: What the agent should do INSTEAD, per guardrail. A refusal that offers no
#: move teaches the model nothing except to escalate.
_GUIDANCE: dict[str, str] = {
    "quiet_hours": ("It is the customer's quiet hours. Use retry_in_hours to "
                    "come back after they end, or wait_for_customer."),
    "frequency_cap": ("This customer has been contacted enough in the last 24 "
                      "hours. Use retry_in_hours to come back tomorrow, or "
                      "wait_for_customer — do not contact them again today."),
    "opt_out": ("This customer has opted out of automated contact. Work the "
                "case without contacting them, or escalate_to_human."),
    "double_debit_lock": ("A payment on this case has already succeeded or is "
                          "pending. Call check_payment_status before anything "
                          "that could debit twice."),
    "monetary_cap": ("The amount is over the single-retry cap. "
                     "escalate_to_human is the correct move for this order."),
    "hard_decline": ("This failure code is a hard decline; card networks "
                     "penalise every retry. Never retry it — "
                     "escalate_to_human."),
    "interceptor": "Choose a different action; this one is against policy.",
}


def enforcement_enabled() -> bool:
    return os.getenv("GUARDRAILS_ENFORCE", "1").strip().lower() not in (
        "0", "false", "no", "off")


def _verdict_log_path() -> Path:
    return Path(os.getenv("STATE_DIR", "data")) / "audit_logs" / "guardrail_verdicts.jsonl"


def log_verdict(payment_id: str, tool: str, outcome: str,
                checks: list[Any] | None = None, reason: str = "") -> None:
    """One line per gate evaluation. Never raises."""
    try:
        from recovery_agent.agent.governance import AGENT_VERSION
        path = _verdict_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "payment_id": payment_id,
            "tool": tool,
            "outcome": outcome,           # "pass" | "blocked"
            "reason": reason,
            "agent_version": AGENT_VERSION,
        }
        if checks:
            entry["checks"] = [
                {"guardrail": c.guardrail, "verdict": c.verdict.value,
                 "reason": c.reason}
                for c in checks
            ]
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def read_verdicts(limit: int = 500) -> list[dict]:
    """Newest-first tail of the verdict log, for the ops view. Never raises."""
    try:
        path = _verdict_log_path()
        if not path.exists():
            return []
        lines = path.read_text().splitlines()[-limit:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        out.reverse()
        return out
    except Exception:
        return []


def _record_refusal(payment_id: str, key: str) -> None:
    """A refusal the next run — and the next briefing — can see."""
    try:
        from recovery_agent.state_store import StateStore
        store = StateStore()
        rec = store.get_payment(payment_id)
        if rec is None:
            return
        refusals = dict(rec.get("refusals") or {})
        refusals[key] = int(refusals.get(key, 0)) + 1
        store.update_payment(payment_id, refusals=refusals)
        store.flush()
    except Exception:
        pass


def _customer_key(payment_id: str) -> str:
    try:
        from recovery_agent.state_store import StateStore
        rec = StateStore().get_payment(payment_id) or {}
        customer = rec.get("customer") or {}
        return (customer.get("email") or rec.get("customer_email")
                or "unknown")
    except Exception:
        return "unknown"


def _profile_for(payment_id: str):
    try:
        from recovery_agent.agent.memory import CustomerMemoryStore
        return CustomerMemoryStore.live().get_or_create_profile(
            _customer_key(payment_id))
    except Exception:
        return None


#: Tools whose whole purpose is to spend a metered external resource.
_BUDGETED_TOOLS = {"send_recovery_notification"}


def _budget_refusal(tool_name: str, payment_id: str) -> dict | None:
    """Refuse a send there is no allowance left for. Never raises — a broken
    meter must never be able to stop a recovery."""
    if tool_name not in _BUDGETED_TOOLS:
        return None
    try:
        from recovery_agent import email_quota
        allowed, why = email_quota.may_send()
    except Exception:
        return None
    if allowed:
        return None
    log_verdict(payment_id, tool_name, "blocked", reason=why)
    _record_refusal(payment_id, f"{tool_name}: email_budget")
    return {
        "status": "blocked",
        "guardrail": "email_budget",
        "reason": why,
        "guidance": ("The allowance resets tomorrow. Use retry_in_hours to "
                     "come back then, put the offer on the page with "
                     "show_page_offer if the customer is still there, or "
                     "escalate_to_human — do not keep trying to send."),
    }


def evaluate_tool_call(engine, case, payment_id: str, tool_name: str,
                       now: datetime | None = None) -> dict | None:
    """Run one proposed tool call through the engine.

    Returns None to let it through, or a refusal payload (JSON-serialisable)
    when a guardrail objects. Shared with the red-team eval, which uses it to
    ask "had the model gone through with this, would the gate have caught it?"
    """
    action = TOOL_ACTION.get(tool_name)
    if action is None or engine is None or case is None:
        return None

    # A channel with nothing left is not a channel. The engine reasons about
    # whether contacting this customer is APPROPRIATE; this asks the separate
    # question of whether the resource to do it still exists. Checked first,
    # because no amount of appropriateness conjures an email allowance.
    budget = _budget_refusal(tool_name, payment_id)
    if budget is not None:
        return budget

    profile = _profile_for(payment_id)
    final_action, checks = engine.validate_action(case, action, profile, now=now)

    offender = next((c for c in checks
                     if c.verdict.value in ("blocked", "modified")
                     and c.guardrail != "interceptor"), None)
    if offender is None and final_action == action:
        log_verdict(payment_id, tool_name, "pass", checks)
        return None

    name = offender.guardrail if offender else "policy"
    reason = offender.reason if offender else "the engine substituted a different action"
    log_verdict(payment_id, tool_name, "blocked", checks, reason=reason)
    _record_refusal(payment_id, f"{tool_name}: {name}")
    refusal = {
        "status": "blocked",
        "guardrail": name,
        "reason": reason,
        "guidance": _GUIDANCE.get(name, _GUIDANCE["interceptor"]),
    }

    # Hand back the exact wait, not "come back later". Live (pay_woo85c9gh),
    # the agent was refused at 06:30 IST, told only that quiet hours were
    # active, guessed 60 minutes, and would have been refused again at 07:30 —
    # the window ran to 08:00. A refusal that hides the number it enforces
    # makes the agent solve a puzzle it has no information for.
    if name == "quiet_hours":
        try:
            from recovery_agent.agent.guardrails import QuietHourGuardrail
            mins = QuietHourGuardrail().minutes_until_end()
            refusal["retry_after_minutes"] = mins
            refusal["guidance"] = (
                f"It is the customer's quiet hours for another {mins} minute(s). "
                f"Call wait_for_customer with expected_within_minutes={mins} "
                f"(or retry_in_hours with hours={max(1, round(mins / 60))}) so "
                f"you come back once they have ended — anything sooner will be "
                f"refused again. In-page surfaces are NOT restricted: if the "
                f"customer is still on the checkout, send_page_push and "
                f"show_page_offer remain available now.")
        except Exception:
            pass
    return refusal


def policy_gate(state, config=None) -> dict:
    """Graph node: validate the pending tool calls before anything executes.

    Mirrors human_approval_gate's contract exactly: refused calls get a
    ToolMessage each (so the model sees why), untouched calls in the same
    batch get "not_executed — call it again on its own", and a SystemMessage
    carrying "[Guardrail]" routes the turn back to the agent instead of the
    ToolNode.
    """
    from recovery_agent.agent.graph import _get_context

    messages = state["messages"]
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}
    if not enforcement_enabled():
        return {}

    ctx = _get_context(config)
    case = getattr(ctx, "case", None) if ctx else None
    engine = getattr(ctx, "guardrail_engine", None) if ctx else None
    if case is None:
        return {}
    if engine is None:
        from recovery_agent.agent.guardrails import GuardrailEngine
        engine = GuardrailEngine()

    payment_id = str(getattr(getattr(case, "payment", None), "payment_id", "") or "")

    refusals: dict[str, dict] = {}
    summary: list[str] = []
    for tc in last.tool_calls:
        refusal = evaluate_tool_call(engine, case, payment_id, tc["name"])
        if refusal is not None:
            refusals[tc["id"]] = refusal
            summary.append(f"{tc['name']} ({refusal['guardrail']}: {refusal['reason']})")

    if not refusals:
        return {}

    return {
        "messages": [
            ToolMessage(
                content=json.dumps(refusals.get(tc["id"]) or {
                    "status": "not_executed",
                    "reason": "another tool in this batch was refused by policy",
                    "guidance": "You may call this tool again on its own.",
                }),
                tool_call_id=tc["id"], name=tc["name"],
            ) for tc in last.tool_calls
        ] + [SystemMessage(
            content=f"[Guardrail] Refused by policy: {'; '.join(summary)}. "
                    f"The refusal message on each tool says what to do instead."
        )],
        "phase": "guard_check",
    }
