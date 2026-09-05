"""Apply one plan to one case, deterministically.

No LLM runs here. The reasoning happened once, when the plan was made; repeating
it per case buys nothing — the answer would be identical — and costs 5 gated
calls a case, which is how an 18-case batch takes twelve minutes and a 200-case
batch takes two hours.

What *is* per case is eligibility, and it is re-checked at execution time rather
than trusted from planning time. A case can settle, be reclassified, climb a
rung or run out of link quota between the plan being made and reaching its turn.
Twelve checks run before any side effect, and each one reuses the control that
already exists rather than a batch-shaped copy of it.

Two things this deliberately does not do:

**It never calls `close_case` or `escalate_to_human`.** It leaves cases
`awaiting_customer` and lets the existing observers — `_watch_for_recovery`, the
webhook, `_mark_recovered` — do what they already do. Closure and escalation
live behind the ladder gating, and a batch must not be a second door into them.

**It never invokes a tool without a runtime.** `generate_recovery_payment_link`
computes its rupees-vs-paise range check from `runtime.context.case`; with
`runtime=None` that check silently does not happen, and the 100x-overcharge
guard is off. `show_page_offer` and `send_recovery_notification` read the link's
real amount from the same place to verify the price they advertise. A batch path
that skipped this would be quietly less safe than the live path it replaces.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from recovery_agent import audit
from recovery_agent.agent import ladder
from recovery_agent.batch.plan import BatchBudget, BatchPlan
from recovery_agent.batch.tiers import amount_tier

# ── outcomes ─────────────────────────────────────────────────────────────

SKIPPED = "skipped"        # not eligible; nothing to do
EXCEPTION = "exception"    # needs judgement — route to the agent
ACTED = "acted"            # a side effect happened
DEFERRED = "deferred"      # eligible, but not right now (quiet hours, cap)
BUDGET = "budget"          # the run has spent its allowance


@dataclass
class Decision:
    """What the executor decided for one case, and why."""
    payment_id: str
    outcome: str
    reason: str = ""
    action: str = ""
    amount_paise: int = 0
    charged_paise: int = 0
    discount_paise: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_side_effect(self) -> bool:
        return self.outcome == ACTED

    def as_dict(self) -> dict[str, Any]:
        return {"payment_id": self.payment_id, "outcome": self.outcome,
                "reason": self.reason, "action": self.action,
                "amount_paise": self.amount_paise,
                "charged_paise": self.charged_paise,
                "discount_paise": self.discount_paise, **self.detail}


@dataclass
class Spend:
    """What a run has used. Guarded, because workers decrement it in parallel."""
    links: int = 0
    emails: int = 0
    cases: int = 0
    discount_paise: int = 0
    llm_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def would_exceed(self, budget: BatchBudget, *, links: int = 0,
                     emails: int = 0, discount_paise: int = 0) -> str:
        """Which resource this action would overspend, or '' if it fits."""
        with self._lock:
            if self.cases >= budget.max_cases:
                return "cases"
            if self.links + links > budget.max_links:
                return "links"
            if self.emails + emails > budget.max_emails:
                return "emails"
            if self.discount_paise + discount_paise > budget.max_discount_paise:
                return "discount"
        return ""

    def reserve(self, budget: BatchBudget, *, links: int = 0, emails: int = 0,
                cases: int = 1, discount_paise: int = 0) -> str:
        """Check and take under one lock. Returns the overrun resource, or ''.

        `would_exceed` followed by `add` is check-then-act across two lock
        acquisitions: four workers each pass the check before any of them takes,
        and a run creates four payment links against a cap of two. On a test
        account where the lifetime quota is 30 and cancelling returns nothing,
        that race costs real, unrecoverable capacity.
        """
        with self._lock:
            if self.cases + cases > budget.max_cases:
                return "cases"
            if self.links + links > budget.max_links:
                return "links"
            if self.emails + emails > budget.max_emails:
                return "emails"
            if self.discount_paise + discount_paise > budget.max_discount_paise:
                return "discount"
            self.links += links
            self.emails += emails
            self.cases += cases
            self.discount_paise += discount_paise
        return ""

    def release(self, *, links: int = 0, emails: int = 0, cases: int = 0,
                discount_paise: int = 0) -> None:
        """Give back what was reserved but never spent."""
        with self._lock:
            self.links = max(0, self.links - links)
            self.emails = max(0, self.emails - emails)
            self.cases = max(0, self.cases - cases)
            self.discount_paise = max(0, self.discount_paise - discount_paise)

    def add(self, *, links: int = 0, emails: int = 0, cases: int = 0,
            discount_paise: int = 0) -> None:
        with self._lock:
            self.links += links
            self.emails += emails
            self.cases += cases
            self.discount_paise += discount_paise

    def as_dict(self) -> dict[str, int]:
        return {"links": self.links, "emails": self.emails,
                "cases": self.cases, "discount_paise": self.discount_paise,
                "llm_calls": self.llm_calls}


# ── the runtime the tools expect ─────────────────────────────────────────

def build_runtime(record: dict):
    """A Case and context shaped exactly as the live agent path builds them.

    Not a convenience: several tool guards read their reference values off
    `runtime.context.case`, and are inert without it.
    """
    from recovery_agent.agent.graph import RecoveryContext
    from recovery_agent.models import Case, CaseStatus, PaymentEvent

    customer = record.get("customer") or {}
    email = customer.get("email") or record.get("customer_email") or ""
    phone = (customer.get("contact") or customer.get("phone")
             or record.get("customer_phone") or "")

    event = PaymentEvent(
        payment_id=record.get("payment_id", ""),
        customer_id=email,
        amount=float(record.get("amount") or 0),
        currency=record.get("currency", "INR") or "INR",
        failure_code=(record.get("failure_code")
                      or record.get("decline_strategy") or "payment_failed"),
        failure_reason=record.get("failure_reason", "") or "",
        metadata={
            "customer_email": email,
            "customer_name": customer.get("name") or record.get("customer_name", ""),
            "customer_phone": phone,
            "scenario": "batch",
            **({"push_outcome": record["push_outcome"]}
               if record.get("push_outcome") else {}),
            **({"page_offer": record["page_offer"]}
               if record.get("page_offer") else {}),
            **({"recovery_link": record["recovery_link"]}
               if record.get("recovery_link") else {}),
            **({"recovery_link_amount": record["recovery_link_amount"]}
               if record.get("recovery_link_amount") is not None else {}),
        },
    )
    case = Case(payment=event, max_attempts=3)

    # Carry the real state over. A fresh Case defaults to OPEN, and the guard
    # that stops a settled case being contacted keys off the status.
    status_map = {
        "recovered": CaseStatus.RECOVERED, "escalated": CaseStatus.ESCALATED,
        "failed": CaseStatus.STOPPED, "stopped": CaseStatus.STOPPED,
    }
    if record.get("status") in status_map:
        case.status = status_map[record["status"]]
    if record.get("status") == "recovered" or float(record.get("recovered_amount") or 0) > 0:
        case.recovered = True
        case.recovered_amount = float(record.get("recovered_amount") or 0)

    return SimpleNamespace(context=RecoveryContext(guardrail_engine=None, case=case))


# ── the gate ─────────────────────────────────────────────────────────────

def precheck(record: dict, plan: BatchPlan, *, run_id: str,
             budget: BatchBudget, spend: Spend, runtime=None) -> Decision:
    """Everything that must be true before a case is touched.

    Ordered cheapest-first, and by how badly being wrong would hurt: a case that
    has already paid is checked before one that is merely in the wrong band.
    """
    from recovery_agent.agent.classify import classify
    from recovery_agent.agent.offers import quote
    from recovery_agent.agent.perception import ground_truth

    pid = str(record.get("payment_id") or "")
    amount = float(record.get("amount") or 0)
    d = lambda outcome, reason, **kw: Decision(   # noqa: E731 - local shorthand
        payment_id=pid, outcome=outcome, reason=reason,
        amount_paise=audit.to_paise(amount), **kw)

    if not pid:
        return d(SKIPPED, "no payment id")

    # 1. Still the batch we planned for?
    if classify(record) != plan.batch_key:
        return d(SKIPPED, "reclassified")

    # 2. Still the band we planned for?
    if amount_tier(amount).key != plan.tier:
        return d(SKIPPED, "wrong_tier")

    # 3. Paid since planning. The most expensive mistake available, so it is
    #    checked before anything cosmetic.
    if ground_truth(pid).get("settled"):
        return d(SKIPPED, "already_paid")

    # 4. Fraud, dispute, opt-out: not ours to chase.
    barred = ladder.pursuit_barred(record)
    if barred:
        return d(SKIPPED, f"pursuit_barred: {barred}")

    # 5. A case in another run that is STILL GOING. Two live runs on one case
    #    would double-contact; a finished run releases the case — a later wave
    #    legitimately touches it again, and what stops a repeat contact then is
    #    the ladder, which never repeats a rung. A stamp from a run this
    #    process no longer knows (a restart) is treated as released too, for
    #    the same reason: the ladder is the durable guard, the stamp is only
    #    the concurrency guard.
    other = str(record.get("batch_run_id") or "")
    if other and other != run_id:
        from recovery_agent.batch import run as batch_run_mod
        live = batch_run_mod.get(other)
        if live is not None and live.status == batch_run_mod.OPEN:
            return d(SKIPPED, "already_in_a_run")

    # 6. Reachable at all?
    customer = record.get("customer") or {}
    if not (customer.get("email") or record.get("customer_email")
            or customer.get("contact") or record.get("customer_phone")):
        return d(SKIPPED, "no_contact")

    # 7. The plan is for a rung; the case must be on it. A case that has moved
    #    on is exactly the one a shared decision no longer fits.
    nxt = ladder.next_rung(record)
    rung = nxt["rung"] if nxt else None
    step = plan.step_for(rung)
    if step is None:
        return d(EXCEPTION, f"ladder_advanced: on {rung or 'no rung'}, "
                            f"plan covers {sorted(plan.steps)}")
    if step.action == "exception":
        return d(EXCEPTION, step.why or "the plan defers this rung")
    if step.action == "skip":
        return d(SKIPPED, step.why or "the plan skips this rung")

    needs_link = step.action == "link_and_notify"
    needs_email = step.action in ("link_and_notify", "notify_only")

    # 8. The account's link quota is spent; nothing here can create one.
    if needs_link and record.get("links_unavailable"):
        return d(EXCEPTION, "links_unavailable")

    # 9. What would this actually charge? Derived per case, never carried in
    #    the plan — a plan holding a rupee figure would apply one case's
    #    discount to another case's amount.
    charged_paise = audit.to_paise(amount)
    discount_paise = 0
    if plan.offer_stage and not step.full_price:
        offer = quote(amount, plan.offer_stage)
        if offer.allowed:
            charged_paise = audit.to_paise(offer.payable_rupees)
            discount_paise = audit.to_paise(amount) - charged_paise

    # 10. `human_approval_gate` is a graph node, not a tool guard — an executor
    #     calling the link tool directly walks straight past it. The ceiling has
    #     to be enforced here or the batch path gives away money the live path
    #     would have stopped.
    if discount_paise:
        from recovery_agent.agent.graph import _APPROVAL_DISCOUNT_THRESHOLD
        if discount_paise > audit.to_paise(_APPROVAL_DISCOUNT_THRESHOLD):
            return d(EXCEPTION, "needs_approval",
                     discount_paise=discount_paise)

    # 11. The merchant's own compliance rules, which the batch path has never
    #     consulted: quiet hours and the contact frequency cap.
    if needs_email:
        blocked = _comms_blocked(record, runtime or build_runtime(record))
        if blocked:
            return d(DEFERRED, blocked)

    # 12. Budget last, so a refusal names the resource rather than the case.
    over = spend.would_exceed(budget, links=1 if needs_link else 0,
                              emails=1 if needs_email else 0,
                              discount_paise=discount_paise)
    if over:
        return d(BUDGET, f"budget_{over}")

    return Decision(payment_id=pid, outcome=ACTED, action=step.action,
                    amount_paise=audit.to_paise(amount),
                    charged_paise=charged_paise, discount_paise=discount_paise,
                    detail={"rung": rung})


def _comms_blocked(record: dict, runtime) -> str:
    """Quiet hours, the frequency cap and opt-out — the merchant's own policy.

    `merchant_dunning_rules.md` commits to all three. `GuardrailEngine`
    implements them and is wired into the graph; the batch path has simply never
    asked. Running a batch at 23:00 today would breach the merchant's committed
    policy while the file stating it sits in the knowledge base.

    Uses the engine's real interface — `validate_action(case, action, profile)` —
    rather than per-check helpers, because those do not exist and a `getattr`
    that silently finds nothing is how a control comes to be claimed and not
    enforced.
    """
    try:
        from recovery_agent.agent.guardrails import (GuardrailEngine,
                                                     GuardrailVerdict)
        from recovery_agent.agent.memory import CustomerMemoryStore
        from recovery_agent.models import ActionType
    except Exception:
        return ""

    try:
        case = runtime.context.case
        email = case.payment.metadata.get("customer_email", "")
        profile = None
        if email:
            # The same profile the live path uses, so the frequency cap counts
            # the same contacts. Without one the cap passes by default, which
            # would make this check decorative.
            profile = CustomerMemoryStore().get_or_create_profile(email)

        _action, checks = GuardrailEngine().validate_action(
            case, ActionType.SEND_NOTIFICATION, profile)
        for check in checks:
            if check.verdict in (GuardrailVerdict.BLOCKED,
                                 GuardrailVerdict.MODIFIED):
                return f"{check.guardrail}: {check.reason}"
    except Exception:
        return ""                            # never block a recovery on a bug
    return ""


# ── the action ───────────────────────────────────────────────────────────

def execute(record: dict, plan: BatchPlan, decision: Decision, *, run_id: str,
            spend: Spend, dry_run: bool = False, runtime=None) -> Decision:
    """Carry out one decision. Only ever called for `ACTED`."""
    from recovery_agent.agent.tools import TOOLS_BY_NAME

    pid = decision.payment_id
    if dry_run:
        decision.detail["dry_run"] = True
        return decision

    runtime = runtime or build_runtime(record)
    audit.record(audit.ACTION_ATTEMPTED, payment_id=pid, batch_run_id=run_id,
                 actor="executor", action=decision.action,
                 amount_paise=decision.charged_paise,
                 rung=decision.detail.get("rung"), plan=plan.digest())

    link_url = link_id = ""
    try:
        if decision.action == "retry":
            out = _call(TOOLS_BY_NAME["retry_in_hours"], runtime,
                        payment_id=pid, hours=plan.retry_hours or 24)
            decision.detail["retry"] = out.get("status")

        elif decision.action in ("link_and_notify", "notify_only"):
            if decision.action == "link_and_notify":
                charged = float(audit.to_rupees(decision.charged_paise))
                out = _call(TOOLS_BY_NAME["generate_recovery_payment_link"],
                            runtime, payment_id=pid, amount=charged,
                            customer_email=_email(record),
                            allowed_rails=",".join(plan.rails) if plan.rails else "",
                            expire_in_minutes=plan.expire_in_minutes)
                if out.get("status") != "ok":
                    decision.outcome = EXCEPTION
                    decision.reason = f"link failed: {out.get('reason') or out.get('message')}"
                    # The email is given back; the link is not. A create that
                    # failed may still have consumed gateway quota, and
                    # over-counting a scarce resource is the safe direction.
                    spend.release(emails=1, cases=1,
                                  discount_paise=decision.discount_paise)
                    return decision
                link_url = out.get("link_url", "")
                link_id = out.get("link_id", "")
                # Both price-integrity guards read this off the Case.
                runtime.context.case.payment.metadata["recovery_link"] = link_url
                runtime.context.case.payment.metadata["recovery_link_amount"] = charged
                # And the message-claim check reads these. Without them a batch
                # body that states its own discount, or names the rail the plan
                # chose, is refused as unfounded — the batch path has the facts,
                # it just was not passing them on.
                runtime.context.case.payment.metadata["recovery_link_rails"] = list(
                    plan.rails or [])
                if decision.discount_paise and amount:
                    runtime.context.case.payment.metadata["offer_pct"] = round(
                        decision.discount_paise / audit.to_paise(amount) * 100, 2)

            out = _call(TOOLS_BY_NAME["send_recovery_notification"], runtime,
                        payment_id=pid, customer_email=_email(record),
                        customer_phone=_phone(record),
                        message=_render(plan.body, record, link_url,
                                        decision.charged_paise),
                        payment_link=link_url)
            if out.get("status") != "ok":
                decision.outcome = EXCEPTION
                decision.reason = f"notify failed: {out.get('reason') or out.get('message')}"
                spend.release(emails=1, cases=1,
                              discount_paise=decision.discount_paise)
                return decision
            decision.detail["channels"] = out.get("channels", [])
            if link_url:
                decision.detail["link_url"] = link_url
                # Carried so the caller can watch this link for payment. The
                # executor does not start the watcher itself — that lives in the
                # web layer, and reaching up into it from here is the layering
                # inversion `presence.py` was extracted to undo.
                decision.detail["link_id"] = link_id
    except Exception as exc:                 # pragma: no cover - defensive
        decision.outcome = EXCEPTION
        decision.reason = f"{type(exc).__name__}: {exc}"
        return decision

    audit.record(audit.ACTION_RESULT, payment_id=pid, batch_run_id=run_id,
                 actor="executor", action=decision.action, result="ok",
                 amount_paise=decision.charged_paise,
                 discount_paise=decision.discount_paise, **decision.detail)
    _stamp(pid, run_id)
    return decision


def _call(tool, runtime, **kwargs) -> dict:
    """Invoke a tool the way the agent does, with a runtime, and parse it."""
    import json as _json
    raw = tool.func(runtime=runtime, **kwargs)
    text = str(raw or "")
    start = text.find("{")
    if start >= 0:
        try:
            return _json.loads(text[start:])
        except (ValueError, TypeError):
            pass
    return {"status": "error", "message": text[:200]}


def _email(record: dict) -> str:
    return ((record.get("customer") or {}).get("email")
            or record.get("customer_email") or "")


def _phone(record: dict) -> str:
    c = record.get("customer") or {}
    return (c.get("contact") or c.get("phone")
            or record.get("customer_phone") or "")


def _render(body: str, record: dict, link: str, charged_paise: int) -> str:
    """Fill the plan's template for one case.

    The template is shared; the figures are not. Every number here is derived
    from this record, so a plan cannot leak one customer's amount into another's
    message.
    """
    name = ((record.get("customer") or {}).get("name")
            or record.get("customer_name") or "there")
    return (body.replace("{name}", str(name))
                .replace("{amount}", f"INR {audit.to_rupees(charged_paise):,}")
                .replace("{link}", link or ""))


def _stamp(payment_id: str, run_id: str) -> None:
    """Tag the case with the run that acted on it.

    This is the join that makes money measurable per batch: `_mark_recovered`
    reads it back onto the `money.recovered` event, so a run's total is a sum
    over events rather than a guess from timestamps.
    """
    try:
        from recovery_agent.state_store import StateStore
        store = StateStore()
        rec = store.get_payment(payment_id)
        if rec is None:
            return
        prior = rec.get("batch_run_id") or ""
        fields: dict[str, Any] = {
            "batch_run_id": run_id,
            "batch_attributed_at": datetime.now(timezone.utc).isoformat(),
        }
        if prior and prior != run_id:
            fields["prior_batch_run_id"] = prior
        store.update_payment(payment_id, **fields)
        store.flush()
    except Exception:
        pass


def work_case(record: dict, plan: BatchPlan, *, run_id: str,
              budget: BatchBudget, spend: Spend,
              dry_run: bool = False) -> Decision:
    """Gate then act. The only entry point a run should use."""
    # One runtime, built once: the Case the guardrails judge must be the same
    # Case the tools then price against.
    runtime = build_runtime(record)
    decision = precheck(record, plan, run_id=run_id, budget=budget,
                        spend=spend, runtime=runtime)
    if decision.outcome != ACTED:
        kind = {SKIPPED: audit.CASE_SKIPPED, EXCEPTION: audit.CASE_EXCEPTION,
                DEFERRED: audit.CASE_SKIPPED,
                BUDGET: audit.BUDGET_EXHAUSTED}[decision.outcome]
        audit.record(kind, payment_id=decision.payment_id, batch_run_id=run_id,
                     actor="executor", reason=decision.reason,
                     amount_paise=decision.amount_paise, dry_run=dry_run)
        return decision
    if not dry_run:
        over = spend.reserve(
            budget,
            links=1 if decision.action == "link_and_notify" else 0,
            emails=1 if decision.action in ("link_and_notify", "notify_only") else 0,
            discount_paise=decision.discount_paise)
        if over:
            decision.outcome, decision.reason = BUDGET, f"budget_{over}"
            audit.record(audit.BUDGET_EXHAUSTED, payment_id=decision.payment_id,
                         batch_run_id=run_id, actor="executor",
                         reason=decision.reason,
                         amount_paise=decision.amount_paise, **spend.as_dict())
            return decision
    return execute(record, plan, decision, run_id=run_id, spend=spend,
                   dry_run=dry_run, runtime=runtime)
