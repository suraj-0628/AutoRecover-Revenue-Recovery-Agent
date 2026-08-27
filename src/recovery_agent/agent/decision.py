"""Decision layer — LLM strategy planner for intervention selection.

Two-tier decision logic:
- Tier 1 (Silent): Background retries, NO customer contact
- Tier 2 (Active): Customer-facing emails/SMS/payment updates

Includes decline-code-specific routing, hard decline prevention,
and payday timing intelligence.

Source: Planning pattern from Agentic AI (Andrew Ng), Module 5
        Tool selection from Evaluating AI Agents
"""
from __future__ import annotations

from datetime import datetime

from recovery_agent.agent.decline_router import DeclineCodeRouter
from recovery_agent.agent.kg_router import RazorpayKnowledgeGraph
from recovery_agent.agent.llm_client import invoke_llm_json
from recovery_agent.agent.memory import CustomerMemoryStore
from recovery_agent.agent.payday_scheduler import PaydayScheduler
from recovery_agent.models import (
    ActionType,
    Case,
    CaseStatus,
    CustomerProfile,
    FailureType,
    HARD_DECLINES,
    RecoveryTier,
    SILENT_ACTIONS,
    ACTIVE_ACTIONS,
)


# Shared KG router instance (built once, reused)
_kg_router: RazorpayKnowledgeGraph | None = None

# Shared DeclineCodeRouter instance (built once, reused)
_decline_router: DeclineCodeRouter | None = None

# Shared PaydayScheduler instance (built once, reused)
_payday_scheduler: PaydayScheduler | None = None


def _get_kg_router() -> RazorpayKnowledgeGraph:
    """Lazy-init the shared KG router."""
    global _kg_router
    if _kg_router is None:
        _kg_router = RazorpayKnowledgeGraph()
    return _kg_router


def _get_decline_router() -> DeclineCodeRouter:
    """Lazy-init the shared decline router."""
    global _decline_router
    if _decline_router is None:
        _decline_router = DeclineCodeRouter()
    return _decline_router


def _get_payday_scheduler() -> PaydayScheduler:
    """Lazy-init the shared payday scheduler."""
    global _payday_scheduler
    if _payday_scheduler is None:
        _payday_scheduler = PaydayScheduler()
    return _payday_scheduler


INSTRUMENT_SWITCH_KEYWORDS = (
    "use another payment instrument",
    "use another payment method",
    "try another method",
    "try another payment",
    "expired",
    "invalid card",
)


def _needs_instrument_switch(case: Case) -> bool:
    """Check if the failure message indicates the customer must switch payment instruments.

    Inspects failure_reason and error_description for keywords that mean
    the current payment method is broken and cannot work on retry.
    """
    text = (
        (case.payment.failure_reason or "")
        + " "
        + (case.payment.metadata.get("error_description", "") or "")
    ).lower()

    if any(kw in text for kw in INSTRUMENT_SWITCH_KEYWORDS):
        return True

    # Also check mandate_revoked via failure code
    if case.diagnosis and case.diagnosis.root_cause == FailureType.MANDATE_REVOKED:
        return True

    return False


def _assign_tier(case: Case) -> RecoveryTier:
    """Assign recovery tier based on failure type, attempt count, and error message.

    Tier 1 (Silent) MUST be set for:
    - Network timeouts (transient — just retry)
    - Initial bank declines (may be temporary)
    - Insufficient funds during payday window (salary credit imminent)

    Tier 2 (Active) MUST be set for:
    - Card expired (customer must update)
    - Mandate revoked (customer must re-authorize)
    - Risk block (needs human review)
    - Hard declines (never retry)
    - Any failure message that says "use another payment instrument/method"
    - After silent tier exhausted

    Inspired by Redux "Silent First" architecture.
    """
    if not case.diagnosis:
        return RecoveryTier.ACTIVE

    cause = case.diagnosis.root_cause

    # Hard declines → always ACTIVE (need immediate escalation)
    if case.payment.failure_code in HARD_DECLINES:
        return RecoveryTier.ACTIVE

    # Card expired → ACTIVE (customer must update payment method)
    if cause == FailureType.CARD_EXPIRED:
        return RecoveryTier.ACTIVE

    # Mandate revoked → ACTIVE (customer must re-authorize)
    if cause == FailureType.MANDATE_REVOKED:
        return RecoveryTier.ACTIVE

    # Risk block → ACTIVE (needs human review)
    if cause == FailureType.RISK_BLOCK:
        return RecoveryTier.ACTIVE

    # Instrument-switching text → ACTIVE (retruing same method is pointless)
    if _needs_instrument_switch(case):
        return RecoveryTier.ACTIVE

    # Network timeout → SILENT (transient, just retry)
    if cause == FailureType.NETWORK_TIMEOUT:
        return RecoveryTier.SILENT

    # Bank declined → SILENT (may be temporary)
    if cause == FailureType.BANK_DECLINED:
        return RecoveryTier.SILENT

    # Insufficient funds → SILENT (payday timing)
    if cause == FailureType.INSUFFICIENT_FUNDS:
        return RecoveryTier.SILENT

    # Default to silent for unclassified — try background retries first
    return RecoveryTier.SILENT


def _check_hard_decline(case: Case) -> ActionType | None:
    """Check if failure code is a hard decline.

    Returns ESCALATE_TO_HUMAN if hard decline, None otherwise.
    Also increments penalties_prevented counter.
    """
    failure_code = case.payment.failure_code

    # Check raw code first
    if failure_code in HARD_DECLINES:
        case.penalties_prevented += 1
        case.payment.metadata["hard_decline_blocked"] = True
        case.payment.metadata["penalty_prevented"] = f"${case.penalties_prevented * 0.10:.2f}"
        return ActionType.ESCALATE_TO_HUMAN

    # Check via failure type
    if case.diagnosis:
        cause = case.diagnosis.root_cause
        if cause == FailureType.CARD_EXPIRED and failure_code in ("54", "14"):
            case.penalties_prevented += 1
            case.payment.metadata["hard_decline_blocked"] = True
            return ActionType.ESCALATE_TO_HUMAN

    return None


STRATEGY_SYSTEM_PROMPT = """You are a Razorpay revenue recovery strategy planner.

Your task: Select the optimal intervention action for a failed payment.
You MUST reason about what each action ACTUALLY DOES before choosing.

═══ RECOVERY TIERS (two-tier architecture) ═══

TIER 1 — SILENT RECOVERY (background, no customer contact):
- retry_payment: Re-charges the SAME card/payment method immediately via Razorpay API.
- wait_and_retry: Pauses for a configurable delay, then retries the SAME payment method.
- Customer stays active, unaware of failure. Multiple silent retries allowed.

TIER 2 — ACTIVE RECOVERY (customer-facing):
- send_notification: Sends email/SMS/WhatsApp explaining the failure.
- update_payment_method: Opens Razorpay checkout for customer to enter NEW card or switch to UPI/netbanking.
- Only triggered when Tier 1 exhausted or failure requires customer action.

═══ INSTRUMENT SWITCH RULE ══

If the failure message contains "use another payment instrument", "use another payment method",
"try another method", "expired", or "invalid card" — the current payment method is BROKEN.
Retrying the same method WILL fail again. You MUST:
  - TIER: ACTIVE
  - ACTION: update_payment_method (let customer switch to a working method)
  - NEVER: retry_payment or wait_and_retry (same broken method → same failure)

══ HARD DECLINE PREVENTION ═══

NEVER retry these codes — they trigger $0.10/attempt Visa/MC network penalties:
- 41 (Lost card), 43 (Stolen card), 54 (Expired card), 14 (Invalid card)
- 04 (Pick up card — fraud), 46 (Closed account), 57 (Transaction not permitted), 93 (Cannot complete)
If detected, IMMEDIATELY escalate to human.

═══ ACTION SEMANTICS (what each tool does mechanically) ═══

retry_payment (SILENT TIER)
  WHAT IT DOES: Re-charges the SAME card/payment method immediately via Razorpay API.
  WHEN TO USE: Only when the failure is TRANSIENT — the same method could succeed on retry.
  GOOD FOR: network_timeout (gateway glitch), bank_declined (temporary bank hiccup), insufficient_funds (funds may have arrived).
  BAD FOR: card_expired (same expired card will fail again!), mandate_revoked (same revoked mandate will fail again!).

send_notification (ACTIVE TIER)
  WHAT IT DOES: Sends an email/SMS/WhatsApp message to the customer explaining the failure and asking them to act.
  WHEN TO USE: When the customer needs to DO something — update card, re-authorize mandate, try a different method.
  GOOD FOR: card_expired (ask to update card), mandate_revoked (ask to re-authorize), first attempt on any failure.
  BAD FOR: network_timeout (no customer action needed — just retry).

update_payment_method (ACTIVE TIER)
  WHAT IT DOES: Opens a Razorpay checkout modal pre-configured for the customer to enter a NEW card or switch to UPI/netbanking.
  WHEN TO USE: When the current payment method is BROKEN and cannot work — expired card, blocked card, revoked mandate.
  GOOD FOR: card_expired (must update card), mandate_revoked (must re-authorize via new method).
  BAD FOR: network_timeout (current method is fine — just retry), insufficient_funds (method is fine — funds are the issue).

wait_and_retry (SILENT TIER)
  WHAT IT DOES: Pauses for a configurable delay, then retries the SAME payment method.
  WHEN TO USE: When the failure is TEMPORARY and conditions will improve with time.
  GOOD FOR: insufficient_funds (salary may arrive), bank_declined (bank block may lift), network_timeout (gateway may recover).
  BAD FOR: card_expired (card won't un-expire by waiting!), mandate_revoked (mandate won't re-authorize by waiting!).

escalate_to_human
  WHAT IT DOES: Transfers the case to a human support agent for manual intervention.
  WHEN TO USE: When automated recovery has failed or the case requires human judgment.
  GOOD FOR: risk_block (fraud review needed), hard declines (41/43/54/14/04/46/57/93), high-value cases after multiple failed attempts.

abandon
  WHAT IT DOES: Stops all recovery attempts. Customer is not contacted again.
  WHEN TO USE: Only as absolute last resort — customer opted out, all methods exhausted, or recovery cost exceeds revenue value.

═══ FAILURE → TIER → ACTION MAPPING (reason about WHY) ═══

CARD_EXPIRED: The card itself is the problem. Retrying the same card WILL fail again.
  → TIER: ACTIVE (customer must update card)
  → First try: send_notification (inform customer, ask to update)
  → Or directly: update_payment_method (let them fix it now)
  → NEVER: retry_payment or wait_and_retry (pointless — card is expired)

INSUFFICIENT_FUNDS: The payment method is fine, but the account lacks balance.
  → TIER: SILENT (funds may arrive)
  → First try: wait_and_retry (funds may arrive — especially if salary-dependent)
  → Then: send_notification (remind customer to add funds)
  → Then: retry_payment (try once funds should be there)
  → NEVER: update_payment_method (method is fine — money is the issue)

BANK_DECLINED: Bank rejected, possibly transient.
  → TIER: SILENT (may be temporary)
  → First try: retry_payment (may be a temporary bank hold)
  → Then: send_notification (ask customer to check with bank or try different method)
  → Then: update_payment_method (switch to a working method)

NETWORK_TIMEOUT: Gateway glitch, nothing wrong with the payment method.
  → TIER: SILENT (transient — just retry)
  → First try: retry_payment (transient — retry immediately)
  → Then: wait_and_retry (if still failing, wait and try again)
  → NEVER: update_payment_method (method is fine — network was the issue)

MANDATE_REVOKED: UPI autopay mandate cancelled by customer.
  → TIER: ACTIVE (customer must re-authorize)
  → First try: send_notification (ask to re-authorize mandate)
  → Then: update_payment_method (switch to manual payment)
  → NEVER: retry_payment or wait_and_retry (revoked mandate won't work)

RISK_BLOCK: Fraud/risk system flagged the payment.
  → TIER: ACTIVE (needs human review)
  → escalate_to_human (requires human fraud review)

You must output EXACTLY this JSON format:
{
  "decided_action": "<one of the 6 action names>",
  "intervention_reasoning": "<step-by-step reasoning explaining WHY this specific action was chosen for this specific failure type, referencing what the action actually does>",
  "tier": "<silent or active>"
}"""


def _build_strategy_prompt(
    case: Case,
    profile: CustomerProfile | None,
    available_rails: list[str],
    optimal_channel: str,
) -> str:
    """Build the strategy planning prompt."""
    diagnosis = case.diagnosis
    payment = case.payment

    diagnosis_context = "No diagnosis available."
    if diagnosis:
        diagnosis_context = (
            f"Root Cause: {diagnosis.root_cause.value}\n"
            f"Confidence: {diagnosis.confidence:.0%}\n"
            f"Diagnostic Reasoning: {diagnosis.reasoning}"
        )

    memory_context = "No customer profile available."
    if profile:
        salary_info = "unknown"
        if profile.salary_window.typical_pay_day > 0:
            current_day = datetime.now().day
            in_window = abs(current_day - profile.salary_window.typical_pay_day) <= 2
            salary_info = (
                f"pay day={profile.salary_window.typical_pay_day}, "
                f"currently {'IN' if in_window else 'OUTSIDE'} liquidity window"
            )

        promise_info = "none"
        if profile.promises:
            unfulfilled = [p for p in profile.promises if not p.fulfilled]
            if unfulfilled:
                promise_info = f"{len(unfulfilled)} unfulfilled (latest: {unfulfilled[-1].promised_date})"

        channel_info = "none"
        if profile.channel_success_rates:
            best = max(profile.channel_success_rates, key=profile.channel_success_rates.get)
            channel_info = f"best={best} ({profile.channel_success_rates[best]:.0%} success)"

        memory_context = (
            f"Total attempts: {profile.total_attempts}\n"
            f"Total recovered: INR {profile.total_recovered:,.2f}\n"
            f"Salary window: {salary_info}\n"
            f"Promise-to-pay: {promise_info}\n"
            f"Channel performance: {channel_info}\n"
            f"Preferred channel: {optimal_channel or 'sms'}\n"
            f"Opted out: {'yes' if profile.opt_out else 'no'}"
        )

    rails_context = "No rails discovered."
    if available_rails:
        rails_context = f"Available recovery rails: {', '.join(available_rails)}"

    attempt_context = (
        f"Current attempt: {case.attempt_count + 1} of {case.max_attempts}"
    )

    return f"""Select the optimal recovery intervention:

FAILURE DATA:
  Payment ID: {payment.payment_id}
  Amount: INR {payment.amount:,.2f}
  Failure Code: {payment.failure_code or 'not provided'}
  Failure Reason: {payment.failure_reason or 'not provided'}

DIAGNOSIS:
{diagnosis_context}

CUSTOMER MEMORY:
{memory_context}

RECOVERY RAILS:
{rails_context}

{attempt_context}

RECOVERY TIER: {case.recovery_tier.value.upper()}
  - SILENT: Background retries, NO customer contact allowed
  - ACTIVE: Customer-facing interventions permitted

GUARDRAIL CONSTRAINTS:
  - Quiet hours (9 PM - 8 AM): Communication actions deferred
  - Frequency cap: Max 2 communications per 24h
  - Double-debit lock: No retry if payment already succeeded or pending
  - Monetary cap: Auto-retry blocked above INR 500,000
  - Opt-out: No messages if customer opted out
  - Hard decline codes (41/43/54/14/04/46/57/93): NEVER retry, escalate immediately

BEFORE CHOOSING, ANSWER THESE QUESTIONS:
1. What is physically wrong with the payment? (expired card? no money? bank said no? network glitch?)
2. Is this a HARD decline code? (41/43/54/14/04/46/57/93) → ESCALATE IMMEDIATELY
3. What tier am I in? (SILENT = only retry_payment/wait_and_retry; ACTIVE = send_notification/update_payment_method allowed)
4. Can waiting fix this? (ONLY if the problem is temporary — funds arriving, bank block lifting)
5. Can retrying the SAME method fix this? (ONLY if the failure was transient — network glitch, temporary bank decline)
6. Does the customer need to DO something? (update card, re-authorize mandate, add funds)
7. Is this action appropriate for THIS failure type? (See the action semantics in your system prompt)

Output your strategy as JSON:"""


def _heuristic_fallback(case: Case) -> ActionType:
    """Simple heuristic fallback when LLM is unavailable.

    Tier-aware decision matrix:
    - Silent tier: only RETRY_PAYMENT and WAIT_AND_RETRY
    - Active tier: SEND_NOTIFICATION, UPDATE_PAYMENT_METHOD, ESCALATE
    - Hard declines: always ESCALATE
    """
    if not case.diagnosis:
        return ActionType.ABANDON

    cause = case.diagnosis.root_cause
    attempts = case.attempt_count
    max_attempts = case.max_attempts
    tier = case.recovery_tier

    # If near max attempts, always escalate
    if attempts >= max_attempts - 1:
        return ActionType.ESCALATE_TO_HUMAN

    # === TIER-AWARE LOGIC ===

    if tier == RecoveryTier.SILENT:
        # Silent tier: linear progression — retry → wait → escalate (no cycling)
        if cause == FailureType.NETWORK_TIMEOUT:
            if attempts == 0:
                return ActionType.RETRY_PAYMENT
            elif attempts == 1:
                return ActionType.WAIT_AND_RETRY
            else:
                return ActionType.ESCALATE_TO_HUMAN

        elif cause == FailureType.INSUFFICIENT_FUNDS:
            if attempts == 0:
                return ActionType.WAIT_AND_RETRY
            elif attempts == 1:
                return ActionType.RETRY_PAYMENT
            else:
                return ActionType.ESCALATE_TO_HUMAN

        elif cause == FailureType.BANK_DECLINED:
            if attempts == 0:
                return ActionType.RETRY_PAYMENT
            elif attempts == 1:
                return ActionType.WAIT_AND_RETRY
            else:
                return ActionType.ESCALATE_TO_HUMAN

        else:
            # Other failures in silent tier — try once, then escalate
            return ActionType.WAIT_AND_RETRY if attempts == 0 else ActionType.ESCALATE_TO_HUMAN

    else:
        # Active tier: customer-facing actions allowed
        if cause == FailureType.CARD_EXPIRED:
            if attempts == 0:
                return ActionType.UPDATE_PAYMENT_METHOD
            elif attempts == 1:
                return ActionType.SEND_NOTIFICATION
            else:
                return ActionType.ESCALATE_TO_HUMAN

        elif cause == FailureType.MANDATE_REVOKED:
            if attempts == 0:
                return ActionType.SEND_NOTIFICATION
            else:
                return ActionType.ESCALATE_TO_HUMAN

        elif cause == FailureType.RISK_BLOCK:
            return ActionType.ESCALATE_TO_HUMAN

        elif cause == FailureType.INSUFFICIENT_FUNDS:
            if attempts == 0:
                return ActionType.SEND_NOTIFICATION
            elif attempts == 1:
                return ActionType.UPDATE_PAYMENT_METHOD
            else:
                return ActionType.ESCALATE_TO_HUMAN

        elif cause == FailureType.NETWORK_TIMEOUT:
            if attempts == 0:
                return ActionType.RETRY_PAYMENT
            elif attempts == 1:
                return ActionType.SEND_NOTIFICATION
            else:
                return ActionType.ESCALATE_TO_HUMAN

        elif cause == FailureType.BANK_DECLINED:
            if attempts == 0:
                return ActionType.SEND_NOTIFICATION
            elif attempts == 1:
                return ActionType.UPDATE_PAYMENT_METHOD
            else:
                return ActionType.ESCALATE_TO_HUMAN

        else:
            if attempts < 2:
                return ActionType.SEND_NOTIFICATION
            else:
                return ActionType.ESCALATE_TO_HUMAN


def decide_intervention(
    case: Case,
    profile: CustomerProfile | None = None,
    memory: CustomerMemoryStore | None = None,
    kg_router: RazorpayKnowledgeGraph | None = None,
) -> ActionType:
    """Choose the appropriate intervention using LLM strategy planning.

    Two-tier decision logic:
    1. Check for hard decline → immediate escalation
    2. Assign tier (silent/active) based on failure type
    3. LLM plans strategy within tier constraints
    4. Enforce tier constraints on final action

    Falls back to heuristic rules if LLM is unavailable.

    Source: Planning pattern — agent creates plan then executes
    https://www.deeplearning.ai/courses/agentic-ai (Module 5)
    """
    if not case.diagnosis:
        return ActionType.ABANDON

    if case.attempt_count >= case.max_attempts - 1:
        case.payment.metadata["strategy_reasoning"] = f"Max recovery attempts reached ({case.attempt_count + 1}/{case.max_attempts}). Escalating to human."
        return ActionType.ESCALATE_TO_HUMAN

    # === STEP 1: Hard decline check ===
    hard_decline_action = _check_hard_decline(case)
    if hard_decline_action:
        case.payment.metadata["strategy_reasoning"] = (
            f"Hard decline code {case.payment.failure_code} detected. "
            f"Retrying would incur $0.10/attempt Visa/MC network penalty. "
            f"Escalating to human immediately."
        )
        return hard_decline_action

    # === STEP 2: Assign tier ===
    tier = _assign_tier(case)
    case.recovery_tier = tier
    case.payment.metadata["recovery_tier"] = tier.value

    # === STEP 3: Discover recovery rails (KG router) ===
    router = kg_router or _get_kg_router()
    cause = case.diagnosis.root_cause

    preferred_channel = ""
    if profile and memory:
        preferred_channel = memory.get_optimal_channel(profile.customer_id)

    available_rails = []
    if cause in (
        FailureType.CARD_EXPIRED,
        FailureType.BANK_DECLINED,
        FailureType.NETWORK_TIMEOUT,
        FailureType.MANDATE_REVOKED,
        FailureType.RISK_BLOCK,
    ):
        available_rails = router.discover_recovery_path(
            failure_code=cause.value,
            customer_id=case.payment.customer_id,
            preferred_channel=preferred_channel,
        )
        recommended_rail = router.recommend_optimal_rail(
            failure_code=cause.value,
            preferred_channel=preferred_channel,
        )
        case.payment.metadata["discovered_rail_path"] = available_rails
        case.payment.metadata["recommended_api_rail"] = recommended_rail

    # === STEP 4: LLM strategy planning ===
    prompt = _build_strategy_prompt(case, profile, available_rails, preferred_channel)
    result = invoke_llm_json(
        prompt=prompt,
        system=STRATEGY_SYSTEM_PROMPT,
        temperature=0,
        max_tokens=1024,
    )

    if result is None:
        action = _heuristic_fallback(case)
    else:
        # Parse decided action
        action_str = result.get("decided_action", "").lower().strip()
        action_map = {
            "retry_payment": ActionType.RETRY_PAYMENT,
            "send_notification": ActionType.SEND_NOTIFICATION,
            "escalate_to_human": ActionType.ESCALATE_TO_HUMAN,
            "update_payment_method": ActionType.UPDATE_PAYMENT_METHOD,
            "wait_and_retry": ActionType.WAIT_AND_RETRY,
            "abandon": ActionType.ABANDON,
        }
        action = action_map.get(action_str)
        if action is None:
            action = _heuristic_fallback(case)

        # Store LLM tier decision for comparison
        llm_tier_str = result.get("tier", "").lower().strip()
        if llm_tier_str in ("silent", "active"):
            llm_tier = RecoveryTier(llm_tier_str)
            if llm_tier != tier:
                case.payment.metadata["tier_override_by_llm"] = True

    # === STEP 5: Enforce tier constraints ===
    if tier == RecoveryTier.SILENT:
        # Silent tier: block customer-facing actions
        if action in ACTIVE_ACTIONS:
            # Override to WAIT_AND_RETRY instead of customer contact
            case.payment.metadata["tier_enforced"] = True
            case.payment.metadata["tier_enforced_reason"] = (
                f"Tier 1 (Silent) active. {action.value} blocked. "
                f"Falling back to wait_and_retry."
            )
            action = ActionType.WAIT_AND_RETRY
    else:
        # Active tier: all actions allowed
        pass

    # Store strategic reasoning in metadata
    reasoning = result.get("intervention_reasoning", "") if result else f"Heuristic fallback: {cause.value} → {action.value}"
    case.payment.metadata["strategy_reasoning"] = reasoning

    return action


def run_decision(
    case: Case,
    profile: CustomerProfile | None = None,
    memory: CustomerMemoryStore | None = None,
    kg_router: RazorpayKnowledgeGraph | None = None,
) -> Case:
    """Run decision layer on a case and update its state.

    Transitions case from DIAGNOSING → DIAGNOSED after LLM selects an intervention.
    Queries KG router for optimal recovery rails and passes them as context.

    Source: Planning with code execution
    https://www.deeplearning.ai/courses/agentic-ai (Module 5)
    """
    case.status = CaseStatus.DIAGNOSED

    action = decide_intervention(case, profile=profile, memory=memory, kg_router=kg_router)

    case.payment.metadata["decided_action"] = action.value

    if memory and profile:
        case.payment.metadata["optimal_channel"] = memory.get_optimal_channel(
            profile.customer_id
        )

    return case
