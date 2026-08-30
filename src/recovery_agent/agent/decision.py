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
from recovery_agent.agent.strategy_metrics import StrategyMetricsStore, ThompsonBandit
from recovery_agent.agent.vector_memory import VectorMemoryStore
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

# Shared StrategyMetricsStore instance (built once, reused)
_strategy_metrics: StrategyMetricsStore | None = None

# Shared ThompsonBandit instance (built once, reused)
_thompson_bandit: ThompsonBandit | None = None


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


def _get_strategy_metrics() -> StrategyMetricsStore:
    """Lazy-init the shared strategy metrics store."""
    global _strategy_metrics
    if _strategy_metrics is None:
        _strategy_metrics = StrategyMetricsStore()
    return _strategy_metrics


def _get_thompson_bandit() -> ThompsonBandit:
    """Lazy-init the shared Thompson Sampling bandit."""
    global _thompson_bandit
    if _thompson_bandit is None:
        _thompson_bandit = ThompsonBandit(_get_strategy_metrics())
    return _thompson_bandit


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

    # User dropoff → ACTIVE (customer must be re-engaged)
    if cause == FailureType.USER_DROPOFF:
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

Select the optimal intervention for a failed payment. Reason about what each action ACTUALLY DOES.

═══ ACTIONS (7 total, each belongs to one tier) ═══

TIER 1 — SILENT (background, customer unaware):
  retry_payment: Re-charges the SAME method immediately. Use for transient failures only.
  wait_and_retry: Delays then retries SAME method. Use when conditions improve with time.

TIER 2 — ACTIVE (customer-facing):
  send_notification: Email/SMS explaining failure + asking customer to act.
  update_payment_method: Opens checkout for customer to enter NEW card/UPI/netbanking.
  voice_call: Initiate an AI voice call to the customer via SuperU. Use for: high-value payments (>INR 1000), user dropoff/abandonment, or when email/SMS notification has already been sent with no response. The AI agent will call the customer, explain the issue, and send a payment link during the call.

ALWAYS:
  escalate_to_human: Transfers to human agent (hard declines, fraud, high-value after failed attempts).
  abandon: Last resort only (opted out, all methods exhausted, cost exceeds revenue).

═══ FAILURE → TIER → ACTION ═══

CARD_EXPIRED → ACTIVE. Card is broken. → update_payment_method (directly) or send_notification (ask customer). NEVER retry same card.
INSUFFICIENT_FUNDS → SILENT. Money issue, not method. → wait_and_retry (funds may arrive) → send_notification → retry_payment. NEVER update_payment_method.
BANK_DECLINED → SILENT. May be temporary. → retry_payment → send_notification → update_payment_method.
NETWORK_TIMEOUT → SILENT. Gateway glitch. → retry_payment → wait_and_retry. NEVER update_payment_method.
MANDATE_REVOKED → ACTIVE. Customer cancelled autopay. → send_notification (re-authorize) → update_payment_method. NEVER retry.
RISK_BLOCK → ACTIVE. Fraud review needed. → escalate_to_human.
USER_DROPOFF → ACTIVE. Customer abandoned checkout. → send_notification (re-engage with payment link) → voice_call (high-value or no response) → update_payment_method. NEVER retry (hostile).

═══ HARD RULES ═══
- Instrument-switch text ("use another payment method", "expired", "invalid card") → ACTIVE + update_payment_method. NEVER retry same method.
- Hard decline codes (41/43/54/14/04/46/57/93) → escalate_to_human IMMEDIATELY. $0.10/attempt penalty.

Output EXACTLY this JSON:
{"decided_action": "<action name>", "intervention_reasoning": "<WHY this action for THIS failure>", "tier": "<silent|active>"}"""


def _build_strategy_prompt(
    case: Case,
    profile: CustomerProfile | None,
    available_rails: list[str],
    optimal_channel: str,
    similar_context: str = "",
    empirical_context: str = "",
) -> str:
    """Build the strategy planning prompt with similar case context and empirical data."""
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

SIMILAR PAST CASES:
{similar_context if similar_context else "No similar cases found in memory."}

EMPIRICAL EVIDENCE:
{empirical_context if empirical_context else "No historical data yet — building baseline."}

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

        elif cause == FailureType.USER_DROPOFF:
            if attempts == 0:
                return ActionType.SEND_NOTIFICATION
            elif attempts == 1:
                return ActionType.UPDATE_PAYMENT_METHOD
            else:
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


def _deterministic_strategy(case: Case) -> ActionType | None:
    """Fast-path for unambiguous failure types — skip LLM entirely.

    Returns a definitive action for well-understood failure patterns
    where the correct intervention is obvious and unambiguous.
    Returns None for ambiguous cases that benefit from LLM reasoning.
    """
    if not case.diagnosis:
        return None

    cause = case.diagnosis.root_cause
    attempts = case.attempt_count

    # Instrument-switch text → always ACTIVE + update_payment_method
    if _needs_instrument_switch(case):
        if attempts == 0:
            return ActionType.UPDATE_PAYMENT_METHOD
        elif attempts == 1:
            return ActionType.SEND_NOTIFICATION
        else:
            return ActionType.ESCALATE_TO_HUMAN

    # Hard decline → always escalate (attempt 0 or not)
    if case.payment.failure_code in HARD_DECLINES:
        return ActionType.ESCALATE_TO_HUMAN

    # Voice call for high-value user dropoffs or second attempt after notification
    if case.payment.amount >= 1000 and cause in (
        FailureType.USER_DROPOFF,
    ):
        # High-value abandonment — voice call has highest conversion
        return ActionType.VOICE_CALL
    if attempts >= 2 and cause not in HARD_DECLINES:
        # Already tried notification — escalate to voice
        return ActionType.VOICE_CALL

    # Card expired → customer must update payment method
    if cause == FailureType.CARD_EXPIRED:
        if attempts == 0:
            return ActionType.UPDATE_PAYMENT_METHOD
        elif attempts == 1:
            return ActionType.SEND_NOTIFICATION
        else:
            return ActionType.ESCALATE_TO_HUMAN

    # Mandate revoked → customer must re-authorize
    if cause == FailureType.MANDATE_REVOKED:
        if attempts == 0:
            return ActionType.SEND_NOTIFICATION
        else:
            return ActionType.ESCALATE_TO_HUMAN

    # Risk block → needs human review
    if cause == FailureType.RISK_BLOCK:
        return ActionType.ESCALATE_TO_HUMAN

    # User dropoff → re-engage customer
    if cause == FailureType.USER_DROPOFF:
        if attempts == 0:
            return ActionType.SEND_NOTIFICATION
        elif attempts == 1:
            return ActionType.UPDATE_PAYMENT_METHOD
        else:
            return ActionType.ESCALATE_TO_HUMAN

    # Network timeout → transient, just retry
    if cause == FailureType.NETWORK_TIMEOUT:
        if attempts == 0:
            return ActionType.RETRY_PAYMENT
        elif attempts == 1:
            return ActionType.WAIT_AND_RETRY
        else:
            return ActionType.ESCALATE_TO_HUMAN

    # Insufficient funds → funds may arrive
    if cause == FailureType.INSUFFICIENT_FUNDS:
        if attempts == 0:
            return ActionType.WAIT_AND_RETRY
        elif attempts == 1:
            return ActionType.RETRY_PAYMENT
        else:
            return ActionType.ESCALATE_TO_HUMAN

    # Bank declined → may be transient
    if cause == FailureType.BANK_DECLINED:
        if attempts == 0:
            return ActionType.RETRY_PAYMENT
        elif attempts == 1:
            return ActionType.SEND_NOTIFICATION
        else:
            return ActionType.ESCALATE_TO_HUMAN

    return None


def decide_intervention(
    case: Case,
    profile: CustomerProfile | None = None,
    memory: CustomerMemoryStore | None = None,
    kg_router: RazorpayKnowledgeGraph | None = None,
    vector_memory: VectorMemoryStore | None = None,
    strategy_metrics: StrategyMetricsStore | None = None,
    bandit: ThompsonBandit | None = None,
) -> ActionType:
    """Choose the appropriate intervention using LLM strategy planning.

    Two-tier decision logic:
    1. Check for hard decline → immediate escalation
    2. Assign tier (silent/active) based on failure type
    3. Query bandit for empirical recommendation
    4. LLM plans strategy within tier constraints + empirical context
    5. Bandit override if high confidence
    6. Enforce tier constraints on final action
    7. Record outcome for future learning

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

    # === STEP 3.5: Query bandit for empirical recommendation ===
    metrics = strategy_metrics or _get_strategy_metrics()
    bandit_inst = bandit or _get_thompson_bandit()
    empirical_ctx = bandit_inst.get_empirical_context(cause)
    bandit_action = bandit_inst.select_action(cause)
    if bandit_action:
        case.payment.metadata["bandit_recommendation"] = bandit_action.value
        bandit_conf = bandit_inst.get_confidence(cause, bandit_action)
        case.payment.metadata["bandit_confidence"] = bandit_conf

    # === STEP 4: Deterministic fast-path (skip LLM for unambiguous cases) ===
    fast_action = _deterministic_strategy(case)
    if fast_action is not None:
        case.payment.metadata["strategy_reasoning"] = (
            f"Fast-path: {cause.value} attempt #{case.attempt_count + 1} → "
            f"{fast_action.value} (deterministic, no LLM needed)"
        )
        case.payment.metadata["strategy_source"] = "deterministic_fast_path"
        action = fast_action

        # Still record outcome for bandit learning
        if bandit_action:
            case.payment.metadata["bandit_recommendation"] = bandit_action.value
        return action

    # === STEP 4.5: LLM strategy planning (only for ambiguous cases) ===
    # Query vector memory for similar past cases
    similar_ctx = ""
    if vector_memory and vector_memory.is_available:
        similar_ctx = vector_memory.get_decision_context(case)

    prompt = _build_strategy_prompt(case, profile, available_rails, preferred_channel, similar_ctx, empirical_ctx)
    result = invoke_llm_json(
        prompt=prompt,
        system=STRATEGY_SYSTEM_PROMPT,
        temperature=0,
        max_tokens=1024,
    )

    if result is None:
        action = _heuristic_fallback(case)
        case.payment.metadata["strategy_source"] = "heuristic_fallback"
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
            case.payment.metadata["strategy_source"] = "heuristic_fallback"
        else:
            case.payment.metadata["strategy_source"] = "llm_strategy_planner"

        # Store LLM tier decision for comparison
        llm_tier_str = result.get("tier", "").lower().strip()
        if llm_tier_str in ("silent", "active"):
            llm_tier = RecoveryTier(llm_tier_str)
            if llm_tier != tier:
                case.payment.metadata["tier_override_by_llm"] = True

    # === STEP 4.5: Bandit override (if high confidence) ===
    # If bandit has high confidence (>80%) and disagrees with LLM, override
    if bandit_action and bandit_action != action:
        bandit_conf = bandit_inst.get_confidence(cause, bandit_action)
        if bandit_conf > 0.80:
            # Bandit is very confident — override LLM
            case.payment.metadata["bandit_override"] = True
            case.payment.metadata["bandit_override_from"] = action.value
            case.payment.metadata["bandit_override_to"] = bandit_action.value
            case.payment.metadata["bandit_override_confidence"] = bandit_conf
            action = bandit_action

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
    vector_memory: VectorMemoryStore | None = None,
    strategy_metrics: StrategyMetricsStore | None = None,
    bandit: ThompsonBandit | None = None,
) -> Case:
    """Run decision layer on a case and update its state.

    Transitions case from DIAGNOSING → DIAGNOSED after LLM selects an intervention.
    Queries KG router for optimal recovery rails and passes them as context.

    Source: Planning with code execution
    https://www.deeplearning.ai/courses/agentic-ai (Module 5)
    """
    case.status = CaseStatus.DIAGNOSED

    action = decide_intervention(
        case, profile=profile, memory=memory, kg_router=kg_router,
        vector_memory=vector_memory, strategy_metrics=strategy_metrics, bandit=bandit,
    )

    case.payment.metadata["decided_action"] = action.value

    if memory and profile:
        case.payment.metadata["optimal_channel"] = memory.get_optimal_channel(
            profile.customer_id
        )

    return case
