"""True ReAct Agent — LangGraph canonical pattern with Real Memory.

Architecture (Context7-grounded):
    START → agent → should_continue → [tools → agent | END]

The LLM reasons, calls tools, observes results, and decides what to do next.
Memory is managed via langmem tools (manage_memory, search_memory) that the
LLM calls during reasoning. No hardcoded reflection chains.

Guardrails: stopping.py enforces tier transitions and max attempts.
Every LLM call gets a manual OTel span sent to Phoenix for full trace visibility.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import nullcontext
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

import opentelemetry
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from recovery_agent.agent.tools import (
    RECOVERY_TOOLS, RecoveryContext, TOOLS_BY_NAME,
)
from recovery_agent.agent.governance import get_allowed_tools, mask_tool_output, AGENT_VERSION
from recovery_agent.models import CaseStatus

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# EXTENDED STATE — adds episode tracking to MessagesState
# ═══════════════════════════════════════════════════════════════

class RecoveryState(MessagesState):
    """Extended state for the recovery agent.

    tool_call_history: tracks (tool_name, args_hash) to detect repetition.
    phase: current recovery phase (planning, diagnosis, execution).
    """
    tool_call_history: list[dict]
    phase: str
    blocked_rounds: int


# ═══════════════════════════════════════════════════════════════
# MEMORY STORE — LangGraph InMemoryStore with semantic search
# ═══════════════════════════════════════════════════════════════

def _build_memory_store():
    """Long-term memory for customer facts, recovery episodes and lessons.

    This was an `InMemoryStore`, which lives and dies with the process. Every
    restart — every deploy — wiped everything the agent had ever learned about
    every customer. Nothing showed it, because a lesson written and recalled
    inside one process run looks identical to one that survives; the loss is
    only visible across a restart, and nobody was looking there.

    That is a real gap and worth naming precisely: it is NOT what caused the
    "recovered but kept going" behaviour. Those were state-ownership bugs
    inside a single process. This is the opposite problem — genuine amnesia,
    silent, and only between runs of the server.

    SQLite-backed now, so a lesson learned on one payment is still there for the
    next one after a restart. Falls back to in-memory rather than refusing to
    start, because losing memory is bad and refusing to recover payments is
    worse.
    """
    import os
    from pathlib import Path

    path = Path(os.getenv("STATE_DIR", "data")) / "agent_memory.db"
    try:
        from langgraph.store.sqlite import SqliteStore
        path.parent.mkdir(parents=True, exist_ok=True)
        cm = SqliteStore.from_conn_string(str(path))
        store = cm.__enter__()
        _KEEP_ALIVE.append(cm)          # the context manager owns the connection
        try:
            store.setup()
        except Exception:
            pass
        print(f"[Agent] long-term memory: sqlite at {path}", flush=True)
        return store
    except Exception as exc:
        from langgraph.store.memory import InMemoryStore
        print(f"[Agent] long-term memory: IN-MEMORY ONLY, lost on restart ({exc})",
              flush=True)
        return InMemoryStore()


#: Context managers whose connections must outlive this function.
_KEEP_ALIVE: list = []


# Module-level store singleton
_memory_store = None


def get_memory_store():
    global _memory_store
    if _memory_store is None:
        _memory_store = _build_memory_store()
    return _memory_store


# ═══════════════════════════════════════════════════════════════
# PHOENIX TRACING — OTel callback handler for LangGraph
# ═══════════════════════════════════════════════════════════════
#
# PROBLEM: Manual OTel spans in agent_node created orphaned traces
# disconnected from LangGraph's execution. ToolNode, guards, and
# stopping_check nodes produced zero Phoenix spans.
#
# FIX: Each custom node creates its own OTel span via _get_otel_tracer().
# ToolNode fires LangChain callbacks automatically (it's a RunnableCallable),
# so tools appear in traces without manual spans.
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# SYSTEM PROMPT — tells the LLM what it is and what tools to use
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a Razorpay revenue recovery agent. A payment failed. Your job is to RECOVER THE MONEY, not to investigate it.

BEFORE ANYTHING ELSE:
A block headed "WHAT IS TRUE RIGHT NOW" appears above, recomputed from the
payment record each turn. It — not the narrative below, and not your own memory
of this case — is what is actually true. When it says the case is SETTLED, the
money is in: write the lesson with manage_memory and stop, whatever anything
else in this conversation asks of you. When it says the case is still open, it
also tells you what has already been tried and what has not.

If you are ever unsure whether the money arrived, call check_payment_status.
It reads the record and confirms against Razorpay. Checking is never wasted.

SCOPE — you are working exactly ONE payment:
Everything you do belongs to the payment named in the case facts below. Memory is
shared across this customer's orders, so a recalled lesson may refer to a
DIFFERENT payment — read the payment id on it before you rely on it, and never
quote another order's amount, link or discount to this customer.

HOW YOUR TURN WORKS — this is not a conversation you stay in:
You act, then your turn ENDS. When anything happens — the customer dismisses a
notification, ignores one, or pays — you are started again with that news, and
your session for this payment carries on from where you left off.

When you have done what you can and the next move is someone else's, call
wait_for_customer. That ends your turn cleanly and is NOT giving up. Never invent
a tool to wait, watch, poll or monitor with; wait_for_customer is the one.

When the case is FINISHED — the money is back, or a person has it, or nothing
further is worth trying — call close_case. That is the ending. It records the
outcome and stores your lesson in one act, and no later signal will reopen the
case. Do not simply stop calling tools: from the outside that is
indistinguishable from running out of ideas, and the case is left looking
abandoned rather than decided.

YOUR TOOLS (these are the only ones that exist — never invent a name):
  Diagnose:  diagnose_payment_failure, check_payment_status,
             get_customer_payment_history, query_knowledge_base,
             discover_recovery_rail, search_memory
  Recover:   send_page_push                 (SILENT: the ONE plain nudge, to a
                                              customer still on the checkout.
                                              Costs nothing, carries no offer
                                              and no link — it reopens the page
                                              they are on. TRY THIS FIRST, and
                                              only once. Offers go by banner)
             (Offers are short-lived: create links with expire_in_minutes=16 and
              show the offer for the same window. Urgency is the point — a
              discount good for two days is not an offer, it is a price cut.)
             show_page_offer                 (put the authorised discount ON the
                                              checkout page — original struck
                                              through, new total, countdown — and
                                              notify at the same time. Use after
                                              the link exists at that price)
             get_recovery_offer              (what discount the store policy
                                              permits — call this BEFORE
                                              generate_recovery_payment_link, so
                                              the link is created at the price you
                                              are going to quote. Never invent a
                                              discount, and never mention one the
                                              link does not charge)
             generate_recovery_payment_link  (customer pays by another method)
             send_recovery_notification      (email/SMS the customer — always
                                              pass the link_url as payment_link)
             initiate_voice_call             (AI phone call; best for an
                                              unresponsive customer or a large
                                              amount. May be switched off, in
                                              which case pick another channel)
             retry_in_hours                  (silent background retry, no contact)
             escalate_to_human               (hand off to a person)
  Learn:     manage_memory                   (store a lesson for next time)
  Finish:    close_case                      (THE ending, every time: outcome +
                                              lesson, and the case is closed for
                                              good. "recovered" needs the money
                                              actually in; "escalated" needs a
                                              ticket to exist; "unrecoverable"
                                              needs the ladder exhausted. The
                                              tool checks all three — it is a
                                              record of a decision, not a
                                              formality)

HOW TO WORK — follow this order:

1. DIAGNOSE ONCE. You may call diagnostic tools in ONE round, at the start.
   Often the failure code in the context is already enough — then skip this.
   Diagnosis does not recover money, so do not linger there.

   The ONE exception is check_payment_status: call it as often as it matters,
   and ALWAYS before you spend money or contact the customer again. It reads
   changing state — the answer is different after the customer acts — so asking
   twice is not repetition, it is the difference between recovering a payment
   and discounting one that was already being paid.

2. THEN ACT. After at most one diagnostic round you MUST call a Recover tool.

   THE LADDER — every payment climbs the same rungs, in this order. You choose
   what to say and when; you do not choose to skip ahead. Each rung is tried
   only after the one before it has failed to get the money.

     1. page_push          A silent nudge on the checkout page. Costs nothing,
                           gives nothing away. Always first. Sending it ENDS
                           YOUR TURN — that is the point of it. You asked the
                           customer to act; give them the chance. You will be
                           started again the moment they do, or when the window
                           closes. Do not prepare the next rung "while you
                           wait": a customer who is mid-payment does not need a
                           discount, and offering one gives away money you were
                           about to receive in full.
     2. offer              The discount the store policy allows: create the link
                           at that price, put it on the page with show_page_offer
                           AND email it with send_recovery_notification. Do both
                           — the customer may be looking at either.
     3. voice_call         Ring them, ask why, negotiate inside policy. Only
                           after the offer has had ~15 minutes to work; the tool
                           refuses earlier and tells you how long is left.
     4. post_call_email    Send the link the call agreed to, by email.
     5. alternate_path     Your call. Something genuinely different from
                           everything already tried — another rail, a retry
                           timed to a payday, a different offer shape. There is
                           no fixed tool for this rung: whatever you choose that
                           you have not already done counts as it. A repeat of
                           something that already failed does not.
     6. escalate_to_human  ONLY when every rung above has been tried and the
                           money is still not back. Filing the ticket is not the
                           end of your work — follow it with close_case
                           (outcome="escalated"). A case handed to a person and
                           never closed looks abandoned, not delegated.

   escalate_to_human refuses while any rung remains, and tells you which one is
   next. It is the last resort, never the answer to a blocked tool. If a tool
   refuses you, read WHY — a refusal usually names what would be allowed instead
   (a smaller discount, a different channel, a wait). Take that, not the exit.

   FIRST, check the context for "FOLLOW-UP". If it says a recovery attempt was
   already delivered and the customer has not paid, the channel is what failed,
   not the message. Move to the next rung. Sending the same email again is the
   one thing that certainly will not work.

   If the customer may still be on the checkout page (the failure just happened,
   or they abandoned it), send_page_push is the cheapest thing that can work.
   Nothing is given away and nobody is interrupted.

   OTHERWISE pick by failure type:
     - Card expired / invalid / lost / stolen / closed account
         -> generate_recovery_payment_link, then send_recovery_notification
            passing that link_url as `payment_link`. An email with no link is
            useless to the customer.
            NEVER retry: the instrument is permanently dead.
     - Insufficient funds
         -> retry_in_hours (try 24-72h, aim for a payday). Do not contact yet.
     - Network / gateway timeout / issuer unavailable
         -> retry_in_hours with a short window (1-6h).
     - Risk, fraud, or dispute
         -> escalate_to_human immediately. This is the ONE case that skips the
            ladder: do not retry, do not contact.
     - Customer was already sent a link and has not paid
         -> escalate the CHANNEL, not the case. Move to the next rung.
            Do NOT simply send the same email again.
     - Anything you cannot classify
         -> generate_recovery_payment_link and work the ladder normally.

   When you have an authorised discount and a link created at that price, put it
   on the page with show_page_offer as well as emailing it — the customer may
   still be looking at the checkout, and an offer they can see beats one sitting
   in an inbox. The page figure and the link must agree; the tool refuses if not.

   If the money is already back — the briefing says SETTLED, or the context
   begins with RECOVERED — your only job is ONE call to close_case with
   outcome="recovered", what happened, and the lesson worth carrying forward.
   That records the outcome and stores the lesson together, so do NOT also call
   manage_memory: passing `lesson` to close_case is how the lesson is saved, and
   doing both stores it twice. Do not create links,
   do not contact the customer again, do not escalate. A recovered case that
   gets another message is a customer being pestered after they have already
   paid you.

   If you are given a `push_outcome`, that is the customer telling you something.
   ACTED means they are paying. DISMISSED quickly usually means the message did
   not give them a reason to act — a plain reminder will not work twice, so
   change what is on offer or change the channel. IGNORED means they may have
   left the page entirely, so reach them somewhere else. Say what you infer and
   why, then choose accordingly — you are reasoning about intent, not matching a
   rule, and you cannot know the real reason until you ask them.

3. END DELIBERATELY. If the case is finished, call close_case — that is what
   finishing looks like. If it is not finished but your work this turn is done,
   call wait_for_customer. Either way, then reply with a short final summary and
   NO further tool calls. Never end a case by falling silent.

HARD RULES:
- Hard decline codes (41, 43, 54, 14, 04, 46, 57, 93) are permanent. NEVER retry.
- Never contact opted-out customers.
- If you have made 3 tool calls without taking a Recover action, take one NOW.
- If a Recover tool fails, move DOWN the ladder to the next rung — not to
  escalation. Escalation is rung 6 and it will refuse you before then.

READING TOOL RESULTS:
- "[TOOL ERROR]" or "status": "error" / "unavailable"  -> that tool FAILED.
- "status": "blocked"  -> a guardrail refused it. Calling it again will fail
  again. Choose a DIFFERENT tool or stop.
- "status": "ok"       -> it worked. Move to the next step.
- "status": "no_data"  -> it worked but found nothing. Do not retry it.

NEVER call a tool that has already failed or been blocked. If you have run out
of moves for THIS turn, call wait_for_customer and end the turn — you will be
started again when something happens. Repeating a failed call wastes the
customer's recovery window."""


# ═══════════════════════════════════════════════════════════════
# MESSAGE SANITIZATION — Gemini API compatibility
# ═══════════════════════════════════════════════════════════════

def _pair_tool_calls(messages: list) -> list:
    """Drop tool_use blocks with no tool_result, and results with no call.

    Anthropic-format models reject a history where an assistant turn requests a
    tool and the next message is not the matching result:

        400 messages.1: `tool_use` ids were found without `tool_result` blocks

    Two things in this graph produce that shape: the repetition guard used to
    return a SystemMessage instead of tool results, and any blind tail-slice of
    the history (self_critique took `messages[-20:]`) can cut a call away from
    its result. Repair it in one place so every caller is safe.
    """
    answered = {
        m.tool_call_id for m in messages
        if isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None)
    }

    out: list = []
    requested: set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            kept = [tc for tc in msg.tool_calls if tc.get("id") in answered]
            if len(kept) == len(msg.tool_calls):
                requested.update(tc["id"] for tc in kept)
                out.append(msg)
                continue
            if not kept and not (msg.content or "").strip():
                continue                      # nothing left worth keeping
            # Rebuild without the unanswered calls rather than dropping the turn.
            repaired = AIMessage(
                content=msg.content or "", tool_calls=kept,
                additional_kwargs=dict(getattr(msg, "additional_kwargs", {}) or {}),
            )
            requested.update(tc["id"] for tc in kept)
            out.append(repaired)
        elif isinstance(msg, ToolMessage):
            if getattr(msg, "tool_call_id", None) in requested:
                out.append(msg)               # orphaned results are dropped
        else:
            out.append(msg)

    while out and isinstance(out[0], ToolMessage):
        out.pop(0)
    return out


def _sanitize_messages_for_gemini(messages: list) -> list:
    """Ensure message sequence is valid for Gemini API.

    Gemini requires: user → ai(tool_calls) → tool_response → ... → ai(final)
    Trailing AIMessage with tool_calls (no ToolMessage after) causes 400.
    """
    if not messages:
        return messages

    # Strip trailing AIMessage with tool_calls (incomplete function call sequence)
    result = list(messages)
    while result and isinstance(result[-1], AIMessage) and result[-1].tool_calls:
        result.pop()

    if not result:
        return [messages[0]]  # fallback to first message

    return result


# ═══════════════════════════════════════════════════════════════
# CONTEXT ASSEMBLY — what the model is shown, and what must survive
# ═══════════════════════════════════════════════════════════════

MAX_TOOL_CONTENT = 500  # chars per tool result shown to the LLM

#: Tool results that must reach the model whole. check_payment_status IS the
#: agent's perception on demand — its output carries the ground-truth briefing
#: and is routinely longer than MAX_TOOL_CONTENT, and a briefing chopped
#: mid-JSON is exactly the half-blindness perception.py exists to prevent.
_NEVER_TRIM_TOOLS = frozenset({"check_payment_status"})


def _trim_tool_results_for_llm(messages: list,
                               max_chars: int = MAX_TOOL_CONTENT) -> list:
    """Shorten large tool results for THIS LLM call only.

    This used to assign to `msg.content`, mutating the message objects that
    live in graph state — so a context-saving measure permanently damaged the
    durable record. `escalate_to_human` returns 638 chars (the ticket carries
    a full hand-over reason), was chopped at 500 mid-JSON, and every reader
    downstream got un-parseable text: the dashboard showed
    "Escalated to Human: N/A" and fell back to printing the observation text
    as the reason, while the real ticket id sat correctly in the queue.
    Copy instead of mutate — the LLM sees less, the record keeps everything.
    """
    trimmed = []
    for msg in messages:
        if (isinstance(msg, ToolMessage) and msg.content
                and len(str(msg.content)) > max_chars
                and (getattr(msg, "name", "") or "") not in _NEVER_TRIM_TOOLS):
            content = str(msg.content)
            short = content[:max_chars] + f"... [truncated, {len(content)} chars total]"
            try:
                msg = msg.model_copy(update={"content": short})
            except AttributeError:          # older langchain-core
                msg = ToolMessage(content=short, tool_call_id=msg.tool_call_id,
                                  name=msg.name, id=msg.id)
        trimmed.append(msg)
    return trimmed


def _assemble_llm_messages(briefing: list, history: list,
                           max_messages: int = 30) -> list:
    """System prompt + perception briefing + (truncated) history, in that order.

    Truncation used to run over the ASSEMBLED list, keeping index 0 (the
    system prompt) and the first HumanMessage. The briefing sat at index 1 —
    so on any case longer than ~30 messages the "WHAT IS TRUE RIGHT NOW" block
    silently fell out of the context, on exactly the long-running multi-run
    cases most at risk of acting on stale beliefs. The head is never trimmed;
    only the history is.
    """
    head = [SystemMessage(content=SYSTEM_PROMPT)] + list(briefing)
    if len(history) <= max_messages:
        return head + list(history)

    # Keep the first HumanMessage — the case facts — plus the newest tail.
    first_human = None
    for i, m in enumerate(history):
        if isinstance(m, HumanMessage):
            first_human = i
            break

    if first_human is None:
        tail = list(history[-max_messages:])
    else:
        tail_start = max(first_human + 1, len(history) - max_messages)
        tail = list(history[tail_start:])
    # A leading ToolMessage answers a call that fell off the slice — drop it.
    # ONLY ToolMessages: the old rule dropped leading AIMessages too, and a
    # tool-heavy tail (call, result, call, result, ...) drained to nothing,
    # leaving the model the case facts and no recent history at all. An
    # AIMessage whose results follow it in the tail is a valid opening.
    while tail and isinstance(tail[0], ToolMessage):
        tail.pop(0)
    if first_human is None:
        return head + tail
    return head + [history[first_human]] + tail


# ═══════════════════════════════════════════════════════════════
# AGENT NODE — LLM with bound tools, every call traced to Phoenix
# ═══════════════════════════════════════════════════════════════

def _build_model(tools: list | None = None, model_name: str | None = None):
    """Build the LLM with tools bound.

    MANDATE 1: Uses ChatOpenAI(cache=True) — langchain_core.caches.InMemoryCache.
    MANDATE 1: Multi-model routing — different models for different complexity levels.
    """
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from langchain_core.globals import set_llm_cache
    from langchain_core.caches import InMemoryCache
    load_dotenv()

    # Initialize LLM cache — safe to call multiple times, InMemoryCache is a singleton
    set_llm_cache(InMemoryCache())

    base_url = os.getenv("LLM_BASE_URL", "http://localhost:20128/v1")
    api_key = os.getenv("LLM_API_KEY", "dummy")
    name = model_name or os.getenv("LLM_MODEL", "antigravity/gemini-2.5-flash")

    model = ChatOpenAI(
        model=name,
        base_url=base_url,
        api_key=api_key,
        temperature=0,
        max_tokens=2048,
        cache=True,  # MANDATE 1: langchain_core.caches.InMemoryCache
    )
    return model.bind_tools(tools or RECOVERY_TOOLS)


# ═════════════════════════════════════════════════════════════
# MODEL ROUTING — configurable model registry (not hardcoded pipeline)
# ═════════════════════════════════════════════════════════════

# CrewAI: "Use the appropriate model sizes and providers to fit each agent's specific task"
# MANDATE 3: This is NOT an LLM decision — model routing is an architectural choice
# made at compile time. The LLM decides WHAT to do; we decide WHICH model runs it.
# This is configurable via env vars, not hardcoded.

def _get_model_for_task(task: str) -> str | None:
    """Get model name for a specific task from env vars.

    MANDATE 1: Uses os.getenv (real SDK) — no hardcoded dict.
    MANDATE 3: This is architectural config, not LLM reasoning.
    """
    import os
    env_key = f"LLM_MODEL_{task.upper()}"
    return os.getenv(env_key)  # None = use primary model


def _build_model_for_task(task: str, tools: list | None = None):
    """Build model with configurable multi-model routing.

    CrewAI pattern: different model sizes for different task complexities.
    Model is resolved from env vars (e.g., LLM_MODEL_SELF_CRITIQUE=...).
    MANDATE 1: Uses os.getenv — no hardcoded dict.
    """
    return _build_model(tools=tools, model_name=_get_model_for_task(task))


# ═══════════════════════════════════════════════════════════════
# OTEL TRACER — one tracer shared across all nodes, sends to Phoenix
# ═══════════════════════════════════════════════════════════════

_OTEL_TRACER = None

def _get_otel_tracer():
    """Get a tracer that sends spans to the Phoenix collector.

    Always checks the global TracerProvider dynamically (never caches
    the provider reference) so that spans created by graph nodes become
    children of the active span from the caller's context.
    """
    global _OTEL_TRACER
    if _OTEL_TRACER is not None:
        return _OTEL_TRACER
    try:
        # Always use the globally registered provider — never create a second one
        _OTEL_TRACER = trace.get_tracer("recovery-agent")
        print(f"[OTel] Tracer initialized → recovery-agent", flush=True)
        return _OTEL_TRACER
    except Exception:
        _OTEL_TRACER = trace.get_tracer("recovery-agent")
        return _OTEL_TRACER


def _get_context(config: RunnableConfig | None):
    """Get the RecoveryContext passed to graph.invoke()/stream().

    Three routes, most stable first. The original code reached straight into
    `config["configurable"]["__pregel_runtime"]` — a private key that would
    silently start returning None on a LangGraph upgrade, taking the Case with
    it (and with the Case goes attempt recording and every stopping rule).

    1. `langgraph.runtime.get_runtime()` — the public API. Reads the ambient
       runtime, so it works even when a node is not handed a config.
    2. `CONFIG_KEY_RUNTIME` — the same private slot, but via LangGraph's own
       constant rather than a string literal we would have to chase.
    3. The literal, for versions that predate the constant.
    """
    try:
        from langgraph.runtime import get_runtime
        runtime = get_runtime()
        if runtime is not None and getattr(runtime, "context", None) is not None:
            return runtime.context
    except Exception:
        pass  # not inside a graph run, or an older langgraph

    if not config:
        return None
    configurable = config.get("configurable") or {}

    try:
        from langgraph.config import CONFIG_KEY_RUNTIME
        runtime = configurable.get(CONFIG_KEY_RUNTIME)
        if runtime is not None:
            return getattr(runtime, "context", None)
    except Exception:
        pass

    runtime = configurable.get("__pregel_runtime")
    return getattr(runtime, "context", None) if runtime else None


def agent_node(state: RecoveryState, config: RunnableConfig) -> dict:
    """LLM reasons about the payment failure and decides which tools to call.

    MANDATE 0: Governance filters tools BEFORE binding — LLM only sees allowed tools.
    MANDATE 1: Uses governance.get_allowed_tools (real SDK, pydantic policy).
    MANDATE 3: LLM decides which allowed tool to call — governance doesn't decide strategy.

    Tracing: LangGraph's Pregel engine fires LangChain callbacks for this node.
    We attach a PhoenixTracingCallbackHandler to the graph stream so every
    agent call, tool call, and guard check appears as a properly nested span
    in Phoenix — no orphaned manual OTel spans.
    """
    from recovery_agent.agent.tools import TOOLS_BY_NAME

    turn_number = len([m for m in state["messages"] if isinstance(m, AIMessage)]) + 1
    tracer = _get_otel_tracer()

    ctx = _get_context(config)
    tier = "silent"
    if ctx:
        case = ctx.case
        # `Case` has no `current_tier` — the field is `recovery_tier`. The old
        # getattr always fell through to its default, so every run was pinned to
        # the silent tier and any tool granted only at active/escalated (voice,
        # for one) was never bound. The agent could not choose what it could not
        # see.
        tier = getattr(getattr(case, "recovery_tier", None), "value", None) or "silent"
        # Update Case status — graph is source of truth
        from recovery_agent.models import CaseStatus
        if case.status == CaseStatus.OPEN:
            case.status = CaseStatus.DIAGNOSING

    # No amount is passed: tool access is tier-gated only. The money control is
    # the give-away ceiling in human_approval_gate — see get_allowed_tools for
    # why an amount gate here was wrong twice over.
    allowed_names = get_allowed_tools(tier=tier)

    # A finished case gets bookkeeping tools ONLY.
    #
    # Sessions are now continuous per payment, so the closing run after a
    # successful payment replays the earlier turns — including the instruction to
    # recover it. Told to stop, the model recorded the lesson and then resumed the
    # older task: it pushed a fresh notification and re-offered 5% to a customer
    # who had already paid INR 37,990.50. Instruction alone is not a control;
    # withholding the tools is.
    # The Case object is rebuilt from scratch on every run, so its status is
    # only as good as whatever the caller put there. The durable record is the
    # one thing that cannot be forgotten between runs, so ask it too — and treat
    # money actually received as decisive regardless of any status string.
    _settled = getattr(case, "status", None) in (
        CaseStatus.RECOVERED, CaseStatus.ESCALATED, CaseStatus.STOPPED)
    if not _settled and case is not None:
        try:
            from recovery_agent.state_store import StateStore
            _rec = StateStore().get_payment(getattr(case.payment, "payment_id", "")) or {}
            _settled = (_rec.get("status") in ("recovered", "escalated")
                        or float(_rec.get("recovered_amount") or 0) > 0)
        except Exception:
            pass

    if ctx is not None and _settled:
        # wait_for_customer stays: it contacts nobody and moves no money, it
        # only ends the turn. Withholding it meant the closing run had no clean
        # way to stop — live, the agent called it and got
        # "wait_for_customer is not a valid tool", which is the graph telling
        # the agent off for doing exactly the right thing.
        allowed_names = [n for n in allowed_names
                         if n in ("manage_memory", "search_memory",
                                  "wait_for_customer", "close_case")]

    allowed_tools = [TOOLS_BY_NAME[n] for n in allowed_names if n in TOOLS_BY_NAME]

    model = _build_model_for_task("agent", tools=allowed_tools)

    # PERCEPTION — recomputed every turn, from the record and not from whatever
    # the caller wrote into a message.
    #
    # Without this the agent knows only what it was told. It stopped when the
    # orchestrator said "RECOVERED" and chased a customer who had already paid
    # whenever the orchestrator said nothing — not because it forgot, but
    # because it could not look. Withholding its tools stopped the damage
    # without addressing that; an agent that behaves only where the bars are is
    # not a careful one.
    #
    # Putting the facts in front of it every turn is the difference between an
    # agent that cannot act and one that knows it should not.
    briefing = []
    try:
        from recovery_agent.agent.perception import ground_truth, as_briefing
        pid = getattr(getattr(case, "payment", None), "payment_id", "") if case else ""
        if pid:
            briefing = [SystemMessage(content=as_briefing(ground_truth(pid)))]
    except Exception:
        pass

    # Context assembly: the head (prompt + briefing) is never trimmed, the
    # history is; large tool results are shortened on copies. The rules and
    # the incidents behind them live on the two helpers.
    messages = _trim_tool_results_for_llm(
        _assemble_llm_messages(briefing, state["messages"]))

    # Rate-limit LLM calls to avoid Google upstream 502s
    # Track last call time across all agent instances
    if not hasattr(agent_node, '_last_call_time'):
        agent_node._last_call_time = 0
    _min_delay = 8.0  # seconds between LLM calls (Google rate limit: ~4 req/20s, need 5s buffer)
    _elapsed = time.time() - agent_node._last_call_time
    if _elapsed < _min_delay:
        time.sleep(_min_delay - _elapsed)
    agent_node._last_call_time = time.time()

    # Fallback models — ordered by rate limit headroom (highest first)
    # gemini-2.5-flash: 5 RPM / 20 RPD (too low)
    # gemini-3.1-flash-lite: 15 RPM / 500 RPD (3x better)
    # gemma-4-31b-it: 16K RPM / 14K RPD (massive headroom)
    # gemma-4-26b-it: 16K RPM / 14K RPD (massive headroom)
    # auto/best-reasoning: OmniRoute smart routing (picks best available)
    # no-think/antigravity/claude-sonnet-4-6: works, different provider
    _fallback_models = [
        "gemma-4-31b-it",
        "gemma-4-26b-it",
        "gemini-3.1-flash-lite",
        "auto/best-reasoning",
        "no-think/antigravity/claude-sonnet-4-6",
        "antigravity/gemini-2.5-flash",  # last resort — 5 RPM limit
    ]

    with tracer.start_as_current_span(f"agent_turn_{turn_number}") if tracer else nullcontext():
        last_error = None
        for attempt in range(len(_fallback_models)):
            try:
                current_model = _fallback_models[attempt]
                if attempt > 0:
                    print(f"[Agent] Trying fallback model: {current_model}", flush=True)
                    fallback_llm = _build_model(tools=allowed_tools, model_name=current_model)
                    response = fallback_llm.invoke(_sanitize_messages_for_gemini(messages))
                else:
                    print(f"[Agent] LLM call #{turn_number} — invoke() returning...", flush=True)
                    response = model.invoke(_sanitize_messages_for_gemini(messages))
                print(f"[Agent] LLM responded: {len(response.tool_calls) if response.tool_calls else 0} tool_calls, content_len={len(response.content) if response.content else 0}", flush=True)

                tool_calls = []
                if hasattr(response, "tool_calls") and response.tool_calls:
                    tool_calls = [tc["name"] for tc in response.tool_calls]

                span = trace.get_current_span()
                if span and span.is_recording():
                    span.set_attribute("agent.turn", turn_number)
                    span.set_attribute("agent.tool_calls", json.dumps(tool_calls))
                    span.set_attribute("agent.status", "tool_call" if tool_calls else "final_response")
                    span.set_status(StatusCode.OK)

                print(f"[Agent] turn={turn_number} tool_calls={tool_calls} status={'tool_call' if tool_calls else 'final_response'}", flush=True)
                return {"messages": [response], "phase": "diagnosing"}
            except Exception as e:
                err_str = str(e).lower()
                last_error = e
                # On 502/429/rate-limit, try next fallback model
                if "502" in str(e) or "429" in str(e) or "rate" in err_str or "server_error" in err_str:
                    if attempt < len(_fallback_models) - 1:
                        print(f"[Agent] Rate limit/error on {current_model} ({type(e).__name__}), trying fallback...", flush=True)
                        continue
                # On 400/bad-request, retry with same model (might be transient)
                if "400" in str(e) or "bad request" in err_str or "invalid_argument" in err_str:
                    wait = 2 ** attempt * 2
                    print(f"[Agent] Retryable error (attempt {attempt+1}). Waiting {wait}s... {type(e).__name__}", flush=True)
                    time.sleep(wait)
                    continue
                # Non-retryable error
                span = trace.get_current_span()
                if span and span.is_recording():
                    span.set_status(StatusCode.ERROR, str(e))
                print(f"[Agent] Error: {e}", flush=True)
                raise

        # All fallback models exhausted — return error response instead of crashing
        error_msg = (
            f"I apologize, but I'm unable to process this recovery right now. "
            f"All available AI models are temporarily unavailable. "
            f"Please try again in a few minutes, or contact support if this persists. "
            f"Last error: {type(last_error).__name__}: {last_error}" if last_error else
            "All AI models are temporarily unavailable. Please try again later."
        )
        error_response = AIMessage(content=error_msg)
        return {"messages": [error_response], "phase": "error"}


# ═══════════════════════════════════════════════════════════════
# TOOL NODE — proper ToolNode (MANDATE 1: real SDK, not hand-rolled)
# ═══════════════════════════════════════════════════════════════

def _build_tool_node(tools=None) -> ToolNode:
    """Build ToolNode with OTel tracing via callbacks.

    MANDATE 1: Uses langgraph.prebuilt.ToolNode (real SDK) instead of
    hand-rolled tool_node_with_trace which broke langmem context.

    ToolNode is a RunnableCallable that LangGraph's Pregel engine
    invokes with proper context, so langmem's get_config() works.
    """
    return ToolNode(
        tools=tools if tools is not None else RECOVERY_TOOLS,
        handle_tool_errors=True,
    )


# ═══════════════════════════════════════════════════════════════
# ROUTING — should_continue checks for tool_calls + stopping rules
# ═══════════════════════════════════════════════════════════════

def should_continue(state: RecoveryState) -> Literal["tool_repetition_guard", "stopping_check", "__end__"]:
    """Decide if we should continue the loop or stop.

    If the LLM made tool calls → check for repetition first.
    If the LLM returned a final response → check stopping rules before ending.
    MAX_TURNS: hard cap to prevent exceeding LLM context/proxy limits.
    """
    MAX_TURNS = 8
    messages = state["messages"]
    last_message = messages[-1]

    # Turns are counted PER RUN, not per session.
    #
    # A session is now one continuous thread per payment, so this used to count
    # every turn the case had ever taken. A case climbing the ladder spends a
    # couple of turns per rung, so the cap tripped part-way up and the graph
    # forced a final response mid-rung: live, `retry_in_hours` was called and
    # never executed, the next call came back "not a valid tool", and a INR
    # 2,499 case went quiet at rung 2 without recovering or escalating. The
    # ladder could not be finished by any case that needed more than about
    # three rungs.
    #
    # Each run begins with a fresh HumanMessage carrying the new facts, so
    # everything after the last one is this run's work. Context size is a
    # separate concern and is handled by the MAX_MESSAGES truncation in
    # agent_node; this cap is only here to stop one run looping.
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            start = i
            break
    ai_msgs = [m for m in messages[start:]
               if isinstance(m, AIMessage) and m.tool_calls]
    if len(ai_msgs) >= MAX_TURNS:
        print(f"[Agent] MAX_TURNS={MAX_TURNS} reached this run, forcing final response",
              flush=True)
        return "stopping_check"

    # Check if we're cycling through the same tools repeatedly
    tool_names_in_history = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                tool_names_in_history.append(tc["name"])

    if len(tool_names_in_history) >= 4:
        # Check if the last 3 tool calls are a repeating pattern
        last_3 = tool_names_in_history[-3:]
        if len(tool_names_in_history) >= 6:
            prev_3 = tool_names_in_history[-6:-3]
            if last_3 == prev_3:
                print(f"[Agent] Cycling pattern detected: {last_3}, forcing stop", flush=True)
                return "stopping_check"

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_repetition_guard"

    # LLM returned final response — check stopping rules before ending
    return "stopping_check"


# ═══════════════════════════════════════════════════════════════
# TOOL REPETITION GUARD — prevent infinite tool loops (P0)
# ═══════════════════════════════════════════════════════════════

def _hash_tool_args(args: dict) -> str:
    """Hash tool args for repetition detection."""
    import hashlib
    return hashlib.md5(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()[:12]


#: Tools that read state which genuinely changes between calls. Repetition is
#: the correct behaviour for these, not a loop.
_RECHECKABLE = {"check_payment_status"}


def tool_repetition_guard(state: RecoveryState, config: RunnableConfig = None) -> dict:
    """Prevent the LLM from looping on the same tools.

    Three layers of protection:
    1. Exact repetition: same tool + same args → blocked (original behavior)
    2. Doom loop: same tool called 3+ times regardless of args → blocked (NEW)
    3. Error limit: 5+ tool errors → force stop (original behavior)

    MANDATE 0: This is a guardrail (acceptable pipeline).
    MANDATE 1: Uses langgraph state (real SDK) for tracking.
    """
    # Set Case.status to ACTING — tools are about to execute
    ctx = _get_context(config)
    if ctx:
        from recovery_agent.models import CaseStatus
        ctx.case.status = CaseStatus.ACTING

    tracer = _get_otel_tracer()
    with tracer.start_as_current_span("tool_repetition_guard") if tracer else nullcontext():
        messages = state["messages"]
        last_message = messages[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            span = trace.get_current_span()
            if span and span.is_recording():
                span.set_attribute("guard.result", "no_tool_calls")
                span.set_status(StatusCode.OK)
            return {"phase": "guard_check"}

        history = state.get("tool_call_history", [])

        # ── Layer 3: Error limit — force stop after repeated failures ──
        error_count = 0
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = str(msg.content) if msg.content else ""
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and parsed.get("status") == "error":
                        error_count += 1
                except (json.JSONDecodeError, TypeError):
                    pass

        if error_count >= 5:
            span = trace.get_current_span()
            if span and span.is_recording():
                span.set_attribute("guard.result", "error_limit")
                span.set_status(StatusCode.OK)
            return {
                "messages": [
                    ToolMessage(
                        content=json.dumps({"status": "blocked",
                                            "reason": f"{error_count} tool errors so far"}),
                        tool_call_id=tc["id"], name=tc["name"],
                    ) for tc in last_message.tool_calls
                ] + [SystemMessage(
                    content=f"[Guardrail] Too many failed tool calls ({error_count} errors). "
                            f"You must stop now. Provide a summary of what happened and what the "
                            f"customer or merchant should do next. Do NOT call any more tools."
                )],
                "phase": "guard_stop",
            }

        # ── Layer 2: Doom loop — block tool after 3 calls regardless of args ──
        # Count how many times each tool has been called in the full history
        tool_call_counts: dict[str, int] = {}
        for entry in history:
            name = entry["name"]
            tool_call_counts[name] = tool_call_counts.get(name, 0) + 1

        # ── Layer 1: Exact repetition — same tool + same args ──
        blocked = []
        blocked_by_doom = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            args_hash = _hash_tool_args(tool_call["args"])

            # Asking again whether the money has arrived is not repetition — the
            # whole point is that the answer changes. Blocking it taught the
            # agent that checking was a mistake, which is precisely the habit
            # that lets it spend money on a customer who has already paid.
            if tool_name in _RECHECKABLE:
                continue

            # Check exact repetition (original behavior)
            for prev in history:
                if prev["name"] == tool_name and prev["args_hash"] == args_hash:
                    blocked.append(tool_name)
                    break

            # Check doom loop (3+ calls regardless of args)
            if tool_call_counts.get(tool_name, 0) >= 3:
                if tool_name not in blocked:
                    blocked_by_doom.append(tool_name)

        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("guard.blocked_tools", json.dumps(blocked))
            span.set_attribute("guard.doom_blocked", json.dumps(blocked_by_doom))
            span.set_attribute("guard.result", "blocked" if blocked or blocked_by_doom else "passed")
            span.set_status(StatusCode.OK)

        all_blocked = blocked + blocked_by_doom

        if all_blocked:
            # Build specific error messages
            reasons = []
            for name in blocked:
                reasons.append(f"{name} (exact same call repeated)")
            for name in blocked_by_doom:
                reasons.append(f"{name} (called {tool_call_counts[name]} times — doom loop)")

            # Every blocked call still needs a tool_result, or the next LLM
            # call fails with "tool_use ids were found without tool_result".
            # That 400 is what actually broke self_critique, and it is why the
            # model never saw the guard's message and looped on the same tools.
            blocked_set = set(all_blocked)
            reason_by_tool = {}
            for name in blocked:
                reason_by_tool[name] = "you already made this exact call"
            for name in blocked_by_doom:
                reason_by_tool[name] = f"called {tool_call_counts[name]} times already"

            tool_results = [
                ToolMessage(
                    content=json.dumps({
                        "status": "blocked",
                        "reason": reason_by_tool.get(tc["name"], "blocked by guardrail"),
                        "guidance": "Do NOT call this tool again. Choose a different "
                                    "tool, or stop and summarise.",
                    }),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
                for tc in last_message.tool_calls
                if tc["name"] in blocked_set
            ]

            # Blocked calls MUST be recorded, or the doom-loop counter never
            # advances and the same batch is re-proposed forever.
            history_entries = [
                {"name": tc["name"], "args_hash": _hash_tool_args(tc["args"])}
                for tc in last_message.tool_calls
            ]

            if len(all_blocked) == len(last_message.tool_calls):
                rounds = int(state.get("blocked_rounds") or 0) + 1
                return {
                    "messages": tool_results + [SystemMessage(
                        content=f"[Guardrail] Every tool you requested is blocked: "
                                f"{'; '.join(reasons)}. Repeating them will not work. "
                                f"Either call a DIFFERENT tool (escalate_to_human, "
                                f"generate_recovery_payment_link, retry_in_hours), or "
                                f"reply with a final summary and no tool calls."
                    )],
                    "tool_call_history": history + history_entries,
                    "blocked_rounds": rounds,
                    "phase": "guard_stop" if rounds >= 2 else "guard_check",
                }

            # Some tools blocked — let the unblocked ones through.
            return {
                "messages": tool_results + [SystemMessage(
                        content=f"[Guardrail] Some tools blocked: {'; '.join(reasons)}. "
                                f"Call ONLY unblocked tools, or stop and provide a summary."
                )],
                "tool_call_history": history + history_entries,
                "phase": "guard_check",
            }

        # All tools allowed — record in history
        new_entries = []
        for tool_call in last_message.tool_calls:
            new_entries.append({
                "name": tool_call["name"],
                "args_hash": _hash_tool_args(tool_call["args"]),
            })

        return {
            "tool_call_history": history + new_entries,
            "blocked_rounds": 0,
            "phase": "acting",
        }


# ═══════════════════════════════════════════════════════════════
# DATA MASKING — PII protection in tool outputs (Governing AI Agents P0)
# ═══════════════════════════════════════════════════════════════

def mask_tool_outputs_node(state: RecoveryState, config: RunnableConfig = None) -> dict:
    """Mask PII in tool message outputs before they reach the LLM.

    Also detects structured tool errors (status=error/unavailable) and prepends
    an explicit [TOOL ERROR] marker so the LLM cannot miss failure signals.

    Governing AI Agents: "mask personal information, build tools that
    provide only the data needed."

    MANDATE 1: Uses governance.mask_tool_output (real SDK, re-based masking).
    MANDATE 3: This is a security guardrail (acceptable pipeline), not LLM reasoning.
    """
    tracer = _get_otel_tracer()
    with tracer.start_as_current_span("mask_tool_outputs") if tracer else nullcontext():
        messages = state["messages"]
        masked_messages = []
        masked_count = 0

        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_name = ""
                if hasattr(msg, "name") and msg.name:
                    tool_name = msg.name
                elif hasattr(msg, "tool_call_id"):
                    for m in reversed(messages):
                        if isinstance(m, AIMessage) and m.tool_calls:
                            for tc in m.tool_calls:
                                if tc.get("id") == msg.tool_call_id:
                                    tool_name = tc.get("name", "")
                                    break
                        if tool_name:
                            break

                original = str(msg.content)
                masked_content = mask_tool_output(tool_name, original)

                # ── Error detection: make failures unmissable for the LLM ──
                # Tools return structured JSON like {"status": "error", "message": "..."}
                # but the LLM often fails to parse this and retries the same tool.
                # Prepend an explicit error marker so the LLM sees the failure clearly.
                error_marker = ""
                try:
                    parsed = json.loads(masked_content)
                    if isinstance(parsed, dict) and parsed.get("status") in ("error", "unavailable"):
                        err_msg = parsed.get("message", "Unknown error")
                        error_marker = (
                            f"[TOOL ERROR] {tool_name} FAILED: {err_msg}\n"
                            f"This tool cannot succeed. Do NOT retry it. Try a DIFFERENT tool.\n"
                        )
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

                final_content = error_marker + masked_content if error_marker else masked_content
                if final_content == original:
                    masked_messages.append(msg)      # unchanged: do not re-emit
                    continue
                masked_count += 1
                masked_messages.append(
                    ToolMessage(
                        content=final_content,
                        tool_call_id=msg.tool_call_id,
                        # `name` was only carried over when the original already
                        # had one, so masked results reached the UI as null and
                        # the LLM lost track of which tool answered.
                        name=msg.name or tool_name or None,
                        # Keeping the id makes add_messages REPLACE this message.
                        # Without it every mask pass appended a fresh copy of each
                        # tool result, so the history grew on every turn.
                        id=msg.id,
                    )
                )
            else:
                masked_messages.append(msg)

        _record_attempts_on_case(messages, config)

        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("mask.masked_count", masked_count)
            span.set_status(StatusCode.OK)

        return {"messages": masked_messages}


# Recovery tools whose outcome is a real recovery attempt on the case.
# An "attempt" is a contact with the customer or a charge against the bank — not
# every tool the agent uses to prepare one. `generate_recovery_payment_link`
# deliberately is NOT here: it creates the instrument, it reaches nobody. Counting
# it burned a third of the attempt budget per cycle, so a single push + link +
# email exhausted max_attempts=3 and the case was reported "failed" while it was
# in fact waiting for the customer to pay.
_ATTEMPT_ACTIONS = {
    "send_page_push": "send_notification",
    "send_recovery_notification": "send_notification",
    "initiate_voice_call": "voice_call",
    "retry_in_hours": "wait_and_retry",
    "escalate_to_human": "escalate_to_human",
}
_OK_STATUSES = {"ok", "scheduled", "escalated", "success", "delivered"}


def _record_attempts_on_case(messages: list, config: RunnableConfig | None) -> None:
    """Write recovery tool outcomes back onto the Case as Attempt records.

    Nothing in the graph did this, so `case.attempts` stayed empty and
    `case.attempt_count` stayed 0 forever. Every rule in `check_stopping_rules`
    reads one of those, so no rule could ever fire and cases could never reach
    RECOVERED / ESCALATED / STOPPED — they sat in ACTING until MAX_TURNS.

    Only the tools that actually try to recover money count. Diagnostics do not.
    """
    ctx = _get_context(config)
    if ctx is None or getattr(ctx, "case", None) is None:
        return
    case = ctx.case

    from recovery_agent.models import ActionType, Attempt

    seen = case.payment.metadata.setdefault("_recorded_tool_calls", [])
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        call_id = getattr(msg, "tool_call_id", "") or ""
        name = getattr(msg, "name", "") or ""
        if not call_id or call_id in seen or name not in _ATTEMPT_ACTIONS:
            continue

        content = str(msg.content or "")
        status = ""
        try:
            parsed = json.loads(content[content.index("{"):]) if "{" in content else {}
            if isinstance(parsed, dict):
                status = str(parsed.get("status", "")).lower()
        except (ValueError, json.JSONDecodeError):
            pass
        succeeded = status in _OK_STATUSES and "[TOOL ERROR]" not in content

        try:
            action = ActionType(_ATTEMPT_ACTIONS[name])
        except ValueError:
            continue

        case.attempts.append(Attempt(
            action_type=action,
            action_details={"tool": name, "status": status or "unknown"},
            result="success" if succeeded else "failed",
            tier=case.recovery_tier,
        ))
        case.attempt_count += 1
        seen.append(call_id)


# ═══════════════════════════════════════════════════════════════
# HUMAN-IN-THE-LOOP — approval gate for high-value recoveries (P1)
# ═══════════════════════════════════════════════════════════════

# CrewAI: "@human_feedback decorator, approval gates"
# Context7: Uses langgraph.types.interrupt to pause the graph

# Threshold for requiring approval (₹50,000 = 5000000 paise)

# Tools that always require approval
# Tools that may not run without a human deciding first.
#
# `escalate_to_human` used to be in here, which deadlocked the graph: asking for
# a human required a human's approval, `interrupt()` fired, and nothing in this
# codebase ever resumes an interrupt — so the tool never ran, `mask_outputs`
# never ran, `stopping_check` never ran, and the case sat in ACTING forever.
# Escalation is the handoff; it can never require approval.
# `initiate_refund` and `update_emi_schedule` are not registered tools at all.
_APPROVAL_REQUIRED_TOOLS: set[str] = set()

# Above this rupee amount a recovery link needs a person. Kept as rupees because
# that is the unit every registered tool actually takes.
# What needs a human is money GIVEN AWAY, not money collected.
#
# This was a cap on the amount charged (INR 50,000), which blocked recovering a
# INR 79,980 order the customer had already chosen to buy: the agent was refused a
# link for INR 75,981 and escalated instead of offering, so the customer never saw
# the discount UI at all. Charging someone what they already owe — or less — is not
# the risk. The risk was the INR 3,999 discount, and that is what is capped now.
_APPROVAL_DISCOUNT_THRESHOLD = float(os.getenv("APPROVAL_DISCOUNT_THRESHOLD", "5000"))

# Only these create a payment obligation, so only these are subject to the
# amount threshold. Diagnostics take an `amount` too and must never be gated.
_MONEY_MOVING_TOOLS = {"generate_recovery_payment_link"}


def _approval_refusal(hr: tuple[float, float, float] | None) -> dict:
    """Refuse a give-away, and say what would be allowed.

    A bare "needs human approval" leaves the agent one move — escalate — even
    when charging a few hundred rupees more would clear the ceiling outright.
    """
    if not hr:
        return {
            "status": "blocked",
            "reason": "this action needs human approval",
            "guidance": "Call escalate_to_human instead.",
        }
    given, owed, min_charge = hr
    return {
        "status": "blocked",
        "reason": (f"gives away ₹{given:,.2f} on a ₹{owed:,.2f} order; "
                   f"₹{_APPROVAL_DISCOUNT_THRESHOLD:,.2f} is the most that may be "
                   f"given away without a human"),
        "max_discount_rupees": round(_APPROVAL_DISCOUNT_THRESHOLD, 2),
        "min_amount_you_may_charge": round(min_charge, 2),
        "guidance": (f"Call this tool again charging at least "
                     f"₹{min_charge:,.2f} if a smaller offer could still recover "
                     f"this. Only call escalate_to_human if it could not."),
    }


def human_approval_gate(state: RecoveryState, config: RunnableConfig = None) -> dict:
    """Pause for human approval on high-value or risky recovery actions.

    MANDATE 3: LLM decides whether to propose actions — this gate only
    checks if the proposed action requires approval based on thresholds.

    MANDATE 1: Uses langgraph.types.interrupt (real SDK) for HITL.
    """
    ctx = _get_context(config)
    tracer = _get_otel_tracer()
    with tracer.start_as_current_span("human_approval_gate") if tracer else nullcontext():
        messages = state["messages"]
        last_message = messages[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            span = trace.get_current_span()
            if span and span.is_recording():
                span.set_attribute("approval.result", "no_tool_calls")
                span.set_status(StatusCode.OK)
            return {}

        needs_approval = False
        action_summary = []

        flagged: set[str] = set()
        headroom: dict[str, tuple[float, float, float]] = {}
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]

            if tool_name in _APPROVAL_REQUIRED_TOOLS:
                needs_approval = True
                flagged.add(tool_call["id"])
                action_summary.append(f"{tool_name}({json.dumps(tool_call['args'], default=str)[:200]})")
                continue

            # Only tools that create a payment obligation are considered, and
            # only the DISCOUNT is measured. `diagnose_payment_failure` also
            # takes an `amount`; gating it once meant a routine diagnosis on a
            # large payment was refused as though it were a charge.
            if tool_name not in _MONEY_MOVING_TOOLS:
                continue

            try:
                charging = float(tool_call["args"].get("amount") or 0)
            except (TypeError, ValueError):
                charging = 0.0

            owed = 0.0
            if ctx is not None and getattr(ctx, "case", None) is not None:
                owed = float(getattr(ctx.case.payment, "amount", 0) or 0)

            given_away = max(0.0, owed - charging) if owed and charging else 0.0
            if given_away > _APPROVAL_DISCOUNT_THRESHOLD:
                needs_approval = True
                flagged.add(tool_call["id"])
                # Tell the agent the headroom, not just "no". On a ₹1,19,970
                # order the policy's own 5% opening offer is ₹5,998 — over this
                # ceiling — so the agent was refused, told to escalate, and a
                # live case went to a human queue that a 4% offer would have
                # cleared. A refusal that hides the number it is enforcing turns
                # every large order into an escalation.
                headroom[tool_call["id"]] = (given_away, owed,
                                             owed - _APPROVAL_DISCOUNT_THRESHOLD)
                action_summary.append(
                    f"{tool_name} (discount ₹{given_away:,.0f} on a ₹{owed:,.0f} order)")

        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("approval.needs_approval", needs_approval)
            span.set_attribute("approval.actions", json.dumps(action_summary))
            span.set_status(StatusCode.OK)

        if needs_approval:
            # Deliberately NOT `interrupt()`. There is no resume path anywhere in
            # this system, so interrupting strands the case with no way forward.
            # Refusing the call and telling the agent to escalate keeps the same
            # safety property and leaves the case recoverable by a person.
            # Only the offending calls are refused. Blocking the whole batch
            # meant one flagged tool took every other tool down with it —
            # live, a flagged `diagnose_payment_failure` also blocked
            # `search_memory`, which takes no amount at all.
            return {
                "messages": [
                    ToolMessage(
                        content=json.dumps(_approval_refusal(headroom.get(tc["id"]))
                        if tc["id"] in flagged else {
                            "status": "not_executed",
                            "reason": "another tool in this batch needs approval",
                            "guidance": "You may call this tool again on its own.",
                        }),
                        tool_call_id=tc["id"], name=tc["name"],
                    ) for tc in last_message.tool_calls
                ] + [SystemMessage(
                    content=f"[Guardrail] Over the give-away ceiling: "
                            f"{'; '.join(action_summary)}. Re-quote within the "
                            f"ceiling if a smaller offer could still work; call "
                            f"escalate_to_human only if it could not."
                )],
                "phase": "guard_check",
            }

        return {}


# ═══════════════════════════════════════════════════════════════
# SELF-CRITIQUE — agent critiques its own recovery (P0)
# ═══════════════════════════════════════════════════════════════

SELF_CRITIQUE_PROMPT = """You are a recovery agent reviewing your own performance.

Review the conversation above and produce a brief self-critique.
Focus on:
1. What recovery strategy did you choose and why?
2. Did it work? What was the outcome?
3. What would you do differently next time for this customer/failure type?

Be specific and actionable. Store lessons that would help future recovery attempts.
Call manage_memory with your critique as the content."""

def self_critique_node(state: RecoveryState) -> dict:
    """After recovery completes, LLM critiques its own performance.

    MANDATE 3: The LLM genuinely decides what to critique — not hardcoded.
    MANDATE 1: Uses langmem manage_memory (real SDK) to store the critique.
    MANDATE 1: Multi-model routing — fast model for self-evaluation (not primary).
    """
    from recovery_agent.agent.tools import TOOLS_BY_NAME

    tracer = _get_otel_tracer()
    with tracer.start_as_current_span("self_critique") if tracer else nullcontext():
        model = _build_model_for_task("self_critique", tools=[TOOLS_BY_NAME["manage_memory"]])

        # A blind tail-slice can cut a tool call away from its result, which is
        # exactly the 400 this node used to die on. Pair first, then slice, then
        # pair again — slicing can orphan a call the first pass had matched.
        history = _pair_tool_calls(state["messages"])[-20:]
        msgs = [SystemMessage(content=SELF_CRITIQUE_PROMPT)] + _pair_tool_calls(history)
        msgs = _sanitize_messages_for_gemini(msgs)

        for attempt in range(3):
            try:
                response = model.invoke(msgs)

                # This node reflects and stores a lesson; `critique_tools` can
                # run nothing else. The model is bound to manage_memory alone
                # and still asked for check_payment_status — the conversation it
                # is reading is full of other tool names, so it reaches for one.
                # The ToolNode then answered "check_payment_status is not a
                # valid tool, try one of [manage_memory]", which reads in the
                # trace as the agent being refused a perfectly sensible call.
                #
                # Dropped from the AIMessage rather than answered with an error,
                # so no orphan tool_use is left behind for the next turn.
                if getattr(response, "tool_calls", None):
                    keep = [tc for tc in response.tool_calls
                            if tc.get("name") == "manage_memory"]
                    if len(keep) != len(response.tool_calls):
                        dropped = [tc.get("name") for tc in response.tool_calls
                                   if tc.get("name") != "manage_memory"]
                        print(f"[SelfCritique] ignored out-of-scope tool calls: "
                              f"{dropped}", flush=True)
                        response = response.model_copy(update={"tool_calls": keep})

                span = trace.get_current_span()
                if span and span.is_recording():
                    span.set_attribute("critique.status", "completed")
                    span.set_status(StatusCode.OK)
                return {"messages": [response], "phase": "self_critique"}
            except Exception as e:
                err_str = str(e).lower()
                if attempt < 2 and ("400" in str(e) or "bad request" in err_str):
                    time.sleep(2 ** attempt)
                    # Re-pair after truncating; a smaller window orphans more.
                    msgs = msgs[:1] + _pair_tool_calls(msgs[-10:])
                    continue
                span = trace.get_current_span()
                if span and span.is_recording():
                    span.set_attribute("critique.status", "error")
                    span.set_status(StatusCode.ERROR, str(e))
                print(f"[SelfCritique] Error: {e}", flush=True)
                return {"phase": "self_critique"}


# ═══════════════════════════════════════════════════════════════
# STOPPING CHECK — guardrail node (MANDATE 0: stopping.py is real)
# ═══════════════════════════════════════════════════════════════

def stopping_check(state: RecoveryState, config: RunnableConfig) -> dict:
    """Check stopping rules and update case status.

    This node runs after the LLM produces a final response.
    It checks if the agent should stop based on:
    - Recovery succeeded
    - Hard decline detected
    - Max attempts reached
    - Escalated to human
    - Tier exhaustion (silent → active transition)

    MANDATE 0: This is real guardrail logic from stopping.py,
    not a pipeline pretending to be agent reasoning.
    """
    from recovery_agent.agent.stopping import check_stopping_rules, transition_to_active_tier

    tracer = _get_otel_tracer()
    with tracer.start_as_current_span("stopping_check") if tracer else nullcontext():
        ctx = _get_context(config)
        if ctx is None:
            return {}

        case = ctx.case

        # Increment silent attempts counter when in silent tier
        # This runs each time stopping_check is reached (end of one agent cycle)
        from recovery_agent.models import RecoveryTier
        if case.recovery_tier == RecoveryTier.SILENT:
            case.silent_attempts += 1

        should_stop, reason = check_stopping_rules(case)

        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("stopping.should_stop", should_stop)
            span.set_attribute("stopping.reason", reason or "none")
            span.set_attribute("stopping.status", case.status.value if case.status else "unknown")

        if reason == "SILENT_RETRY_FAILED":
            case = transition_to_active_tier(case, reason=reason)
            ctx.case = case
            return {
                "messages": [SystemMessage(
                    content=f"[Guardrail] Silent tier retry failed. Transitioning to Active tier. "
                            f"Customer must switch payment instruments."
                )],
                "phase": "tier_transition",
            }

        if reason == "AWAITING_CUSTOMER":
            case.status = CaseStatus.AWAITING_CUSTOMER
            ctx.case = case
            if span and span.is_recording():
                span.set_attribute("stopping.final_status", case.status.value)
                span.set_status(StatusCode.OK)
            return {"phase": "awaiting_customer"}

        if reason == "SILENT_TIER_EXHAUSTED":
            case = transition_to_active_tier(case, reason=reason)
            ctx.case = case
            return {
                "messages": [SystemMessage(
                    content=f"[Guardrail] Silent tier exhausted after {case.silent_attempts} attempts. "
                            f"Transitioning to Active tier for customer-facing recovery."
                )],
                "phase": "tier_transition",
            }

        if should_stop:
            if case.recovered:
                case.status = CaseStatus.RECOVERED
            elif any(a.action_type.value == "escalate_to_human" for a in case.attempts):
                case.status = CaseStatus.ESCALATED
            else:
                case.status = CaseStatus.STOPPED
            ctx.case = case

        if span and span.is_recording():
            span.set_attribute("stopping.final_status", case.status.value if case.status else "unknown")
            span.set_status(StatusCode.OK)

        return {"phase": "complete"}


# ═══════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════

def _route_after_tools(state: RecoveryState) -> Literal["agent", "stopping_check"]:
    """`wait_for_customer` ends the turn — it does not loop back for another think.

    The model kept inventing a tool to wait with (`wait_for_page_push_response`,
    `wait_for_payment_recovery`) because ending a turn with plain text is, from
    its side, indistinguishable from abandoning the case. Now that the verb
    exists, calling it has to actually do what it says, or the model learns that
    it does not mean anything.
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "")
            if name in ("wait_for_customer", "close_case"):
                return "stopping_check"
            # A delivered push IS a wait. Its entire purpose is to elicit a
            # response, so carrying on in the same turn pre-empts the customer
            # you just asked to act.
            #
            # Live: the push went out at 04:16:46, the customer clicked it at
            # 04:16:51, and by 04:17:02 the agent — still in the same turn — had
            # created a 5% discounted link for someone who paid full price six
            # seconds later. It gave away INR 4,998.75 to a customer who was
            # already paying, and spent one of thirty lifetime payment links to
            # do it.
            if name == "send_page_push":
                try:
                    if json.loads(str(msg.content))\
                            .get("status") == "delivered":
                        return "stopping_check"
                except (ValueError, TypeError):
                    pass
            return "agent"
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            break
    return "agent"


def _route_after_approval(state: RecoveryState) -> Literal["tools", "agent"]:
    """A refused action must not reach ToolNode.

    The gate used to edge unconditionally into `tools`, so anything it objected
    to executed anyway. Now a refusal goes back to the agent with the reason.
    """
    last = state["messages"][-1]
    if isinstance(last, SystemMessage) and "[Guardrail]" in (last.content or ""):
        return "agent"
    return "tools"


def _route_after_guard(
    state: RecoveryState,
) -> Literal["human_approval_gate", "agent", "stopping_check"]:
    """Execute the tools, hand back to the agent, or leave the loop.

    Sending a blocked agent back to `agent` every time is what produced the doom
    loop: turns 4-8 of a live run were byte-identical. After two consecutive
    rounds where *everything* was blocked, the agent has no move left, so the
    graph exits instead of burning turns until MAX_TURNS.
    """
    if state.get("phase") == "guard_stop":
        return "stopping_check"

    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, SystemMessage) and "[Guardrail]" in last_message.content:
        return "agent"

    return "human_approval_gate"


def _route_after_critique(state: RecoveryState) -> Literal["critique_tools", "__end__"]:
    """Route self_critique output — if it made tool calls, execute them."""
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "critique_tools"
    return "__end__"


def build_graph():
    """Build the ReAct agent graph with real memory and guardrails.

    Pattern:
        START → agent → should_continue
            → [tool_repetition_guard → [human_approval_gate → tools → agent | agent] | stopping_check → END]
            → self_critique → END

    Guardrails:
        - tool_repetition_guard: prevents same tool+args called twice
        - human_approval_gate: pauses for approval on high-value recoveries (HITL)
        - stopping_check: enforces tier transitions and max attempts
        - self_critique: agent critiques its own performance after recovery
    """
    builder = StateGraph(RecoveryState, context_schema=RecoveryContext)

    # Nodes
    builder.add_node("agent", agent_node)
    builder.add_node("tool_repetition_guard", tool_repetition_guard)
    builder.add_node("human_approval_gate", human_approval_gate)
    builder.add_node("tools", _build_tool_node())
    builder.add_node("stopping_check", stopping_check)
    builder.add_node("self_critique", self_critique_node)

    # Edges
    builder.add_edge(START, "agent")

    # Agent → should_continue → guard or stopping
    builder.add_conditional_edges("agent", should_continue, {
        "tool_repetition_guard": "tool_repetition_guard",
        "stopping_check": "stopping_check",
    })

    # Guard → route (back to agent if blocked, to approval gate if ok)
    builder.add_conditional_edges("tool_repetition_guard", _route_after_guard, {
        "human_approval_gate": "human_approval_gate",
        "agent": "agent",
        "stopping_check": "stopping_check",
    })

    # Approval gate → tools, or back to the agent if it refused the action.
    builder.add_conditional_edges("human_approval_gate", _route_after_approval, {
        "tools": "tools",
        "agent": "agent",
    })

    # Tools → mask_outputs → agent (loop back with PII masking)
    builder.add_node("mask_outputs", mask_tool_outputs_node)
    builder.add_edge("tools", "mask_outputs")
    builder.add_conditional_edges("mask_outputs", _route_after_tools, {
        "agent": "agent",
        "stopping_check": "stopping_check",
    })

    # Stopping check → self-critique → critique_tools → end
    # ToolNode after self_critique allows manage_memory to actually execute
    builder.add_edge("stopping_check", "self_critique")
    from recovery_agent.agent.tools import TOOLS_BY_NAME as _TBNAME
    builder.add_node("critique_tools", _build_tool_node(tools=[_TBNAME["manage_memory"]]))
    builder.add_conditional_edges("self_critique", _route_after_critique, {
        "critique_tools": "critique_tools",
        "__end__": END,
    })
    builder.add_edge("critique_tools", END)

    # Working memory — the message history of each case's session.
    #
    # `MemorySaver` is LangGraph's dev checkpointer and keeps everything in the
    # process. A restart mid-recovery therefore lost the entire conversation for
    # every in-flight case: the customer's dismissal, what had been offered, the
    # lot. The case would carry on from the durable record and repeat itself.
    import os
    from pathlib import Path
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        cpath = Path(os.getenv("STATE_DIR", "data")) / "agent_sessions.db"
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cm = SqliteSaver.from_conn_string(str(cpath))
        checkpointer = cm.__enter__()
        _KEEP_ALIVE.append(cm)
        print(f"[Agent] sessions: sqlite at {cpath}", flush=True)
    except Exception as exc:
        checkpointer = MemorySaver()
        print(f"[Agent] sessions: IN-MEMORY ONLY, lost on restart ({exc})", flush=True)

    store = get_memory_store()
    return builder.compile(checkpointer=checkpointer, store=store)


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ═══════════════════════════════════════════════════════════════
# INITIAL STATE — builds the first HumanMessage from a Case
# ═══════════════════════════════════════════════════════════════

def build_initial_state(case) -> dict:
    """Build initial messages state from a Case object.

    The first message is a HumanMessage containing all failure context.
    The LLM will read this and start calling tools to diagnose and recover.
    """
    payment = case.payment
    context_parts = [
        f"A payment has failed and needs recovery.",
        f"",
        f"Payment ID: {payment.payment_id}",
        f"Amount: {payment.currency} {payment.amount:,.2f}"
        f"  (when a tool asks for `amount`, pass {payment.amount:.2f} — RUPEES, never paise)",
        f"Customer ID: {payment.customer_id}",
        f"Failure Code: {payment.failure_code or 'not provided'}",
        f"Failure Reason: {payment.failure_reason or 'not provided'}",
        f"Attempt: {case.attempt_count + 1} of {case.max_attempts}",
        f"Recovery Tier: {case.recovery_tier.value}",
    ]

    if payment.metadata.get("scenario") == "followup" or payment.failure_code == "no_response":
        context_parts[0] = (
            "FOLLOW-UP: a recovery attempt was ALREADY delivered to this customer "
            "and they still have not paid."
        )
        context_parts.extend([
            "",
            "The first channel did not work. Repeating it will not work either.",
            "Escalate the CHANNEL, not the case: move to the next rung of the "
            "ladder. Do NOT send another email with the same link, and do not "
            "reach for escalate_to_human — it is the last rung and will refuse "
            "you until the ones above it are done.",
        ])

    if payment.metadata.get("customer_email"):
        context_parts.append(f"Customer Email: {payment.metadata['customer_email']}")
    if payment.metadata.get("customer_phone"):
        context_parts.append(f"Customer Phone: {payment.metadata['customer_phone']}")

    # Where this case stands on the ladder, read from the durable record rather
    # than the Case — a Case is rebuilt on every hand-off run, so left to itself
    # the agent starts each rung not knowing which ones it has already climbed,
    # and "do not repeat a channel that failed" is advice it cannot act on.
    try:
        from recovery_agent.agent import ladder as _ladder
        from recovery_agent.state_store import StateStore
        rec = StateStore().get_payment(payment.payment_id) or {}
        st = _ladder.state(rec)
        tried = _ladder.actions_tried(rec)
        if st["climbed"] or tried:
            context_parts += [
                "",
                "WHERE THIS CASE STANDS ON THE LADDER:",
                f"  climbed:       {', '.join(st['climbed']) or 'nothing yet'}",
                f"  already tried: {', '.join(tried) or 'nothing yet'}"
                f"   <- do NOT repeat any of these",
            ]
            nxt = st["remaining"][0] if st["remaining"] else None
            context_parts.append(
                f"  next rung:     {nxt['rung']} — {nxt['what']}" if nxt else
                "  next rung:     none left; escalate_to_human is now permitted"
            )
            for u in st["unavailable"]:
                context_parts.append(f"  unavailable:   {u['rung']} ({u['why_not']})")
    except Exception:
        pass

    return {
        "messages": [HumanMessage(content="\n".join(context_parts))],
        "tool_call_history": [],
        "blocked_rounds": 0,
        "phase": "initializing",
    }
