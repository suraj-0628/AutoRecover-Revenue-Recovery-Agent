"""Replay — put a recorded briefing back in front of the model, k times.

A journey through the full stack takes minutes and burns proxy quota on every
tool round-trip. A decision point needs none of that: the perception facts
were logged the moment the live agent decided, so the exact briefing can be
re-rendered and shown to the current model in one call. That turns "did the
prompt change break the agent's judgment?" from an hour of journeys into
seconds per decision — and makes stability measurable, because the same
briefing can be asked k times at sampling temperature.

Transport failures (the proxy 429ing, a model 404ing) are INCONCLUSIVE, never
a failed eval — same contract as the integration rig.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Any

from recovery_agent.evals.conformance import judge_decision


def build_replay_messages(facts: dict, bait: str = "") -> list:
    """The decision context, reconstructed: system prompt, briefing, case stub.

    The stub carries only what build_initial_state would have said that the
    briefing does not already say better; `bait` is adversarial text appended
    as if the orchestrator relayed it — red-team scenarios ride this.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from recovery_agent.agent.graph import SYSTEM_PROMPT
    from recovery_agent.agent.perception import as_briefing

    owed = facts.get("owed") or 0
    stub = [
        "A payment has failed and needs recovery.",
        "",
        f"Payment ID: {facts.get('payment_id', 'pay_replay')}",
        f"Amount: INR {float(owed):,.2f}"
        f"  (when a tool asks for `amount`, pass {float(owed):.2f} — RUPEES, never paise)",
        f"Failure Code: {facts.get('failure_code') or 'not provided'}",
        "Customer Email: customer@example.com",
        "Customer Phone: +91 99999 00000",
    ]
    if bait:
        stub += ["", bait]
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=as_briefing(facts)),
        HumanMessage(content="\n".join(stub)),
    ]


def _sampling_model(model_name: str | None = None):
    """A model for SAMPLING: no cache (k identical answers would measure the
    cache, not the model) and a small temperature so stability means something."""
    from langchain_openai import ChatOpenAI
    from recovery_agent.agent.tools import RECOVERY_TOOLS
    try:
        temperature = float(os.getenv("EVAL_TEMPERATURE", "0.3"))
    except ValueError:
        temperature = 0.3
    model = ChatOpenAI(
        model=model_name or os.getenv("EVAL_MODEL")
        or os.getenv("LLM_MODEL", "antigravity/gemini-2.5-flash"),
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:20128/v1"),
        api_key=os.getenv("LLM_API_KEY", "dummy"),
        temperature=temperature,
        max_tokens=1024,
        timeout=int(os.getenv("EVAL_LLM_TIMEOUT", "45")),
        cache=False,
    )
    return model.bind_tools(list(RECOVERY_TOOLS))


def proxy_reachable() -> bool:
    from recovery_agent.agent.llm_client import _check_llm_reachable
    return _check_llm_reachable()


def replay_decision(facts: dict, k: int = 3, bait: str = "",
                    model_name: str | None = None) -> dict[str, Any]:
    """Ask the model k times; judge every sample; measure agreement.

    Returns {samples, majority, agreement, ok_rate, transport_errors,
    inconclusive}. A sample is the list of tool calls the model proposed
    (possibly empty — prose is a decision too, and a conformant one).
    """
    messages = build_replay_messages(facts, bait=bait)
    model = _sampling_model(model_name)

    samples: list[dict] = []
    transport_errors = 0
    for _ in range(max(1, k)):
        try:
            response = model.invoke(messages)
        except Exception as exc:
            transport_errors += 1
            samples.append({"error": str(exc)[:160]})
            continue
        chosen = [{"name": tc["name"], "args": tc.get("args") or {}}
                  for tc in (getattr(response, "tool_calls", None) or [])]
        verdicts = judge_decision(facts, chosen)
        samples.append({
            "chosen": chosen,
            "first_tool": chosen[0]["name"] if chosen else "(prose)",
            "ok": all(v.ok for v in verdicts),
            "violations": [{"rule": v.rule, "reason": v.reason,
                            "enforced_by": v.enforced_by}
                           for v in verdicts if not v.ok],
        })

    real = [s for s in samples if "error" not in s]
    inconclusive = not real
    firsts = Counter(s["first_tool"] for s in real)
    majority, majority_n = (firsts.most_common(1)[0] if firsts else (None, 0))
    return {
        "samples": samples,
        "majority": majority,
        "agreement": round(majority_n / len(real), 3) if real else None,
        "ok_rate": (round(sum(1 for s in real if s["ok"]) / len(real), 3)
                    if real else None),
        "transport_errors": transport_errors,
        "inconclusive": inconclusive,
    }
