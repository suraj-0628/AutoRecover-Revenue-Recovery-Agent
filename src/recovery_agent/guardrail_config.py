"""Runtime-tunable guardrail policy, with the reason for each knob attached.

The six guardrails were tunable only by env var, which means only by someone
with shell access and a restart. That is the wrong audience: the person who
should be moving a contact cap is the merchant running the recovery, and the
moment they need to move it is usually mid-demo or mid-incident.

Live 2026-09-04 (pay_ls1dep23k, INR 12,495): a genuine bank-decline recovery was
refused at "4/3 contacts in 24h" — the cap counted every earlier test against
the same customer, so the agent could not even create the link. Nothing was
wrong with the agent's judgement; the ceiling was simply too low for the day,
and there was no way to raise it without editing the environment and
restarting four services.

Each setting carries its own control type, because not everything is a slider:
a contact count has a natural range and slides; money is typed; an hour is
picked; a policy is switched on or off.

PRECEDENCE: stored value (this file) > environment variable > built-in default.
So an operator's deliberate change wins, the deployment's env still configures
a fresh install, and the code default is the policy of last resort.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _path() -> Path:
    return Path(os.getenv("STATE_DIR", "data")) / "guardrail_config.json"


#: Every knob: what it does, how it is edited, and what it costs to get wrong.
#: `kind` drives the control the dashboard renders.
#:   slider  — a bounded count, dragged
#:   number  — money, typed, because a rupee figure has no natural range
#:   hour    — 0-23, picked
#:   toggle  — a policy that is on or off
SETTINGS: list[dict[str, Any]] = [
    {
        "key": "max_contacts_24h", "env": "GUARDRAIL_MAX_CONTACTS_24H",
        "label": "Contacts per customer / 24h", "kind": "slider",
        "min": 1, "max": 30, "step": 1, "default": 5, "group": "Contact",
        "what": "How many emails, SMS and calls one customer may receive in a "
                "rolling day. In-page notifications never count — the customer "
                "is already on the page.",
        "why": "The ladder itself can legitimately reach someone three or four "
               "times in an evening (offer email, a call, the agreed link, a "
               "rail switch). Set below that and the cap blocks your own "
               "policy rather than spam.",
    },
    {
        "key": "min_contact_gap_minutes", "env": "GUARDRAIL_MIN_CONTACT_GAP_MIN",
        "label": "Minimum gap between contacts", "kind": "slider",
        "min": 0, "max": 60, "step": 1, "default": 5, "group": "Contact",
        "what": "How long the agent must leave a customer alone after "
                "contacting them, before it may contact them again on the "
                "same case.",
        "why": "The daily cap counts contacts but says nothing about their "
               "spacing, so a whole day's allowance can be spent in a minute. "
               "Live (pay_qttkbl2b3): a 5% offer was emailed, the status was "
               "checked forty seconds later, and because the money had not yet "
               "arrived a second email went out at 20% -- two offers and two "
               "links inside ninety seconds. Nobody decides on a payment that "
               "fast, so the second one bought nothing the first had not "
               "already bought. Set to 0 to disable.",
    },
    {
        "key": "quiet_enabled", "env": "", "label": "Quiet hours",
        "kind": "toggle", "default": True, "group": "Contact",
        "what": "Hold back contact that INTERRUPTS overnight — voice calls "
                "ring and SMS buzzes. Email still goes (it waits in an inbox) "
                "and in-page offers are never affected.",
        "why": "Switch off for a demo at an odd hour. Leave on in production: "
               "a 2am phone call costs more goodwill than the order is worth.",
    },
    {
        "key": "quiet_start", "env": "GUARDRAIL_QUIET_START",
        "label": "Quiet from", "kind": "hour", "default": 21, "group": "Contact",
        "what": "Hour of the evening the quiet window opens (IST).",
        "why": "",
    },
    {
        "key": "quiet_end", "env": "GUARDRAIL_QUIET_END",
        "label": "Quiet until", "kind": "hour", "default": 8, "group": "Contact",
        "what": "Hour of the morning it closes (IST). The agent is told exactly "
                "how many minutes remain, so it waits once rather than "
                "retrying into the same refusal.",
        "why": "",
    },
    {
        "key": "max_discount_giveaway", "env": "APPROVAL_DISCOUNT_THRESHOLD",
        "label": "Give-away ceiling", "kind": "number", "unit": "₹",
        "default": 5000.0, "group": "Money",
        "what": "The most the agent may discount on one order without a human. "
                "It caps what is GIVEN AWAY, never what is collected.",
        "why": "Capping the charge instead once refused a full-price link on a "
               "₹79,980 order and escalated a recovery that would have "
               "succeeded. The risk is the discount, not the debt.",
    },
    {
        "key": "max_single_retry", "env": "GUARDRAIL_MAX_SINGLE_RETRY",
        "label": "Max single retry", "kind": "number", "unit": "₹",
        "default": 500000.0, "group": "Money",
        "what": "Above this order value the agent may not retry a charge on its "
                "own; the case goes to a person.",
        "why": "A mistake on a large order is expensive and slow to unwind.",
    },
    {
        "key": "voice_enabled", "env": "VOICE_CALLS_ENABLED",
        "label": "AI voice calls", "kind": "toggle", "default": False,
        "group": "Channels",
        "what": "Whether the agent may place SuperU voice calls at all.",
        "why": "Calls cost real credits from a small allowance. Off by default; "
               "turn on for the demo and back off afterwards.",
    },
    {
        "key": "voice_min_amount", "env": "VOICE_MIN_AMOUNT_RUPEES",
        "label": "Minimum order to call", "kind": "number", "unit": "₹",
        "default": 5000.0, "group": "Channels",
        "what": "Below this the agent uses a cheaper channel instead of ringing.",
        "why": "A call can cost more than the order it is chasing.",
    },
    {
        "key": "double_debit_enabled", "env": "", "label": "Double-debit lock",
        "kind": "toggle", "default": True, "danger": True, "group": "Safety",
        "what": "Refuses any action that could charge a customer whose payment "
                "already succeeded or is still pending.",
        "why": "Switching this off risks taking money twice from a real "
               "customer. There is almost no reason to.",
    },
    {
        "key": "hard_decline_enabled", "env": "", "label": "Hard-decline block",
        "kind": "toggle", "default": True, "danger": True, "group": "Safety",
        "what": "Never retries a permanently dead instrument (lost, stolen, "
                "closed account, do-not-honour).",
        "why": "Card networks fine merchants for retrying these. Off means "
               "paying penalties to attempt something that cannot succeed.",
    },
    {
        "key": "opt_out_enabled", "env": "", "label": "Respect opt-out",
        "kind": "toggle", "default": True, "locked": True, "group": "Safety",
        "what": "Never contacts a customer who has asked not to be contacted.",
        "why": "Not switchable. This is a legal obligation, not a tuning "
               "parameter, and a dashboard that can turn it off is a liability.",
    },
]

BY_KEY = {s["key"]: s for s in SETTINGS}


def _stored() -> dict:
    try:
        return json.loads(_path().read_text())
    except (OSError, ValueError):
        return {}


def _coerce(spec: dict, value: Any) -> Any:
    kind = spec["kind"]
    if kind == "toggle":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if kind == "number":
        return float(value)
    number = int(float(value))
    if kind in ("slider", "hour"):
        lo = spec.get("min", 0 if kind == "hour" else 1)
        hi = spec.get("max", 23 if kind == "hour" else 10 ** 9)
        number = max(lo, min(hi, number))
    return number


def get(key: str) -> Any:
    """Stored value, else environment, else the built-in default."""
    spec = BY_KEY.get(key)
    if spec is None:
        raise KeyError(key)
    stored = _stored()
    if key in stored:
        try:
            return _coerce(spec, stored[key])
        except (TypeError, ValueError):
            pass
    env_name = spec.get("env")
    if env_name:
        raw = os.getenv(env_name, "")
        if raw != "":
            try:
                return _coerce(spec, raw)
            except (TypeError, ValueError):
                pass
    return spec["default"]


def all_values() -> dict[str, Any]:
    return {s["key"]: get(s["key"]) for s in SETTINGS}


def describe() -> list[dict[str, Any]]:
    """The full spec plus current values, for the dashboard to render."""
    out = []
    for s in SETTINGS:
        entry = dict(s)
        entry["value"] = get(s["key"])
        entry["source"] = ("operator" if s["key"] in _stored() else
                           "env" if s.get("env") and os.getenv(s["env"], "") != ""
                           else "default")
        out.append(entry)
    return out


def update(changes: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Apply operator changes. Returns (new values, rejected keys with reasons).

    A locked setting is refused rather than silently ignored — an operator who
    thinks they turned off opt-out and did not is worse off than one who was
    told no.
    """
    rejected: list[str] = []
    with _LOCK:
        stored = _stored()
        for key, raw in (changes or {}).items():
            spec = BY_KEY.get(key)
            if spec is None:
                rejected.append(f"{key}: unknown setting")
                continue
            if spec.get("locked"):
                rejected.append(f"{key}: {spec['label']} cannot be changed — "
                                f"{spec['why']}")
                continue
            try:
                stored[key] = _coerce(spec, raw)
            except (TypeError, ValueError):
                rejected.append(f"{key}: {raw!r} is not a valid value")
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stored, indent=2))
    return all_values(), rejected


def reset() -> dict[str, Any]:
    """Drop every operator override, back to env/defaults."""
    with _LOCK:
        try:
            _path().unlink()
        except OSError:
            pass
    return all_values()
