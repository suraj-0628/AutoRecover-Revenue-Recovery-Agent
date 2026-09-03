"""A holdout arm, so "we recovered INR X" can become "we caused INR X".

Every recovery number this system has produced counts cases where money arrived
after the agent acted. It cannot separate that from money that would have
arrived anyway — and the distinction is not hypothetical. On `pay_9frnh6kjb` the
customer paid in full while the agent sat blocked by a stale flag, having done
nothing across three runs. That case counts as a recovery.

So a fraction of cases are deliberately not worked. They are recorded,
classified and watched exactly like the rest; the agent is simply never started
on them. What they recover on their own is the baseline, and the difference is
the only honest claim this system can make about its own value.

Two design points that matter more than they look:

**Assignment is deterministic**, from a hash of the payment id — not a coin
flip. A case is re-entered whenever a customer acts, a retry fires or a run
hands off, and a fresh random draw each time would silently move cases between
arms and destroy the comparison.

**Cases with no arm are excluded, not counted as treated.** Everything recorded
before this existed was worked by an agent under active development, with its
bugs and its fixes. Folding that history into the treated arm would inflate the
denominator with runs that no longer represent the system.
"""
from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Iterable

TREATED = "treated"
CONTROL = "control"


def holdout_fraction() -> float:
    """Share of cases held back. 0 disables the experiment entirely."""
    try:
        f = float(os.getenv("HOLDOUT_FRACTION", "0.2"))
    except ValueError:
        return 0.2
    return min(max(f, 0.0), 0.5)          # never hold back more than half


def assign(payment_id: str) -> str:
    """Which arm this payment belongs to. Same id, same answer, always."""
    fraction = holdout_fraction()
    if fraction <= 0 or not payment_id:
        return TREATED
    digest = hashlib.sha256(payment_id.encode("utf-8")).digest()
    # 16 bits is plenty of resolution for a 0-50% split and keeps the
    # arithmetic obvious.
    bucket = int.from_bytes(digest[:2], "big") / 65535.0
    return CONTROL if bucket < fraction else TREATED


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — honest at the sample sizes this will actually see.

    A normal-approximation interval on 4 successes out of 15 produces bounds
    outside [0, 1], which would be worse than showing nothing.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    margin = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _recovered(rec: dict) -> bool:
    return (rec.get("status") == "recovered"
            or float(rec.get("recovered_amount") or 0) > 0)


def results(records: Iterable[dict]) -> dict[str, Any]:
    """Recovery rates per arm, the difference, and how much to trust it."""
    arms: dict[str, dict[str, Any]] = {
        TREATED: {"n": 0, "recovered": 0, "value": 0.0},
        CONTROL: {"n": 0, "recovered": 0, "value": 0.0},
    }
    for rec in records:
        arm = rec.get("arm")
        if arm not in arms:               # unassigned history is not evidence
            continue
        a = arms[arm]
        a["n"] += 1
        if _recovered(rec):
            a["recovered"] += 1
            a["value"] += float(rec.get("recovered_amount") or 0)

    for a in arms.values():
        a["rate"] = (a["recovered"] / a["n"]) if a["n"] else 0.0
        lo, hi = _wilson(a["recovered"], a["n"])
        a["ci"] = [round(lo * 100, 1), round(hi * 100, 1)]
        a["value"] = round(a["value"], 2)

    t, c = arms[TREATED], arms[CONTROL]
    lift = (t["rate"] - c["rate"]) * 100

    # Say plainly when the arms cannot yet be told apart. Overlapping intervals
    # is not "no effect", it is "not enough evidence" — and reporting a lift
    # without that caveat is the thing this module exists to stop.
    conclusive = bool(t["n"] and c["n"] and t["ci"][0] > c["ci"][1])
    if not c["n"]:
        verdict = "no control cases yet — lift cannot be computed"
    elif not t["n"]:
        verdict = "no treated cases yet"
    elif conclusive:
        verdict = (f"treated recovers more, and the intervals do not overlap "
                   f"(n={t['n']}+{c['n']})")
    else:
        verdict = (f"not yet separable at n={t['n']}+{c['n']} — the intervals "
                   f"overlap, so this lift is suggestive, not evidence")

    return {
        "enabled": holdout_fraction() > 0,
        "holdout_fraction": holdout_fraction(),
        "treated": t,
        "control": c,
        "lift_pp": round(lift, 1),
        "conclusive": conclusive,
        "verdict": verdict,
    }
