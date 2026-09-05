"""Whether a score is worth believing — separate from what the score says.

An eval suite that reports 92.9% and nothing else invites exactly one
mistake: treating that number as a measurement. On 2026-09-04 this project's
corpus held 28 decisions and the report said so — but they were 28 TURNS
across 4 CASES, two of which contributed 24 of them, and the corpus contained
zero `method` and zero `funds` cases. The two failure families with the most
intricate policy were entirely unmeasured, and the headline number was
essentially "two drop-offs went well".

Nothing was wrong with the arithmetic. The problem was that the report gave no
way to notice, and the CI gate happily compared two equally unfounded numbers
and called stability "no regression".

So every metric here carries its own credibility alongside it:

  * the UNIT of analysis (turns are not independent samples; cases are)
  * a confidence interval, so 26/28 is never mistaken for a precise 92.9%
  * COVERAGE — which parts of the policy the corpus never exercises
  * PROVENANCE — when this mode last actually ran, and against what

A metric that fails these is reported, not gated. Refusing to gate on a number
you cannot defend is the point: a green build bought with a stale, underpowered
score is worse than no build gate at all, because it is trusted.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any

#: Below this many independent CASES a rate is noise, not a measurement. Four
#: cases move a percentage by 25 points each; a 2% regression tolerance
#: against that is theatre.
MIN_CASES_TO_GATE = 20

#: A mode older than this is stale: it was measured before changes it is now
#: being used to approve. The scorecard merges modes forward, so without this
#: a red-team score from last week silently gates today's prompt rewrite.
MAX_AGE_HOURS = 24

#: Every failure family the policy treats differently. A corpus missing one is
#: not a smaller corpus — it is a corpus that cannot see that policy at all.
REQUIRED_KINDS = ("dropoff", "method", "funds", "transient", "risk")


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval. Honest at small n, where normal approximation is not."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def unit_of_analysis(rows: list[dict]) -> dict[str, Any]:
    """Turns are not independent; cases are. Report both and the concentration.

    A corpus of 28 turns where one case supplies 12 is closer to n=4 than to
    n=28, and a rate computed over turns inherits that case's idiosyncrasies
    12 times over.
    """
    per_case = collections.Counter(r.get("payment_id", "?") for r in rows)
    turns = len(rows)
    cases = len(per_case)
    top = per_case.most_common(1)[0][1] if per_case else 0
    return {
        "turns": turns,
        "cases": cases,
        "turns_per_case": round(turns / cases, 1) if cases else 0.0,
        "largest_case_share": round(top / turns, 3) if turns else 0.0,
        "effective_n": cases,
        "note": ("turns are correlated within a case; `cases` is the unit that "
                 "supports a rate"),
    }


def coverage(rows: list[dict]) -> dict[str, Any]:
    """Which parts of the policy this corpus can and cannot see."""
    kinds = collections.Counter(
        (r.get("facts") or {}).get("failure_kind") or "unknown" for r in rows)
    case_kinds = {}
    for r in rows:
        case_kinds[r.get("payment_id")] = (r.get("facts") or {}).get("failure_kind")
    by_kind_cases = collections.Counter(v or "unknown" for v in case_kinds.values())
    missing = [k for k in REQUIRED_KINDS if by_kind_cases.get(k, 0) == 0]
    return {
        "turns_by_kind": dict(kinds),
        "cases_by_kind": dict(by_kind_cases),
        "missing_kinds": missing,
        "covered": len(REQUIRED_KINDS) - len(missing),
        "of": len(REQUIRED_KINDS),
        "blind_spots": [
            f"no {k} case — the policy for it is never exercised" for k in missing
        ],
    }


def fingerprint(rows: list[dict]) -> str:
    """Identity of the exact corpus a score was computed over.

    A score is only comparable to another score computed over the same data.
    Without this the gate compares yesterday's number on yesterday's corpus to
    today's on today's and calls the difference a regression.
    """
    key = "|".join(sorted(f"{r.get('payment_id')}:{r.get('turn')}" for r in rows))
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def stamp(result: dict, rows: list[dict] | None = None) -> dict:
    """Attach provenance to a mode result, so staleness is visible later."""
    out = dict(result)
    out["ran_at"] = datetime.now(timezone.utc).isoformat()
    if rows is not None:
        out["corpus_fingerprint"] = fingerprint(rows)
        out["unit"] = unit_of_analysis(rows)
        out["coverage"] = coverage(rows)
    return out


def age_hours(result: dict) -> float | None:
    ran = result.get("ran_at")
    if not ran:
        return None
    try:
        when = datetime.fromisoformat(str(ran))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - when).total_seconds() / 3600, 2)


def gateability(mode: str, result: dict) -> dict[str, Any]:
    """May this metric fail a build? Every refusal names its reason.

    Three ways a number loses the right to gate: nobody knows when it was
    measured, it was measured too long ago to describe the current code, or it
    rests on too few independent cases to distinguish a regression from noise.
    """
    reasons: list[str] = []
    age = age_hours(result)
    if result.get("inconclusive"):
        reasons.append("the run itself was inconclusive")
    if age is None:
        reasons.append("no run timestamp — provenance unknown")
    elif age > MAX_AGE_HOURS:
        reasons.append(f"stale: last measured {age:.0f}h ago, over the "
                       f"{MAX_AGE_HOURS}h limit")
    n = (result.get("unit") or {}).get("effective_n")
    if n is None:
        n = result.get("cases")
    if isinstance(n, int) and n < MIN_CASES_TO_GATE:
        reasons.append(f"underpowered: {n} independent case(s), "
                       f"{MIN_CASES_TO_GATE} needed to gate")
    return {"mode": mode, "gateable": not reasons, "reasons": reasons,
            "age_hours": age, "effective_n": n}


def describe_rate(successes: int, n: int) -> str:
    """A rate that shows its own uncertainty."""
    if n <= 0:
        return "no data"
    lo, hi = wilson(successes, n)
    return (f"{100.0 * successes / n:.1f}% ({successes}/{n}, "
            f"95% CI {100 * lo:.0f}–{100 * hi:.0f}%)")
