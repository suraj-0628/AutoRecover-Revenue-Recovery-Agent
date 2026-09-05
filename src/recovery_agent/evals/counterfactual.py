"""Score the agent against scripted policies on the decisions it actually made.

Reads the recorded decision corpus, replays every scripted baseline through
the SAME perception facts, and reports conformance and money side by side.

    python -m recovery_agent.evals.counterfactual
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recovery_agent.evals.baselines import (BASELINES, contacts_spent,
                                            giveaway_rupees)
from recovery_agent.evals.conformance import judge_decision


def corpus_path() -> Path:
    return Path(__file__).resolve().parents[3] / "evals" / "corpus" / "decisions.jsonl"


def load(path: Path | None = None) -> list[dict]:
    p = path or corpus_path()
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def score(name: str, decisions: list[tuple[dict, list]]) -> dict[str, Any]:
    """Conformance, money and contacts for one policy over the same points."""
    total = conformant = 0
    giveaway = 0.0
    contacts = 0
    broken: dict[str, int] = {}
    acted = 0
    for facts, chosen in decisions:
        total += 1
        verdicts = judge_decision(facts, chosen)
        if all(v.ok for v in verdicts):
            conformant += 1
        for v in verdicts:
            if not v.ok:
                broken[v.rule] = broken.get(v.rule, 0) + 1
        giveaway += giveaway_rupees(facts, chosen)
        contacts += contacts_spent(chosen)
        if chosen:
            acted += 1
    return {
        "policy": name,
        "decisions": total,
        "acted": acted,
        "conformant": conformant,
        "conformance_pct": round(100.0 * conformant / total, 1) if total else 0.0,
        "giveaway_rupees": round(giveaway, 2),
        "contacts": contacts,
        "violations": dict(sorted(broken.items(), key=lambda kv: -kv[1])),
    }


def compare(rows: list[dict] | None = None) -> dict[str, Any]:
    rows = rows if rows is not None else load()
    agent_points = [(r.get("facts") or {}, r.get("chosen") or []) for r in rows]
    results = [score("agent", agent_points)]
    for name, policy in BASELINES.items():
        pts = [(f, policy(f)) for f, _ in agent_points]
        results.append(score(name, pts))
    return {"decisions": len(rows), "results": results}


def render(out: dict[str, Any]) -> str:
    lines = [
        "# Counterfactual — the agent vs. a script",
        "",
        f"Every one of the **{out['decisions']} recorded decisions** replayed "
        f"through each policy, scored on the SAME perception facts the agent "
        f"decided under. Entirely offline: no traffic was split and no customer "
        f"was experimented on.",
        "",
        "**What this shows:** whether the agent chooses better than a script at "
        "the points where a choice existed.",
        "**What it does NOT show:** that either policy *caused* a recovery. "
        "That needs a live control arm, which this deliberately is not.",
        "",
        "| Policy | Conformance | Acted | Given away | Contacts | Top violation |",
        "|---|---|---|---|---|---|",
    ]
    for r in out["results"]:
        worst = next(iter(r["violations"]), "—")
        lines.append(
            f"| `{r['policy']}` | {r['conformance_pct']}% "
            f"({r['conformant']}/{r['decisions']}) | {r['acted']} | "
            f"₹{r['giveaway_rupees']:,.2f} | {r['contacts']} | {worst} |")
    lines += [
        "",
        "`do_nothing` is conformant by construction — it never acts, so it "
        "never breaks a rule. It is here to keep the conformance column "
        "honest: a score a do-nothing policy can win is not measuring "
        "recovery, and the money and action columns are what separate them.",
    ]
    return "\n".join(lines)


def main() -> int:
    out = compare()
    text = render(out)
    print(text)
    results_dir = Path(__file__).resolve().parents[3] / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "counterfactual.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
