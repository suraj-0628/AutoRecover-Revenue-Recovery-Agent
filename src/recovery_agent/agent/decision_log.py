"""Decision log — every (perceived facts → chosen action) pair, durably.

The agent's unit tests exercise the machinery around the model; nothing
exercised the model's actual decisions, because nothing kept them. This log
is the corpus: one JSONL line per LLM turn recording what the agent could see
(the perception facts its briefing was rendered from) and what it chose to do
about it. The eval harness replays these briefings against the current
prompt/model and scores the choices against the money and ladder invariants —
so a prompt tweak that makes the agent start discounting bank declines fails
an eval instead of a customer.

Append-only, one file per STATE_DIR, alongside the per-case audit logs.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def decisions_path(state_dir: str | None = None) -> Path:
    base = Path(state_dir or os.getenv("STATE_DIR", "data"))
    return base / "audit_logs" / "decisions.jsonl"


def log_decision(payment_id: str, turn: int, model: str,
                 facts: dict, chosen: list[dict]) -> None:
    """Append one decision point. Never raises — losing a log line must never
    cost a recovery run."""
    try:
        path = decisions_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "payment_id": payment_id,
            "turn": turn,
            "model": str(model or "unknown"),
            "facts": facts or {},
            # A turn with no tool calls is a decision too — the decision to
            # answer in prose, which the conformance rules also judge.
            "chosen": [{"name": c.get("name", ""), "args": c.get("args") or {}}
                       for c in (chosen or [])],
        }
        with open(path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass


def load_decisions(state_dir: str | None = None,
                   limit: int | None = None) -> list[dict]:
    """Read the corpus, oldest first, skipping unparseable lines."""
    entries = read_decision_file(decisions_path(state_dir))
    if limit is not None:
        entries = entries[-limit:]
    return entries


def read_decision_file(path: Path) -> list[dict]:
    """Parse any decisions JSONL file — tolerant of junk lines."""
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and entry.get("payment_id"):
            out.append(entry)
    return out


def merge_decisions(*sources: list[dict]) -> list[dict]:
    """Union decision lists into one corpus, oldest first.

    Identity is (payment_id, turn): payment ids are minted fresh per case and
    turn numbers only ever grow within a case's continuous thread, so a
    collision means the same decision seen twice (a re-synced file, a
    duplicated line) — the later-timestamped copy wins. This is what lets the
    committed corpus absorb every rig run without ever double-counting one.
    """
    by_key: dict[tuple, dict] = {}
    for source in sources:
        for entry in source:
            key = (entry.get("payment_id"), entry.get("turn"))
            held = by_key.get(key)
            if held is None or str(entry.get("ts", "")) >= str(held.get("ts", "")):
                by_key[key] = entry
    return sorted(by_key.values(),
                  key=lambda e: (str(e.get("ts", "")),
                                 str(e.get("payment_id", "")),
                                 int(e.get("turn") or 0)))
