"""Backfill llm_usage for cases worked before token capture existed.

The OPS view prices LLM spend from `llm_usage` on each case record — a field
that only started accumulating when the capture hooks landed. Every earlier
case shows "0 calls", which reads as free when it was not.

The real history was never lost: LangGraph's checkpointer serialized every
AIMessage — usage_metadata, response model name and all — into
agent_sessions.db, one thread per case. This script walks those threads,
rebuilds each case's token history from what the model actually returned at
the time, and writes it onto the record. Wall-clock seconds are not in the
checkpoints and are left alone.

    .venv/bin/python -m recovery_agent.scripts.backfill_llm_usage             # data/
    .venv/bin/python -m recovery_agent.scripts.backfill_llm_usage --state-dir data-test
    .venv/bin/python -m recovery_agent.scripts.backfill_llm_usage --dry-run

Safe to re-run, and safe alongside a live stack: records are updated through
StateStore (cross-process FileLock + merge), and a case whose recorded usage
already meets or exceeds the checkpoint-derived figures is left untouched.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def usage_from_messages(messages: list) -> dict:
    """Token history as the checkpointed AIMessages tell it. Pure.

    Every AIMessage is one LLM response, so one call. Tokens come from
    usage_metadata where the provider returned it; a response without usage
    still counts as a call — a call with unknown tokens was not a free call.
    """
    calls = 0
    input_tokens = 0
    output_tokens = 0
    by_model: dict[str, int] = {}
    for m in messages:
        if type(m).__name__ != "AIMessage":
            continue
        calls += 1
        usage = getattr(m, "usage_metadata", None) or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
        model = str((getattr(m, "response_metadata", None) or {})
                    .get("model_name") or "unknown")
        by_model[model] = by_model.get(model, 0) + 1
    return {"calls": calls, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "by_model": by_model}


def should_backfill(existing: dict | None, derived: dict) -> bool:
    """Only write when the checkpoints know MORE than the record does.

    The checkpoint is a superset of live capture (every captured turn's
    AIMessage is in it), so derived >= existing means fill; an existing
    record that somehow exceeds the checkpoints is trusted and kept.
    """
    if derived["calls"] <= 0:
        return False
    existing = existing or {}
    return (derived["calls"] > int(existing.get("calls") or 0)
            or derived["input_tokens"] > int(existing.get("input_tokens") or 0))


def backfill(state_dir: str, dry_run: bool = False) -> list[dict]:
    db_path = Path(state_dir) / "agent_sessions.db"
    if not db_path.exists():
        print(f"[backfill] no checkpoint db at {db_path}", file=sys.stderr)
        return []

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        threads = [r[0] for r in conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints "
            "WHERE thread_id LIKE 'case:%'")]
    finally:
        conn.close()
    if not threads:
        print(f"[backfill] no case threads in {db_path}")
        return []

    os.environ["STATE_DIR"] = state_dir
    from langgraph.checkpoint.sqlite import SqliteSaver
    from recovery_agent.state_store import StateStore

    store = StateStore(Path(state_dir))
    rows: list[dict] = []
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        for thread_id in sorted(threads):
            payment_id = thread_id.split("case:", 1)[1]
            try:
                tup = saver.get_tuple({"configurable": {"thread_id": thread_id}})
            except Exception as exc:
                rows.append({"payment_id": payment_id, "action": "unreadable",
                             "why": str(exc)[:120]})
                continue
            if tup is None:
                continue
            messages = (tup.checkpoint.get("channel_values") or {}).get("messages") or []
            derived = usage_from_messages(messages)

            rec = store.get_payment(payment_id)
            if rec is None:
                rows.append({"payment_id": payment_id, "action": "no_record",
                             **derived})
                continue
            existing = rec.get("llm_usage") or {}
            if not should_backfill(existing, derived):
                rows.append({"payment_id": payment_id, "action": "kept",
                             **derived})
                continue

            merged = dict(existing)
            merged.update({
                "calls": derived["calls"],
                "input_tokens": derived["input_tokens"],
                "output_tokens": derived["output_tokens"],
                "by_model": derived["by_model"],
                "backfilled_at": datetime.now(timezone.utc).isoformat(),
            })
            rows.append({"payment_id": payment_id, "action": "backfilled",
                         **derived})
            if not dry_run:
                store.update_payment(payment_id, llm_usage=merged)
    if not dry_run:
        store.flush()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", default=os.getenv("STATE_DIR", "data"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = backfill(args.state_dir, dry_run=args.dry_run)
    filled = [r for r in rows if r["action"] == "backfilled"]
    kept = [r for r in rows if r["action"] == "kept"]
    orphans = [r for r in rows if r["action"] in ("no_record", "unreadable")]
    for r in filled:
        print(f"[backfill] {r['payment_id']}: {r['calls']} calls, "
              f"{r['input_tokens']:,} in / {r['output_tokens']:,} out tokens"
              + (" (dry-run)" if args.dry_run else ""))
    print(f"[backfill] {args.state_dir}: {len(filled)} backfilled, "
          f"{len(kept)} already current, {len(orphans)} without a record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
