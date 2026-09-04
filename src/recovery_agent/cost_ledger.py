"""What each recovery actually cost, and how we know.

`economics.py` prices what we metered: tokens counted, messages delivered,
links minted, discounts accepted — multiplied by rates from env. Good numbers,
but every one of them is OUR arithmetic. When a judge asks "is that real?",
"we multiplied our own counter by a constant we chose" is a weak answer.

Some of it is not arithmetic at all. SuperU bills per call and will tell you
the figure; that is not an estimate, it is an invoice line. This ledger is
where those two kinds of number stop being confused with each other. Every
entry declares its provenance:

    BILLED     the provider told us this number (an invoice line)
    MEASURED   we counted the real quantity, priced at a configured rate
    ESTIMATED  neither — a stand-in until something better exists

Append-only JSONL next to the audit logs. `source_ref` is the provider's own
id for the thing (a SuperU call uuid), which makes reconciliation idempotent:
re-running it can never double-count a call, so it is safe to poll.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BILLED = "BILLED"
MEASURED = "MEASURED"
ESTIMATED = "ESTIMATED"

#: Surfaces that cost money. Kept as constants so a typo cannot silently
#: create a fifth surface nothing ever reads.
SURFACE_LLM = "llm"
SURFACE_VOICE = "voice"
SURFACE_EMAIL = "email"
SURFACE_LINK = "link"


def ledger_path(state_dir: str | None = None) -> Path:
    base = Path(state_dir or os.getenv("STATE_DIR", "data"))
    return base / "audit_logs" / "cost_ledger.jsonl"


def record(surface: str, inr: float, provenance: str,
           payment_id: str = "", source_ref: str = "",
           qty: float = 0.0, unit: str = "",
           raw: dict | None = None, state_dir: str | None = None) -> bool:
    """Append one cost event. Returns False if `source_ref` is already
    recorded, so callers can poll without fear. Never raises.

    `raw` keeps the provider's own fields verbatim. Currency assumptions can
    be wrong — SuperU returns two different meters and labels neither — so
    the ledger keeps what was actually said alongside what we made of it.
    A corrected assumption then means re-deriving, not re-fetching.
    """
    try:
        if source_ref and has_source_ref(source_ref, state_dir):
            return False
        path = ledger_path(state_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "surface": surface,
            "payment_id": payment_id,
            "inr": round(float(inr or 0), 4),
            "provenance": provenance,
            "source_ref": source_ref,
            "qty": float(qty or 0),
            "unit": unit,
        }
        if raw:
            entry["raw"] = raw
        with open(path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return True
    except Exception:
        return False


def read_events(state_dir: str | None = None,
                surface: str = "") -> list[dict]:
    """Every event, oldest first, skipping unparseable lines. Never raises."""
    try:
        path = ledger_path(state_dir)
        if not path.exists():
            return []
        out = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if not isinstance(entry, dict):
                continue
            if surface and entry.get("surface") != surface:
                continue
            out.append(entry)
        return out
    except Exception:
        return []


def has_source_ref(source_ref: str, state_dir: str | None = None) -> bool:
    if not source_ref:
        return False
    return any(e.get("source_ref") == source_ref
               for e in read_events(state_dir))


def drop_surface(surface: str, state_dir: str | None = None) -> int:
    """Remove every event for one surface, returning how many went.

    Only for re-deriving after a pricing assumption is corrected — the raw
    provider figures are kept on each entry precisely so this is possible
    without re-fetching. Rewrites the file atomically. Never raises.
    """
    try:
        path = ledger_path(state_dir)
        if not path.exists():
            return 0
        kept, dropped = [], 0
        for e in read_events(state_dir):
            if e.get("surface") == surface:
                dropped += 1
            else:
                kept.append(e)
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(e, default=str) + "\n" for e in kept))
        tmp.replace(path)
        return dropped
    except Exception:
        return 0


def by_payment(surface: str = "", state_dir: str | None = None) -> dict[str, float]:
    """Total INR per payment id, for one surface or all of them."""
    totals: dict[str, float] = {}
    for e in read_events(state_dir, surface=surface):
        pid = e.get("payment_id") or ""
        if not pid:
            continue
        totals[pid] = round(totals.get(pid, 0.0) + float(e.get("inr") or 0), 4)
    return totals


def summarise(state_dir: str | None = None) -> dict:
    """Per-surface totals with the provenance mix — what the ops view needs to
    say "this figure is billed" instead of "this figure is ours"."""
    events = read_events(state_dir)
    surfaces: dict[str, dict[str, Any]] = {}
    for e in events:
        s = surfaces.setdefault(e.get("surface") or "unknown", {
            "inr": 0.0, "events": 0, "provenance": {},
        })
        s["inr"] = round(s["inr"] + float(e.get("inr") or 0), 4)
        s["events"] += 1
        prov = e.get("provenance") or ESTIMATED
        s["provenance"][prov] = s["provenance"].get(prov, 0) + 1
    return {
        "surfaces": surfaces,
        "events": len(events),
        "billed_inr": round(sum(float(e.get("inr") or 0) for e in events
                                if e.get("provenance") == BILLED), 4),
    }
