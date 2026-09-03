"""Delivery hook for in-page pushes.

Why this exists rather than `from recovery_agent.frontend import deliver_page_push`
-----------------------------------------------------------------------------------
The frontend runs as `python -m recovery_agent.frontend`, so that file is loaded
under the name `__main__`. Importing `recovery_agent.frontend` from a tool then
executes the module a *second* time under its real name, producing a second Flask
app and a second Socket.IO server. Emitting on that second instance reaches
nobody: the customer's browser is attached to the first.

The failure is silent — the tool reports "delivered" and no push ever appears.

A tiny neutral module both sides import by the same name avoids it. The frontend
registers its real delivery function at startup; tools call whatever is
registered.
"""
from __future__ import annotations

from typing import Any, Callable

_deliver: Callable[[dict], dict] | None = None


def register_delivery(fn: Callable[[dict], dict]) -> None:
    """Called by the frontend at import time with its real emit function."""
    global _deliver
    _deliver = fn


def deliver(payload: dict[str, Any]) -> dict[str, Any]:
    """Send a push, or say plainly that there is nowhere to send it."""
    if _deliver is None:
        return {
            "status": "no_active_session",
            "note": "no checkout page is connected — the customer has left, "
                    "so reach them on another channel",
        }
    return _deliver(payload)


def is_available() -> bool:
    return _deliver is not None
