"""Test-wide defaults: the suite must not touch the outside world.

Two live incidents on 2026-09-04, same root cause — a test process inherits a
production configuration and quietly acts on it.

1. TRACING. The collector defaults to localhost:6006, so on any machine where
   Phoenix is up (every machine running the demo) the suite exported into the
   real project. Eight runs in an afternoon buried 25 genuine agent traces
   under ~5,600 fragments.

2. EMAIL. `frontend.py` calls load_dotenv() at import, so the moment any test
   imported it, SMTP_HOST became the real Brevo relay for the whole session.
   One suite run sent five real emails with live payment links to the
   developer's own inbox and spent five of a 300/day allowance.

The values set here are chosen so that load_dotenv() cannot undo them: it does
not override a key already present in os.environ, and an empty string counts
as present.

Escape hatches, for the rare case of deliberately exercising a real integration:
    PHOENIX_TRACE_TESTS=1   allow tracing to reach the collector
    ALLOW_TEST_EMAIL=1      allow SMTP delivery
"""
from __future__ import annotations

import os


def _on(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


# ── tracing ────────────────────────────────────────────────────────────
if not _on("PHOENIX_TRACE_TESTS"):
    # Set before any recovery_agent import: init_observability() latches on
    # first call and every later call returns the first decision.
    os.environ["PHOENIX_DISABLED"] = "1"

# ── email ──────────────────────────────────────────────────────────────
if not _on("ALLOW_TEST_EMAIL"):
    # Empty host is the existing "write the .eml, do not deliver" path, so
    # tests that assert on eml_path keep working — they just stop mailing
    # real people. Blanking it here beats load_dotenv() because dotenv does
    # not override keys that already exist.
    os.environ["SMTP_HOST"] = ""

# ── paid and quota-bearing outbound ────────────────────────────────────
# Both are service-only capabilities: start.sh exports them, an ad-hoc process
# does not inherit them, and a test must never be the thing that spends a
# lifetime payment-link quota or a paid voice credit. Explicitly cleared
# rather than merely "probably absent".
for _capability in ("RAZORPAY_WRITES_OK", "SUPERU_CALLS_OK"):
    os.environ.pop(_capability, None)
