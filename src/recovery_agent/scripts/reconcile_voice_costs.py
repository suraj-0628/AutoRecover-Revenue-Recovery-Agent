"""Pull SuperU's billed call costs into the ledger.

Voice was the one surface spending real money against a guessed constant.
This reads SuperU's own per-call charges and records them as BILLED against
the case that caused each call.

    .venv/bin/python -m recovery_agent.scripts.reconcile_voice_costs
    .venv/bin/python -m recovery_agent.scripts.reconcile_voice_costs --state-dir data-test
    .venv/bin/python -m recovery_agent.scripts.reconcile_voice_costs --dry-run

READ ONLY against SuperU: it fetches call logs and places no calls, so it
spends no credits and is safe to run on a timer. Idempotent — each call is
keyed on its own uuid, so running it twice records nothing twice.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-dir", default=os.getenv("STATE_DIR", "data"))
    ap.add_argument("--limit", type=int, default=100,
                    help="how many recent calls to examine")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be recorded, write nothing")
    ap.add_argument("--rederive", action="store_true",
                    help="drop existing voice entries and re-record them at "
                         "the current pricing assumptions (the provider is "
                         "re-read, but nothing is charged twice)")
    args = ap.parse_args()

    os.environ["STATE_DIR"] = args.state_dir

    if args.rederive and not args.dry_run:
        from recovery_agent import cost_ledger
        gone = cost_ledger.drop_surface(cost_ledger.SURFACE_VOICE,
                                        args.state_dir)
        print(f"[voice-costs] dropped {gone} voice entries for re-derivation")

    from recovery_agent.integrations.superu_client import get_superu_client
    from recovery_agent.integrations import superu_reconcile as rec

    client = get_superu_client()
    if not client.can_read:
        print("[voice-costs] SUPERU_API_KEY is not set — nothing to reconcile",
              file=sys.stderr)
        return 1

    if args.dry_run:
        result = client.get_call_logs(limit=args.limit)
        if result.get("status") != "ok":
            print(f"[voice-costs] read failed: "
                  f"{result.get('error') or result.get('reason')}", file=sys.stderr)
            return 1
        from recovery_agent import cost_ledger
        total = 0.0
        for call in result.get("calls") or []:
            pid = rec.payment_id_from_campaign(call.get("campaign_id"))
            if not pid:
                continue
            inr, raw = rec.call_cost_inr(call)
            known = cost_ledger.has_source_ref(str(call.get("id") or ""),
                                               args.state_dir)
            total += 0 if known else inr
            print(f"[voice-costs] {'known ' if known else 'NEW   '} {pid}: "
                  f"INR {inr:.2f} ({raw['call_duration_seconds']:.1f}s, "
                  f"cost={raw['cost']} {raw['cost_currency']}, "
                  f"telecom={raw['telecom_total_cost']} {raw['telecom_currency']})")
        print(f"[voice-costs] would record INR {total:.2f} (dry run)")
        return 0

    out = rec.reconcile(client=client, limit=args.limit,
                        state_dir=args.state_dir)
    if out.get("status") != "ok":
        print(f"[voice-costs] {out.get('status')}: {out.get('reason')}",
              file=sys.stderr)
        return 1
    for row in out.get("rows") or []:
        print(f"[voice-costs] {row['payment_id']}: INR {row['inr']:.2f} "
              f"({row['seconds']:.1f}s) call {row['call_id'][:8]}")
    print(f"[voice-costs] recorded {out['recorded']} new "
          f"(INR {out['inr']:.2f}), {out['already_known']} already known, "
          f"{out['unmatched']} not from this system. "
          f"SuperU reports {out['provider_calls']} calls / "
          f"{out['provider_total_cost']} platform cost in its own units.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
