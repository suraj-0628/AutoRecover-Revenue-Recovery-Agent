#!/usr/bin/env bash
# Grow the decision corpus until the eval gate can actually gate.
#
# The evals' bottleneck is data, not machinery: 7 cases across 4 families
# cannot fail a build honestly, so the suite reports NOT VERIFIED. This drives
# real cases through the sandboxed rig — fake gateway, isolated state, no
# Razorpay links spent, no SuperU credits, no real email — and folds the
# decisions the agent actually makes into the committed corpus.
#
# Every family the policy treats differently is driven, because a corpus that
# misses one cannot see that policy at all:
#   D1 risk · C1 funds · C2 transient · B1/B3 method · A1/A2/A4 dropoff
#
# Usage:  ./tools/grow_corpus.sh [extra drive_cases args]
set -euo pipefail
cd "$(dirname "$0")/.."

PROXY="${LLM_PROXY_URL:-http://localhost:20128/v1/models}"
# Retry rather than probe once. A cold OmniRoute takes ~5s to answer its first
# request, so a single `-m 3` probe reported "not answering" for a proxy that
# was merely starting -- and aborted a job that takes minutes over a blip that
# lasts seconds.
proxy_up=""
for attempt in 1 2 3 4 5; do
  if curl -sf -m 10 "$PROXY" >/dev/null 2>&1; then proxy_up=1; break; fi
  [ "$attempt" = 5 ] || { echo "   proxy not answering yet (attempt $attempt/5), waiting..."; sleep 3; }
done
if [ -z "$proxy_up" ]; then
  echo "The LLM proxy is not answering on $PROXY (5 attempts over ~40s)."
  echo "The agent cannot make decisions without it, and decisions are what"
  echo "this collects. Start OmniRoute/antigravity and re-run."
  exit 1
fi

# Listening is not the same as having quota. A proxy that 429s every request
# still answers /v1/models, and the driver would spend minutes producing
# nothing but errors. This is a WARNING, not a gate: the model chain falls
# back, so an exhausted primary means slower/cheaper decisions, not no
# decisions -- but the operator should know which model the corpus will hold.
MODEL="${LLM_MODEL:-$(grep -E '^LLM_MODEL=' .env 2>/dev/null | cut -d= -f2-)}"
if [ -n "$MODEL" ]; then
  probe=$(curl -s -m 30 -X POST "${PROXY%/models}/chat/completions" \
    -H 'Content-Type: application/json' -H 'Authorization: Bearer sk-local' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" 2>/dev/null || true)
  case "$probe" in
    *exhausted*|*rate_limit*|*"429"*)
      echo "   WARNING: $MODEL has no quota right now."
      echo "   The run will fall back down the model chain, so the decisions"
      echo "   this collects will be a cheaper model's, not $MODEL's."
      echo "   Ctrl-C now if you want the corpus to reflect the primary model."
      sleep 5 ;;
  esac
fi

# Never run two drivers at once: the rig is shared, and resumability treats a
# missing observation as work to do — a second driver silently starts driving
# the cases the first has not reached yet.
if pgrep -f "python.*drive_cases" >/dev/null 2>&1; then
  echo "A case driver is already running. Wait for it to finish."
  exit 1
fi

echo "==> starting the sandboxed rig (port 6002, STATE_DIR=data-test)"
pkill -f "fake_stack.py" 2>/dev/null || true
sleep 1
nohup .venv/bin/python tests/integration/fake_stack.py frontend >/tmp/grow_frontend.log 2>&1 &
nohup .venv/bin/python tests/integration/fake_stack.py daemon   >/tmp/grow_daemon.log   2>&1 &
for _ in $(seq 1 20); do
  curl -sf -m 2 http://localhost:6002/health >/dev/null 2>&1 && break
  sleep 1
done
curl -sf -m 2 http://localhost:6002/health >/dev/null || { echo "rig did not come up"; exit 1; }

# One case per family, plus the drop-off variants that produce the most turns.
CASES="${*:-D1 C1 C2 B1 B3 A1 A2 A4 A7 E1}"
echo "==> driving: $CASES"
.venv/bin/python tests/integration/drive_cases.py $CASES || true

echo "==> folding new decisions into the committed corpus"
.venv/bin/python -m recovery_agent.evals.run --mode recorded --sync-corpus

echo
echo "==> where the corpus now stands"
.venv/bin/python - <<'PY'
import json, sys
sys.path.insert(0, "src")
from recovery_agent.evals import quality
rows = [json.loads(l) for l in open("evals/corpus/decisions.jsonl") if l.strip()]
unit, cov = quality.unit_of_analysis(rows), quality.coverage(rows)
print(f"  cases: {unit['cases']}  (need {quality.MIN_CASES_TO_GATE} to gate)")
print(f"  families covered: {cov['covered']}/{cov['of']}")
for b in cov["blind_spots"]:
    print(f"  BLIND SPOT — {b}")
if unit["cases"] >= quality.MIN_CASES_TO_GATE and not cov["blind_spots"]:
    print("\n  The corpus can now support a real gate. Freeze it:")
    print("    .venv/bin/python -m recovery_agent.evals.run --mode all "
          "--write-baseline")
    print("  then switch CI to `make ci-strict`.")
else:
    print("\n  Still short. Re-run to add more cases, or drive more variants:")
    print("    ./tools/grow_corpus.sh A3 A6 B4 C4 E2 E4")
PY

pkill -f "fake_stack.py" 2>/dev/null || true
