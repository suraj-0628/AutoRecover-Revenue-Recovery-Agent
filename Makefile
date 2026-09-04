# Recovery Agent — dev commands
# Usage: make test, make start, make stop

.PHONY: test start stop check ci evals evals-full evals-baseline

test:
	@echo "Running tests with venv Python..."
	.venv/bin/python3 -m pytest tests/ -v --tb=short

test-quick:
	.venv/bin/python3 -m pytest tests/ -x --tb=short -q

check:
	@echo "Verifying critical imports..."
	.venv/bin/python3 -c "from langchain.tools import tool; from langgraph.graph import StateGraph; from langgraph.prebuilt import ToolNode; print('All imports OK')"

# Fold new rig-run decisions into the committed corpus, score them, and
# red-team the model. Recorded mode is free; red-team needs the LLM proxy
# (INCONCLUSIVE without it, never FAIL).
evals:
	.venv/bin/python3 -m recovery_agent.evals.run --mode recorded --sync-corpus --check
	.venv/bin/python3 -m recovery_agent.evals.run --mode redteam --k 3 --check

# The whole battery: recorded, replay, red-team, memory A/B.
evals-full:
	.venv/bin/python3 -m recovery_agent.evals.run --mode all --k 3 --sync-corpus --check

evals-baseline:
	.venv/bin/python3 -m recovery_agent.evals.run --mode all --k 3 --sync-corpus --write-baseline

# What CI runs on a fresh checkout: the unit suite plus the recorded-decision
# conformance gate over the COMMITTED corpus (evals/corpus/). No LLM, no rig,
# no data-test — deterministic and quota-free.
ci:
	.venv/bin/python3 -m pytest tests/ -q
	.venv/bin/python3 -m recovery_agent.evals.run --mode recorded --check

start:
	bash start.sh

stop:
	pkill -f "recovery_agent" 2>/dev/null
	pkill -f "phoenix" 2>/dev/null
	echo "All services stopped"
