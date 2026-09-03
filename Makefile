# Recovery Agent — dev commands
# Usage: make test, make start, make stop

.PHONY: test start stop check

test:
	@echo "Running tests with venv Python..."
	.venv/bin/python3 -m pytest tests/ -v --tb=short

test-quick:
	.venv/bin/python3 -m pytest tests/ -x --tb=short -q

check:
	@echo "Verifying critical imports..."
	.venv/bin/python3 -c "from langchain.tools import tool; from langgraph.graph import StateGraph; from langgraph.prebuilt import ToolNode; print('All imports OK')"

start:
	bash start.sh

stop:
	pkill -f "recovery_agent" 2>/dev/null
	pkill -f "phoenix" 2>/dev/null
	echo "All services stopped"
