"""Behavioural evals for the recovery agent.

Four suites, one vocabulary:

- recorded  — score every decision the live agent actually made (free, no LLM)
- replay    — put recorded briefings back in front of the current model k
              times; measure conformance and stability (spends proxy calls)
- redteam   — synthetic briefings engineered to bait a violation; measure
              whether the model holds, and which rail catches it when it does
- memory-ab — the same decision with and without the memory lines; measure
              whether memory changes what the model does

The rules live in conformance.py and are the same money/ladder invariants the
runtime enforces — the evals measure whether the MODEL chooses correctly
before the rails have to catch it.

Run: .venv/bin/python -m recovery_agent.evals.run --mode all
"""
