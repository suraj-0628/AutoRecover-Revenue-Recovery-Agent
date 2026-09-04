"""One provider going dark must not end the run.

`agent_node` carries a six-model fallback chain for exactly this, but the
branch that walks it matched only 502/429/rate/server_error. A provider losing
its credentials returns **404** — `No active credentials for provider: gemini`
— which fell through to `raise`, so the whole graph died on the FIRST model
with five healthy fallbacks below it never tried.

Found live: it killed four cases of the 2026-09-03 integration matrix (B3, A6,
A4, A3), which had been recorded as "LLM quota" starvation until the trail was
read and showed a 404. AUDIT-FINDINGS.md had listed this as an S0 blocker.
"""
import pytest

from recovery_agent.agent.graph import _UNAVAILABLE_MODEL


REAL_OUTAGES = [
    # The exact error that killed the integration run.
    "Error code: 404 - {'error': {'message': 'No active credentials for "
    "provider: gemini', 'type': 'invalid_request_error', 'code': 'model_not_found'}}",
    "Error code: 429 - rate limit exceeded",
    "Error code: 502 - Bad Gateway",
    "Error code: 503 - upstream unavailable",
    "Error code: 504 - Gateway Timeout",
    "openai.NotFoundError: model_not_found",
    "insufficient_quota: You exceeded your current quota",
    "The model is overloaded. Please try again later.",
]

#: A bad REQUEST is not a bad model — walking the chain would repeat the same
#: mistake five more times and hide the real error.
BAD_REQUESTS = [
    "Error code: 400 - messages.1: `tool_use` ids were found without `tool_result` blocks",
    "Error code: 401 - invalid api key",
    "Error code: 403 - permission denied",
    "ValueError: unexpected message order",
]


@pytest.mark.parametrize("err", REAL_OUTAGES)
def test_an_unavailable_model_moves_to_the_next_fallback(err):
    assert _UNAVAILABLE_MODEL.search(err), (
        f"this error must walk the fallback chain, not end the run: {err[:80]}")


@pytest.mark.parametrize("err", BAD_REQUESTS)
def test_a_bad_request_does_not_burn_the_whole_chain(err):
    assert not _UNAVAILABLE_MODEL.search(err), (
        f"this is a request problem, not a model outage: {err[:80]}")


def test_the_agent_node_actually_consults_the_matcher():
    """The regex is only worth anything if the retry branch uses it."""
    import inspect
    from recovery_agent.agent import graph

    src = inspect.getsource(graph.agent_node)
    assert "_UNAVAILABLE_MODEL.search" in src
    # The old literal checks must be gone, or 404 slips through again.
    assert '"502" in str(e)' not in src
    assert '"429" in str(e)' not in src
