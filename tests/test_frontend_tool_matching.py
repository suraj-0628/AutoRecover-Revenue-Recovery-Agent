"""Tool results must be matched to their calls by tool_call_id.

The agent issues 3-4 tool calls per turn. The previous matcher assigned each
result to the first tool that still had `result is None`, in dict insertion
order, so results landed on the wrong tools. The visible consequence was that
`generate_recovery_payment_link` carried another tool's output, so
`sdk_res["link_url"]` was empty, `_watch_for_recovery` never started, and a
customer who actually paid was never noticed.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from recovery_agent.frontend import _parse_tool_result


def _match(all_messages):
    """The matching logic under test, mirrored from _run_agent_for_payment_inner."""
    calls_by_id, order = {}, []
    for msg in all_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["id"] not in calls_by_id:
                    calls_by_id[tc["id"]] = {"name": tc["name"], "args": tc["args"],
                                             "result": None}
                    order.append(tc["id"])
        elif isinstance(msg, ToolMessage):
            entry = calls_by_id.get(getattr(msg, "tool_call_id", None))
            if entry is not None:
                entry["result"] = _parse_tool_result(msg.content)

    out = {}
    for cid in order:
        e = calls_by_id[cid]
        if out.get(e["name"]) is None or e["result"] is not None:
            out[e["name"]] = {"args": e["args"], "result": e["result"]}
    return out


def _call(name, cid, **args):
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


def test_parallel_results_go_to_the_right_tools():
    """Four calls in one batch, results returned out of order."""
    msgs = [
        HumanMessage(content="payment failed"),
        AIMessage(content="", tool_calls=[
            _call("diagnose_payment_failure", "c1"),
            _call("get_customer_payment_history", "c2"),
            _call("search_memory", "c3"),
            _call("generate_recovery_payment_link", "c4", amount=2999.0),
        ]),
        # deliberately not in call order — Razorpay/LLM do not guarantee it
        ToolMessage(content=json.dumps({"status": "ok", "link_url": "https://rzp.io/L",
                                        "link_id": "plink_1"}),
                    tool_call_id="c4", name="generate_recovery_payment_link"),
        ToolMessage(content=json.dumps({"status": "no_data"}),
                    tool_call_id="c3", name="search_memory"),
        ToolMessage(content=json.dumps({"root_cause": "bank_declined"}),
                    tool_call_id="c1", name="diagnose_payment_failure"),
        ToolMessage(content=json.dumps({"status": "ok", "total_payments": 2}),
                    tool_call_id="c2", name="get_customer_payment_history"),
    ]
    got = _match(msgs)

    assert got["generate_recovery_payment_link"]["result"]["link_url"] == "https://rzp.io/L"
    assert got["diagnose_payment_failure"]["result"]["root_cause"] == "bank_declined"
    assert got["search_memory"]["result"]["status"] == "no_data"
    assert got["get_customer_payment_history"]["result"]["total_payments"] == 2


def test_the_link_url_survives_so_the_watcher_can_start():
    """The exact failure seen live: link_url empty -> nothing polls for payment."""
    msgs = [
        AIMessage(content="", tool_calls=[
            _call("diagnose_payment_failure", "a1"),
            _call("generate_recovery_payment_link", "a2", amount=3798.0),
        ]),
        ToolMessage(content=json.dumps({"root_cause": "bank_declined"}),
                    tool_call_id="a1", name="diagnose_payment_failure"),
        ToolMessage(content=json.dumps({"status": "ok", "link_url": "https://rzp.io/X",
                                        "link_id": "plink_X"}),
                    tool_call_id="a2", name="generate_recovery_payment_link"),
    ]
    sdk_res = _match(msgs)["generate_recovery_payment_link"]["result"]
    assert sdk_res.get("link_url"), "watcher would never start"
    assert sdk_res.get("link_id") == "plink_X"


def test_an_unanswered_call_keeps_a_none_result():
    msgs = [
        AIMessage(content="", tool_calls=[_call("escalate_to_human", "z1")]),
    ]
    assert _match(msgs)["escalate_to_human"]["result"] is None


def test_a_later_call_to_the_same_tool_wins():
    msgs = [
        AIMessage(content="", tool_calls=[_call("generate_recovery_payment_link", "b1")]),
        ToolMessage(content=json.dumps({"status": "error"}), tool_call_id="b1",
                    name="generate_recovery_payment_link"),
        AIMessage(content="", tool_calls=[_call("generate_recovery_payment_link", "b2")]),
        ToolMessage(content=json.dumps({"status": "ok", "link_url": "https://rzp.io/2"}),
                    tool_call_id="b2", name="generate_recovery_payment_link"),
    ]
    assert _match(msgs)["generate_recovery_payment_link"]["result"]["link_url"].endswith("/2")


def test_masked_error_results_are_still_parsed():
    """mask_tool_outputs_node prepends [TOOL ERROR]; a startswith('{') check lost it."""
    body = '[TOOL ERROR] generate_recovery_payment_link FAILED: quota\n{"status": "error", "message": "quota"}'
    assert _parse_tool_result(body) == {"status": "error", "message": "quota"}


def test_non_json_results_are_not_lost():
    assert _parse_tool_result("created memory abc-123")["raw"].startswith("created memory")
