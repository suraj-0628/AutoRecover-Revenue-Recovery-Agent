"""Seed Phoenix's model price table with the OmniRoute fleet.

Phoenix computes span/trace/session cost from token counts the moment a
span's `llm.model_name` matches a pricing entry. Its built-in table knows the
big providers' official names — not the aliases and served names our proxy
returns ("gemma-4-31b-it", "antigravity/gemini-2.5-flash", ...). This script
adds pattern-matched entries for each family, idempotently: a model whose
name already exists, or whose served name an existing pattern already
matches, is skipped.

Prices are USD per million tokens (Phoenix's native unit) — public list
rates for the flash/sonnet classes, estimates for hosted open models. The
OPS dashboard's INR figures come from OUR ledger; Phoenix is the drill-down
and cross-check. Override any of it:

    PHOENIX_MODEL_COSTS_JSON='[{"name":"x","name_pattern":"x.*","prompt":0.1,"completion":0.4}]'

Run (server must be up):
    .venv/bin/python -m recovery_agent.scripts.seed_phoenix_model_costs
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

#: name, regex matched against llm.model_name, USD per 1M prompt/completion
#: tokens, and a served-name probe used for the already-covered check.
DEFAULT_COSTS = [
    {"name": "gemini-flash-lite (proxy)",
     "name_pattern": r".*gemini-3.*flash-lite.*",
     "prompt": 0.10, "completion": 0.40, "probe": "gemini-3.1-flash-lite"},
    {"name": "gemini-2.5-flash (proxy)",
     "name_pattern": r".*gemini-2\.5-flash.*",
     "prompt": 0.30, "completion": 2.50, "probe": "antigravity/gemini-2.5-flash"},
    {"name": "gemini-3.6-flash (proxy)",
     "name_pattern": r".*gemini-3\.6-flash.*",
     "prompt": 0.30, "completion": 2.50, "probe": "antigravity/gemini-3.6-flash-high"},
    {"name": "gemma-4 (proxy, est.)",
     "name_pattern": r".*gemma-4-\d+b.*",
     "prompt": 0.03, "completion": 0.09, "probe": "gemma-4-31b-it"},
    {"name": "claude-sonnet-4-6 (proxy)",
     "name_pattern": r".*claude-sonnet-4-6.*",
     "prompt": 3.00, "completion": 15.00,
     "probe": "no-think/antigravity/claude-sonnet-4-6"},
]


def _graphql(base_url: str, query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/graphql", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        out = json.loads(r.read().decode())
    if out.get("errors"):
        raise RuntimeError(json.dumps(out["errors"])[:300])
    return out.get("data") or {}


def existing_models(base_url: str) -> list[dict]:
    data = _graphql(base_url, """
        query { generativeModels(first: 200) {
            edges { node { name namePattern } } } }""")
    return [e["node"] for e in data["generativeModels"]["edges"]]


def already_covered(entry: dict, existing: list[dict]) -> str:
    """Why this entry needn't be created — or "" if it should be."""
    for model in existing:
        if model["name"] == entry["name"]:
            return f"name exists ({model['name']!r})"
        try:
            if re.fullmatch(model["namePattern"], entry["probe"]):
                return (f"served name {entry['probe']!r} already matched by "
                        f"{model['name']!r}")
        except re.error:
            continue
    return ""


def create_model(base_url: str, entry: dict) -> None:
    _graphql(base_url, """
        mutation($input: CreateModelMutationInput!) {
            createModel(input: $input) { model { name } } }""", {
        "input": {
            "name": entry["name"],
            "namePattern": entry["name_pattern"],
            # Phoenix requires exactly these token_type names: "input" and
            # "output" (the kind enum still says PROMPT/COMPLETION).
            "costs": [
                {"tokenType": "input", "kind": "PROMPT",
                 "costPerMillionTokens": float(entry["prompt"])},
                {"tokenType": "output", "kind": "COMPLETION",
                 "costPerMillionTokens": float(entry["completion"])},
            ],
        }})


def seed(base_url: str | None = None) -> dict:
    from recovery_agent.observability import phoenix_base_url
    base = base_url or phoenix_base_url()
    entries = DEFAULT_COSTS
    override = os.getenv("PHOENIX_MODEL_COSTS_JSON", "").strip()
    if override:
        entries = json.loads(override)
        for e in entries:
            e.setdefault("probe", e["name"])
    existing = existing_models(base)
    created, skipped = [], []
    for entry in entries:
        why = already_covered(entry, existing)
        if why:
            skipped.append({"name": entry["name"], "why": why})
            continue
        create_model(base, entry)
        created.append(entry["name"])
    return {"created": created, "skipped": skipped, "existing": len(existing)}


def main() -> int:
    try:
        result = seed()
    except Exception as exc:
        print(f"[phoenix-costs] failed: {exc}", file=sys.stderr)
        return 1
    for name in result["created"]:
        print(f"[phoenix-costs] created: {name}")
    for s in result["skipped"]:
        print(f"[phoenix-costs] skipped: {s['name']} — {s['why']}")
    print(f"[phoenix-costs] price table now covers the proxy fleet "
          f"({result['existing']} entries existed before)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
