"""Amount bands, read from the merchant's own dunning policy.

A batch shares a *cause*. Within one cause, what the merchant has committed to
doing still varies by size: `merchant_dunning_rules.md` prescribes a different
retry window, channel set, incentive and escalation trigger for a INR 400 order
than for a INR 60,000 one. So the planning unit is (cause x band), not cause
alone — a single plan across a batch spanning both would apply one band's policy
to the other band's customers.

The bands are parsed from the knowledge base rather than hardcoded here, for the
same reason `offers.py` parses its caps from it: the file is the policy. A
hardcoded copy is a second source of truth that drifts the first time someone
edits the markdown and nothing fails.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Used only when the knowledge base cannot be read. Deliberately the same
#: numbers, so a parse failure degrades quietly rather than changing policy.
_FALLBACK_BANDS: tuple[tuple[str, float, float], ...] = (
    ("micro", 0.0, 500.0),
    ("small", 500.0, 5_000.0),
    ("medium", 5_000.0, 50_000.0),
    ("large", 50_000.0, 500_000.0),
    ("high_value", 500_000.0, float("inf")),
)

_SLUGS = {
    "micro-transactions": "micro",
    "small transactions": "small",
    "medium transactions": "medium",
    "large transactions": "large",
    "high-value transactions": "high_value",
}

_HEADING = re.compile(
    r"^###\s+(?P<name>[\w\s-]+?)\s*\((?P<range>[^)]*)\)\s*$", re.M)


@dataclass(frozen=True)
class Tier:
    """One band of the merchant's dunning policy."""
    key: str
    title: str
    lower_rupees: float          # inclusive
    upper_rupees: float          # exclusive
    retry_window: str = ""
    communication: str = ""
    incentive: str = ""
    escalation: str = ""

    def contains(self, rupees: float) -> bool:
        return self.lower_rupees <= float(rupees or 0) < self.upper_rupees

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "title": self.title,
            "lower_rupees": self.lower_rupees,
            "upper_rupees": (None if self.upper_rupees == float("inf")
                             else self.upper_rupees),
            "retry_window": self.retry_window,
            "communication": self.communication,
            "incentive": self.incentive,
            "escalation": self.escalation,
        }


def _kb_path() -> Path:
    return (Path(os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge_base"))
            / "merchant_dunning_rules.md")


def _money(text: str) -> float:
    """`₹5,00,000` -> 500000.0. Indian grouping, so a plain int() will not do."""
    digits = re.sub(r"[^\d.]", "", text or "")
    return float(digits) if digits else 0.0


def _bullet(block: str, label: str) -> str:
    m = re.search(rf"^-\s+\*\*{label}\*\*:\s*(.+)$", block, re.M)
    return m.group(1).strip() if m else ""


_cache: list[Tier] | None = None


def load_tiers(refresh: bool = False) -> list[Tier]:
    """The bands, in ascending order."""
    global _cache
    if _cache is not None and not refresh:
        return _cache

    tiers: list[Tier] = []
    try:
        text = _kb_path().read_text()
        matches = list(_HEADING.finditer(text))
        for i, m in enumerate(matches):
            slug = _SLUGS.get(m.group("name").strip().lower())
            if slug is None:
                continue                     # a heading from another section
            body = text[m.end(): matches[i + 1].start() if i + 1 < len(matches)
                        else len(text)]
            rng = m.group("range")
            numbers = [_money(p) for p in re.findall(r"₹\s*[\d,]+", rng)]
            if ">" in rng and numbers:
                lower, upper = numbers[0], float("inf")
            elif len(numbers) >= 2:
                lower, upper = numbers[0], numbers[1]
            elif numbers:
                lower, upper = 0.0, numbers[0]
            else:
                continue
            # The headings overlap at the boundaries (₹1-₹500 then ₹500-₹5,000).
            # Treat every band as [lower, upper) so ₹500 lands in exactly one.
            tiers.append(Tier(
                key=slug, title=m.group("name").strip(),
                lower_rupees=(0.0 if slug == "micro" else lower),
                upper_rupees=upper,
                retry_window=_bullet(body, "Retry window"),
                communication=_bullet(body, "Communication"),
                incentive=_bullet(body, "Incentive"),
                escalation=_bullet(body, "Escalation"),
            ))
    except Exception:
        tiers = []

    if len(tiers) != len(_FALLBACK_BANDS):
        tiers = [Tier(key=k, title=k.replace("_", " ").title(),
                      lower_rupees=lo, upper_rupees=hi)
                 for k, lo, hi in _FALLBACK_BANDS]

    tiers.sort(key=lambda t: t.lower_rupees)
    _cache = tiers
    return tiers


def amount_tier(rupees: Any) -> Tier:
    """Which band this amount falls in. Never raises; never returns None."""
    try:
        value = float(rupees or 0)
    except (TypeError, ValueError):
        value = 0.0
    tiers = load_tiers()
    for tier in tiers:
        if tier.contains(value):
            return tier
    return tiers[-1]                 # above every band: the top one


def group_by_tier(records: list[dict]) -> dict[str, list[dict]]:
    """Cases grouped by band, ascending. The planning unit."""
    out: dict[str, list[dict]] = {}
    for rec in records:
        out.setdefault(amount_tier(rec.get("amount")).key, []).append(rec)
    order = [t.key for t in load_tiers()]
    return {k: out[k] for k in order if k in out}
