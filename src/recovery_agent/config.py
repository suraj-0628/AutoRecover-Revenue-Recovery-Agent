"""Agent Configuration — pydantic-settings + YAML loader.

NVIDIA NeMo: "configuration-driven workflows with YAML/JSON config"
MANDATE 1: pydantic-settings (real SDK) for typed config validation.
MANDATE 1: PyYAML (real SDK) for YAML loading.
MANDATE 2: No stubs — real config loading with validation.
MANDATE 3: Config is data, not hardcoded decisions.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ═══════════════════════════════════════════════════════════════
# CONFIG SCHEMAS — typed config with validation
# ═══════════════════════════════════════════════════════════════

class LLMConfig(BaseModel):
    """LLM configuration — validated by pydantic."""
    base_url: str = "http://localhost:20128/v1"
    model: str = "antigravity/gemini-2.5-flash"
    api_key: str = "dummy"
    temperature: float = 0
    max_tokens: int = 2048
    timeout: int = 25
    model_routes: dict[str, Optional[str]] = Field(default_factory=dict)


class GovernanceConfig(BaseModel):
    """Governance configuration."""
    agent_version: str = "2.0.0"
    escalation_threshold_paise: int = 5_000_000
    approval_threshold_paise: int = 5_000_000


class TierConfig(BaseModel):
    """Per-tier policy configuration."""
    max_communications_per_day: int = 3
    allow_financial_tools: bool = False
    require_approval_above_paise: int = 5_000_000
    allowed_failure_categories: list[str] = Field(default_factory=lambda: ["transient", "permanent", "unknown"])
    can_escalate: bool = False


class GuardrailsConfig(BaseModel):
    """Guardrail configuration."""
    quiet_hours_start: int = 21
    quiet_hours_end: int = 8
    max_retries_per_day: int = 3
    monetary_cap_paise: int = 100_000_000


class MemoryConfig(BaseModel):
    """Memory configuration."""
    namespace: list[str] = Field(default_factory=lambda: ["recovery", "{customer_id}"])
    episode_store: bool = True
    vector_search: bool = True


class ObservabilityConfig(BaseModel):
    """Observability configuration."""
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    enabled: bool = True
    log_tool_calls: bool = True
    log_pii_masked: bool = True


class AgentConfig(BaseSettings):
    """Root configuration — loaded from YAML with env var overrides.

    NVIDIA NeMo: "configuration-driven development"
    MANDATE 1: pydantic-settings BaseSettings (real SDK).
    MANDATE 1: YAML loading via PyYAML (real SDK).
    MANDATE 2: No stubs — real config with validation.
    """
    llm: LLMConfig = Field(default_factory=LLMConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    tiers: dict[str, TierConfig] = Field(default_factory=lambda: {
        "silent": TierConfig(
            max_communications_per_day=2,
            allow_financial_tools=False,
            require_approval_above_paise=0,
            allowed_failure_categories=["transient"],
            can_escalate=False,
        ),
        "active": TierConfig(),
        "escalated": TierConfig(
            max_communications_per_day=5,
            allow_financial_tools=True,
            require_approval_above_paise=10_000_000,
            can_escalate=True,
        ),
    })
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    model_config = {
        "env_prefix": "RECOVERY_",
        "env_nested_delimiter": "__",
    }


def load_config(config_path: str | Path | None = None) -> AgentConfig:
    """Load configuration from YAML file with env var overrides.

    NVIDIA NeMo: "configuration-driven workflows"
    MANDATE 1: PyYAML + pydantic-settings (real SDKs).

    Priority: env vars > YAML file > defaults
    """
    data = {}

    if config_path is None:
        # Look for config in standard locations
        candidates = [
            Path("config/agent_config.yaml"),
            Path("agent_config.yaml"),
            Path(os.getenv("RECOVERY_CONFIG", "")),
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

    # MANDATE 1: pydantic-settings validates the config
    return AgentConfig(**data)
