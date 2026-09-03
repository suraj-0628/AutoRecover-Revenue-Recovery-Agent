"""FastAPI Production Deployment — REST API with auth, rate limiting, caching.

NVIDIA NeMo: "deploy to production with authentication, rate limiting,
caching, and professional web interface."

MANDATE 1: FastAPI (real SDK, installed) for REST API.
MANDATE 1: pydantic (real SDK) for request/response schemas.
MANDATE 2: No stubs — real auth, real rate limiting, real endpoints.
MANDATE 3: API is infrastructure, not agent reasoning.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# APP — FastAPI application
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Recovery Agent API",
    description="Payment recovery agent with governance, observability, and evaluation",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════
# AUTH — API key authentication (NVIDIA NeMo pattern)
# ═══════════════════════════════════════════════════════════════

API_KEYS: dict[str, str] = {}  # key -> role (populated from env)

def _load_api_keys():
    """Load API keys from environment variables."""
    keys_env = os.getenv("RECOVERY_API_KEYS", "")
    if keys_env:
        for pair in keys_env.split(","):
            if "=" in pair:
                key, role = pair.split("=", 1)
                API_KEYS[key.strip()] = role.strip()
    # Default key for development
    if not API_KEYS:
        API_KEYS["dev-key-12345"] = "admin"


def verify_api_key(request: Request) -> str:
    """Verify API key from header. Returns role."""
    _load_api_keys()
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return API_KEYS[api_key]


# ═══════════════════════════════════════════════════════════════
# RATE LIMITING — per-key rate limiting (NVIDIA NeMo pattern)
# ═══════════════════════════════════════════════════════════════

class RateLimiter:
    """Token bucket rate limiter per API key."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        cutoff = now - self.window_seconds
        self.requests[key] = [t for t in self.requests[key] if t > cutoff]
        if len(self.requests[key]) >= self.max_requests:
            return False
        self.requests[key].append(now)
        return True

    def get_remaining(self, key: str) -> int:
        """Get remaining requests in window."""
        now = time.time()
        cutoff = now - self.window_seconds
        recent = [t for t in self.requests[key] if t > cutoff]
        return max(0, self.max_requests - len(recent))


rate_limiter = RateLimiter(max_requests=60, window_seconds=60)


def check_rate_limit(request: Request, role: str = Depends(verify_api_key)):
    """Dependency that checks rate limit."""
    api_key = request.headers.get("X-API-Key", "anonymous")
    if not rate_limiter.check(api_key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": str(rate_limiter.window_seconds)},
        )
    return role


# ═══════════════════════════════════════════════════════════════
# REQUEST/RESPONSE SCHEMAS — pydantic (real SDK)
# ═══════════════════════════════════════════════════════════════

class RecoveryRequest(BaseModel):
    """Request to run recovery on a payment failure."""
    payment_id: str = Field(..., description="Payment identifier")
    customer_id: str = Field(..., description="Customer identifier")
    amount: int = Field(..., gt=0, description="Amount in paise")
    currency: str = "INR"
    failure_code: str = Field(..., description="Failure error code")
    failure_reason: str = Field(..., description="Human-readable failure reason")
    attempt_count: int = 0


class RecoveryResponse(BaseModel):
    """Response from recovery agent."""
    payment_id: str
    status: str
    tools_called: list[str]
    summary: str
    agent_version: str
    eval_score: Optional[float] = None
    duration_ms: int


class EvalResponse(BaseModel):
    """Evaluation result."""
    scenario: str
    overall_score: float
    tool_selection_score: float
    recovery_score: float
    efficiency_score: float
    issues: list[str]
    suggestions: list[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime_seconds: float


# ═══════════════════════════════════════════════════════════════
# CACHING — LLM response caching (NVIDIA NeMo pattern)
# ═══════════════════════════════════════════════════════════════

class ResponseCache:
    """Simple in-memory response cache for identical requests."""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self.cache: dict[str, tuple[float, dict]] = {}

    def _key(self, req: RecoveryRequest) -> str:
        data = f"{req.payment_id}:{req.failure_code}:{req.amount}:{req.attempt_count}"
        return hashlib.md5(data.encode()).hexdigest()

    def get(self, req: RecoveryRequest) -> dict | None:
        key = self._key(req)
        if key in self.cache:
            ts, data = self.cache[key]
            if time.time() - ts < self.ttl:
                return data
            del self.cache[key]
        return None

    def set(self, req: RecoveryRequest, data: dict):
        key = self._key(req)
        self.cache[key] = (time.time(), data)

    def clear(self):
        self.cache.clear()


response_cache = ResponseCache(ttl_seconds=300)


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS — REST API
# ═══════════════════════════════════════════════════════════════

_start_time = time.time()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        uptime_seconds=int(time.time() - _start_time),
    )


@app.post("/v1/recovery/run", response_model=RecoveryResponse)
async def run_recovery(
    request: RecoveryRequest,
    role: str = Depends(check_rate_limit),
):
    """Run recovery agent on a payment failure.

    NVIDIA NeMo: "deploy with authentication, rate limiting, caching."
    """
    # Check cache first
    cached = response_cache.get(request)
    if cached:
        return RecoveryResponse(**cached)

    # Build case and run agent
    from recovery_agent.agent import RecoveryAgent
    from recovery_agent.models import Case, PaymentEvent, RecoveryTier

    case = Case(
        payment=PaymentEvent(
            event_type="payment_failed",
            payment_id=request.payment_id,
            amount=request.amount,
            currency=request.currency,
            failure_code=request.failure_code,
            failure_reason=request.failure_reason,
            customer_id=request.customer_id,
        ),
        attempt_count=request.attempt_count,
    )

    agent = RecoveryAgent()
    start = time.time()
    case = agent.run(case)
    duration_ms = int((time.time() - start) * 1000)

    # Auto-evaluate against gold standard
    # Custom evaluation (deterministic F1) from recovery_eval.py
    from recovery_agent.eval.recovery_eval import auto_evaluate_agent_run
    tools_called = case.payment.metadata.get("tool_calls", [])
    eval_result = auto_evaluate_agent_run(
        payment_id=request.payment_id,
        failure_code=request.failure_code,
        tools_called=tools_called,
        recovered=case.recovered,
        attempt_count=request.attempt_count,
    )

    response_data = {
        "payment_id": request.payment_id,
        "status": case.status.value,
        "tools_called": tools_called,
        "summary": case.payment.metadata.get("agent_summary", ""),
        "agent_version": "2.0.0",
        "eval_score": eval_result.overall_score if eval_result else None,
        "duration_ms": duration_ms,
    }

    # Cache the response
    response_cache.set(request, response_data)

    return RecoveryResponse(**response_data)


@app.get("/v1/evaluation/summary")
async def get_eval_summary(role: str = Depends(check_rate_limit)):
    """Get evaluation summary across all runs."""
    from recovery_agent.eval.recovery_eval import ImprovementTracker
    tracker = ImprovementTracker()
    return tracker.get_summary()


@app.get("/v1/evaluation/regressions")
async def detect_regressions(role: str = Depends(check_rate_limit)):
    """Detect performance regressions."""
    from recovery_agent.eval.recovery_eval import ImprovementTracker
    tracker = ImprovementTracker()
    regressions = tracker.detect_regression()
    return {"regressions": regressions, "count": len(regressions)}


@app.post("/v1/cache/clear")
async def clear_cache(role: str = Depends(verify_api_key)):
    """Clear response cache (admin only)."""
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    response_cache.clear()
    return {"status": "cache cleared"}


@app.get("/v1/config")
async def get_config(role: str = Depends(check_rate_limit)):
    """Get current agent configuration (non-sensitive)."""
    from recovery_agent.config import load_config
    config = load_config()
    return {
        "llm_model": config.llm.model,
        "agent_version": config.governance.agent_version,
        "tiers": {k: v.model_dump() for k, v in config.tiers.items()},
        "observability": config.observability.model_dump(),
    }
