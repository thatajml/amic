"""Shared data types for the frozen AMIC interfaces (brief section 8, plan section 3).

Every layer imports these. Changing a field here is a contract change: flag it.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

ArmName = Literal["always", "never", "amic", "oracle"]


class Route(StrEnum):
    SKIP = "SKIP"
    SHALLOW = "SHALLOW"
    DEEP = "DEEP"
    WRITE = "WRITE"


class ResolvedBy(StrEnum):
    CACHE = "cache"
    TIER0 = "tier0"
    TIER1 = "tier1"
    TIER2 = "tier2"
    FAILOPEN = "failopen"  # band / timeout fallback, never SKIP
    FIXED = "fixed"  # always / never / oracle arms


class MemoryTier(StrEnum):
    M1 = "M1"
    M2 = "M2"
    M3 = "M3"


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    ts: datetime


class Budget(BaseModel):
    max_prompt_tokens: int = 4096
    max_memory_tokens: int = 1024
    gate_timeout_ms: float = 50.0


class RequestContext(BaseModel):
    request_id: str
    tenant_id: str  # = learner id (P8)
    session_id: str
    query: str
    m0_turns: list[Turn] = Field(default_factory=list)
    arm: ArmName
    budget: Budget = Field(default_factory=Budget)
    q_vec: list[float] | None = None  # filled ONCE by the signal bus (P2)
    oracle_label: bool | None = None  # experiments only, read by the oracle arm
    oracle_depth: Route | None = None  # experiments only: SHALLOW or DEEP


class RetrievalPlan(BaseModel):
    stores: list[MemoryTier] = Field(default_factory=list)
    k: int = 0
    filters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    mmr: bool = False
    mmr_lambda: float = 0.5
    rerank: bool = False
    hops: int = 0
    token_budget: int = 0

    @property
    def is_empty(self) -> bool:
        return self.k == 0 or not self.stores


class MemoryDecision(BaseModel):
    route: Route
    resolved_by: ResolvedBy
    p: float | None = None  # calibrated P(memory needed)
    depth_p: float | None = None  # P(DEEP | retrieve)
    tau_low: float | None = None
    tau_high: float | None = None
    plan: RetrievalPlan = Field(default_factory=RetrievalPlan)
    token_budget: int = 0
    rationale: str = ""


class MemoryItem(BaseModel):
    id: str
    tier: MemoryTier
    text: str
    score: float
    ts: datetime
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class LatencySplit(BaseModel):
    """Milliseconds. Every term of the break-even inequality (brief 9.3) maps to a field."""

    embed: float = 0.0
    signals: float = 0.0  # feature extraction excluding the embedding (C_feat)
    cache: float = 0.0
    tier0: float = 0.0
    tier1: float = 0.0
    tier2: float = 0.0
    retrieval: float = 0.0
    generation: float = 0.0
    verify: float = 0.0
    total: float = 0.0

    @property
    def gate(self) -> float:
        return self.signals + self.cache + self.tier0 + self.tier1 + self.tier2


class DecisionTrace(BaseModel):
    request_id: str
    ts: datetime
    tenant_id: str
    session_id: str
    arm: str
    config_hash: str
    query: str
    route: Route
    resolved_by: ResolvedBy
    p: float | None = None
    tau_low: float | None = None
    tau_high: float | None = None
    latency: LatencySplit = Field(default_factory=LatencySplit)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    memory_tokens: int = 0
    n_items: int = 0
    escalated: bool = False
    cache_hit: bool = False
    features: dict[str, float] = Field(default_factory=dict)
    oracle_label: bool | None = None


class AgentResponse(BaseModel):
    answer: str
    decision: MemoryDecision
    items: list[MemoryItem] = Field(default_factory=list)
    trace_id: str
