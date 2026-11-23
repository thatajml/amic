"""The five frozen interfaces (brief section 8). Signatures must not change without flagging."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from amic.contracts.models import (
    AgentResponse,
    DecisionTrace,
    MemoryDecision,
    MemoryItem,
    RequestContext,
    RetrievalPlan,
)


@runtime_checkable
class AMICController(Protocol):
    def decide(self, ctx: RequestContext) -> MemoryDecision: ...


@runtime_checkable
class RetrievalEngine(Protocol):
    def retrieve(
        self, plan: RetrievalPlan, q_vec: np.ndarray, user_id: str, query: str | None = None
    ) -> list[MemoryItem]: ...  # `query` (additive, optional): text for the DEEP cross-encoder


@runtime_checkable
class Sketch(Protocol):
    def affinity(self, q_vec: np.ndarray) -> tuple[float, float]:
        """Return (max_sim, margin) against the learner's centroids. Reads no documents."""
        ...


@runtime_checkable
class SketchStore(Protocol):
    def get(self, user_id: str) -> Sketch | None:
        """None means the learner has no long-term memory yet (Tier-0 rule R1)."""
        ...


@runtime_checkable
class TelemetryCollector(Protocol):
    def record(self, trace: DecisionTrace) -> None: ...


@runtime_checkable
class IPipelineArm(Protocol):
    def answer(self, ctx: RequestContext) -> AgentResponse: ...
