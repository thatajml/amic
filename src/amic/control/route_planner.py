"""RoutePlanner: Route -> RetrievalPlan + token budget. Shared by all arms so routes mean the
same thing everywhere (the `always` arm's DEEP is exactly AMIC's DEEP).

Stores per route come from RouteConfig (default M1 only, as in the first results; with M2:
SHALLOW = top facts, DEEP = wide M1 + MMR plus facts)."""

from __future__ import annotations

from amic.contracts.config import RouteConfig
from amic.contracts.models import MemoryTier, RetrievalPlan, Route


class RoutePlanner:
    def __init__(self, cfg: RouteConfig) -> None:
        self.cfg = cfg

    def plan(self, route: Route) -> RetrievalPlan:
        c = self.cfg
        if route == Route.SHALLOW:
            return RetrievalPlan(
                stores=[MemoryTier(s) for s in c.shallow_stores],
                k=c.shallow_k,
                token_budget=c.shallow_token_budget,
            )
        if route == Route.DEEP:
            return RetrievalPlan(
                stores=[MemoryTier(s) for s in c.deep_stores],
                k=c.deep_k,
                mmr=c.deep_mmr,
                token_budget=c.deep_token_budget,
            )
        return RetrievalPlan()  # SKIP / WRITE: zero vector-store contact
