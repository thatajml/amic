"""Baseline arms: always / never / oracle. They share the pipeline; only `decide` differs."""

from __future__ import annotations

from typing import Any

from amic.contracts.models import MemoryDecision, ResolvedBy, Route
from amic.orchestration.graph import PipelineArm, RunState


class _FixedRouteArm(PipelineArm):
    def _fixed(self, route: Route, rationale: str) -> dict[str, Any]:
        plan = self.c.planner.plan(route)
        return {
            "decision": MemoryDecision(
                route=route,
                resolved_by=ResolvedBy.FIXED,
                plan=plan,
                token_budget=plan.token_budget,
                rationale=rationale,
            )
        }


class AlwaysArm(_FixedRouteArm):
    """Conventional baseline: retrieve on every query (route set by `routes.always_route`)."""

    name = "always"

    def decide(self, state: RunState) -> dict[str, Any]:
        route = Route(self.c.settings.routes.always_route)
        return self._fixed(route, f"always-retrieve baseline ({route.value})")


class NeverArm(_FixedRouteArm):
    """Cost floor: M0 only, never embeds, never touches the vector store."""

    name = "never"

    def decide(self, state: RunState) -> dict[str, Any]:
        return self._fixed(Route.SKIP, "never-retrieve baseline")


class OracleArm(_FixedRouteArm):
    """Upper bound: retrieve iff the ground-truth label says memory is needed."""

    name = "oracle"

    def decide(self, state: RunState) -> dict[str, Any]:
        ctx = state["ctx"]
        if ctx.oracle_label is None:
            raise ValueError("oracle arm needs ctx.oracle_label (experiments only)")
        if not ctx.oracle_label:
            return self._fixed(Route.SKIP, "oracle: memory not required")
        route = ctx.oracle_depth if ctx.oracle_depth in (Route.SHALLOW, Route.DEEP) else Route.DEEP
        return self._fixed(route, f"oracle: memory required ({route.value})")
