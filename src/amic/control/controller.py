"""AMICController (L4): Tier 0 -> (decision cache) -> Tier 1 -> Tier 2 -> RoutePlanner.

`decide_signals` works from a Signals object, so the SAME code path serves the live request
(features computed by the bus) and the offline replay (features loaded from parquet).

Tier 2 (ablation.tiers = "full") adjudicates Tier 1's uncertain band. Without it, or when it
fails open (timeout, error, rate cap), an abstention goes to `abstain_fallback` (SHALLOW by
default), never to SKIP (brief 6.2). Every decision is reported to Tier 2's rate cap, which
bounds Tier 2 to `gate.tier2_max_rate` of ALL traffic.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from amic.contracts.config import Settings
from amic.contracts.models import MemoryDecision, RequestContext, ResolvedBy, Route
from amic.control.cache import DecisionCache, Key
from amic.control.route_planner import RoutePlanner
from amic.control.signals.bus import S3_FEATURES, SignalBus, Signals
from amic.control.thresholds import ThresholdPolicy
from amic.control.tier0 import Tier0
from amic.control.tier1 import Tier1Model
from amic.control.tier2 import Tier2Result, _Tier2Base


@dataclass
class TierTimings:
    tier0: float = 0.0
    cache: float = 0.0
    tier1: float = 0.0
    tier2: float = 0.0
    tier2_status: str = ""  # "", or the Tier-2 result status when it was consulted


class Controller:
    def __init__(
        self,
        settings: Settings,
        planner: RoutePlanner,
        tier1: Tier1Model | None,
        bus: SignalBus | None = None,
        tier2: _Tier2Base | None = None,
        cache: DecisionCache | None = None,
        version_fn: Callable[[str], int] | None = None,
    ) -> None:
        self.s = settings
        self.planner = planner
        self.tier0 = Tier0()
        self.tier1 = tier1
        self.bus = bus
        self.tier2 = tier2
        self.cache = cache
        self.version_fn = version_fn  # the learner's memory version: part of the cache key
        self.policy = ThresholdPolicy.from_config(settings.gate)
        if settings.ablation.tiers != "t0" and tier1 is None:
            raise ValueError("tiers != t0 needs a trained Tier-1 model")
        if tier1 is not None and not settings.ablation.use_sketch_probe:
            leaked = [f for f in tier1.features if f in S3_FEATURES and f != "s3_has_memory"]
            if leaked:
                raise ValueError(f"S3 ablation but the Tier-1 model uses {leaked}")

    # AMICController (frozen interface)
    def decide(self, ctx: RequestContext) -> MemoryDecision:
        if self.bus is None:
            raise RuntimeError("Controller.decide needs a SignalBus")
        return self.decide_signals(self.bus.extract(ctx), ctx)[0]

    def _decision(
        self, route: Route, by: ResolvedBy, rationale: str, **kw: float | None
    ) -> MemoryDecision:
        plan = self.planner.plan(route)
        return MemoryDecision(
            route=route,
            resolved_by=by,
            plan=plan,
            token_budget=plan.token_budget,
            rationale=rationale,
            tau_low=self.policy.tau_low,
            tau_high=self.policy.tau_high,
            **kw,  # type: ignore[arg-type]
        )

    def _remember(self, key: Key | None, d: MemoryDecision) -> MemoryDecision:
        if key is not None and self.cache is not None:
            self.cache.put(key, d)
        return d

    def _observe(self) -> None:
        """A decision that did not consult Tier 2 still counts as traffic for its rate cap."""
        if self.tier2 is not None:
            self.tier2.observe(False)

    def decide_signals(
        self, sig: Signals, ctx: RequestContext | None = None
    ) -> tuple[MemoryDecision, TierTimings]:
        tm = TierTimings()
        t = time.perf_counter()
        r0 = self.tier0.decide(sig)
        tm.tier0 = (time.perf_counter() - t) * 1000
        if r0.route is not None:
            self._observe()
            return self._decision(r0.route, ResolvedBy.TIER0, r0.rule), tm

        fallback = Route(self.s.gate.abstain_fallback)
        if self.s.ablation.tiers == "t0" or self.tier1 is None:
            self._observe()
            return self._decision(fallback, ResolvedBy.FAILOPEN, "tier0 abstained; no tier1"), tm

        key: Key | None = None
        if self.cache is not None and ctx is not None and sig.q_vec.size > 1:
            t = time.perf_counter()
            ver = self.version_fn(ctx.tenant_id) if self.version_fn is not None else 0
            m0 = sig.features.get("s2_m0_turns", 0.0) > 0
            key = self.cache.key(ctx.tenant_id, ver, sig.q_vec, m0)
            hit = self.cache.get(key)
            tm.cache = (time.perf_counter() - t) * 1000
            if hit is not None:
                self._observe()
                why = f"cache hit ({hit.resolved_by.value}): {hit.rationale}"
                return hit.model_copy(
                    update={"resolved_by": ResolvedBy.CACHE, "rationale": why}
                ), tm

        t = time.perf_counter()
        p, depth_p = self.tier1.predict(sig.features)
        # band on: in-band p goes to Tier 2, or (Tier 2 off) fails open to the fallback route.
        # band off (gate.use_band=False): hard decision at tau (ablation).
        band = self.s.ablation.tiers == "full" or self.s.gate.use_band
        v = self.policy.verdict(p, band=band)
        tm.tier1 = (time.perf_counter() - t) * 1000
        kw = {"p": p, "depth_p": depth_p}
        if v == "skip":
            self._observe()
            d = self._decision(Route.SKIP, ResolvedBy.TIER1, f"p={p:.3f} < tau", **kw)
            return self._remember(key, d), tm
        if v == "retrieve":
            self._observe()
            route = Route.DEEP if depth_p >= 0.5 else Route.SHALLOW
            d = self._decision(route, ResolvedBy.TIER1, f"p={p:.3f} > tau", **kw)
            return self._remember(key, d), tm
        # abstain: Tier 2 adjudicates (tiers = full); otherwise fail open, never SKIP
        if self.s.ablation.tiers == "full" and self.tier2 is not None:
            res: Tier2Result = self.tier2.adjudicate(sig, ctx, p)
            tm.tier2, tm.tier2_status = res.ms, res.status
            if res.route is not None:
                why = f"p={p:.3f} in band; tier2 -> {res.route.value}"
                return self._remember(
                    key, self._decision(res.route, ResolvedBy.TIER2, why, **kw)
                ), tm
            why = f"p={p:.3f} in band; tier2 {res.status}"
            return self._decision(fallback, ResolvedBy.FAILOPEN, why, **kw), tm
        self._observe()
        d = self._decision(fallback, ResolvedBy.FAILOPEN, f"p={p:.3f} in band", **kw)
        return self._remember(key, d), tm
