"""The proposed arm: signal bus (embeds once) -> cascaded controller -> shared pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from amic.control.controller import Controller
from amic.control.signals.bus import SignalBus
from amic.control.tier1 import Tier1Model
from amic.control.tier2 import Tier2Adjudicator, _Tier2Base
from amic.orchestration.graph import PipelineArm, RunState

if TYPE_CHECKING:
    from amic.app import Components


def build_tier2(comps: Components) -> Tier2Adjudicator:
    """Live Tier 2 from settings: the gate model with the HTTP timeout = tier2_timeout_ms."""
    from amic.app import gate_llm_from_settings
    from amic.control.distill import DistillationBuffer

    g = comps.settings.gate
    llm = comps.extras.get("tier2_llm") or gate_llm_from_settings(
        comps.settings, timeout_s=g.tier2_timeout_ms / 1000
    )
    return Tier2Adjudicator(
        llm,
        timeout_ms=g.tier2_timeout_ms,
        max_rate=g.tier2_max_rate,
        buffer=DistillationBuffer(g.distill_path) if g.distill_path else None,
    )


class AmicArm(PipelineArm):
    name = "amic"

    def __init__(
        self,
        comps: Components,
        tier1: Tier1Model | None = None,
        tier2: _Tier2Base | None = None,
    ) -> None:
        super().__init__(comps)
        s = comps.settings
        if tier1 is None and s.ablation.tiers != "t0":
            tier1 = Tier1Model.load(s.gate.tier1_artifact)
        if tier2 is None and s.ablation.tiers == "full":
            tier2 = build_tier2(comps)
        self.tier2 = tier2
        self.bus = SignalBus(
            comps.embedder, comps.sketches, comps.sessions, s.ablation.use_sketch_probe
        )
        self.controller = Controller(
            s,
            comps.planner,
            tier1,
            self.bus,
            tier2,
            cache=comps.decision_cache,
            version_fn=comps.sessions.mem_version,
        )

    def decide(self, state: RunState) -> dict[str, Any]:
        sig = self.bus.extract(state["ctx"])
        decision, tm = self.controller.decide_signals(sig, state["ctx"])
        lat = {
            **state.get("lat", {}),
            "embed": sig.embed_ms,
            "signals": sig.feat_ms,
            "tier0": tm.tier0,
            "cache": tm.cache,
            "tier1": tm.tier1,
            "tier2": tm.tier2,
        }
        return {"decision": decision, "q_vec": sig.q_vec, "features": sig.features, "lat": lat}
