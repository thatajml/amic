"""The proposed arm: signal bus (embeds once) -> cascaded controller -> shared pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from amic.control.controller import Controller
from amic.control.signals.bus import SignalBus
from amic.control.tier1 import Tier1Model
from amic.orchestration.graph import PipelineArm, RunState

if TYPE_CHECKING:
    from amic.app import Components


class AmicArm(PipelineArm):
    name = "amic"

    def __init__(self, comps: Components, tier1: Tier1Model | None = None) -> None:
        super().__init__(comps)
        s = comps.settings
        if tier1 is None and s.ablation.tiers != "t0":
            tier1 = Tier1Model.load(s.gate.tier1_artifact)
        self.bus = SignalBus(
            comps.embedder, comps.sketches, comps.sessions, s.ablation.use_sketch_probe
        )
        self.controller = Controller(s, comps.planner, tier1, self.bus)

    def decide(self, state: RunState) -> dict[str, Any]:
        sig = self.bus.extract(state["ctx"])
        decision, tm = self.controller.decide_signals(sig)
        lat = {
            **state.get("lat", {}),
            "embed": sig.embed_ms,
            "signals": sig.feat_ms,
            "tier0": tm.tier0,
            "tier1": tm.tier1,
        }
        return {"decision": decision, "q_vec": sig.q_vec, "features": sig.features, "lat": lat}
