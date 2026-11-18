from datetime import UTC, datetime

import numpy as np
import pytest

from amic.contracts import interfaces as I
from amic.contracts.config import Settings, load_settings
from amic.contracts.models import (
    AgentResponse,
    DecisionTrace,
    LatencySplit,
    MemoryDecision,
    MemoryItem,
    MemoryTier,
    RequestContext,
    ResolvedBy,
    RetrievalPlan,
    Route,
    Turn,
)

NOW = datetime(2026, 11, 18, tzinfo=UTC)


def _samples():
    plan = RetrievalPlan(stores=[MemoryTier.M1, MemoryTier.M3], k=12, mmr=True, token_budget=1000)
    decision = MemoryDecision(
        route=Route.DEEP,
        resolved_by=ResolvedBy.TIER1,
        p=0.71,
        tau_low=0.12,
        tau_high=0.35,
        plan=plan,
        token_budget=1000,
        rationale="p > tau_high",
    )
    item = MemoryItem(id="m1", tier=MemoryTier.M1, text="loops", score=0.8, ts=NOW)
    return [
        RequestContext(
            request_id="r1",
            tenant_id="u1",
            session_id="s1",
            query="what did I get wrong?",
            m0_turns=[Turn(role="user", text="hi", ts=NOW)],
            arm="amic",
            q_vec=[0.1, 0.2],
        ),
        plan,
        decision,
        item,
        DecisionTrace(
            request_id="r1",
            ts=NOW,
            tenant_id="u1",
            session_id="s1",
            arm="amic",
            config_hash="abc",
            query="q",
            route=Route.DEEP,
            resolved_by=ResolvedBy.TIER1,
            latency=LatencySplit(signals=0.2, tier0=0.01, tier1=1.5, retrieval=12.0),
            features={"s1_deictic": 1.0},
        ),
        AgentResponse(answer="a", decision=decision, items=[item], trace_id="r1"),
    ]


@pytest.mark.parametrize("obj", _samples(), ids=lambda o: type(o).__name__)
def test_json_round_trip(obj):
    assert type(obj).model_validate_json(obj.model_dump_json()) == obj


def test_latency_gate_excludes_retrieval_and_generation():
    lat = LatencySplit(signals=1, cache=1, tier0=1, tier1=1, tier2=1, retrieval=100, generation=999)
    assert lat.gate == 5


def test_empty_plan():
    assert RetrievalPlan().is_empty
    assert not RetrievalPlan(stores=[MemoryTier.M1], k=3).is_empty


def test_load_yaml_and_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("gate:\n  lam: 9.0\nablation:\n  tiers: t0\n")
    s = load_settings(p, gate={"tau_mode": "fixed"})
    assert s.gate.lam == 9.0 and s.gate.tau_mode == "fixed" and s.ablation.tiers == "t0"


def test_ablation_files_load():
    for name in ["t0_only", "t0_t1", "no_s3", "tau_fixed"]:
        load_settings(f"configs/ablations/{name}.yaml")


def test_config_hash_changes_with_behaviour():
    assert Settings().config_hash() != load_settings(gate={"lam": 2.0}).config_hash()
    assert Settings().config_hash() == Settings().config_hash()


def test_protocols_are_structural():
    class Dummy:
        def affinity(self, q_vec: np.ndarray) -> tuple[float, float]:
            return 0.0, 0.0

    assert isinstance(Dummy(), I.Sketch)
    assert not isinstance(Dummy(), I.AMICController)
