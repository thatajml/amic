from __future__ import annotations

from datetime import UTC, datetime

import pytest

from amic.chatui import (
    CUSTOM,
    OLLAMA,
    PRESETS,
    decision_summary,
    find_trace,
    list_ollama_models,
    make_gateway,
)
from amic.contracts.models import (
    AgentResponse,
    DecisionTrace,
    LatencySplit,
    MemoryDecision,
    MemoryItem,
    MemoryTier,
    ResolvedBy,
    RetrievalPlan,
    Route,
)


def _resp(route: Route = Route.SHALLOW, rid: str = "r1") -> AgentResponse:
    d = MemoryDecision(
        route=route,
        resolved_by=ResolvedBy.TIER1,
        p=0.42,
        tau_low=0.12,
        tau_high=0.35,
        plan=RetrievalPlan(k=3),
        token_budget=400,
        rationale="p above band",
    )
    item = MemoryItem(
        id="m1", tier=MemoryTier.M1, text="loops", score=0.81, ts=datetime(2026, 1, 1, tzinfo=UTC)
    )
    return AgentResponse(answer="hi", decision=d, items=[item], trace_id=rid)


def _trace(rid: str) -> DecisionTrace:
    return DecisionTrace(
        request_id=rid,
        ts=datetime.now(UTC),
        tenant_id="t",
        session_id="s",
        query="q",
        arm="amic",
        config_hash="h",
        route=Route.SHALLOW,
        resolved_by=ResolvedBy.TIER1,
        p=0.42,
        tau_low=0.12,
        tau_high=0.35,
        latency=LatencySplit(embed=5, tier1=1.5, retrieval=2, generation=900, total=910),
        prompt_tokens=300,
        completion_tokens=50,
        memory_tokens=120,
        n_items=1,
        features={"s1_deixis": 1.0, "s3_margin": -0.2, "x_affinity_personal": 0.5, "s2_m0": 0.0},
    )


def test_presets_cover_the_listed_providers():
    assert "Gemini" in PRESETS and CUSTOM in PRESETS
    assert PRESETS["Gemini"].key_env == "GEMINI_API_KEY"


def test_ollama_unreachable_lists_nothing():
    assert list_ollama_models("http://127.0.0.1:9") == []


def test_make_gateway_requires_a_model():
    with pytest.raises(ValueError):
        make_gateway(OLLAMA, "")


def test_make_gateway_openai_compatible_local_server_needs_no_key():
    gw = make_gateway("LM Studio / vLLM (local)", "qwen2.5-7b-instruct")
    assert gw.model == "qwen2.5-7b-instruct"


def test_make_gateway_custom_requires_base_url():
    with pytest.raises(ValueError):
        make_gateway(CUSTOM, "some-model")


def test_reasoning_models_get_room_for_the_answer():
    gw = make_gateway("Gemini", "gemini-3.5-flash-lite", api_key="k", max_tokens=256)
    assert gw._max_tokens >= 1024  # type: ignore[attr-defined]


def test_find_trace_matches_request_id():
    traces = [_trace("a"), _trace("b")]
    assert find_trace(traces, _resp(rid="a")).request_id == "a"  # type: ignore[union-attr]
    assert find_trace(traces, _resp(rid="zzz")) is None  # never another request's trace
    assert find_trace([], _resp()) is None


def test_decision_summary_fields():
    sm = decision_summary(_resp(), _trace("r1"))
    assert sm["route"] == "SHALLOW" and sm["resolved_by"] == "tier1"
    assert sm["band"] == (0.12, 0.35)
    assert sm["memory_tokens"] == 120 and sm["latency_ms"]["total"] == 910
    names = [k for k, _ in sm["signals"]]
    assert names[0] == "s1_deixis"  # strongest first
    assert "x_affinity_personal" not in names and "s2_m0" not in names  # interactions / zeros
    assert sm["items"][0]["text"] == "loops"


def test_decision_summary_without_trace():
    sm = decision_summary(_resp(Route.SKIP), None)
    assert sm["route"] == "SKIP" and "latency_ms" not in sm
