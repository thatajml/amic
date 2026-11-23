from __future__ import annotations

import json
import time

import numpy as np
import pytest

from amic.contracts.config import load_settings
from amic.contracts.models import RequestContext, ResolvedBy, Route
from amic.control.controller import Controller
from amic.control.distill import DistillationBuffer, load_rows
from amic.control.route_planner import RoutePlanner
from amic.control.signals.bus import FEATURES, Signals
from amic.control.tier2 import (
    RateCap,
    ReplayTier2,
    Tier2Adjudicator,
    build_prompt,
    parse_route,
)
from amic.generation.gateway.base import LLMResult
from amic.memory.sketch import build_sketch


class ScriptedLLM:
    model = "scripted"

    def __init__(self, text='{"route": "SKIP"}', delay_s=0.0, fail=None):
        self.text, self.delay_s, self.fail, self.calls = text, delay_s, fail, 0

    def chat(self, messages, **kw):
        self.calls += 1
        self.last_prompt = messages[0]["content"]
        if self.fail is not None:
            raise self.fail
        time.sleep(self.delay_s)
        return LLMResult(text=self.text, prompt_tokens=80, completion_tokens=6, total_ms=1.0)


class StubTier1:
    features: list[str] = []

    def __init__(self, p: float, depth: float = 0.3) -> None:
        self.p, self.depth = p, depth

    def predict(self, feats):
        return self.p, self.depth


SECRET = "my capstone parrot simulator crashed with a KeyError on line 42"


def _sig() -> Signals:
    feats = dict.fromkeys(FEATURES, 0.0)
    feats.update(s3_has_memory=1.0, s2_n_sessions_log=np.log1p(3), s2_days_since_log=np.log1p(5))
    vecs = np.eye(6, 8, dtype=np.float32)
    sketch = build_sketch(vecs, [SECRET] + ["loops and lists"] * 5)
    return Signals(np.zeros(8), feats, sketch, False, 0.0, 0.0)


def _ctx() -> RequestContext:
    return RequestContext(
        request_id="r1", tenant_id="t", session_id="s", query="what should I practise?", arm="amic"
    )


def test_parse_route():
    assert parse_route('{"route": "deep"}') == Route.DEEP
    assert parse_route('[{"route": "SKIP"}]') == Route.SKIP
    with pytest.raises(ValueError):
        parse_route('{"route": "MAYBE"}')


def test_prompt_uses_the_sketch_summary_not_memory_documents():
    prompt = build_prompt(_sig(), _ctx())
    assert SECRET not in prompt  # no document text, only aggregate topics (P3)
    assert "6 notes from 3 past sessions" in prompt and "5 days ago" in prompt
    assert "loops" in prompt and "what should I practise?" in prompt


def test_rate_cap_allows_the_budget_then_refuses():
    cap = RateCap(max_rate=0.1, window=50)  # 5 calls per 50 decisions
    allowed = 0
    for _ in range(50):
        if cap.allow():
            cap.observe(True)
            allowed += 1
        else:
            cap.observe(False)
    assert allowed == 5
    for _ in range(50):  # the window slides: old fires age out
        cap.observe(False)
    assert cap.allow()


def test_ok_verdict_is_buffered(tmp_path):
    buf = DistillationBuffer(tmp_path / "d.jsonl")
    adj = Tier2Adjudicator(ScriptedLLM('{"route": "SHALLOW"}'), buffer=buf, max_rate=1.0)
    res = adj.adjudicate(_sig(), _ctx(), p=0.2)
    assert res.route == Route.SHALLOW and res.status == "ok"
    rows = load_rows(tmp_path / "d.jsonl")
    assert len(rows) == 1 and rows[0]["route"] == "SHALLOW" and rows[0]["p"] == 0.2


@pytest.mark.parametrize(
    ("llm", "status"),
    [
        (ScriptedLLM("not json"), "unparseable"),
        (ScriptedLLM(fail=TimeoutError("timed out")), "timeout"),
        (ScriptedLLM(fail=ConnectionError("refused")), "error"),
        (ScriptedLLM('{"route": "SKIP"}', delay_s=0.05), "timeout"),  # answered too late
    ],
)
def test_failures_return_no_route(llm, status, tmp_path):
    buf = DistillationBuffer(tmp_path / "d.jsonl")
    adj = Tier2Adjudicator(llm, timeout_ms=20, buffer=buf, max_rate=1.0)
    res = adj.adjudicate(_sig(), _ctx(), p=0.2)
    assert res.route is None and res.status == status
    assert load_rows(tmp_path / "d.jsonl") == []  # failures are not training data


def _controller(p: float, tier2, tiers: str = "full") -> Controller:
    s = load_settings(ablation={"tiers": tiers})
    return Controller(s, RoutePlanner(s.routes), StubTier1(p), tier2=tier2)  # type: ignore[arg-type]


def test_tier2_fires_only_inside_the_band():
    llm = ScriptedLLM('{"route": "SKIP"}')
    adj = Tier2Adjudicator(llm, max_rate=1.0)
    d, tm = _controller(0.2, adj).decide_signals(_sig(), _ctx())  # band = [0.12, 0.35]
    assert d.route == Route.SKIP and d.resolved_by == ResolvedBy.TIER2 and llm.calls == 1
    assert tm.tier2 > 0 and tm.tier2_status == "ok"
    d, _ = _controller(0.05, adj).decide_signals(_sig(), _ctx())
    assert d.resolved_by == ResolvedBy.TIER1 and llm.calls == 1
    d, _ = _controller(0.9, adj).decide_signals(_sig(), _ctx())
    assert d.resolved_by == ResolvedBy.TIER1 and llm.calls == 1


def test_tier2_failure_fails_open_never_skip():
    adj = Tier2Adjudicator(ScriptedLLM(fail=TimeoutError("timed out")), max_rate=1.0)
    d, _ = _controller(0.2, adj).decide_signals(_sig(), _ctx())
    assert d.route == Route.SHALLOW and d.resolved_by == ResolvedBy.FAILOPEN
    assert "timeout" in d.rationale


def test_tier2_off_unless_tiers_full():
    llm = ScriptedLLM()
    d, _ = _controller(0.2, Tier2Adjudicator(llm), tiers="t0_t1").decide_signals(_sig(), _ctx())
    assert d.resolved_by == ResolvedBy.FAILOPEN and llm.calls == 0


def test_cap_counts_all_traffic_in_the_controller():
    adj = Tier2Adjudicator(ScriptedLLM('{"route": "DEEP"}'), max_rate=0.1)  # 20 per 200
    in_band, out_band = _controller(0.2, adj), _controller(0.9, adj)
    fired = 0
    for i in range(200):
        ctl = in_band if i % 2 == 0 else out_band  # half the traffic is in the band
        d, _ = ctl.decide_signals(_sig(), _ctx())
        fired += d.resolved_by == ResolvedBy.TIER2
    assert fired == 20 and adj.capped == 80


def test_replay_applies_the_timeout_to_recorded_latency(tmp_path):
    path = tmp_path / "v.jsonl"
    rows = [
        {
            "probe_id": "fast",
            "model": "m",
            "prompt_version": 1,
            "route": "SKIP",
            "status": "ok",
            "ms": 300,
        },
        {
            "probe_id": "slow",
            "model": "m",
            "prompt_version": 1,
            "route": "SKIP",
            "status": "ok",
            "ms": 900,
        },
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    rep = ReplayTier2.from_jsonl(str(path), "m", max_rate=1.0)
    ctx = _ctx()
    assert (
        rep.adjudicate(_sig(), ctx.model_copy(update={"request_id": "fast"}), 0.2).route
        == Route.SKIP
    )
    slow = rep.adjudicate(_sig(), ctx.model_copy(update={"request_id": "slow"}), 0.2)
    assert slow.route is None and slow.status == "timeout"
    assert rep.adjudicate(_sig(), ctx, 0.2).status == "replay-miss"
