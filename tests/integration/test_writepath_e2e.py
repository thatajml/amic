"""Async write path, M2 facts and the decision cache, end to end with fakes at the edges."""

from __future__ import annotations

import uuid

import fakeredis
from fastapi.testclient import TestClient

from amic.app import build_arm, build_components, make_context
from amic.arms.amic import AmicArm
from amic.contracts.config import load_settings
from amic.contracts.models import MemoryTier, ResolvedBy, Route
from amic.generation.gateway.base import FakeLLM, LLMResult
from amic.memory.embedder import HashEmbedder
from amic.service.api import create_app
from amic.telemetry.store import InMemoryTelemetry
from amic.writepath.worker import build_worker
from tests.conftest import T0

FACTS = (
    '{"facts": [{"key": "misconception:range_end", "kind": "misconception", '
    '"text": "Thinks range(n) includes n", "confidence": 0.9}]}'
)


class FactLLM:
    model = "facts-fake"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, **kw) -> LLMResult:
        self.calls += 1
        return LLMResult(text=FACTS, prompt_tokens=100, completion_tokens=30, total_ms=1.0)


def _comps(chroma_client, tmp_path, **over):
    s = load_settings(
        chroma_collection=f"t_{uuid.uuid4().hex[:10]}",
        metrics_db_url="sqlite://",
        **over,
    )
    c = build_components(
        s,
        embedder=HashEmbedder(),
        llm=FakeLLM(),
        gate_llm=FactLLM(),  # extraction never calls a real model in tests
        redis=fakeredis.FakeRedis(),
        chroma_client=chroma_client,
        telemetry=InMemoryTelemetry(),
    )
    return c


def test_queued_write_is_consolidated_by_the_worker(chroma_client, tmp_path):
    c = _comps(
        chroma_client,
        tmp_path,
        writepath={
            "queue": "sqlite",
            "queue_db": str(tmp_path / "q.sqlite"),
            "extract_facts": True,
        },
    )
    arm = build_arm("never", c)
    ctx = make_context(c, "alice", "s1", "why does range(5) stop at 4 in my loop?", "never", now=T0)
    resp, job = arm.answer_deferred(ctx)
    job()  # the response path: enqueue only
    assert c.write_queue.pending() == 1
    assert c.episodic.count("alice") == 0  # no read-your-writes on long-term stores...
    assert c.sessions.get_turns("s1")[-1].text == resp.answer  # ...but M0 has the turn now
    v0 = c.sessions.mem_version("alice")
    worker = build_worker(c)
    stats = worker.run(c.write_queue, once=True)
    assert stats.processed == 1 and stats.failed == 0 and c.write_queue.pending() == 0
    assert c.episodic.count("alice") == 1  # M1
    assert c.semantic.count("alice") == 1  # M2: the extracted fact
    assert c.sessions.mem_version("alice") > v0  # sketch + cache invalidated everywhere
    assert c.sketches.get("alice") is not None


def test_inline_queue_keeps_the_old_synchronous_behaviour(chroma_client, tmp_path):
    c = _comps(chroma_client, tmp_path)  # default: inline, no extraction
    arm = build_arm("never", c)
    arm.answer(make_context(c, "bob", "s1", "what is a dict?", "never", now=T0))
    assert c.episodic.count("bob") == 1 and c.semantic.count("bob") == 0


def test_m2_routes_inject_current_facts(chroma_client, tmp_path):
    c = _comps(
        chroma_client,
        tmp_path,
        writepath={"extract_facts": True},
        routes={"shallow_stores": ["M2"], "always_route": "SHALLOW"},
    )
    c.write_queue.enqueue(  # consolidate one turn inline -> one fact
        __import__("amic.contracts.models", fromlist=["WriteIntent"]).WriteIntent(
            tenant_id="alice",
            session_id="old",
            record_id="r1",
            query="my loop over range(5) crashed again with an IndexError",
            answer="range(5) stops at 4",
            ts=T0,
        )
    )
    r = build_arm("always", c).answer(
        make_context(c, "alice", "s2", "what was my range mistake?", "always", now=T0)
    )
    assert r.items and all(i.tier == MemoryTier.M2 for i in r.items)
    assert "range(n) includes n" in r.items[0].text and "memory=yes" in r.answer


class StubTier1:
    features: list[str] = []

    def predict(self, feats):
        return 0.9, 0.1  # always retrieve, SHALLOW


def test_decision_cache_hits_then_invalidates_on_write(chroma_client, tmp_path):
    c = _comps(chroma_client, tmp_path, ablation={"use_cache": True})
    c.writer.write_turn("alice", "old", "h1", "I keep mixing up append and extend", "...", T0)
    arm = AmicArm(c, tier1=StubTier1())  # type: ignore[arg-type]
    q = "how do I add two lists together?"
    first = arm.answer_deferred(make_context(c, "alice", "n1", q, "amic", now=T0))[0]
    second = arm.answer_deferred(make_context(c, "alice", "n2", q, "amic", now=T0))[0]
    assert first.decision.resolved_by == ResolvedBy.TIER1
    assert (
        second.decision.resolved_by == ResolvedBy.CACHE and second.decision.route == Route.SHALLOW
    )
    assert c.telemetry.traces[-1].cache_hit and c.telemetry.traces[-1].latency.cache > 0
    c.writer.write_turn("alice", "old", "h2", "lists again", "...", T0)  # memory changed
    third = arm.answer_deferred(make_context(c, "alice", "n3", q, "amic", now=T0))[0]
    assert third.decision.resolved_by == ResolvedBy.TIER1


def test_erasure_clears_m1_m2_and_the_sketch(chroma_client, tmp_path):
    c = _comps(chroma_client, tmp_path, writepath={"extract_facts": True})
    build_arm("never", c).answer(
        make_context(c, "carol", "s1", "my parrot simulator crashes on line 42", "never", now=T0)
    )
    assert c.episodic.count("carol") == 1 and c.semantic.count("carol") == 1
    with TestClient(create_app(c)) as client:
        out = client.delete("/learners/carol").json()
    assert out == {"deleted_records": 1, "deleted_facts": 1}
    assert c.sketches.get("carol") is None  # R1 (no memory) fires again
