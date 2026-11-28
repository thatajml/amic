"""Fact extraction in the worker (failure handling) and the M2 back-fill of stored turns."""

from __future__ import annotations

import uuid
from datetime import timedelta

import fakeredis

from amic.app import build_components
from amic.contracts.config import load_settings
from amic.contracts.models import WriteIntent
from amic.generation.gateway.base import FakeLLM, LLMResult
from amic.memory.embedder import HashEmbedder
from amic.telemetry.store import InMemoryTelemetry
from amic.writepath.backfill import backfill_learner, parse_turn
from amic.writepath.extract import ConflictResolver, FactExtractor
from amic.writepath.worker import build_worker
from amic.writepath.writer import record_text
from tests.conftest import T0

FACTS = (
    '{"facts": [{"key": "misconception:range_end", "kind": "misconception", '
    '"text": "Thinks range(n) includes n", "confidence": 0.9}]}'
)


class FactLLM:
    model = "facts-fake"

    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def chat(self, messages, **kw) -> LLMResult:
        self.calls += 1
        if self.fail:
            raise TimeoutError("ollama timed out")
        return LLMResult(text=FACTS, prompt_tokens=10, completion_tokens=5, total_ms=1.0)


def _comps(chroma_client, llm):
    s = load_settings(
        chroma_collection=f"t_{uuid.uuid4().hex[:10]}",
        metrics_db_url="sqlite://",
        writepath={"extract_facts": True},
    )
    return build_components(
        s,
        embedder=HashEmbedder(),
        llm=FakeLLM(),
        gate_llm=llm,
        redis=fakeredis.FakeRedis(server=fakeredis.FakeServer()),
        chroma_client=chroma_client,
        telemetry=InMemoryTelemetry(),
    )


def _intent(i: int, q: str, sid: str = "s1") -> WriteIntent:
    return WriteIntent(
        tenant_id="alice",
        session_id=sid,
        record_id=f"r{i}",
        query=q,
        answer="an answer",
        ts=T0 + timedelta(minutes=i),
    )


def test_a_failing_extractor_does_not_lose_the_turn(chroma_client):
    c = _comps(chroma_client, FactLLM(fail=True))
    w = build_worker(c)
    w.process(_intent(1, "why does range(5) stop at 4 in my loop?"))
    assert c.episodic.count("alice") == 1  # the turn is stored
    assert c.semantic.count("alice") == 0
    assert w.stats.extract_failed == 1 and w.stats.failed == 0 and w.stats.processed == 1


def test_salient_turns_get_facts_and_phatic_ones_do_not(chroma_client):
    llm = FactLLM()
    c = _comps(chroma_client, llm)
    w = build_worker(c)
    w.process(_intent(1, "thanks"))
    assert llm.calls == 0
    w.process(_intent(2, "why does range(5) stop at 4 in my loop?"))
    assert llm.calls == 1 and c.semantic.count("alice") == 1
    assert w.stats.extracted == 1 and w.stats.facts["insert"] == 1


def test_parse_turn_inverts_record_text():
    q, a = "what is a list?\nand a tuple?", "A list is mutable."
    assert parse_turn(record_text(q, a)) == (q, a)
    assert parse_turn("Student asked me to remember: likes teal") is None


def test_backfill_extracts_from_stored_turns_once(chroma_client):
    # a learner who chatted while extraction was off: only raw turns in M1
    off = load_settings(chroma_collection=f"t_{uuid.uuid4().hex[:10]}", metrics_db_url="sqlite://")
    c = build_components(
        off,
        embedder=HashEmbedder(),
        llm=FakeLLM(),
        redis=fakeredis.FakeRedis(server=fakeredis.FakeServer()),
        chroma_client=chroma_client,
        telemetry=InMemoryTelemetry(),
    )
    for i, q in enumerate(
        ["why does range(5) stop at 4 in my loop?", "thanks", "how do I loop over a dict?"]
    ):
        c.writer.write_turn("alice", "s1", f"r{i}", q, "an answer", T0 + timedelta(minutes=i))
    llm = FactLLM()
    ex, res = FactExtractor(llm), ConflictResolver(c.semantic, c.embedder)
    kw = dict(
        episodic=c.episodic,
        semantic=c.semantic,
        extractor=ex,
        resolver=res,
        profiles=c.profiles,
        sessions=c.sessions,
    )
    v0 = c.sessions.mem_version("alice")
    r1 = backfill_learner("alice", **kw)  # type: ignore[arg-type]
    assert r1["insert"] == 1 and r1["sessions"] == 1 and llm.calls == 1  # one call per session
    assert c.semantic.count("alice") == 1
    assert c.sessions.mem_version("alice") > v0
    prof = c.profiles.get("alice")
    assert prof is not None and prof.misconceptions  # the fact reached the M3 profile
    r2 = backfill_learner("alice", **kw)  # type: ignore[arg-type]
    assert r2["sessions"] == 0 and llm.calls == 1  # idempotent: nothing left to do
