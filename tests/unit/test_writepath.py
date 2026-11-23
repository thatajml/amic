"""L7 write path: queues, salience, fact parsing, conflict resolution with validity intervals."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import fakeredis
import pytest

from amic.contracts.models import WriteIntent
from amic.memory.embedder import HashEmbedder
from amic.memory.stores.semantic import SemanticStore
from amic.writepath.extract import ConflictResolver, ExtractedFact, is_salient, parse_facts
from amic.writepath.queue import InlineQueue, RedisStreamQueue, SqliteWriteQueue

T = datetime(2026, 11, 1, 10, 0, tzinfo=UTC)


def _intent(i: int = 0, tenant: str = "alice") -> WriteIntent:
    return WriteIntent(
        tenant_id=tenant,
        session_id="s1",
        record_id=f"r{i}",
        query=f"question number {i} about lists",
        answer="an answer",
        ts=T,
    )


# ----------------------------------------------------------------------------- queues
def test_inline_queue_runs_the_handler_immediately():
    seen = []
    q = InlineQueue(seen.append)
    q.enqueue(_intent(1))
    assert [x.record_id for x in seen] == ["r1"] and q.pending() == 0


def test_sqlite_queue_leases_acks_and_redelivers(tmp_path):
    q = SqliteWriteQueue(
        str(tmp_path / "q.sqlite"), lease_s=2.0
    )  # long enough that a slow machine never expires it early
    for i in range(3):
        q.enqueue(_intent(i))
    first = q.read(count=2, block_ms=0)
    assert [m.record_id for _, m in first] == ["r0", "r1"]
    assert [m.record_id for _, m in q.read(count=5, block_ms=0)] == ["r2"]  # r0, r1 leased
    q.ack(first[0][0])
    import time

    time.sleep(2.1)  # lease expired: the unacked r1 comes back (at-least-once)
    again = q.read(count=5, block_ms=0)
    assert {m.record_id for _, m in again} == {"r1", "r2"} and q.pending() == 2


def test_redis_stream_queue_roundtrip_and_reclaim():
    r = fakeredis.FakeRedis()
    stream = f"s-{uuid.uuid4().hex[:6]}"
    a = RedisStreamQueue(r, stream=stream, consumer="a", reclaim_idle_ms=0)
    b = RedisStreamQueue(r, stream=stream, consumer="b", reclaim_idle_ms=0)  # same group
    a.enqueue(_intent(1))
    a.enqueue(_intent(2))
    got = a.read(count=1, block_ms=0)
    assert [m.record_id for _, m in got] == ["r1"]
    # consumer a "crashes" without acking r1: b reclaims it before reading new messages
    reclaimed = b.read(count=1, block_ms=0)
    assert [m.record_id for _, m in reclaimed] == ["r1"]
    b.ack(reclaimed[0][0])
    rest = b.read(count=5, block_ms=0)
    assert [m.record_id for _, m in rest] == ["r2"]


# ----------------------------------------------------------------------------- extraction
@pytest.mark.parametrize(
    ("q", "ok"),
    [("thanks!", False), ("ok", False), ("why", False), ("why does my loop skip items?", True)],
)
def test_salience(q, ok):
    assert is_salient(q, "") is ok


def test_parse_facts_normalises_and_tolerates_junk():
    raw = (
        '{"facts": [{"key": "Misconception: Range End", "kind": "misconception", '
        '"text": "Thinks range(n) includes n.", "confidence": 1.4}, {"text": ""}, '
        '{"key": "x", "kind": "weird", "text": "Likes analogies", "confidence": "high"}]}'
    )
    fs = parse_facts(raw)
    assert [f.key for f in fs] == ["misconception:_range_end", "x"]
    assert fs[0].confidence == 1.0 and fs[1].kind == "other" and fs[1].confidence == 0.7


# ----------------------------------------------------------------------------- resolution
@pytest.fixture
def store(chroma_client):
    return SemanticStore(chroma_client, f"t_{uuid.uuid4().hex[:8]}")


def _fact(key: str, text: str) -> ExtractedFact:
    return ExtractedFact(key, key.split(":")[0], text, 0.9)


def test_resolver_inserts_refreshes_updates_and_skips(store):
    res = ConflictResolver(store, HashEmbedder())
    c = res.resolve("alice", [_fact("misconception:range", "thinks range n includes n")], T)
    assert c["insert"] == 1
    c = res.resolve("alice", [_fact("misconception:range", "thinks range n includes n")], T)
    assert c["refresh"] == 1 and store.count("alice") == 1
    later = T + timedelta(days=3)
    c = res.resolve(
        "alice", [_fact("misconception:range", "now understands range stops early")], later
    )
    assert c["update"] == 1
    current = store.count("alice")
    every = store.all("alice")
    assert current == 1 and len(every) == 2  # the old fact is closed, never deleted
    closed = [m for _, _, m in every if m["valid_to"] != 0.0]
    assert len(closed) == 1 and closed[0]["valid_to"] == later.timestamp()
    c = res.resolve("alice", [_fact("progress:range", "now understands range stops early")], later)
    assert c["skip"] == 1  # same content under another key: a duplicate


def test_retrieval_reads_current_facts_only(store):
    res = ConflictResolver(store, HashEmbedder())
    res.resolve("alice", [_fact("goal", "wants to pass the exam")], T)
    res.resolve("alice", [_fact("goal", "wants to build a budgeting tool")], T + timedelta(days=1))
    hits = store.current("alice", HashEmbedder().encode("goal exam budgeting tool"), 5)
    assert [h.text for h in hits] == ["wants to build a budgeting tool"]


def test_facts_are_isolated_per_learner_and_erasable(store):
    res = ConflictResolver(store, HashEmbedder())
    res.resolve("alice", [_fact("project", "builds a parrot simulator")], T)
    assert store.current("bob", HashEmbedder().encode("parrot simulator"), 5) == []
    assert store.delete_tenant("alice") == 1 and store.count("alice", current_only=False) == 0
