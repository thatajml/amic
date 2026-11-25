"""The Redis-backed parts against a REAL Redis (fakeredis elsewhere): the WriteIntent stream
(XADD / XREADGROUP / XACK / XAUTOCLAIM), session state + memory version, M3 profiles, and the
whole write path with the worker in a separate thread.

Skipped unless AMIC_TEST_REDIS_URL is set (CI runs a redis:7 service container), e.g.
    docker run -d --name amic-test-redis -p 6390:6379 redis:7-alpine
    AMIC_TEST_REDIS_URL=redis://localhost:6390/15 pytest -m redis
Uses a database index of its own and deletes only the keys it creates.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import UTC, datetime

import pytest

from amic.contracts.models import WriteIntent
from amic.memory.stores.profile import ProfileStore, RedisKV
from amic.service.session import SessionManager
from amic.writepath.queue import RedisStreamQueue

URL = os.environ.get("AMIC_TEST_REDIS_URL")
pytestmark = [
    pytest.mark.redis,
    pytest.mark.skipif(not URL, reason="AMIC_TEST_REDIS_URL not set (needs a real Redis)"),
]
T = datetime(2026, 11, 1, 10, 0, tzinfo=UTC)


@pytest.fixture
def r():
    import redis

    client = redis.Redis.from_url(URL)
    client.ping()
    prefix = f"amictest-{uuid.uuid4().hex[:8]}"
    yield client, prefix
    for k in client.scan_iter(match=f"{prefix}*"):
        client.delete(k)


def _intent(i: int, tenant: str = "t") -> WriteIntent:
    return WriteIntent(
        tenant_id=tenant,
        session_id="s",
        record_id=f"r{i}",
        query=f"q {i} about lists",
        answer="a",
        ts=T,
    )


def test_stream_queue_ack_and_reclaim_on_real_redis(r):
    client, prefix = r
    stream = f"{prefix}:writes"
    a = RedisStreamQueue(client, stream=stream, group="g", consumer="a", reclaim_idle_ms=50)
    b = RedisStreamQueue(client, stream=stream, group="g", consumer="b", reclaim_idle_ms=50)
    for i in range(3):
        a.enqueue(_intent(i))
    got = a.read(count=2, block_ms=100)
    assert [m.record_id for _, m in got] == ["r0", "r1"]
    a.ack(got[0][0])  # r1 stays pending: consumer a "crashes"
    time.sleep(0.1)
    reclaimed = b.read(count=1, block_ms=100)
    assert [m.record_id for _, m in reclaimed] == ["r1"]  # XAUTOCLAIM after the idle time
    b.ack(reclaimed[0][0])
    rest = b.read(count=5, block_ms=100)
    assert [m.record_id for _, m in rest] == ["r2"]
    b.ack(rest[0][0])
    assert a.pending() == 0


def test_sessions_version_and_profiles_on_real_redis(r):
    client, prefix = r
    sm = SessionManager(client, m0_turns=6, prefix=prefix)
    v0 = sm.mem_version("ann")
    v1 = sm.bump_mem_version("ann")
    assert v1 > v0 and sm.mem_version("ann") == v1
    sm.touch("ann", "s1", T)
    sm.erase_tenant("ann")
    assert sm.mem_version("ann") > v1 and sm.tenant_meta("ann") == {}  # a new value, never reset
    kv = RedisKV(client)
    kv.PREFIX = f"{prefix}:profile:"
    store = ProfileStore(kv)
    p = store.get_or_new("ann")
    p.observe_turn("my dict lookup keeps giving an error", T)
    store.put(p)
    assert store.get("ann").mastery == p.mastery and store.tenants() == ["ann"]
    assert store.delete("ann") == 1


def test_write_path_with_a_worker_thread_on_real_redis(r, chroma_client):
    """API-style enqueue -> a worker in another thread consolidates -> M1 + M3 updated, memory
    version bumped, queue drained; then /forget through the same stream."""
    client, prefix = r
    from amic.app import build_components
    from amic.contracts.config import load_settings
    from amic.generation.gateway.base import FakeLLM
    from amic.memory.embedder import HashEmbedder
    from amic.telemetry.store import InMemoryTelemetry
    from amic.writepath.worker import build_worker

    s = load_settings(
        chroma_collection=f"t_{uuid.uuid4().hex[:10]}",
        metrics_db_url="sqlite://",
        writepath={"queue": "redis", "stream": f"{prefix}:writes", "group": "workers"},
    )
    c = build_components(
        s,
        embedder=HashEmbedder(),
        llm=FakeLLM(),
        redis=client,
        chroma_client=chroma_client,
        telemetry=InMemoryTelemetry(),
    )
    worker = build_worker(c)
    from amic.app import build_queue

    wq = build_queue(c, for_worker=True)
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            for mid, it in wq.read(count=10, block_ms=100):
                worker.process(it)
                wq.ack(mid)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    try:
        v0 = c.sessions.mem_version("bea")
        c.write_queue.enqueue(
            _intent(1, "bea").model_copy(update={"query": "my list index is out of range error"})
        )
        c.write_queue.enqueue(
            WriteIntent(
                tenant_id="bea",
                session_id="s",
                record_id="r2",
                query="likes cats",
                answer="",
                ts=T,
                op="remember",
            )
        )
        deadline = time.time() + 10
        while c.episodic.count("bea") < 2 and time.time() < deadline:
            time.sleep(0.05)
        assert c.episodic.count("bea") == 2
        assert c.sessions.mem_version("bea") > v0
        assert "lists" in c.profiles.get("bea").mastery
        c.write_queue.enqueue(
            WriteIntent(
                tenant_id="bea",
                session_id="s",
                record_id="r3",
                query="likes cats",
                answer="",
                ts=T,
                op="forget",
            )
        )
        deadline = time.time() + 10
        while c.episodic.count("bea") > 1 and time.time() < deadline:
            time.sleep(0.05)
        assert c.episodic.count("bea") == 1
    finally:
        stop.set()
        th.join(timeout=5)
        c.profiles.delete("bea")
        for k in client.scan_iter(match="amic:tenant:bea*"):
            client.delete(k)
