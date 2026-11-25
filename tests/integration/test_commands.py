"""Route WRITE end to end: `/remember` and `/forget` (Tier 0 R4) through the pipeline and the
consolidation worker. No LLM call is made for a command; writes go through the queue (P5)."""

from __future__ import annotations

import uuid

import fakeredis

from amic.app import build_components, make_context
from amic.arms.amic import AmicArm
from amic.contracts.config import load_settings
from amic.contracts.models import ResolvedBy, Route
from amic.generation.gateway.base import FakeLLM
from amic.memory.embedder import HashEmbedder
from amic.telemetry.store import InMemoryTelemetry
from amic.writepath.commands import parse_command
from tests.conftest import T0


class StubTier1:
    features: list[str] = []

    def predict(self, feats):
        return 0.9, 0.1


def _comps(chroma_client, **over):
    s = load_settings(
        chroma_collection=f"t_{uuid.uuid4().hex[:10]}", metrics_db_url="sqlite://", **over
    )
    return build_components(
        s,
        embedder=HashEmbedder(),
        llm=FakeLLM(),
        redis=fakeredis.FakeRedis(server=fakeredis.FakeServer()),
        chroma_client=chroma_client,
        telemetry=InMemoryTelemetry(),
    )


def test_parse_command():
    assert parse_command("/remember I prefer short examples").text == "I prefer short examples"
    assert parse_command("  /FORGET: my old project").op == "forget"
    assert parse_command("please remember this") is None
    assert parse_command("/remember").text == ""


def test_remember_then_forget(chroma_client):
    c = _comps(chroma_client)
    arm = AmicArm(c, tier1=StubTier1())  # type: ignore[arg-type]
    calls_before = len(c.llm.calls)
    resp = arm.answer(
        make_context(c, "rae", "s1", "/remember I am building a recipe scaler", "amic", now=T0)
    )
    d = resp.decision
    assert d.route == Route.WRITE and d.resolved_by == ResolvedBy.TIER0
    assert "recipe scaler" in resp.answer and len(c.llm.calls) == calls_before  # no LLM call
    assert c.telemetry.traces[-1].completion_tokens == 0
    # the (inline) worker stored it in M1, M2 and the M3 profile
    assert c.episodic.count("rae") == 1
    assert c.semantic.count("rae") == 1
    prof = c.profiles.get("rae")
    assert "I am building a recipe scaler" in prof.render()
    # a later SHALLOW request with M3 in the route sees the note
    # forget it: every tier
    arm.answer(make_context(c, "rae", "s1", "/forget recipe scaler", "amic", now=T0))
    assert c.episodic.count("rae") == 0 and c.semantic.count("rae", current_only=False) == 0
    assert "recipe scaler" not in c.profiles.get("rae").render()


def test_empty_command_is_answered_but_writes_nothing(chroma_client):
    c = _comps(chroma_client)
    arm = AmicArm(c, tier1=StubTier1())  # type: ignore[arg-type]
    resp = arm.answer(make_context(c, "sam", "s1", "/remember", "amic", now=T0))
    assert resp.decision.route == Route.WRITE and "/remember" in resp.answer
    assert c.episodic.count("sam") == 0


def test_commands_go_through_the_queue(chroma_client, tmp_path):
    c = _comps(chroma_client, writepath={"queue": "sqlite", "queue_db": str(tmp_path / "q.db")})
    arm = AmicArm(c, tier1=StubTier1())  # type: ignore[arg-type]
    arm.answer(make_context(c, "tim", "s1", "/remember my exam is in May", "amic", now=T0))
    assert c.episodic.count("tim") == 0 and c.write_queue.pending() == 1  # nothing written yet
    from amic.writepath.worker import build_worker

    build_worker(c).run(c.write_queue, once=True)
    assert c.episodic.count("tim") == 1 and c.write_queue.pending() == 0


def test_erasure_drops_queued_and_in_flight_writes(chroma_client, tmp_path):
    """Found by scripts/smoke_service.py: a turn still in the queue when the learner was erased
    was consolidated afterwards and resurrected an M1 record. Intents from before the erasure
    are now dropped, and an erasure that overtakes one mid-consolidation purges its writes."""
    from datetime import UTC, datetime

    from amic.contracts.models import WriteIntent
    from amic.writepath.worker import build_worker

    c = _comps(chroma_client, writepath={"queue": "sqlite", "queue_db": str(tmp_path / "q.db")})
    now = datetime.now(UTC)
    it = WriteIntent(
        tenant_id="vic",
        session_id="s",
        record_id="r1",
        query="my dict lookup gives an error",
        answer="use .get()",
        ts=now,
    )
    c.write_queue.enqueue(it)
    c.sessions.erase_tenant("vic")  # erased while the turn waits in the queue
    w = build_worker(c)
    w.run(c.write_queue, once=True)
    assert c.episodic.count("vic") == 0 and c.profiles.get("vic") is None
    assert w.stats.dropped_erased == 1 and c.write_queue.pending() == 0
    # in flight: the erasure lands while the worker is consolidating the intent
    later = it.model_copy(update={"record_id": "r2", "ts": datetime.now(UTC)})
    c.writer.on_written.append(lambda t: c.sessions.erase_tenant(t))
    w.process(later)
    assert c.episodic.count("vic") == 0 and c.profiles.get("vic") is None
    assert w.stats.dropped_erased == 2
