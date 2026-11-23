"""Shared fixtures: the real pipeline with fakes at the edges (no models, Ollama or Redis)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import fakeredis
import pytest

from amic.app import Components, build_components
from amic.contracts.config import Settings
from amic.generation.gateway.base import FakeLLM
from amic.memory.embedder import HashEmbedder
from amic.memory.stores.episodic import make_chroma_client
from amic.telemetry.store import InMemoryTelemetry

T0 = datetime(2026, 11, 1, 10, 0, tzinfo=UTC)


@pytest.fixture(scope="session")
def chroma_client():
    return make_chroma_client("ephemeral")


@pytest.fixture
def comps(chroma_client) -> Components:
    # unique collection per test: the in-process Chroma client is shared
    s = Settings(chroma_collection=f"t_{uuid.uuid4().hex[:10]}", metrics_db_url="sqlite://")
    return build_components(
        s,
        embedder=HashEmbedder(),
        llm=FakeLLM(),
        redis=fakeredis.FakeRedis(server=fakeredis.FakeServer()),
        chroma_client=chroma_client,
        telemetry=InMemoryTelemetry(),
    )


@pytest.fixture
def seed_history():
    """Write past turns straight through the real writer, days in the past."""

    def _seed(c: Components, tenant: str, pairs: list[tuple[str, str]], days_ago: int = 3) -> None:
        for i, (q, a) in enumerate(pairs):
            c.writer.write_turn(
                tenant,
                f"{tenant}-old",
                f"{tenant}-h{i}",
                q,
                a,
                T0 - timedelta(days=days_ago, minutes=-i),
            )

    return _seed
