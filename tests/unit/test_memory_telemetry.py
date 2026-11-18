from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from amic.contracts.models import (
    DecisionTrace,
    LatencySplit,
    MemoryItem,
    MemoryTier,
    ResolvedBy,
    Route,
)
from amic.memory.assembler import ContextAssembler, estimate_tokens
from amic.memory.embedder import HashEmbedder
from amic.memory.retrieval import mmr
from amic.telemetry.store import SqliteTelemetry

NOW = datetime(2026, 11, 18, tzinfo=UTC)


def test_hash_embedder_similarity():
    e = HashEmbedder()
    a, b, c = e.encode("python list append"), e.encode("list append python"), e.encode("zebra")
    assert np.isclose(np.linalg.norm(a), 1.0)
    assert a @ b > 0.99 and abs(a @ c) < 0.5


def test_mmr_prefers_diversity():
    q = np.array([1.0, 0.0])
    cand = np.array([[1.0, 0.0], [0.999, 0.045], [0.6, 0.8]])
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    assert mmr(q, cand, 2, lam=0.3) == [0, 2]  # 0.5 ties when cand[0] == q
    assert mmr(q, cand, 2, lam=1.0) == [0, 1]
    assert mmr(q, cand[:0], 2, 0.5) == []


def test_assembler_respects_budget_and_orders_chronologically():
    items = [
        MemoryItem(id=str(i), tier=MemoryTier.M1, text="x" * 40, score=s, ts=NOW.replace(day=d))
        for i, (s, d) in enumerate([(0.9, 10), (0.5, 2), (0.7, 5)])
    ]
    per = estimate_tokens(f"- [{NOW:%Y-%m-%d}] " + "x" * 40)
    block, used, tok = ContextAssembler().assemble(items, token_budget=2 * per)
    assert [u.id for u in used] == ["2", "0"]  # top-2 by score, then by time
    assert tok <= 2 * per and block.count("\n") == 1
    assert ContextAssembler().assemble(items, 0) == ("", [], 0)


def test_sqlite_round_trip(tmp_path):
    tel = SqliteTelemetry(f"sqlite:///{tmp_path / 'm.db'}")
    t = DecisionTrace(
        request_id="r1",
        ts=NOW,
        tenant_id="u",
        session_id="s",
        arm="always",
        config_hash="h",
        query="q",
        route=Route.DEEP,
        resolved_by=ResolvedBy.FIXED,
        latency=LatencySplit(embed=5.0, retrieval=2.0, generation=1500.0, total=1510.0),
        prompt_tokens=400,
        completion_tokens=90,
        memory_tokens=300,
        n_items=5,
        features={"s1": 1.0},
    )
    tel.record(t)
    assert tel.fetch() == [t]
    assert tel.fetch(arm="never") == []
    assert tel.to_parquet(tmp_path / "t.parquet") == 1
