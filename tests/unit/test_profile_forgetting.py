"""M3 learner profile (EWMA mastery, notes, rendering, decay), DEEP v2 retrieval and the
ForgettingManager (decay, TTL, compaction, erasure)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import fakeredis
import numpy as np
import pytest

from amic.contracts.config import ForgettingConfig, RouteConfig
from amic.contracts.models import MemoryTier, RetrievalPlan
from amic.memory.embedder import HashEmbedder
from amic.memory.rerank import hop_entities
from amic.memory.retrieval import PROFILE_SCORE, ChromaRetrievalEngine
from amic.memory.stores.episodic import EpisodicStore
from amic.memory.stores.profile import (
    LearnerProfile,
    ProfileStore,
    RedisKV,
    SqliteKV,
    detect_concepts,
)
from amic.memory.stores.semantic import SemanticStore
from amic.writepath.extract import ExtractedFact
from amic.writepath.forgetting import ForgettingManager, duplicate_ids

T = datetime(2026, 11, 1, 10, 0, tzinfo=UTC)
E = HashEmbedder()


def _f(key: str, kind: str, text: str) -> ExtractedFact:
    return ExtractedFact(key, kind, text, 0.9)


# ----------------------------------------------------------------------------- M3 profile
def test_concepts_are_detected_from_the_curriculum_lexicon():
    assert "for loops and range" in detect_concepts("why does range(5) stop at 4?")
    assert "dictionaries" in detect_concepts("I got a KeyError from my dict")
    assert detect_concepts("hello there") == []


def test_ewma_mastery_moves_with_evidence_and_progress_closes_misconceptions():
    p = LearnerProfile(tenant_id="a")
    mis = _f("misconception:range_end", "misconception", "Thinks range(n) includes n")
    assert p.observe_turn("my loop with range crashes", T, [mis], alpha=0.5)
    st = p.mastery["for loops and range"]
    assert st.m == pytest.approx(0.25) and st.n == 1  # 0.5 -> 0.5*0.5 + 0.5*0
    assert "misconception:range_end" in p.misconceptions
    prog = _f("progress:range_end", "progress", "Now knows range stops before n")
    p.observe_turn("got it, range stops early", T + timedelta(days=1), [prog], alpha=0.5)
    assert p.mastery["for loops and range"].m == pytest.approx(0.625)
    assert p.misconceptions == {}  # progress on the same topic closes it


def test_lexical_evidence_without_facts_and_exposure_only_questions():
    p = LearnerProfile(tenant_id="a")
    p.observe_turn("my dict lookup keeps giving an error", T, alpha=0.5)
    assert p.mastery["dictionaries"].m < 0.5
    p.observe_turn("what is a while loop?", T)  # a plain question: exposure, no evidence
    st = p.mastery["while loops"]
    assert st.seen == 1 and st.n == 0 and st.m == 0.5


def test_notes_keep_the_latest_per_kind_and_render_is_bounded():
    p = LearnerProfile(tenant_id="a")
    for i in range(5):
        p.observe_turn("", T, [_f(f"goal:g{i}", "goal", f"goal number {i}")], max_notes=3)
    assert list(p.notes["goal"]) == ["goal:g2", "goal:g3", "goal:g4"]
    p.observe_turn("", T, [_f("preference", "preference", "short worked examples")])
    text = p.render(max_tokens=200)
    assert text.startswith("Learner profile.") and "goal number 4" in text
    assert "short worked examples" in text and len(text) <= 800
    assert len(p.render(max_tokens=10)) <= 40


def test_mastery_decays_back_to_the_prior():
    p = LearnerProfile(tenant_id="a")
    p.observe_turn("", T, [_f("misconception:x", "misconception", "range bug")], alpha=1.0)
    assert p.mastery["for loops and range"].m == 0.0
    assert p.decay(T + timedelta(days=30), half_life_days=30)
    assert p.mastery["for loops and range"].m == pytest.approx(0.25)


@pytest.mark.parametrize("kv", ["sqlite", "redis"])
def test_profile_store_roundtrip(kv, tmp_path):
    store = ProfileStore(
        SqliteKV(str(tmp_path / "p.sqlite")) if kv == "sqlite" else RedisKV(fakeredis.FakeRedis())
    )
    p = store.get_or_new("bob")
    p.observe_turn("my list index is out of range error", T)
    store.put(p)
    assert store.get("bob").mastery == p.mastery and store.tenants() == ["bob"]
    assert store.delete("bob") == 1 and store.get("bob") is None


# ----------------------------------------------------------------------------- retrieval
@pytest.fixture
def stores(chroma_client):
    name = f"t_{uuid.uuid4().hex[:8]}"
    ep = EpisodicStore(chroma_client, name)
    sem = SemanticStore(chroma_client, name)
    prof = ProfileStore(SqliteKV(":memory:"))
    return ep, sem, prof


def _add(ep, tenant, i, text, ts=T, **meta):
    ep.add(tenant, f"{tenant}-{i}", text, E.encode(text), ts, {"kind": "turn", **meta})


def test_shallow_m3_puts_the_profile_block_first(stores):
    ep, sem, prof = stores
    p = prof.get_or_new("ann")
    p.observe_turn("", T, [_f("goal", "goal", "build a budgeting tool")])
    prof.put(p)
    eng = ChromaRetrievalEngine(ep, RouteConfig(), sem, profiles=prof)
    items = eng.retrieve(
        RetrievalPlan(stores=[MemoryTier.M3, MemoryTier.M2], k=3), E.encode("x"), "ann"
    )
    assert items[0].tier == MemoryTier.M3 and items[0].score == PROFILE_SCORE
    assert "budgeting tool" in items[0].text
    # another learner never sees it
    assert eng.retrieve(RetrievalPlan(stores=[MemoryTier.M3], k=3), E.encode("x"), "bob") == []


def test_hop_entities_are_new_entities_from_the_top_candidates():
    texts = ["Student asked: my Football League table crashes with a KeyError in the dict"]
    ents = hop_entities(texts, "why does my table crash?")
    assert "dictionaries" in ents and "Football League" in ents


def test_deep_v2_reranks_hops_and_respects_k(stores):
    ep, sem, prof = stores
    for i in range(8):
        _add(ep, "cy", i, f"Student asked: question {i} about lists and append")
    _add(ep, "cy", 99, "Student asked: my Recipe Scaler project reads a file", T)
    calls = {}

    def rerank(q: str, texts: list[str]) -> list[float]:
        calls["n"] = len(texts)
        return [10.0 if "Recipe" in t else float(i) for i, t in enumerate(texts)]

    eng = ChromaRetrievalEngine(
        ep, RouteConfig(deep_fetch_k=5, hop_k=4), sem, profiles=prof, embedder=E, reranker=rerank
    )
    plan = RetrievalPlan(stores=[MemoryTier.M1], k=3, mmr=True, mmr_lambda=1.0, rerank=True, hops=1)
    items = eng.retrieve(plan, E.encode("lists append question"), "cy", query="lists append?")
    assert len(items) == 3 and calls["n"] >= 5
    assert eng.last_debug["pool"] >= 5
    assert max(it.score for it in items) == pytest.approx(1.0)  # rerank scores, normalised


# ----------------------------------------------------------------------------- forgetting
def test_duplicate_ids_keep_the_newest():
    v = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    assert duplicate_ids(["old", "new", "other"], [1.0, 2.0, 3.0], v, 0.97) == ["old"]


class _Sessions:
    def __init__(self) -> None:
        self.bumps: list[str] = []
        self.erased: list[str] = []

    def bump_mem_version(self, t: str) -> int:
        self.bumps.append(t)
        return len(self.bumps)

    def erase_tenant(self, t: str) -> None:
        self.erased.append(t)


def test_forgetting_decays_ttl_compacts_and_bumps(stores):
    ep, sem, prof = stores
    now = T + timedelta(days=400)
    _add(ep, "dee", 1, "Student asked: how do I append to a list?", T)
    _add(ep, "dee", 2, "Student asked: how do I append to a list?", T + timedelta(days=1))
    _add(ep, "dee", 3, "Student asked me to remember: likes cats", T, kind="remember")
    _add(ep, "dee", 4, "Student asked: how do I append to a list?", T, kind="remember")
    sem.add(
        "dee",
        "f-old",
        "old goal",
        E.encode("old goal"),
        key="goal",
        kind="goal",
        confidence=0.9,
        valid_from=T,
    )
    sem.add(
        "dee",
        "f-closed",
        "older goal",
        E.encode("older goal"),
        key="goal2",
        kind="goal",
        confidence=0.9,
        valid_from=T,
    )
    sem.close("dee", "f-closed", T)
    p = prof.get_or_new("dee")
    p.observe_turn("", T, [_f("misconception:x", "misconception", "range bug")], alpha=1.0)
    prof.put(p)
    sess = _Sessions()
    fm = ForgettingManager(
        ForgettingConfig(fact_half_life_days=60, closed_fact_ttl_days=365, episode_ttl_days=None),
        ep,
        sem,
        prof,
        sess,
    )
    dry = fm.run_tenant("dee", now, dry_run=True)
    assert dry == {
        "facts_closed": 1,
        "facts_deleted": 1,
        "episodes_expired": 0,
        "episodes_compacted": 1,
        "profiles_decayed": 1,
    }
    assert ep.count("dee") == 4 and sess.bumps == []  # dry run changes nothing
    assert fm.run_tenant("dee", now) == dry and sess.bumps == ["dee"]
    assert ep.count("dee") == 3  # the older duplicate turn went; remember notes are kept
    assert sem.count("dee") == 0 and len(sem.all("dee")) == 1  # decayed fact closed, not deleted
    closed = sem.all("dee")[0][2]
    assert closed["closed_by"] == "decay" and closed["valid_to"] == now.timestamp()
    assert prof.get("dee").mastery["for loops and range"].m > 0.45
    assert "dee" in fm.tenants()


def test_episode_ttl_and_erasure(stores):
    ep, sem, prof = stores
    _add(ep, "eve", 1, "ancient question", T - timedelta(days=100))
    _add(ep, "eve", 2, "recent question", T)
    fm = ForgettingManager(ForgettingConfig(episode_ttl_days=30), ep, sem, prof, _Sessions())
    assert fm.run_tenant("eve", T)["episodes_expired"] == 1 and ep.count("eve") == 1
    prof.put(prof.get_or_new("eve"))
    s = fm.sessions
    assert fm.erase("eve") == {"deleted_records": 1, "deleted_facts": 0, "deleted_profile": 1}
    assert s.erased == ["eve"] and ep.count("eve") == 0
