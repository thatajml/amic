from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from amic.contracts.models import RequestContext, Route, Turn
from amic.control.signals.bus import FEATURES, S3_FEATURES, SignalBus, compute_features
from amic.control.tier0 import Tier0
from amic.memory.embedder import HashEmbedder
from amic.memory.sketch import BloomFilter, EpisodicSketchStore, build_sketch

NOW = datetime(2026, 11, 18, 12, tzinfo=UTC)
E = HashEmbedder()


def ctx(q: str, m0: list[str] | None = None, session: str = "s-now") -> RequestContext:
    turns = [
        Turn(role="user" if i % 2 == 0 else "assistant", text=t, ts=NOW)
        for i, t in enumerate(m0 or [])
    ]
    return RequestContext(
        request_id="r",
        tenant_id="u1",
        session_id=session,
        query=q,
        m0_turns=turns,
        arm="amic",
        now=NOW,
    )


def sketch_from(texts: list[str]):
    return build_sketch(E.encode_many(texts), texts)


HIST = [
    "Student asked: my recipe scaler crashes with IndexError in the for loop",
    "Student asked: append vs extend gave nested lists in my recipe scaler",
    "Student asked: opened the file in w mode and lost my saved ingredients",
]


def feats(q, m0=None, sk="hist", meta=None, use_sketch=True):
    s = sketch_from(HIST) if sk == "hist" else sk
    c = ctx(q, m0)
    return compute_features(c, E.encode(q), s, meta or {}, use_sketch), s


# ----------------------------------------------------------------------------- features
def test_feature_vector_complete_and_ordered():
    f, _ = feats("what is a tuple?")
    assert list(f) == FEATURES and all(isinstance(v, float) for v in f.values())
    assert set(S3_FEATURES) <= set(FEATURES)


@pytest.mark.parametrize(
    ("q", "key"),
    [
        ("What did I get wrong last time?", "s1_explicit_ref"),
        ("remind me what we covered", "s1_explicit_ref"),
        ("the same bug is back again", "s1_temporal"),
        ("that error came back in my project", "s1_deixis"),
        ("what did you mean by that last part?", "s1_followup"),
        ("what should I practice next?", "s1_request_plan"),
        ("How do I reverse a list?", "s1_general_form"),
        ("why does print(x[3]) fail", "s1_has_code"),
    ],
)
def test_lexical_markers(q, key):
    assert feats(q)[0][key] > 0


def test_general_question_has_no_personal_markers():
    f, _ = feats("What is the difference between a list and a tuple?")
    assert f["s1_explicit_ref"] == f["s1_possessive"] == f["s1_deixis"] == 0
    assert f["x_general_no_personal"] == 1.0 and f["s4_concept_hits"] >= 2


def test_affinity_higher_for_query_about_stored_topics():
    on, _ = feats("my recipe scaler append nested lists")
    off, _ = feats("explain recursion with factorial")
    assert on["s3_max_sim"] > off["s3_max_sim"]
    assert on["s3_term_hit_frac"] > off["s3_term_hit_frac"]


def test_s3_ablation_zeroes_affinity_but_keeps_has_memory():
    f, _ = feats("my recipe scaler append nested lists", use_sketch=False)
    assert f["s3_has_memory"] == 1.0
    assert f["s3_max_sim"] == f["s3_margin"] == f["s3_term_hits"] == 0.0


def test_m0_overlap_and_days_since():
    meta = {"last_session_id": "old", "last_seen": (NOW - timedelta(days=6)).timestamp()}
    f, _ = feats(
        "show the dictionary example again", m0=["what is a dictionary", "a mapping"], meta=meta
    )
    assert f["s2_m0_overlap"] > 0 and f["s2_m0_turns"] == 2
    assert f["s2_days_since_log"] == pytest.approx(np.log1p(6.0))


def test_days_since_uses_previous_session_when_already_touched():
    meta = {
        "last_session_id": "s-now",
        "last_seen": NOW.timestamp(),
        "prev_session_end": (NOW - timedelta(days=3)).timestamp(),
    }
    f, _ = feats("hi", meta=meta)
    assert f["s2_days_since_log"] == pytest.approx(np.log1p(3.0))


# ----------------------------------------------------------------------------- sketch
def test_bloom_no_false_negatives():
    b = BloomFilter()
    words = [f"word{i}" for i in range(500)]
    for w in words:
        b.add(w)
    assert all(w in b for w in words)
    assert sum(f"other{i}" in b for i in range(1000)) < 50  # low false-positive rate


def test_sketch_affinity_shapes():
    s = sketch_from(HIST * 4)
    assert 1 <= s.centroids.shape[0] <= 8 and s.n_records == 12
    ms, mg = s.affinity(E.encode(HIST[0]))
    assert 0.0 <= mg <= ms <= 1.0001
    assert build_sketch(np.zeros((0, 384)), []).affinity(E.encode("x")) == (0.0, 0.0)


def test_sketch_store_none_without_memory_and_invalidation(comps, seed_history):
    store = EpisodicSketchStore(comps.episodic)
    assert store.get("nobody") is None
    assert store.get("amy") is None
    seed_history(comps, "amy", [("loops", "a"), ("lists", "b")])
    assert store.get("amy") is None  # cached None until invalidated
    store.invalidate("amy")
    sk = store.get("amy")
    assert sk is not None and sk.n_records == 2
    assert store.get("amy") is sk and store.builds == 3  # cached


def test_sketch_reads_no_documents_at_query_time(comps, seed_history):
    seed_history(comps, "amy", [("loops", "a")])
    store = EpisodicSketchStore(comps.episodic)
    sk = store.get("amy")
    calls = []
    comps.episodic.query = lambda *a, **k: calls.append(1)  # type: ignore[method-assign]
    sk.affinity(E.encode("loops"))
    assert calls == []


def test_bus_embeds_exactly_once(comps, seed_history):
    seed_history(comps, "amy", [("loops", "a")])
    n = {"encode": 0}
    real = comps.embedder.encode

    def counting(text):
        n["encode"] += 1
        return real(text)

    comps.embedder.encode = counting  # type: ignore[method-assign]
    bus = SignalBus(comps.embedder, EpisodicSketchStore(comps.episodic), comps.sessions)
    c = ctx("loops again")
    c = c.model_copy(update={"tenant_id": "amy"})
    sig = bus.extract(c)
    assert n["encode"] == 1 and sig.q_vec.shape == (384,) and sig.sketch is not None
    assert sig.embed_ms >= 0 and sig.feat_ms >= 0


# ----------------------------------------------------------------------------- tier 0
def sig_for(q, m0=None, sk="hist"):
    s = sketch_from(HIST) if sk == "hist" else sk
    c = ctx(q, m0)
    from amic.control.signals.bus import COMMAND, Signals

    return Signals(
        E.encode(q), compute_features(c, E.encode(q), s, {}), s, bool(COMMAND.match(q)), 0, 0
    )


@pytest.mark.parametrize(
    ("q", "m0", "sk", "route", "rule"),
    [
        ("/remember I prefer short answers", None, None, Route.WRITE, "R4"),
        ("What did I get wrong last time?", None, None, Route.SKIP, "R1"),
        (
            "what did you mean by that last part?",
            ["what is a set", "a collection"],
            "hist",
            Route.SKIP,
            "R2",
        ),
        ("What did I get wrong last time?", None, "hist", Route.DEEP, "R3"),
        (
            "what did you mean, and what did we discuss last week?",
            ["x", "y"],
            "hist",
            Route.DEEP,
            "R3",
        ),
        ("How do I reverse a list?", None, "hist", None, "abstain"),
    ],
)
def test_tier0_rules(q, m0, sk, route, rule):
    r = Tier0().decide(sig_for(q, m0, sk))
    assert r.route == route and r.rule.startswith(rule)


def test_followup_without_m0_is_not_r2():
    assert Tier0().decide(sig_for("what did you mean by that last part?")).route is None
