"""M1 episodic store (Chroma) contract, and the DEEP-route rerank / entity-hop helpers."""

from __future__ import annotations

import sys
import types
import uuid
from datetime import timedelta

import numpy as np

from amic.memory.embedder import HashEmbedder
from amic.memory.rerank import CrossEncoderReranker, hop_entities, names
from amic.memory.stores.episodic import EpisodicStore
from tests.conftest import T0

EMB = HashEmbedder()


def _store(chroma_client, mode="single") -> EpisodicStore:
    return EpisodicStore(chroma_client, f"e_{uuid.uuid4().hex[:10]}", mode=mode)


def _add(s: EpisodicStore, tenant: str, rid: str, text: str, minutes: int = 0, **meta) -> None:
    s.add(tenant, rid, text, EMB.encode(text), T0 + timedelta(minutes=minutes), meta or None)


# ------------------------------------------------------------------ EpisodicStore
def test_add_query_roundtrip_returns_hits_with_metadata(chroma_client):
    s = _store(chroma_client)
    _add(s, "alice", "r1", "range(5) stops at 4", session_id="s1")
    _add(s, "alice", "r2", "dictionaries map keys to values")
    hits = s.query("alice", EMB.encode("range(5) stops at 4"), k=2)
    assert [h.id for h in hits][0] == "r1" and hits[0].score > hits[1].score
    assert hits[0].metadata["session_id"] == "s1" and hits[0].metadata["tenant_id"] == "alice"
    assert hits[0].ts == T0 and hits[0].embedding.shape == EMB.encode("x").shape
    assert s.query("alice", EMB.encode("x"), k=0) == []


def test_learners_are_isolated_in_a_shared_collection(chroma_client):
    s = _store(chroma_client)
    _add(s, "alice", "r1", "alice private note")
    _add(s, "bob", "r2", "bob private note")
    assert s.count("alice") == 1 and s.count("bob") == 1
    assert {h.id for h in s.query("alice", EMB.encode("private note"), k=5)} == {"r1"}
    assert s.tenants() == ["alice", "bob"]


def test_metadata_filter_narrows_the_query(chroma_client):
    s = _store(chroma_client)
    _add(s, "alice", "a", "loops and ranges", kind="turn")
    _add(s, "alice", "b", "loops and ranges too", kind="remember")
    hits = s.query("alice", EMB.encode("loops"), k=5, filters={"kind": "remember"})
    assert [h.id for h in hits] == ["b"]


def test_add_is_idempotent_by_record_id(chroma_client):
    s = _store(chroma_client)
    _add(s, "alice", "r1", "first text")
    _add(s, "alice", "r1", "replaced text")
    assert s.count("alice") == 1
    assert s.records("alice")[0][1] == "replaced text"


def test_records_and_vectors_expose_ids_texts_and_embeddings(chroma_client):
    s = _store(chroma_client)
    _add(s, "alice", "r1", "one")
    _add(s, "alice", "r2", "two")
    recs = {rid: (text, meta, emb) for rid, text, meta, emb in s.records("alice")}
    assert set(recs) == {"r1", "r2"} and recs["r1"][0] == "one"
    assert recs["r1"][1]["tenant_id"] == "alice" and recs["r1"][2].dtype == np.float32
    ids, arr = s.vectors("alice")
    assert sorted(ids) == ["r1", "r2"] and arr.shape[0] == 2
    ids2, arr2, texts = s.vectors_and_texts("alice")
    assert sorted(texts) == ["one", "two"] and arr2.shape == arr.shape and len(ids2) == 2


def test_an_unknown_learner_has_no_vectors(chroma_client):
    s = _store(chroma_client)
    assert s.count("ghost") == 0 and s.records("ghost") == []
    ids, arr = s.vectors("ghost")
    assert ids == [] and arr.shape[0] == 0
    assert s.vectors_and_texts("ghost")[0] == []
    assert s.tenants() == []


def test_add_many_bulk_upserts_in_batches(chroma_client):
    s = _store(chroma_client)
    texts = [f"memory number {i}" for i in range(7)]
    s.add_many(
        "alice",
        [f"r{i}" for i in range(7)],
        texts,
        np.stack([EMB.encode(t) for t in texts]),
        [T0 + timedelta(minutes=i) for i in range(7)],
        [{"kind": "turn"}] * 7,
        batch=3,
    )
    assert s.count("alice") == 7
    assert all(m["kind"] == "turn" for _, _, m, _ in s.records("alice"))


def test_delete_ids_only_removes_the_owners_records(chroma_client):
    s = _store(chroma_client)
    _add(s, "alice", "r1", "alice note")
    _add(s, "bob", "r2", "bob note")
    assert s.delete_ids("alice", []) == 0
    assert s.delete_ids("alice", ["r2", "missing"]) == 0  # bob's id is not alice's to delete
    assert s.count("bob") == 1
    assert s.delete_ids("alice", ["r1", "r2"]) == 1 and s.count("alice") == 0


def test_update_metadata_merges_fields_and_ignores_foreign_ids(chroma_client):
    s = _store(chroma_client)
    _add(s, "alice", "r1", "note", kind="turn")
    s.update_metadata("alice", "r1", flagged=True)
    meta = s.records("alice")[0][2]
    assert meta["flagged"] is True and meta["kind"] == "turn" and meta["tenant_id"] == "alice"
    s.update_metadata("bob", "r1", flagged=False)  # not bob's record: unchanged
    assert s.records("alice")[0][2]["flagged"] is True


def test_delete_tenant_erases_one_learner_only(chroma_client):
    s = _store(chroma_client)
    _add(s, "alice", "r1", "a")
    _add(s, "alice", "r2", "b")
    _add(s, "bob", "r3", "c")
    s.delete_tenant("alice")
    assert s.count("alice") == 0 and s.count("bob") == 1


def test_per_tenant_mode_uses_a_collection_per_learner(chroma_client):
    s = _store(chroma_client, mode="per_tenant")
    _add(s, "alice", "r1", "a")
    _add(s, "bob/../x", "r2", "b")  # unsafe characters are sanitised in the collection name
    assert s.count("alice") == 1 and s.count("bob/../x") == 1
    assert s.tenants() == ["alice", "bob/../x"]
    assert s.query("alice", EMB.encode("a"), k=3)[0].id == "r1"


# ------------------------------------------------------------------ rerank helpers
def test_names_finds_mid_sentence_capitalised_names_only():
    assert names("I met Priya Sharma yesterday") == {"Priya Sharma"}
    assert names("Python is great. Today we cover loops.") == set()  # sentence starts, stop words
    assert "Student" not in names("The Student asked about Django")
    assert names("see Django docs") == {"Django"}


def test_hop_entities_excludes_the_query_and_orders_by_frequency():
    texts = [
        "I used Django with a for loop",
        "Django project, list comprehension",
        "my Flask app uses a for loop",
    ]
    out = hop_entities(texts, "how do I write a for loop?")
    assert "for loop" not in out  # named by the query: not a new association
    assert out[0] == "Django"  # most frequent first
    assert len(hop_entities(texts * 3, "unrelated question", max_entities=1)) == 1
    assert hop_entities([], "anything") == []


def test_cross_encoder_scores_pairs_and_loads_lazily_once(monkeypatch):
    loads: list[tuple] = []

    class FakeCE:
        def __init__(self, name, device, max_length):
            loads.append((name, device, max_length))

        def predict(self, pairs, show_progress_bar):
            return np.array([float(len(t)) for _, t in pairs])

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=FakeCE)
    )
    rr = CrossEncoderReranker("fake-model", device="cpu", max_length=64)
    assert rr("q", []) == [] and loads == []  # nothing to score: the model is never loaded
    assert rr("q", ["ab", "abcd"]) == [2.0, 4.0]
    assert rr("q", ["x"]) == [1.0]
    assert loads == [("fake-model", "cpu", 64)]  # loaded once, reused
