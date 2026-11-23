"""Per-learner memory version: the invalidation signal for the M4 sketch and decision caches."""

from __future__ import annotations

import fakeredis
import pytest

from amic.service.session import SessionManager, SqliteSessionManager
from tests.conftest import T0


@pytest.fixture(params=["redis", "sqlite"])
def sessions(request):
    if request.param == "redis":
        return SessionManager(fakeredis.FakeRedis())
    return SqliteSessionManager(":memory:")


def test_version_starts_at_zero_and_only_increases(sessions):
    assert sessions.mem_version("a") == 0
    v1 = sessions.bump_mem_version("a")
    v2 = sessions.bump_mem_version("a")
    assert 0 < v1 < v2 == sessions.mem_version("a")
    assert sessions.mem_version("b") == 0  # per learner


def test_erasure_sets_a_new_version_instead_of_resetting(sessions):
    v = sessions.bump_mem_version("a")
    sessions.erase_tenant("a")
    assert sessions.mem_version("a") > v  # an entry cached before the erasure can never match


def test_writes_bump_the_version(comps):
    assert comps.sessions.mem_version("alice") == 0
    comps.writer.write_turn("alice", "s0", "r1", "what is a list", "an ordered collection", T0)
    assert comps.sessions.mem_version("alice") > 0


def test_erasure_makes_the_sketch_disappear(comps):
    """Regression: erasure used to leave the cached sketch behind, so R1 never fired again."""
    comps.writer.write_turn("alice", "s0", "r1", "what is a list", "an ordered collection", T0)
    assert comps.sketches.get("alice") is not None
    comps.episodic.delete_tenant("alice")
    comps.sessions.erase_tenant("alice")
    assert comps.sketches.get("alice") is None


def test_a_write_from_another_process_rebuilds_the_sketch(comps):
    comps.writer.write_turn("alice", "s0", "r1", "what is a list", "an ordered collection", T0)
    before = comps.sketches.get("alice")
    assert before is not None and before.n_records == 1
    # simulate the consolidation worker: write to the store + bump, without touching this cache
    v = comps.embedder.encode("dict keys")
    comps.episodic.add("alice", "r2", "dicts map keys to values", v, T0)
    comps.sessions.bump_mem_version("alice")
    assert comps.sketches.get("alice").n_records == 2


def test_sketch_top_terms_are_deterministic(comps):
    for i, q in enumerate(["loops loops lists", "loops dicts", "lists strings"]):
        comps.writer.write_turn("bob", "s0", f"r{i}", q, "", T0)
    sk = comps.sketches.get("bob")
    assert sk.top_terms[:2] == ["lists", "loops"]  # df 2 each, alphabetical tie-break
