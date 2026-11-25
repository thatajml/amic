from __future__ import annotations

import numpy as np

from amic.contracts.models import MemoryDecision, ResolvedBy, Route
from amic.control.cache import DecisionCache
from amic.memory.embedder import HashEmbedder


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _d(route: Route = Route.SHALLOW) -> MemoryDecision:
    return MemoryDecision(route=route, resolved_by=ResolvedBy.TIER1, p=0.4)


def test_identical_query_hits_and_different_versions_miss():
    e = HashEmbedder()
    c = DecisionCache()
    v = e.encode("why does my loop skip items")
    k = c.key("alice", 1, v, False)
    assert c.get(k) is None
    c.put(k, _d())
    assert (
        c.get(c.key("alice", 1, e.encode("why does my loop skip items"), False)).route
        == Route.SHALLOW
    )
    assert c.get(c.key("alice", 2, v, False)) is None  # a write bumped the version
    assert c.get(c.key("bob", 1, v, False)) is None  # per learner
    assert c.get(c.key("alice", 1, v, True)) is None  # M0 present now
    assert c.hits == 1 and c.misses == 4


def test_ttl_expires_entries():
    clock = Clock()
    c = DecisionCache(ttl_s=10, clock=clock)
    k = c.key("a", 1, HashEmbedder().encode("x y z"), False)
    c.put(k, _d())
    clock.t = 9
    assert c.get(k) is not None
    clock.t = 20
    assert c.get(k) is None and len(c) == 0


def test_buckets_are_deterministic_and_spread():
    rng = np.random.default_rng(3)
    vecs = rng.standard_normal((200, 384)).astype(np.float32)
    a, b = DecisionCache(nbits=16), DecisionCache(nbits=16)
    ba = [a.bucket(v) for v in vecs]
    assert ba == [b.bucket(v) for v in vecs]  # same seed, same hyperplanes
    assert len(set(ba)) > 190  # unrelated vectors almost never share a 16-bit bucket


def test_lru_bound():
    c = DecisionCache(max_entries=2)
    e = HashEmbedder()
    keys = [c.key("a", 1, e.encode(f"query {w}"), False) for w in ("one", "two", "three")]
    for k in keys:
        c.put(k, _d())
    assert len(c) == 2 and c.get(keys[0]) is None
