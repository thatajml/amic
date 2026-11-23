"""DecisionCache (brief 6.4): reuse a recent gate decision for a near-identical query.

Key = (learner, memory version, LSH bucket of q_vec, M0 present). The LSH is `nbits` random
hyperplanes (seeded), so identical queries always collide and paraphrases sometimes do.
Invalidation needs no messages: every long-term write or erasure changes the learner's memory
version (service/session.py), so entries made before it can never match again; the TTL bounds
staleness and size. Deviation from the brief (PLAN R4, flagged): checked AFTER Tier 0, which is
cheaper than a lookup and session-dependent (R2), and only Tier-1 / Tier-2 / band outcomes are
stored, never Tier 0's or a Tier-2 failure.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable

import numpy as np

from amic.contracts.models import MemoryDecision

Key = tuple[str, int, int, bool]


class DecisionCache:
    def __init__(
        self,
        nbits: int = 16,
        ttl_s: float = 900.0,
        max_entries: int = 50_000,
        seed: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.nbits = nbits
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self.seed = seed
        self.clock = clock
        self._planes: np.ndarray | None = None
        self._store: OrderedDict[Key, tuple[float, MemoryDecision]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def bucket(self, q_vec: np.ndarray) -> int:
        if self._planes is None or self._planes.shape[1] != q_vec.shape[0]:
            rng = np.random.default_rng(self.seed)
            self._planes = rng.standard_normal((self.nbits, q_vec.shape[0])).astype(np.float32)
        bits = ((self._planes @ q_vec) > 0).astype(np.int64)  # one sign bit per hyperplane
        return int(bits @ (np.int64(1) << np.arange(self.nbits, dtype=np.int64)))

    def key(self, tenant_id: str, version: int, q_vec: np.ndarray, m0_present: bool) -> Key:
        return (tenant_id, version, self.bucket(q_vec), m0_present)

    def get(self, key: Key) -> MemoryDecision | None:
        hit = self._store.get(key)
        if hit is None or self.clock() - hit[0] > self.ttl_s:
            if hit is not None:
                del self._store[key]
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return hit[1]

    def put(self, key: Key, decision: MemoryDecision) -> None:
        self._store[key] = (self.clock(), decision)
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)
