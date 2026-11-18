"""RetrievalEngine: executes a RetrievalPlan against the memory stores (frozen interface)."""

from __future__ import annotations

import numpy as np

from amic.contracts.config import RouteConfig
from amic.contracts.models import MemoryItem, MemoryTier, RetrievalPlan
from amic.memory.stores.episodic import EpisodicStore, Hit


def mmr(q: np.ndarray, cand: np.ndarray, k: int, lam: float) -> list[int]:
    """Maximal marginal relevance over unit vectors; returns selected candidate indices."""
    if len(cand) == 0 or k <= 0:
        return []
    rel = cand @ q
    chosen: list[int] = []
    rest = list(range(len(cand)))
    while rest and len(chosen) < k:
        div = (cand[rest] @ cand[chosen].T).max(axis=1) if chosen else np.zeros(len(rest))
        best = rest[int(np.argmax(lam * rel[rest] - (1 - lam) * div))]
        chosen.append(best)
        rest.remove(best)
    return chosen


class ChromaRetrievalEngine:
    def __init__(self, episodic: EpisodicStore, routes: RouteConfig) -> None:
        self._episodic = episodic
        self._routes = routes
        self.calls = 0  # observable: lets tests assert SKIP made zero store contact

    def retrieve(self, plan: RetrievalPlan, q_vec: np.ndarray, user_id: str) -> list[MemoryItem]:
        if plan.is_empty:
            return []
        self.calls += 1
        items: list[MemoryItem] = []
        if MemoryTier.M1 in plan.stores:
            fetch = max(plan.k, self._routes.deep_fetch_k) if plan.mmr else plan.k
            hits = self._episodic.query(user_id, q_vec, fetch, plan.filters or None)
            if plan.mmr and len(hits) > plan.k:
                idx = mmr(q_vec, np.stack([h.embedding for h in hits]), plan.k, plan.mmr_lambda)
                hits = [hits[i] for i in idx]
            items += [_to_item(h, MemoryTier.M1) for h in hits[: plan.k]]
        # M2 (facts) and M3 (profile) stores plug in here; not built in this sprint.
        return items


def _to_item(h: Hit, tier: MemoryTier) -> MemoryItem:
    meta = {k: v for k, v in h.metadata.items() if k not in ("tenant_id",)}
    return MemoryItem(id=h.id, tier=tier, text=h.text, score=h.score, ts=h.ts, metadata=meta)
