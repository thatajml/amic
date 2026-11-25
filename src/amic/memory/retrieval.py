"""RetrievalEngine: executes a RetrievalPlan against the memory stores (frozen interface).

Stores: M1 episodes, M2 current facts, M3 the learner profile (one rendered block, always first).
DEEP with `rerank` or `hops` (brief section 7) builds a wide candidate pool from M1 and M2, adds
one hop of entity expansion, scores the pool with a cross-encoder, then picks k with MMR over the
rerank relevance. Without those flags the original M1(+M2) path is unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np

from amic.contracts.config import RouteConfig
from amic.contracts.models import MemoryItem, MemoryTier, RetrievalPlan
from amic.memory.rerank import Reranker, hop_entities
from amic.memory.stores.episodic import EpisodicStore, Hit
from amic.memory.stores.semantic import SemanticStore

PROFILE_SCORE = 2.0  # above any similarity: the profile block is assembled first


def mmr(
    q: np.ndarray, cand: np.ndarray, k: int, lam: float, rel: np.ndarray | None = None
) -> list[int]:
    """Maximal marginal relevance over unit vectors; returns selected candidate indices.
    `rel` overrides the relevance term (default: cosine to q), e.g. with rerank scores."""
    if len(cand) == 0 or k <= 0:
        return []
    rel = cand @ q if rel is None else rel
    chosen: list[int] = []
    rest = list(range(len(cand)))
    while rest and len(chosen) < k:
        div = (cand[rest] @ cand[chosen].T).max(axis=1) if chosen else np.zeros(len(rest))
        best = rest[int(np.argmax(lam * rel[rest] - (1 - lam) * div))]
        chosen.append(best)
        rest.remove(best)
    return chosen


class ChromaRetrievalEngine:
    def __init__(
        self,
        episodic: EpisodicStore,
        routes: RouteConfig,
        semantic: SemanticStore | None = None,
        profiles: Any = None,
        embedder: Any = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._episodic = episodic
        self._routes = routes
        self._semantic = semantic
        self._profiles = profiles
        self._embedder = embedder
        self._reranker = reranker
        self.calls = 0  # observable: lets tests assert SKIP made zero store contact
        self.last_debug: dict[str, Any] = {}

    def retrieve(
        self, plan: RetrievalPlan, q_vec: np.ndarray, user_id: str, query: str | None = None
    ) -> list[MemoryItem]:
        if plan.is_empty:
            return []
        self.calls += 1
        items: list[MemoryItem] = []
        if MemoryTier.M3 in plan.stores and self._profiles is not None:
            prof = self._profiles.get(user_id)
            if prof is not None and not prof.is_empty():
                items.append(
                    MemoryItem(
                        id=f"profile:{user_id}",
                        tier=MemoryTier.M3,
                        text=prof.render(self._routes.profile_tokens),
                        score=PROFILE_SCORE,
                        ts=datetime.fromtimestamp(prof.updated or 0, UTC),
                    )
                )
        if plan.rerank or plan.hops > 0:
            return items + self._deep_v2(plan, q_vec, user_id, query or "")
        if MemoryTier.M1 in plan.stores:
            fetch = max(plan.k, self._routes.deep_fetch_k) if plan.mmr else plan.k
            hits = self._episodic.query(user_id, q_vec, fetch, plan.filters or None)
            if plan.mmr and len(hits) > plan.k:
                idx = mmr(q_vec, np.stack([h.embedding for h in hits]), plan.k, plan.mmr_lambda)
                hits = [hits[i] for i in idx]
            items += [_to_item(h, MemoryTier.M1) for h in hits[: plan.k]]
        if MemoryTier.M2 in plan.stores and self._semantic is not None:
            # facts alone: k facts; next to M1 (DEEP): m2_k facts on top of the episodes
            k = self._routes.m2_k if MemoryTier.M1 in plan.stores else plan.k
            items += [_to_item(h, MemoryTier.M2) for h in self._semantic.current(user_id, q_vec, k)]
        return items

    def _pool(
        self, plan: RetrievalPlan, vec: np.ndarray, user_id: str, k1: int, k2: int
    ) -> list[tuple[Hit, MemoryTier]]:
        out: list[tuple[Hit, MemoryTier]] = []
        if MemoryTier.M1 in plan.stores and k1 > 0:
            out += [
                (h, MemoryTier.M1)
                for h in self._episodic.query(user_id, vec, k1, plan.filters or None)
            ]
        if MemoryTier.M2 in plan.stores and self._semantic is not None and k2 > 0:
            out += [(h, MemoryTier.M2) for h in self._semantic.current(user_id, vec, k2)]
        return out

    def _deep_v2(
        self, plan: RetrievalPlan, q_vec: np.ndarray, user_id: str, query: str
    ) -> list[MemoryItem]:
        r = self._routes
        cands = self._pool(plan, q_vec, user_id, max(plan.k, r.deep_fetch_k), 2 * r.m2_k)
        hop_ids: set[str] = set()
        ents: list[str] = []
        if plan.hops > 0 and self._embedder is not None and cands:
            seed = sorted(cands, key=lambda c: -c[0].score)[: r.hop_seed]
            ents = hop_entities([h.text for h, _ in seed], query)
            if ents:
                hv = self._embedder.encode("; ".join(ents))
                seen = {h.id for h, _ in cands}
                for h, tier in self._pool(plan, hv, user_id, r.hop_k, r.hop_k):
                    if h.id not in seen:
                        seen.add(h.id)
                        hop_ids.add(h.id)
                        cands.append((h, tier))
        if not cands:
            return []
        if plan.rerank and self._reranker is not None and query:
            raw = np.asarray(self._reranker(query, [h.text for h, _ in cands]), dtype=np.float32)
            span = float(raw.max() - raw.min())
            rel = (raw - raw.min()) / span if span > 0 else np.zeros_like(raw)
        else:
            rel = np.asarray([h.score for h, _ in cands], dtype=np.float32)
        emb = np.stack([h.embedding for h, _ in cands])
        idx = mmr(q_vec, emb, plan.k, plan.mmr_lambda if plan.mmr else 1.0, rel=rel)
        self.last_debug = {"pool": len(cands), "hop_entities": ents, "hop_added": len(hop_ids)}
        out = []
        for i in idx:
            h, tier = cands[i]
            it = _to_item(h, tier)
            it.score = float(rel[i])
            if h.id in hop_ids:
                it.metadata["hop"] = 1
            out.append(it)
        return out


def _to_item(h: Hit, tier: MemoryTier) -> MemoryItem:
    meta = {k: v for k, v in h.metadata.items() if k not in ("tenant_id",)}
    return MemoryItem(id=h.id, tier=tier, text=h.text, score=h.score, ts=h.ts, metadata=meta)
