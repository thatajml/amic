"""M2 semantic store: distilled atomic facts about a learner, with validity intervals.

Facts live in the Chroma collection `<collection>_facts` (same tenant filter as M1). Each fact
has a `key` (e.g. "misconception:range", "goal") used for conflict resolution, a `kind`, a
confidence, and a validity interval: `valid_from`, and `valid_to` = 0 while the fact is current.
A superseded fact is never deleted, only closed (valid_to set), so history stays auditable;
retrieval reads current facts only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from amic.memory.stores.episodic import EpisodicStore, Hit

CURRENT = {"valid_to": 0.0}


class SemanticStore:
    def __init__(self, client: Any, collection: str, mode: str = "single") -> None:
        self._store = EpisodicStore(client, f"{collection}_facts", mode)  # type: ignore[arg-type]

    def add(
        self,
        tenant_id: str,
        fact_id: str,
        text: str,
        vec: np.ndarray,
        *,
        key: str,
        kind: str,
        confidence: float,
        valid_from: datetime,
        source: str = "",
    ) -> None:
        self._store.add(
            tenant_id,
            fact_id,
            text,
            vec,
            valid_from,
            {
                "key": key,
                "kind": kind,
                "confidence": float(confidence),
                "valid_from": valid_from.timestamp(),
                "valid_to": 0.0,
                "source": source,
            },
        )

    def current(self, tenant_id: str, q_vec: np.ndarray, k: int) -> list[Hit]:
        """Top-k CURRENT facts by similarity (superseded facts are never returned)."""
        if k <= 0:
            return []
        try:
            return self._store.query(tenant_id, q_vec, k, CURRENT)
        except Exception:  # noqa: BLE001 - an empty/new collection must not break retrieval
            return []

    def current_by_key(self, tenant_id: str, key: str) -> list[tuple[str, np.ndarray, dict]]:
        col = self._store._col(tenant_id)
        got = col.get(
            where=self._store._where(tenant_id, {"key": key, **CURRENT}),
            include=["embeddings", "metadatas"],
        )
        embs = got["embeddings"]
        return [
            (got["ids"][i], np.asarray(embs[i], dtype=np.float32), dict(got["metadatas"][i] or {}))
            for i in range(len(got["ids"]))
        ]

    def _patch(self, tenant_id: str, fact_id: str, **fields: float | str) -> None:
        col = self._store._col(tenant_id)
        got = col.get(ids=[fact_id], include=["metadatas"])
        if not got["ids"]:
            return
        meta = {**dict(got["metadatas"][0] or {}), **fields}
        col.update(ids=[fact_id], metadatas=[meta])

    def close(self, tenant_id: str, fact_id: str, valid_to: datetime) -> None:
        """End a fact's validity (superseded by a newer fact with the same key)."""
        self._patch(tenant_id, fact_id, valid_to=valid_to.timestamp())

    def refresh(self, tenant_id: str, fact_id: str, seen: datetime, confidence: float) -> None:
        """The same fact was observed again: keep it current, record when and how sure."""
        self._patch(tenant_id, fact_id, last_seen=seen.timestamp(), confidence=float(confidence))

    def all(self, tenant_id: str) -> list[tuple[str, str, dict]]:
        """(id, text, metadata) of every fact, current or not (audit, tests)."""
        col = self._store._col(tenant_id)
        got = col.get(where=self._store._where(tenant_id), include=["documents", "metadatas"])
        return [
            (got["ids"][i], got["documents"][i] or "", dict(got["metadatas"][i] or {}))
            for i in range(len(got["ids"]))
        ]

    def count(self, tenant_id: str, current_only: bool = True) -> int:
        col = self._store._col(tenant_id)
        where = self._store._where(tenant_id, CURRENT if current_only else None)
        return len(col.get(where=where, include=[])["ids"])

    def records(self, tenant_id: str) -> list[tuple[str, str, dict, np.ndarray]]:
        """(id, text, metadata, embedding) of every fact, current or not."""
        return self._store.records(tenant_id)

    def patch(self, tenant_id: str, fact_id: str, **fields: float | str) -> None:
        self._patch(tenant_id, fact_id, **fields)

    def tenants(self) -> list[str]:
        return self._store.tenants()

    def delete_ids(self, tenant_id: str, ids: list[str]) -> int:
        """Hard delete (an explicit /forget or a TTL on closed history), not a validity close."""
        return self._store.delete_ids(tenant_id, ids)

    def delete_tenant(self, tenant_id: str) -> int:
        return self._store.delete_tenant(tenant_id)
