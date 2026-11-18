"""M1 episodic store: raw turn records in Chroma, namespaced per learner (P8).

Default layout is one collection + a `tenant_id` metadata filter (plan R9); `per_tenant`
gives each learner its own collection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np

Metadata = dict[str, str | int | float | bool]


@dataclass
class Hit:
    id: str
    text: str
    score: float  # cosine similarity
    ts: datetime
    metadata: Metadata
    embedding: np.ndarray


def make_chroma_client(
    kind: Literal["embedded", "http", "ephemeral"],
    path: str = "data/chroma",
    host: str = "localhost",
    port: int = 8000,
) -> Any:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    cs = ChromaSettings(anonymized_telemetry=False, allow_reset=True)
    if kind == "embedded":
        return chromadb.PersistentClient(path=path, settings=cs)
    if kind == "http":
        return chromadb.HttpClient(host=host, port=port, settings=cs)
    return chromadb.EphemeralClient(settings=cs)


_SAFE = re.compile(r"[^a-zA-Z0-9_-]")


class EpisodicStore:
    def __init__(
        self, client: Any, collection: str, mode: Literal["single", "per_tenant"] = "single"
    ) -> None:
        self._client = client
        self._name = collection
        self._mode = mode
        self._cols: dict[str, Any] = {}

    # ------------------------------------------------------------------ internals
    def _col(self, tenant_id: str) -> Any:
        name = (
            self._name if self._mode == "single" else f"{self._name}__{_SAFE.sub('_', tenant_id)}"
        )
        col = self._cols.get(name)
        if col is None:
            col = self._client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
            self._cols[name] = col
        return col

    def _where(self, tenant_id: str, filters: Metadata | None = None) -> dict[str, Any]:
        conds: list[dict[str, Any]] = [{"tenant_id": tenant_id}]
        conds += [{k: v} for k, v in (filters or {}).items()]
        return conds[0] if len(conds) == 1 else {"$and": conds}

    # ------------------------------------------------------------------ API
    def add(
        self,
        tenant_id: str,
        record_id: str,
        text: str,
        vec: np.ndarray,
        ts: datetime,
        metadata: Metadata | None = None,
    ) -> None:
        meta: Metadata = {**(metadata or {}), "tenant_id": tenant_id, "ts": ts.timestamp()}
        self._col(tenant_id).upsert(
            ids=[record_id], embeddings=[vec.tolist()], documents=[text], metadatas=[meta]
        )

    def query(
        self, tenant_id: str, q_vec: np.ndarray, k: int, filters: Metadata | None = None
    ) -> list[Hit]:
        if k <= 0:
            return []
        res = self._col(tenant_id).query(
            query_embeddings=[q_vec.tolist()],
            n_results=k,
            where=self._where(tenant_id, filters),
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        ids = res["ids"][0] if res["ids"] else []
        out: list[Hit] = []
        for i, rid in enumerate(ids):
            meta = dict(res["metadatas"][0][i] or {})
            out.append(
                Hit(
                    id=rid,
                    text=res["documents"][0][i] or "",
                    score=1.0 - float(res["distances"][0][i]),
                    ts=datetime.fromtimestamp(float(meta.get("ts", 0.0)), tz=UTC),
                    metadata=meta,
                    embedding=np.asarray(res["embeddings"][0][i], dtype=np.float32),
                )
            )
        return out

    def count(self, tenant_id: str) -> int:
        return len(self._col(tenant_id).get(where=self._where(tenant_id), include=[])["ids"])

    def vectors(self, tenant_id: str) -> tuple[list[str], np.ndarray]:
        """All of a learner's record ids + embeddings (for the M4 sketch builder)."""
        got = self._col(tenant_id).get(where=self._where(tenant_id), include=["embeddings"])
        embs = got["embeddings"]
        arr = (
            np.asarray(embs, dtype=np.float32)
            if embs is not None and len(embs)
            else np.zeros((0, 0))
        )
        return list(got["ids"]), arr

    def vectors_and_texts(self, tenant_id: str) -> tuple[list[str], np.ndarray, list[str]]:
        """Ids, embeddings and documents for one learner (sketch builder input)."""
        got = self._col(tenant_id).get(
            where=self._where(tenant_id), include=["embeddings", "documents"]
        )
        embs = got["embeddings"]
        if embs is None or len(embs) == 0:
            return [], np.zeros((0, 0), dtype=np.float32), []
        return list(got["ids"]), np.asarray(embs, dtype=np.float32), list(got["documents"] or [])

    def delete_tenant(self, tenant_id: str) -> int:
        col = self._col(tenant_id)
        ids = col.get(where=self._where(tenant_id), include=[])["ids"]
        if ids:
            col.delete(ids=ids)
        return len(ids)
