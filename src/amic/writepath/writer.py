"""Long-term memory writer (L7). Runs AFTER the response is produced (P5).

This sprint: embed the turn and upsert it into M1. The full pipeline (salience filter, fact
extraction, conflict resolution, M3 EWMA, sketch rebuild, cache invalidation) is future work;
`on_written` hooks let the sketch builder (Day 4) subscribe without changing callers.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from amic.memory.embedder import Embedder
from amic.memory.stores.episodic import EpisodicStore

_MAX_ANSWER_CHARS = 400


def record_text(query: str, answer: str) -> str:
    a = answer if len(answer) <= _MAX_ANSWER_CHARS else answer[:_MAX_ANSWER_CHARS] + "..."
    return f"Student asked: {query}\nTutor answered: {a}"


class MemoryWriter:
    def __init__(self, embedder: Embedder, episodic: EpisodicStore) -> None:
        self._embedder = embedder
        self._episodic = episodic
        self.on_written: list[Callable[[str], None]] = []

    def write_turn(
        self,
        tenant_id: str,
        session_id: str,
        record_id: str,
        query: str,
        answer: str,
        ts: datetime,
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        text = record_text(query, answer)
        self._episodic.add(
            tenant_id,
            record_id,
            text,
            self._embedder.encode(text),
            ts,
            {**(metadata or {}), "session_id": session_id, "kind": "turn"},
        )
        for hook in self.on_written:
            hook(tenant_id)
