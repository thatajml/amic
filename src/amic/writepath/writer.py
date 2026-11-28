"""Long-term memory writer (L7). Runs AFTER the response is produced (P5).

Embeds a turn (or an explicit /remember note) and upserts it into M1. The rest of the
consolidation pipeline (salience, fact extraction, conflict resolution, M3 EWMA) is in
writepath/worker.py; `on_written` hooks bump the learner's memory version, which rebuilds the M4
sketch and invalidates cached decisions in every process.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from amic.memory.embedder import Embedder
from amic.memory.stores.episodic import EpisodicStore

_MAX_ANSWER_CHARS = 400
_MAX_QUERY_CHARS = 2000


def record_text(query: str, answer: str) -> str:
    a = answer if len(answer) <= _MAX_ANSWER_CHARS else answer[:_MAX_ANSWER_CHARS] + "..."
    q = query if len(query) <= _MAX_QUERY_CHARS else query[:_MAX_QUERY_CHARS] + "..."
    return f"Student asked: {q}\nTutor answered: {a}"


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
        self.write_record(
            tenant_id, session_id, record_id, record_text(query, answer), ts, metadata, "turn"
        )

    def write_record(
        self,
        tenant_id: str,
        session_id: str,
        record_id: str,
        text: str,
        ts: datetime,
        metadata: dict[str, str | int | float | bool] | None = None,
        kind: str = "turn",
    ) -> None:
        """Upsert one M1 record (a turn, or an explicit /remember note) and fire the hooks."""
        self._episodic.add(
            tenant_id,
            record_id,
            text,
            self._embedder.encode(text),
            ts,
            {**(metadata or {}), "session_id": session_id, "kind": kind},
        )
        for hook in self.on_written:
            hook(tenant_id)
