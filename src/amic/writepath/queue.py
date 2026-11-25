"""WriteIntentQueue (L7): the response path only enqueues; a ConsolidationWorker writes (P5).

Three backends behind one interface:
- InlineQueue: processes on enqueue (the pre-queue behaviour; tests and frozen-memory
  experiments, where long-term writes are dropped anyway).
- SqliteWriteQueue: a durable table, for the no-Docker dev setup (AMIC_SESSION_BACKEND=sqlite).
- RedisStreamQueue: Redis Streams with a consumer group (XADD / XREADGROUP / XACK), at-least-once
  delivery; messages a crashed consumer left unacknowledged are reclaimed with XAUTOCLAIM.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from amic.contracts.models import WriteIntent


class WriteIntentQueue(Protocol):
    def enqueue(self, intent: WriteIntent) -> str: ...

    def read(self, count: int = 10, block_ms: int = 1000) -> list[tuple[str, WriteIntent]]: ...

    def ack(self, msg_id: str) -> None: ...

    def pending(self) -> int: ...


class InlineQueue:
    """Synchronous: `enqueue` runs the handler immediately (no worker process)."""

    def __init__(self, handler: Callable[[WriteIntent], None]) -> None:
        self.handler = handler
        self._n = 0

    def enqueue(self, intent: WriteIntent) -> str:
        self.handler(intent)
        self._n += 1
        return str(self._n)

    def read(self, count: int = 10, block_ms: int = 1000) -> list[tuple[str, WriteIntent]]:
        return []

    def ack(self, msg_id: str) -> None:
        return None

    def pending(self) -> int:
        return 0


class SqliteWriteQueue:
    """Durable queue in one SQLite file. A read leases rows for `lease_s`; unacked rows return."""

    def __init__(self, path: str = "data/writequeue.sqlite", lease_s: float = 60.0) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS intents (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "body TEXT NOT NULL, leased_until REAL NOT NULL DEFAULT 0, "
            "done INTEGER NOT NULL DEFAULT 0)"
        )
        self._lock = threading.Lock()
        self.lease_s = lease_s

    def enqueue(self, intent: WriteIntent) -> str:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO intents (body) VALUES (?)", (intent.model_dump_json(),)
            )
        return str(cur.lastrowid)

    def read(self, count: int = 10, block_ms: int = 1000) -> list[tuple[str, WriteIntent]]:
        deadline = time.monotonic() + block_ms / 1000
        while True:
            now = time.time()
            with self._lock:
                rows = self._db.execute(
                    "SELECT id, body FROM intents WHERE done=0 AND leased_until < ? "
                    "ORDER BY id LIMIT ?",
                    (now, count),
                ).fetchall()
                if rows:
                    self._db.executemany(
                        "UPDATE intents SET leased_until=? WHERE id=?",
                        [(now + self.lease_s, r[0]) for r in rows],
                    )
            if rows or time.monotonic() >= deadline:
                return [(str(r[0]), WriteIntent.model_validate_json(r[1])) for r in rows]
            time.sleep(0.05)

    def ack(self, msg_id: str) -> None:
        with self._lock:
            self._db.execute("UPDATE intents SET done=1 WHERE id=?", (int(msg_id),))

    def pending(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM intents WHERE done=0").fetchone()[0])


class RedisStreamQueue:
    def __init__(
        self,
        redis: Any,
        stream: str = "amic:writes",
        group: str = "consolidators",
        consumer: str = "worker-1",
        reclaim_idle_ms: int = 60_000,
    ) -> None:
        self._r = redis
        self.stream, self.group, self.consumer = stream, group, consumer
        self.reclaim_idle_ms = reclaim_idle_ms
        try:
            self._r.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception as e:  # noqa: BLE001 - BUSYGROUP: the group already exists
            if "BUSYGROUP" not in str(e):
                raise

    def enqueue(self, intent: WriteIntent) -> str:
        msg_id = self._r.xadd(self.stream, {"body": intent.model_dump_json()})
        return msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)

    @staticmethod
    def _decode(entries: Any) -> list[tuple[str, WriteIntent]]:
        out = []
        for msg_id, fields in entries or []:
            body = fields.get(b"body", fields.get("body"))
            if body is None:
                continue
            mid = msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
            out.append((mid, WriteIntent.model_validate_json(body)))
        return out

    def read(self, count: int = 10, block_ms: int = 1000) -> list[tuple[str, WriteIntent]]:
        # first, messages another consumer took but never acknowledged (it crashed)
        claimed = self._r.xautoclaim(
            self.stream, self.group, self.consumer, self.reclaim_idle_ms, "0-0", count=count
        )
        got = self._decode(claimed[1] if claimed else [])
        if got:
            return got
        res = self._r.xreadgroup(
            self.group, self.consumer, {self.stream: ">"}, count=count, block=block_ms
        )
        return self._decode(res[0][1]) if res else []

    def ack(self, msg_id: str) -> None:
        self._r.xack(self.stream, self.group, msg_id)

    def pending(self) -> int:
        info = self._r.xpending(self.stream, self.group)
        n = info.get("pending", 0) if isinstance(info, dict) else info[0]
        undelivered = 0
        for g in self._r.xinfo_groups(self.stream):
            name = g.get("name", g.get(b"name"))
            if name in (self.group, self.group.encode()):
                undelivered = int(g.get("lag") or 0)
        return int(n) + undelivered
