"""SessionManager: M0 working buffer + per-learner session metadata in Redis.

M0 is written synchronously on the response path (one Redis RPUSH), so turn N is visible
to turn N+1 immediately. Long-term stores have no read-your-writes guarantee (brief section 7).

Memory version: a per-learner number that changes on every long-term write and on erasure.
The M4 sketch cache and the decision cache key on it, so a write made by another process (the
consolidation worker) invalidates them without any message passing. Erasure sets a NEW value
rather than deleting it, so an entry cached before the erasure can never match again.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from amic.contracts.models import Turn

_KEEP = 50  # turns retained per session in Redis; prompts use the last `m0_turns`


def _next_version(prev: int) -> int:
    """Strictly increasing, and wall-clock based so it survives process restarts."""
    return max(prev + 1, time.time_ns())


class SessionManager:
    def __init__(self, redis: Any, m0_turns: int = 6, prefix: str = "amic") -> None:
        self._r = redis
        self._n = m0_turns
        self._p = prefix

    def _turns_key(self, session_id: str) -> str:
        return f"{self._p}:sess:{session_id}:turns"

    def _tenant_key(self, tenant_id: str) -> str:
        return f"{self._p}:tenant:{tenant_id}"

    def get_turns(self, session_id: str, n: int | None = None) -> list[Turn]:
        n = self._n if n is None else n
        if n <= 0:
            return []
        raw = self._r.lrange(self._turns_key(session_id), -n, -1)
        return [Turn.model_validate_json(x) for x in raw]

    def append(self, session_id: str, *turns: Turn) -> None:
        key = self._turns_key(session_id)
        pipe = self._r.pipeline()
        pipe.rpush(key, *[t.model_dump_json() for t in turns])
        pipe.ltrim(key, -_KEEP, -1)
        pipe.execute()

    def touch(self, tenant_id: str, session_id: str, ts: datetime | None = None) -> dict[str, Any]:
        """Record activity. Returns the PREVIOUS metadata (S2 needs 'days since last session')."""
        ts = ts or datetime.now(UTC)
        key = self._tenant_key(tenant_id)
        prev = self.tenant_meta(tenant_id)
        mapping: dict[str, str | float] = {
            "last_session_id": session_id,
            "last_seen": ts.timestamp(),
        }
        if prev.get("last_session_id") != session_id:
            mapping["prev_session_end"] = prev.get("last_seen", ts.timestamp())
            self._r.hincrby(key, "n_sessions", 1)
        self._r.hset(key, mapping=mapping)
        self._r.sadd(f"{key}:sessions", session_id)
        return prev

    def tenant_meta(self, tenant_id: str) -> dict[str, Any]:
        raw = self._r.hgetall(self._tenant_key(tenant_id))
        out: dict[str, Any] = {}
        for k, v in raw.items():
            k = k.decode() if isinstance(k, bytes) else k
            v = v.decode() if isinstance(v, bytes) else v
            out[k] = v if k == "last_session_id" else float(v)
        return out

    def erase_tenant(self, tenant_id: str) -> None:
        key = self._tenant_key(tenant_id)
        sessions = self._r.smembers(f"{key}:sessions")
        for s in sessions:
            s = s.decode() if isinstance(s, bytes) else s
            self._r.delete(self._turns_key(s))
        self._r.delete(key, f"{key}:sessions")
        # tombstone: queued/in-flight writes from before this moment are dropped by the worker
        self._r.set(f"{self._p}:tenant:{tenant_id}:erased_at", repr(time.time()))
        self.bump_mem_version(tenant_id)  # a new value, never a reset (see module docstring)

    def erased_at(self, tenant_id: str) -> float:
        """Epoch seconds of the learner's last erasure (0 = never erased)."""
        v = self._r.get(f"{self._p}:tenant:{tenant_id}:erased_at")
        return float(v) if v is not None else 0.0

    # ----------------------------------------------------------------- memory version
    def _ver_key(self, tenant_id: str) -> str:
        return f"{self._p}:tenant:{tenant_id}:memver"

    def mem_version(self, tenant_id: str) -> int:
        v = self._r.get(self._ver_key(tenant_id))
        return int(v) if v is not None else 0

    def bump_mem_version(self, tenant_id: str) -> int:
        v = _next_version(self.mem_version(tenant_id))
        self._r.set(self._ver_key(tenant_id), v)
        return v


class SqliteSessionManager:
    """Same contract as SessionManager, backed by one SQLite file (no Docker needed).

    Used for offline labelling/experiments on the dev laptop, where Docker Desktop's WSL VM was
    killed under memory pressure (2026-11-18). The service keeps Redis by default.
    """

    def __init__(self, path: str = "data/sessions.sqlite", m0_turns: int = 6) -> None:
        import sqlite3
        import threading
        from pathlib import Path

        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(
            """CREATE TABLE IF NOT EXISTS turns (
                   id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, body TEXT);
               CREATE INDEX IF NOT EXISTS turns_s ON turns(session_id, id);
               CREATE TABLE IF NOT EXISTS tenant (
                   tenant_id TEXT PRIMARY KEY, last_session_id TEXT, last_seen REAL,
                   prev_session_end REAL, n_sessions REAL);
               CREATE TABLE IF NOT EXISTS tenant_sessions (
                   tenant_id TEXT, session_id TEXT, PRIMARY KEY (tenant_id, session_id));
               CREATE TABLE IF NOT EXISTS mem_version (
                   tenant_id TEXT PRIMARY KEY, v INTEGER NOT NULL);
               CREATE TABLE IF NOT EXISTS erased (
                   tenant_id TEXT PRIMARY KEY, ts REAL NOT NULL);"""
        )
        self._lock = threading.Lock()
        self._n = m0_turns

    def get_turns(self, session_id: str, n: int | None = None) -> list[Turn]:
        n = self._n if n is None else n
        if n <= 0:
            return []
        with self._lock:
            rows = self._db.execute(
                "SELECT body FROM turns WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, n),
            ).fetchall()
        return [Turn.model_validate_json(r[0]) for r in reversed(rows)]

    def append(self, session_id: str, *turns: Turn) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT INTO turns (session_id, body) VALUES (?, ?)",
                [(session_id, t.model_dump_json()) for t in turns],
            )
            self._db.execute(
                "DELETE FROM turns WHERE session_id=? AND id NOT IN "
                "(SELECT id FROM turns WHERE session_id=? ORDER BY id DESC LIMIT ?)",
                (session_id, session_id, _KEEP),
            )

    def touch(self, tenant_id: str, session_id: str, ts: datetime | None = None) -> dict[str, Any]:
        ts = ts or datetime.now(UTC)
        prev = self.tenant_meta(tenant_id)
        new_session = prev.get("last_session_id") != session_id
        with self._lock:
            self._db.execute(
                """INSERT INTO tenant VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(tenant_id) DO UPDATE SET
                     last_session_id=excluded.last_session_id, last_seen=excluded.last_seen,
                     prev_session_end=CASE WHEN ? THEN excluded.prev_session_end
                                           ELSE tenant.prev_session_end END,
                     n_sessions=tenant.n_sessions + ?""",
                (
                    tenant_id,
                    session_id,
                    ts.timestamp(),
                    prev.get("last_seen", ts.timestamp()) if new_session else None,
                    1.0 if new_session else 0.0,
                    int(new_session),
                    1.0 if new_session else 0.0,
                ),
            )
            self._db.execute(
                "INSERT OR IGNORE INTO tenant_sessions VALUES (?, ?)", (tenant_id, session_id)
            )
        return prev

    def tenant_meta(self, tenant_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute(
                "SELECT last_session_id, last_seen, prev_session_end, n_sessions "
                "FROM tenant WHERE tenant_id=?",
                (tenant_id,),
            ).fetchone()
        if row is None:
            return {}
        keys = ("last_session_id", "last_seen", "prev_session_end", "n_sessions")
        return {k: v for k, v in zip(keys, row, strict=True) if v is not None}

    def erase_tenant(self, tenant_id: str) -> None:
        with self._lock:
            sess = [
                r[0]
                for r in self._db.execute(
                    "SELECT session_id FROM tenant_sessions WHERE tenant_id=?", (tenant_id,)
                )
            ]
            self._db.executemany("DELETE FROM turns WHERE session_id=?", [(s,) for s in sess])
            self._db.execute("DELETE FROM tenant_sessions WHERE tenant_id=?", (tenant_id,))
            self._db.execute("DELETE FROM tenant WHERE tenant_id=?", (tenant_id,))
            self._db.execute(
                "INSERT OR REPLACE INTO erased VALUES (?, ?)", (tenant_id, time.time())
            )
        self.bump_mem_version(tenant_id)  # a new value, never a reset (see module docstring)

    def erased_at(self, tenant_id: str) -> float:
        """Epoch seconds of the learner's last erasure (0 = never erased)."""
        with self._lock:
            row = self._db.execute(
                "SELECT ts FROM erased WHERE tenant_id=?", (tenant_id,)
            ).fetchone()
        return float(row[0]) if row else 0.0

    def mem_version(self, tenant_id: str) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT v FROM mem_version WHERE tenant_id=?", (tenant_id,)
            ).fetchone()
        return int(row[0]) if row else 0

    def bump_mem_version(self, tenant_id: str) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT v FROM mem_version WHERE tenant_id=?", (tenant_id,)
            ).fetchone()
            v = _next_version(int(row[0]) if row else 0)
            self._db.execute("INSERT OR REPLACE INTO mem_version VALUES (?, ?)", (tenant_id, v))
        return v
