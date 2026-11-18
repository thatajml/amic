"""SessionManager: M0 working buffer + per-learner session metadata in Redis.

M0 is written synchronously on the response path (one Redis RPUSH), so turn N is visible
to turn N+1 immediately. Long-term stores have no read-your-writes guarantee (brief section 7).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from amic.contracts.models import Turn

_KEEP = 50  # turns retained per session in Redis; prompts use the last `m0_turns`


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
