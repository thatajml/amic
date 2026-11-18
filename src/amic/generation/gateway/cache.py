"""Response cache for EXPERIMENTS ONLY (never in the service).

Arms share one deterministic pipeline, so when two arms build a byte-identical prompt at the
same seed (e.g. AMIC chose DEEP, exactly like `always`), the model output is the same. The cache
stores the LLMResult *including the timings measured on first generation*, so latency tables
stay faithful while GPU time drops ~4x. Everything else (embedding, gate, retrieval) is measured
live on every run. Documented in docs/results.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path

from amic.generation.gateway.base import LLMGateway, LLMResult, Message


class CachingGateway:
    def __init__(self, inner: LLMGateway, path: str | Path = "data/llm_cache.sqlite") -> None:
        self.inner = inner
        self.model = inner.model
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _key(self, messages: list[Message], **opts: object) -> str:
        blob = json.dumps({"m": self.model, "msgs": messages, "opts": opts}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        k = self._key(
            messages, max_tokens=max_tokens, temperature=temperature, seed=seed, json=json_mode
        )
        with self._lock:
            row = self._db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        if row:
            self.hits += 1
            return LLMResult(**json.loads(row[0]))
        self.misses += 1
        res = self.inner.chat(
            messages, max_tokens=max_tokens, temperature=temperature, seed=seed, json_mode=json_mode
        )
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?)", (k, json.dumps(asdict(res)))
            )
            self._db.commit()
        return res
