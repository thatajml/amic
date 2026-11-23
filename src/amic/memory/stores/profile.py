"""M3 learner profile (brief section 7): one small structured record per learner.

Holds per-concept mastery (an EWMA of evidence in [0, 1], prior 0.5), open misconceptions, and
the latest goals, preferences, projects and personal context. It is updated by the consolidation
worker (never on the response path) and injected as ONE short block (~100-200 tokens) on SHALLOW
and DEEP routes when RouteConfig lists "M3".

Evidence, per consolidated turn:
- concepts are found in the student's query and in extracted facts by a curriculum lexicon
  (CONCEPT_PATTERNS; the education domain's 18 Python concepts);
- an extracted `misconception` fact on a concept -> observation 0.0 (and it is kept as open);
- a `progress` fact -> 1.0 (and closes open misconceptions on the same concept);
- without facts, lexical cues in the query: struggle ("error", "doesn't work", "stuck") -> 0.25,
  success ("got it", "works now", "makes sense") -> 0.85; a plain question only counts exposure.
mastery <- (1 - alpha) * mastery + alpha * observation.

Storage: a key-value backend (SQLite file, or Redis next to the sessions), JSON per learner.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

CONCEPT_PATTERNS: dict[str, str] = {
    "variables and types": r"\bvariables?\b|\btypes?\b|\bint\b|\bfloat\b|\bstr\b|\bcasting\b",
    "input and type conversion": r"\binput\(?|\bconvert|\bconversion\b|\bint\(",
    "strings": r"\bstrings?\b|\bslic(e|ing)\b|\bf-strings?\b|\bimmutable\b",
    "conditionals": r"\bif\b.*\b(else|elif)\b|\bconditionals?\b|\belif\b|\bboolean\b",
    "while loops": r"\bwhile\b|\binfinite loop\b",
    "for loops and range": r"\bfor loops?\b|\brange\(?|\bloop(s|ing)? over\b|\biterat",
    "lists": r"\blists?\b|\bappend\b|\bextend\b|\bindex(es|ing)?\b",
    "dictionaries": r"\bdict(ionary|ionaries)?\b|\bkeyerror\b|\.get\(|\bkeys?\b",
    "functions and return values": r"\bfunctions?\b|\breturn(s|ing)?\b|\bdef\b|\bparameters?\b"
    r"|\barguments?\b|\bdefault argument",
    "variable scope": r"\bscope\b|\bglobal\b|\blocal variable",
    "exceptions and try/except": r"\bexceptions?\b|\btry\b|\bexcept\b|\braise\b",
    "file reading and writing": r"\bfiles?\b|\bopen\(|\bcsv\b|\bwith open\b",
    "list comprehensions": r"\bcomprehensions?\b",
    "reading tracebacks and debugging": r"\btracebacks?\b|\bdebug(ging)?\b|\berror message",
}
_CONCEPT_RX = {c: re.compile(p, re.I) for c, p in CONCEPT_PATTERNS.items()}
STRUGGLE = re.compile(
    r"\b(error|wrong|doesn'?t work|does not work|not working|stuck|confus\w*|bug|broke|"
    r"keeps? (giving|getting|crashing)|why (does|is|do)n'?t|instead of)\b",
    re.I,
)
SUCCESS = re.compile(
    r"\b(got it|makes sense|i (now )?understand|now i see|works now|that worked|figured it out|"
    r"i fixed|it worked)\b",
    re.I,
)
TEXT_KINDS = ("goal", "preference", "project", "personal")


def detect_concepts(text: str) -> list[str]:
    return [c for c, rx in _CONCEPT_RX.items() if rx.search(text)]


class ConceptState(BaseModel):
    m: float = 0.5  # mastery estimate (EWMA)
    n: int = 0  # evidence observations
    seen: int = 0  # times the concept came up
    last: float = 0.0  # epoch seconds of the last update


class LearnerProfile(BaseModel):
    tenant_id: str
    mastery: dict[str, ConceptState] = Field(default_factory=dict)
    misconceptions: dict[str, str] = Field(default_factory=dict)  # fact key -> text (open)
    notes: dict[str, dict[str, str]] = Field(default_factory=dict)  # kind -> {key: text}
    updated: float = 0.0

    # ------------------------------------------------------------------ updates
    def _observe(self, concept: str, obs: float | None, ts: float, alpha: float) -> None:
        st = self.mastery.setdefault(concept, ConceptState())
        st.seen += 1
        if obs is not None:
            st.m = (1 - alpha) * st.m + alpha * obs
            st.n += 1
        st.last = max(st.last, ts)

    def observe_turn(
        self,
        query: str,
        ts: datetime,
        facts: Iterable[Any] = (),
        alpha: float = 0.3,
        max_notes: int = 3,
    ) -> bool:
        """Update from one consolidated turn (+ the facts extracted from it). True if changed."""
        t = ts.timestamp()
        before = self.model_dump_json()
        facts = list(facts)
        fact_concepts: set[str] = set()
        for f in facts:
            kind, key, text = f.kind, f.key, f.text
            cs = detect_concepts(f"{key.replace('_', ' ')} {text}")
            if kind == "misconception":
                self.misconceptions[key] = text
                for c in cs:
                    self._observe(c, 0.0, t, alpha)
            elif kind == "progress":
                for c in cs:
                    self._observe(c, 1.0, t, alpha)
                topic = key.split(":", 1)[-1]
                for mk in list(self.misconceptions):
                    if mk.split(":", 1)[-1] == topic or set(
                        detect_concepts(mk.replace("_", " ") + " " + self.misconceptions[mk])
                    ) & set(cs):
                        del self.misconceptions[mk]
            elif kind in TEXT_KINDS:
                bucket = self.notes.setdefault(kind, {})
                bucket.pop(key, None)
                bucket[key] = text  # latest last
                while len(bucket) > max_notes:
                    bucket.pop(next(iter(bucket)))
            fact_concepts |= set(cs) if kind in ("misconception", "progress") else set()
        lexical = 0.25 if STRUGGLE.search(query) else 0.85 if SUCCESS.search(query) else None
        for c in detect_concepts(query):
            if c not in fact_concepts:  # facts are stronger evidence than lexical cues
                self._observe(c, lexical, t, alpha)
        changed = self.model_dump_json() != before
        if changed:
            self.updated = max(self.updated, t)
        return changed

    def decay(self, now: datetime, half_life_days: float, prior: float = 0.5) -> bool:
        """Forgetting curve: mastery drifts back to the prior with the time since evidence."""
        changed = False
        for st in self.mastery.values():
            if st.last <= 0 or st.n == 0:
                continue
            dt_days = max(0.0, (now.timestamp() - st.last) / 86400)
            f = 0.5 ** (dt_days / half_life_days)
            m = prior + (st.m - prior) * f
            if abs(m - st.m) > 1e-4:
                st.m, st.last, changed = m, now.timestamp(), True
        return changed

    # ------------------------------------------------------------------ rendering
    def is_empty(self) -> bool:
        return not (self.mastery or self.misconceptions or self.notes)

    def render(self, max_tokens: int = 200) -> str:
        """The M3 block. Priority order (truncated at the budget): the latest goal, project and
        preference; concepts still being learned; open misconceptions; strong concepts; other
        notes; concepts only asked about."""

        def latest(kind: str, n: int) -> list[str]:
            return list(reversed(list(self.notes.get(kind, {}).values())))[:n]

        parts: list[str] = []
        for kind, label in (("goal", "Goal"), ("project", "Project"), ("preference", "Prefers")):
            if texts := latest(kind, 1):
                parts.append(f"{label}: {texts[0]}")
        evid = [(c, s) for c, s in self.mastery.items() if s.n > 0]
        weak = sorted((x for x in evid if x[1].m < 0.45), key=lambda x: x[1].m)
        strong = sorted((x for x in evid if x[1].m > 0.65), key=lambda x: -x[1].m)
        if weak:
            parts.append("Still working on: " + ", ".join(f"{c} ({s.m:.1f})" for c, s in weak))
        if self.misconceptions:
            recent = list(self.misconceptions.values())[-3:]
            parts.append("Open misconceptions: " + "; ".join(reversed(recent)))
        if strong:
            parts.append("Comfortable with: " + ", ".join(c for c, _ in strong))
        if notes := latest("personal", 2):
            parts.append("Note: " + "; ".join(notes))
        studied = [c for c, s in self.mastery.items() if s.n == 0 and s.seen > 0]
        if studied:
            parts.append("Has asked about: " + ", ".join(studied))
        text = "Learner profile. " + ". ".join(p.rstrip(". ") for p in parts) + "."
        limit = max_tokens * 4  # ~4 chars/token, the assembler's estimate
        return text if len(text) <= limit else text[: limit - 3].rsplit(" ", 1)[0] + "..."


# ---------------------------------------------------------------------- storage
class _KV(Protocol):
    def get(self, key: str) -> str | None: ...
    def put(self, key: str, value: str) -> None: ...
    def delete(self, key: str) -> int: ...
    def keys(self) -> list[str]: ...


class SqliteKV:
    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS profiles (k TEXT PRIMARY KEY, v TEXT)")
            self._db.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM profiles WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO profiles (k, v) VALUES (?, ?)", (key, value))
            self._db.commit()

    def delete(self, key: str) -> int:
        with self._lock:
            n = self._db.execute("DELETE FROM profiles WHERE k=?", (key,)).rowcount
            self._db.commit()
        return n

    def keys(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._db.execute("SELECT k FROM profiles ORDER BY k")]


class RedisKV:
    PREFIX = "amic:profile:"

    def __init__(self, redis: Any) -> None:
        self._r = redis

    def get(self, key: str) -> str | None:
        v = self._r.get(self.PREFIX + key)
        return v.decode() if isinstance(v, bytes) else v

    def put(self, key: str, value: str) -> None:
        self._r.set(self.PREFIX + key, value)

    def delete(self, key: str) -> int:
        return int(self._r.delete(self.PREFIX + key))

    def keys(self) -> list[str]:
        out = []
        for k in self._r.scan_iter(match=self.PREFIX + "*"):
            k = k.decode() if isinstance(k, bytes) else k
            out.append(k.removeprefix(self.PREFIX))
        return sorted(out)


class ProfileStore:
    def __init__(self, kv: _KV) -> None:
        self._kv = kv

    def get(self, tenant_id: str) -> LearnerProfile | None:
        raw = self._kv.get(tenant_id)
        return LearnerProfile.model_validate_json(raw) if raw else None

    def get_or_new(self, tenant_id: str) -> LearnerProfile:
        return self.get(tenant_id) or LearnerProfile(tenant_id=tenant_id)

    def put(self, profile: LearnerProfile) -> None:
        if not profile.updated:
            profile.updated = datetime.now(UTC).timestamp()
        self._kv.put(profile.tenant_id, profile.model_dump_json())

    def delete(self, tenant_id: str) -> int:
        return self._kv.delete(tenant_id)

    def tenants(self) -> list[str]:
        return self._kv.keys()

    def dump(self, tenant_id: str) -> dict[str, Any]:
        p = self.get(tenant_id)
        return json.loads(p.model_dump_json()) if p else {}
