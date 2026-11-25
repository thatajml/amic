"""Consolidation steps (L7, brief section 7): salience filter -> LLM fact extraction ->
conflict resolution with validity intervals -> M2.

Resolution is deterministic (no LLM): a new fact is compared with the learner's CURRENT facts.
- same key, near-identical text (cosine >= same_key_sim): refresh the existing fact;
- same key, different content: the new fact supersedes it (old interval closed at the new
  fact's time, new fact inserted) - e.g. a misconception that has since been fixed;
- different key but a near-duplicate of a current fact (cosine >= dup_sim): skip;
- otherwise: insert.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from amic.generation.gateway.base import LLMGateway
from amic.memory.stores.semantic import SemanticStore

KINDS = ("misconception", "progress", "goal", "preference", "project", "personal", "other")
PHATIC = re.compile(
    r"^\s*(?:thanks?(?: you)?|thank you so much|ok(?:ay)?|cool|great|got it"
    r"|bye|hi|hello|yes|no|sure)"
    r"[\s!.?]*$",
    re.I,
)

PROMPT = """Extract durable facts about the STUDENT from this tutoring conversation (date: {date}).
Keep only what a tutor should remember for future sessions: misconceptions or mistakes the
student made, topics they have now understood (progress), goals, learning preferences, projects,
and personal context they shared. Ignore general Python knowledge and anything the tutor said
that is not about this student. Write each fact as one short sentence about the student.

{transcript}

For each fact give a short stable key "<kind>:<topic>" in snake_case (kinds: {kinds}), e.g.
"misconception:range_end", "goal", "preference:explanations", "project". Use the SAME key when a
later fact updates an earlier one on the same topic. Confidence: 0 to 1.
Return JSON only: {{"facts": [{{"key": "...", "kind": "...", "text": "...", "confidence": 0.9}}]}}
Return {{"facts": []}} if there is nothing worth remembering."""


@dataclass(frozen=True)
class ExtractedFact:
    key: str
    kind: str
    text: str
    confidence: float


def is_salient(query: str, answer: str, min_words: int = 4) -> bool:
    """Worth an extraction call? Phatic or very short student turns are not."""
    q = query.strip()
    return not PHATIC.match(q) and len(q.split()) >= min_words


def _norm_key(key: str, kind: str) -> str:
    k = re.sub(r"[^a-z0-9_:]+", "_", key.strip().lower()).strip("_:")
    return k or kind


def parse_facts(raw: str) -> list[ExtractedFact]:
    j: Any = json.loads(raw)
    if isinstance(j, dict):
        j = j.get("facts", [])
    out = []
    for f in j if isinstance(j, list) else []:
        if not isinstance(f, dict) or not str(f.get("text", "")).strip():
            continue
        kind = str(f.get("kind", "other")).strip().lower()
        kind = kind if kind in KINDS else "other"
        try:
            conf = min(1.0, max(0.0, float(f.get("confidence", 0.7))))
        except (TypeError, ValueError):
            conf = 0.7
        out.append(
            ExtractedFact(
                _norm_key(str(f.get("key", kind)), kind), kind, str(f["text"]).strip(), conf
            )
        )
    return out


class FactExtractor:
    def __init__(self, llm: LLMGateway, max_tokens: int = 500) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.calls = 0

    def extract(self, exchanges: list[tuple[str, str]], when: datetime) -> list[ExtractedFact]:
        transcript = "\n".join(
            f"STUDENT: {q}\nTUTOR: {a[:600]}" if a else f"STUDENT: {q}" for q, a in exchanges
        )
        prompt = PROMPT.format(
            date=f"{when:%Y-%m-%d}", transcript=transcript, kinds=", ".join(KINDS)
        )
        self.calls += 1
        raw = self.llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.0,
            seed=0,
            json_mode=True,
        ).text
        try:
            return parse_facts(raw)
        except (ValueError, TypeError):
            return []


class ConflictResolver:
    def __init__(
        self,
        store: SemanticStore,
        embedder: Any,
        same_key_sim: float = 0.9,
        dup_sim: float = 0.92,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.same_key_sim = same_key_sim
        self.dup_sim = dup_sim

    def resolve(
        self, tenant_id: str, facts: list[ExtractedFact], when: datetime, source: str = ""
    ) -> dict[str, int]:
        counts = {"insert": 0, "update": 0, "refresh": 0, "skip": 0}
        for f in facts:
            vec = self.embedder.encode(f.text)
            same = self.store.current_by_key(tenant_id, f.key)
            if same:
                best_id, best_sim = max(
                    ((fid, float(v @ vec)) for fid, v, _ in same), key=lambda t: t[1]
                )
                if best_sim >= self.same_key_sim:
                    self.store.refresh(tenant_id, best_id, when, f.confidence)
                    counts["refresh"] += 1
                    continue
                for fid, _, _ in same:  # superseded: close, never delete
                    self.store.close(tenant_id, fid, when)
                self._insert(tenant_id, f, vec, when, source)
                counts["update"] += 1
                continue
            near = self.store.current(tenant_id, vec, 1)
            if near and near[0].score >= self.dup_sim:
                counts["skip"] += 1
                continue
            self._insert(tenant_id, f, vec, when, source)
            counts["insert"] += 1
        return counts

    def _insert(
        self, tenant_id: str, f: ExtractedFact, vec: Any, when: datetime, source: str
    ) -> None:
        self.store.add(
            tenant_id,
            f"{tenant_id}-f-{uuid.uuid4().hex[:12]}",
            f.text,
            vec,
            key=f.key,
            kind=f.kind,
            confidence=f.confidence,
            valid_from=when,
            source=source,
        )
