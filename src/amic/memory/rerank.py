"""DEEP-route helpers (brief section 7): cross-encoder rerank and one hop of entity expansion.

The cross-encoder (default `cross-encoder/ms-marco-MiniLM-L-6-v2`, ~90 MB, sentence-transformers)
scores (query, memory) pairs jointly, which a bi-encoder cosine cannot. It loads lazily on the
first DEEP request that asks for rerank, so SKIP/SHALLOW never pay for it.

Entity hop: the entities named in the top candidates (curriculum concepts, plus capitalised
names for open-domain chats) that the query itself does not name are embedded together and used
as a second query, which pulls in memories one association away (e.g. the project a misconception
was made in). One hop only, never recursive.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterable
from typing import Any

from amic.memory.stores.profile import detect_concepts

Reranker = Callable[[str, list[str]], list[float]]

_NAME = re.compile(r"\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})*\b")
_STOP_NAMES = {"Student", "Tutor", "User", "Assistant", "Python"} | {
    w.capitalize()
    for w in [
        "the",
        "this",
        "that",
        "these",
        "those",
        "what",
        "how",
        "why",
        "when",
        "where",
        "which",
        "who",
        "here",
        "there",
        "let",
        "yes",
        "no",
        "now",
        "but",
        "and",
        "or",
        "ok",
        "okay",
        "hey",
        "hi",
        "great",
        "good",
        "sure",
        "thanks",
        "you",
        "your",
        "it",
        "its",
        "in",
        "if",
        "for",
        "so",
        "also",
        "then",
        "try",
        "just",
        "like",
        "remember",
        "once",
        "instead",
        "note",
        "see",
        "look",
        "first",
        "next",
        "finally",
        "well",
        "because",
        "since",
        "while",
        "with",
        "without",
        "after",
        "before",
        "every",
        "each",
        "all",
        "any",
        "some",
        "maybe",
        "perhaps",
        "think",
        "got",
    ]
}


def names(text: str) -> set[str]:
    """Capitalised names that do not start a sentence (a crude, dependency-free NER)."""
    out = set()
    for m in _NAME.finditer(text):
        before = text[: m.start()].rstrip()
        if before and before[-1] not in ".!?:\n\"'" and m.group() not in _STOP_NAMES:
            out.add(m.group())
    return out


def hop_entities(texts: Iterable[str], query: str, max_entities: int = 4) -> list[str]:
    """Entities mentioned by the top candidates but not by the query (most frequent first)."""
    in_query = set(detect_concepts(query)) | {n.lower() for n in names(query)}
    counts: dict[str, int] = {}
    for t in texts:
        for e in set(detect_concepts(t)) | names(t):
            if e.lower() not in in_query and e not in in_query:
                counts[e] = counts.get(e, 0) + 1
    return [e for e, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:max_entities]


class CrossEncoderReranker:
    """Callable reranker; thread-safe lazy load."""

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 256) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name, device=self.device, max_length=self.max_length
                )
        return self._model

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self._load().predict([(query, t) for t in texts], show_progress_bar=False)
        return [float(s) for s in scores]
