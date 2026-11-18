"""M4 memory sketch (gate-only; never injected into a prompt).

Per learner: k cluster centroids over their M1 embeddings + a Bloom filter over salient terms.
`affinity(q_vec)` is a handful of dot products: no ANN search, no documents read (P3, S3).
Rebuilt by the write path (MemoryWriter.on_written hook) and cached in-process.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

from amic.memory.stores.episodic import EpisodicStore

_WORD = re.compile(r"[a-z_][a-z0-9_]{2,}")
STOPWORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "you",
        "your",
        "with",
        "that",
        "this",
        "have",
        "what",
        "how",
        "why",
        "can",
        "are",
        "was",
        "but",
        "not",
        "just",
        "like",
        "when",
        "then",
        "from",
        "they",
        "them",
        "there",
        "their",
        "about",
        "into",
        "out",
        "its",
        "it's",
        "would",
        "could",
        "should",
        "did",
        "does",
        "get",
        "got",
        "make",
        "made",
        "use",
        "used",
        "using",
        "want",
        "need",
        "know",
        "think",
        "really",
        "also",
        "some",
        "any",
        "one",
        "two",
        "very",
        "student",
        "asked",
        "tutor",
        "answered",
        "here",
        "let's",
        "lets",
        "sure",
        "okay",
        "yes",
        "thanks",
    ]
)


def terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in STOPWORDS}


class BloomFilter:
    def __init__(self, n_bits: int = 8192, n_hashes: int = 4) -> None:
        self.n_bits = n_bits
        self.k = n_hashes
        self.bits = np.zeros(n_bits, dtype=bool)

    def _idx(self, item: str) -> list[int]:
        h = hashlib.blake2b(item.encode(), digest_size=16).digest()
        a, b = int.from_bytes(h[:8], "little"), int.from_bytes(h[8:], "little")
        return [(a + i * b) % self.n_bits for i in range(self.k)]

    def add(self, item: str) -> None:
        self.bits[self._idx(item)] = True

    def __contains__(self, item: str) -> bool:
        return bool(self.bits[self._idx(item)].all())


@dataclass
class MemorySketch:
    centroids: np.ndarray  # (k, d), unit rows
    weights: np.ndarray  # (k,) cluster sizes
    bloom: BloomFilter
    n_records: int
    vocab_size: int = 0
    _term_df: dict[str, int] = field(default_factory=dict, repr=False)

    def affinity(self, q_vec: np.ndarray) -> tuple[float, float]:
        """(max_sim, margin): best centroid cosine and its gap to the runner-up."""
        if self.centroids.shape[0] == 0:
            return 0.0, 0.0
        sims = self.centroids @ q_vec
        if sims.shape[0] == 1:
            return float(sims[0]), float(sims[0])
        top2 = np.partition(sims, -2)[-2:]
        return float(top2[1]), float(top2[1] - top2[0])

    def term_hits(self, text: str) -> tuple[int, int]:
        """(#query terms found in the learner's memory, #query terms)."""
        ts = terms(text)
        return sum(1 for t in ts if t in self.bloom), len(ts)


def _kmeans(x: np.ndarray, k: int, iters: int = 15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Spherical k-means (cosine) with k-means++ init; small n, so plain numpy is enough."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    c = [x[rng.integers(n)]]
    for _ in range(1, k):
        d = 1.0 - np.max(x @ np.stack(c).T, axis=1)
        p = np.clip(d, 0, None)
        p = p / p.sum() if p.sum() > 0 else np.full(n, 1 / n)
        c.append(x[rng.choice(n, p=p)])
    cent = np.stack(c)
    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        assign = np.argmax(x @ cent.T, axis=1)
        for j in range(k):
            m = x[assign == j]
            if len(m):
                v = m.sum(axis=0)
                cent[j] = v / (np.linalg.norm(v) or 1.0)
    weights = np.bincount(assign, minlength=k).astype(np.float32)
    return cent.astype(np.float32), weights


def build_sketch(vectors: np.ndarray, texts: list[str], max_k: int = 8) -> MemorySketch:
    n = vectors.shape[0]
    bloom = BloomFilter()
    vocab: set[str] = set()
    for t in texts:
        for w in terms(t):
            bloom.add(w)
            vocab.add(w)
    if n == 0:
        return MemorySketch(np.zeros((0, 0), np.float32), np.zeros(0), bloom, 0, len(vocab))
    k = max(1, min(max_k, n // 5 or 1))
    cent, w = _kmeans(vectors.astype(np.float32), k)
    return MemorySketch(cent, w, bloom, n, len(vocab))


class EpisodicSketchStore:
    """SketchStore over the episodic store. `get` returns None when a learner has no memory (R1)."""

    def __init__(self, episodic: EpisodicStore, max_k: int = 8) -> None:
        self._episodic = episodic
        self._max_k = max_k
        self._cache: dict[str, MemorySketch | None] = {}
        self.builds = 0

    def get(self, user_id: str) -> MemorySketch | None:
        if user_id not in self._cache:
            self.rebuild(user_id)
        return self._cache[user_id]

    def rebuild(self, user_id: str) -> None:
        ids, vecs, texts = self._episodic.vectors_and_texts(user_id)
        self.builds += 1
        self._cache[user_id] = build_sketch(vecs, texts, self._max_k) if ids else None

    def invalidate(self, user_id: str) -> None:
        """Called by the write path; next `get` rebuilds lazily."""
        self._cache.pop(user_id, None)
