"""Query/record embedding. The request path embeds the query ONCE (P2) and reuses the vector."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def encode(self, text: str) -> np.ndarray: ...

    def encode_many(self, texts: list[str]) -> np.ndarray: ...


class STEmbedder:
    """Sentence-Transformers model, L2-normalised float32 output (dot product = cosine)."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.encode("dimension probe").shape[0])  # also warms the model up

    def encode(self, text: str) -> np.ndarray:
        v = self._model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(v, dtype=np.float32)

    def encode_many(self, texts: list[str]) -> np.ndarray:
        v = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64
        )
        return np.asarray(v, dtype=np.float32)


_TOKEN = re.compile(r"[a-z0-9_]+")


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for tests and CI (no model download).

    Texts sharing words get positive cosine similarity, so retrieval tests stay meaningful.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN.findall(text.lower()):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = float(np.linalg.norm(v))
        if n == 0:
            v[0] = 1.0
            return v
        return v / n

    def encode_many(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.encode(t) for t in texts]) if texts else np.zeros((0, self.dim))
