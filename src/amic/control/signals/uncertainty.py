"""S5 uncertainty signal (brief 6.1, optional): token entropy of a short draft.

The tutor model drafts the first `max_tokens` tokens of the memory-less answer (the SKIP prompt,
greedy) and returns the top-k log-probabilities per token. Per token, the entropy of the top-k
distribution (renormalised) is computed; the features are its mean and max. High entropy = the
model is unsure how to answer without memory. Needs logprobs, so local models only (Ollama);
cost: one short generation (tens to hundreds of ms on the dev GPU), on the gate path.
"""

from __future__ import annotations

import math
from typing import Any

S5_FEATURES = ["s5_entropy_mean", "s5_entropy_max"]


def token_entropy(top_logprobs: list[float]) -> float:
    if not top_logprobs:
        return 0.0
    m = max(top_logprobs)
    ps = [math.exp(lp - m) for lp in top_logprobs]
    z = sum(ps)
    return float(-sum((p / z) * math.log(p / z) for p in ps if p > 0))


def s5_features(per_token: list[list[float]]) -> dict[str, float]:
    ents = [token_entropy(t) for t in per_token]
    if not ents:
        return dict.fromkeys(S5_FEATURES, 0.0)
    return {"s5_entropy_mean": sum(ents) / len(ents), "s5_entropy_max": max(ents)}


class UncertaintySignal:
    """`llm` must offer draft_logprobs(messages, max_tokens, top) (OllamaGateway does)."""

    def __init__(self, llm: Any, prompts: Any, max_tokens: int = 16, top: int = 5) -> None:
        self.llm = llm
        self.prompts = prompts
        self.max_tokens = max_tokens
        self.top = top
        self.last_ms = 0.0

    def features(self, query: str, m0_turns: list[Any], now: Any = None) -> dict[str, float]:
        msgs = self.prompts.build(query, m0_turns, "", now=now)
        _, per_token, ms = self.llm.draft_logprobs(msgs, self.max_tokens, self.top)
        self.last_ms = ms
        return s5_features(per_token)
