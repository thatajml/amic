"""Ollama adapter (local tutor and Tier-2 models). Reports prefill vs decode time separately."""

from __future__ import annotations

import time

from amic.generation.gateway.base import LLMResult, Message


class OllamaGateway:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        seed: int = 0,
        max_tokens: int = 256,
        keep_alive: str = "30m",
        num_ctx: int | None = None,
        timeout_s: float | None = None,
    ) -> None:
        """`num_ctx` None keeps Ollama's default context; calls sharing one model must agree on
        it or Ollama reloads the model. `timeout_s` bounds the HTTP call (Tier 2 fails open)."""
        import ollama

        self.model = model
        kw = {"timeout": timeout_s} if timeout_s is not None else {}
        self._client = ollama.Client(host=base_url, **kw)
        self._defaults: dict[str, float | int] = {
            "temperature": temperature,
            "seed": seed,
            "num_predict": max_tokens,
        }
        if num_ctx is not None:
            self._defaults["num_ctx"] = num_ctx
        self._keep_alive = keep_alive

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        opts = dict(self._defaults)
        if max_tokens is not None:
            opts["num_predict"] = max_tokens
        if temperature is not None:
            opts["temperature"] = temperature
        if seed is not None:
            opts["seed"] = seed
        t = time.perf_counter()
        r = self._client.chat(
            model=self.model,
            messages=messages,
            options=opts,
            format="json" if json_mode else None,
            keep_alive=self._keep_alive,
        )
        wall = (time.perf_counter() - t) * 1000
        return LLMResult(
            text=r["message"]["content"],
            prompt_tokens=int(r.get("prompt_eval_count") or 0),
            completion_tokens=int(r.get("eval_count") or 0),
            total_ms=wall,
            prefill_ms=(r.get("prompt_eval_duration") or 0) / 1e6,
            decode_ms=(r.get("eval_duration") or 0) / 1e6,
        )

    def draft_logprobs(
        self, messages: list[Message], max_tokens: int = 16, top: int = 5
    ) -> tuple[str, list[list[float]], float]:
        """A short greedy draft with the top-`top` log-probabilities of every generated token
        (S5 uncertainty signal; needs Ollama >= 0.12). Returns (text, per-token top logprobs,
        wall ms)."""
        opts = {**self._defaults, "num_predict": max_tokens, "temperature": 0.0}
        t = time.perf_counter()
        r = self._client.chat(
            model=self.model,
            messages=messages,
            options=opts,
            keep_alive=self._keep_alive,
            logprobs=True,
            top_logprobs=top,
        )
        wall = (time.perf_counter() - t) * 1000
        per_token = []
        for tok in r.get("logprobs") or []:
            tops = tok.get("top_logprobs") or [{"logprob": tok.get("logprob", 0.0)}]
            per_token.append([float(x["logprob"]) for x in tops])
        return r["message"]["content"], per_token, wall
