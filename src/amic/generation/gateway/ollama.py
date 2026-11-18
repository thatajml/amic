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
    ) -> None:
        import ollama

        self.model = model
        self._client = ollama.Client(host=base_url)
        self._defaults = {"temperature": temperature, "seed": seed, "num_predict": max_tokens}
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
