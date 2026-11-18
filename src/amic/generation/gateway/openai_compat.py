"""OpenAI-compatible adapter. Used for the offline judge (Gemini's OpenAI-compatible endpoint)."""

from __future__ import annotations

import os
import time

from tenacity import retry, stop_after_attempt, wait_exponential

from amic.generation.gateway.base import LLMResult, Message


class OpenAICompatGateway:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        from openai import OpenAI

        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} is not set (put it in .env)")
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=key)
        self._temperature = temperature
        self._max_tokens = max_tokens

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, max=60), reraise=True)
    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        t = time.perf_counter()
        kwargs: dict[str, object] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        r = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens or self._max_tokens,
            temperature=self._temperature if temperature is None else temperature,
            **kwargs,  # type: ignore[arg-type]
        )
        wall = (time.perf_counter() - t) * 1000
        usage = r.usage
        return LLMResult(
            text=r.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_ms=wall,
        )
