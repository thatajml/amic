"""OpenAI-compatible adapter. Used offline for the judge and dataset generation (Gemini's
OpenAI-compatible endpoint). Retries only transient failures (429 / 5xx / timeouts)."""

from __future__ import annotations

import os
import re
import time
from datetime import UTC, datetime, timedelta

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from amic.generation.gateway.base import LLMResult, Message


class DailyQuotaExceeded(RuntimeError):
    """The provider's per-DAY request cap is spent. Not retried: waiting minutes cannot help."""

    def __init__(self, message: str, retry_after_s: float) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


_RETRY_IN = re.compile(r"retry in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.I)


def daily_quota_retry_after(e: BaseException) -> float | None:
    """Seconds until the daily quota resets if `e` is a per-day 429, else None."""
    import openai

    if not isinstance(e, openai.RateLimitError):
        return None
    msg = str(e)
    if "per day" not in msg.lower() and "perday" not in msg.lower():
        return None  # per-minute limit: transient, let tenacity back off
    m = _RETRY_IN.search(msg)
    if m and any(m.groups()):
        h, mi, se = (float(x) if x else 0.0 for x in m.groups())
        return h * 3600 + mi * 60 + se
    # fallback: next midnight US-Pacific (Gemini's reset), ~08:00 UTC in winter
    now = datetime.now(UTC)
    reset = now.replace(hour=8, minute=0, second=0, microsecond=0)
    if reset <= now:
        reset += timedelta(days=1)
    return (reset - now).total_seconds()


def _transient(e: BaseException) -> bool:
    import openai

    if daily_quota_retry_after(e) is not None:
        return False
    if isinstance(e, openai.RateLimitError | openai.APITimeoutError | openai.APIConnectionError):
        return True
    return isinstance(e, openai.APIStatusError) and e.status_code >= 500


class OpenAICompatGateway:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        reasoning_effort: str | None = "low",
        timeout_s: float = 120.0,
    ) -> None:
        from openai import OpenAI

        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} is not set (put it in .env)")
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=key, timeout=timeout_s, max_retries=0)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._reasoning = reasoning_effort

    @retry(
        retry=retry_if_exception(_transient),
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=2, min=4, max=90),
        reraise=True,
    )
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
        extra: dict[str, object] = {}
        if self._reasoning:
            extra["reasoning_effort"] = self._reasoning
        kwargs: dict[str, object] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            r = self._client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens or self._max_tokens,
                temperature=self._temperature if temperature is None else temperature,
                extra_body=extra or None,
                **kwargs,  # type: ignore[arg-type]
            )
        except Exception as e:
            wait = daily_quota_retry_after(e)
            if wait is not None:
                raise DailyQuotaExceeded(f"{self.model}: daily quota exhausted", wait) from e
            raise
        wall = (time.perf_counter() - t) * 1000
        usage = r.usage
        return LLMResult(
            text=r.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_ms=wall,
        )


def judge_from_settings(models: object) -> OpenAICompatGateway:
    """Build the judge gateway from ModelConfig, loading .env first."""
    from dotenv import load_dotenv

    load_dotenv(".env")
    return OpenAICompatGateway(
        model=models.judge,  # type: ignore[attr-defined]
        base_url=models.judge_base_url,  # type: ignore[attr-defined]
        reasoning_effort=models.judge_reasoning,  # type: ignore[attr-defined]
    )
