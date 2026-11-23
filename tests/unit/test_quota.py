from __future__ import annotations

import httpx
import openai
import pytest

from amic.eval.quota import run_with_quota
from amic.generation.gateway.openai_compat import DailyQuotaExceeded, daily_quota_retry_after


def _rate_limit(msg: str) -> openai.RateLimitError:
    req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return openai.RateLimitError(msg, response=httpx.Response(429, request=req), body=None)


def test_daily_quota_detected_and_parsed():
    e = _rate_limit(
        "Quota exceeded for metric: generate_content_free_tier_requests, limit: 500 ... "
        "Generate content requests per day limit exceeded on Free Tier. Please retry in 21h7m30.3s."
    )
    assert daily_quota_retry_after(e) == pytest.approx(21 * 3600 + 7 * 60 + 30.3)


def test_daily_quota_detected_when_details_only_in_list_body():
    # Gemini's OpenAI-compatible endpoint: body is a JSON list, so str(e) == "Error code: 429".
    req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    body = [
        {
            "error": {
                "code": 429,
                "message": "You exceeded your current quota ... Generate content requests per "
                "day limit exceeded on Free Tier.\nPlease retry in 5h55m39.39s.",
                "status": "RESOURCE_EXHAUSTED",
            }
        }
    ]
    e = openai.RateLimitError(
        "Error code: 429", response=httpx.Response(429, request=req), body=body
    )
    assert daily_quota_retry_after(e) == pytest.approx(5 * 3600 + 55 * 60 + 39.39)


class _StubCompletions:
    def __init__(self, daily_capped: bool) -> None:
        self.daily_capped = daily_capped
        self.calls = 0

    def create(self, **_kw: object) -> object:
        from types import SimpleNamespace

        self.calls += 1
        if self.daily_capped:
            raise _rate_limit("requests per day limit exceeded. Please retry in 1h0m0s.")
        msg = SimpleNamespace(content='{"ok": true}')
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def _gateway_with(*capped: bool):  # noqa: ANN202
    from types import SimpleNamespace

    from amic.generation.gateway.openai_compat import OpenAICompatGateway

    gw = OpenAICompatGateway("m", "https://example.invalid/v1", api_key="k")
    stubs = [_StubCompletions(c) for c in capped]
    gw._clients = [SimpleNamespace(chat=SimpleNamespace(completions=s)) for s in stubs]  # type: ignore[assignment]
    return gw, stubs


def test_rotates_to_next_key_when_daily_cap_reached():
    gw, stubs = _gateway_with(True, False)
    assert gw.chat([{"role": "user", "content": "x"}]).text == '{"ok": true}'
    assert stubs[0].calls == 1 and stubs[1].calls == 1 and gw._active == 1
    gw.chat([{"role": "user", "content": "y"}])  # stays on the working key
    assert stubs[0].calls == 1 and stubs[1].calls == 2


def test_raises_daily_quota_when_all_keys_spent():
    gw, _ = _gateway_with(True, True)
    with pytest.raises(DailyQuotaExceeded) as ei:
        gw.chat([{"role": "user", "content": "x"}])
    assert ei.value.retry_after_s == pytest.approx(3600)


def test_keys_from_env_order_and_dedup(monkeypatch: pytest.MonkeyPatch):
    from amic.generation.gateway.openai_compat import _keys_from_env

    monkeypatch.setenv("TESTKEY", "a")
    monkeypatch.setenv("TESTKEY_2", "b")
    monkeypatch.setenv("TESTKEY_3", "a")
    monkeypatch.setenv("TESTKEY_4", " ")
    assert _keys_from_env("TESTKEY") == ["a", "b"]


def test_per_minute_limit_is_not_daily():
    assert (
        daily_quota_retry_after(_rate_limit("Requests per minute exceeded, retry in 12s")) is None
    )
    assert daily_quota_retry_after(ValueError("x")) is None


def test_daily_quota_without_retry_hint_falls_back_to_next_reset():
    wait = daily_quota_retry_after(_rate_limit("requests per day limit exceeded"))
    assert wait is not None and 0 < wait <= 24 * 3600


class _Provider:
    """Allows `cap` calls, then raises the daily-quota error until `reset()`."""

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.used = 0
        self.order: list[int] = []

    def __call__(self, job: int) -> int:
        if self.used >= self.cap:
            raise DailyQuotaExceeded("cap", retry_after_s=10.0)
        self.used += 1
        self.order.append(job)
        return job * 10

    def reset(self, _seconds: float) -> None:
        self.used = 0


def test_runs_in_priority_order_and_resumes_after_reset():
    prov = _Provider(cap=4)
    out: list[int] = []
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        prov.reset(s)

    n = run_with_quota(
        list(range(10)), prov, out.append, workers=1, wait=True, sleep=sleep, margin_s=5
    )
    assert n == 10 and sorted(out) == [j * 10 for j in range(10)]
    assert prov.order == list(range(10))  # priority order kept across resets
    assert slept == [15.0, 15.0]  # 4 + 4 + 2 jobs: two resets, retry_after + margin


def test_without_wait_stops_cleanly_at_cap():
    prov = _Provider(cap=3)
    out: list[int] = []
    n = run_with_quota(list(range(8)), prov, out.append, workers=2, wait=False)
    assert n == 3 and len(out) == 3


def test_other_failures_do_not_stop_the_run():
    def flaky(j: int) -> int:
        if j == 2:
            raise ValueError("bad output")
        return j

    out: list[int] = []
    assert run_with_quota([1, 2, 3], flaky, out.append, workers=1) == 2
    assert out == [1, 3]
