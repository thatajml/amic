"""LLM gateways: the experiment response cache and the Ollama adapter (no Ollama needed)."""

from __future__ import annotations

import sys
import types

import pytest

from amic.generation.gateway.base import LLMResult
from amic.generation.gateway.cache import CachingGateway


class CountingLLM:
    model = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, **kw) -> LLMResult:
        self.calls += 1
        return LLMResult(
            text=f"answer {self.calls}",
            prompt_tokens=7,
            completion_tokens=3,
            total_ms=12.5,
            prefill_ms=2.0,
            decode_ms=10.0,
        )


MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "what is a list?"}]


# ------------------------------------------------------------------ CachingGateway
def test_cache_returns_the_first_result_including_its_timings(tmp_path):
    inner = CountingLLM()
    g = CachingGateway(inner, tmp_path / "c.sqlite")
    first = g.chat(MSGS, seed=1)
    second = g.chat(MSGS, seed=1)
    assert inner.calls == 1 and (g.hits, g.misses) == (1, 1)
    assert second == first and second.total_ms == 12.5 and second.decode_ms == 10.0


def test_cache_key_covers_messages_and_every_generation_option(tmp_path):
    inner = CountingLLM()
    g = CachingGateway(inner, tmp_path / "c.sqlite")
    g.chat(MSGS, seed=1)
    g.chat(MSGS, seed=2)  # seed
    g.chat(MSGS, seed=1, temperature=0.7)  # temperature
    g.chat(MSGS, seed=1, max_tokens=8)  # token cap
    g.chat(MSGS, seed=1, json_mode=True)  # json mode
    g.chat([*MSGS[:1], {"role": "user", "content": "other"}], seed=1)  # prompt
    assert inner.calls == 6 and g.hits == 0


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "nested" / "c.sqlite"  # parent directory is created on demand
    a = CountingLLM()
    CachingGateway(a, path).chat(MSGS, seed=1)
    b = CountingLLM()
    g = CachingGateway(b, path)
    assert g.chat(MSGS, seed=1).text == "answer 1" and b.calls == 0 and g.hits == 1


def test_cache_is_keyed_by_model(tmp_path):
    a, b = CountingLLM(), CountingLLM()
    b.model = "another"
    CachingGateway(a, tmp_path / "c.sqlite").chat(MSGS, seed=1)
    CachingGateway(b, tmp_path / "c.sqlite").chat(MSGS, seed=1)
    assert a.calls == 1 and b.calls == 1


# ------------------------------------------------------------------ OllamaGateway
class FakeClient:
    """Stands in for ollama.Client and records the request it receives."""

    last: dict = {}
    reply: dict = {}

    def __init__(self, host=None, **kw) -> None:
        FakeClient.last = {"host": host, "client_kw": kw}

    def chat(self, **kw):
        FakeClient.last.update(kw)
        return FakeClient.reply


@pytest.fixture
def ollama_gw(monkeypatch):
    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(Client=FakeClient))
    from amic.generation.gateway.ollama import OllamaGateway

    FakeClient.reply = {
        "message": {"content": "hello"},
        "prompt_eval_count": 11,
        "eval_count": 5,
        "prompt_eval_duration": 4_000_000,  # ns
        "eval_duration": 9_000_000,
    }
    return OllamaGateway


def test_ollama_maps_usage_and_splits_prefill_from_decode(ollama_gw):
    r = ollama_gw("llama3.1:8b").chat(MSGS)
    assert r.text == "hello" and (r.prompt_tokens, r.completion_tokens) == (11, 5)
    assert r.prefill_ms == pytest.approx(4.0) and r.decode_ms == pytest.approx(9.0)
    assert r.total_ms >= 0


def test_ollama_per_call_options_override_the_defaults(ollama_gw):
    gw = ollama_gw("llama3.1:8b", temperature=0.2, seed=3, max_tokens=64, num_ctx=2048)
    gw.chat(MSGS)
    assert FakeClient.last["options"] == {
        "temperature": 0.2,
        "seed": 3,
        "num_predict": 64,
        "num_ctx": 2048,
    }
    assert FakeClient.last["format"] is None
    gw.chat(MSGS, max_tokens=8, temperature=0.9, seed=9, json_mode=True)
    o = FakeClient.last["options"]
    assert (o["num_predict"], o["temperature"], o["seed"]) == (8, 0.9, 9)
    assert FakeClient.last["format"] == "json"


def test_ollama_timeout_is_passed_to_the_client_only_when_set(ollama_gw):
    ollama_gw("m")
    assert FakeClient.last["client_kw"] == {}
    ollama_gw("m", timeout_s=0.8)
    assert FakeClient.last["client_kw"] == {"timeout": 0.8}


def test_reasoning_models_run_without_thinking_and_strip_think_blocks(ollama_gw):
    FakeClient.reply = {"message": {"content": "<think>hidden\nchain</think>\n  The answer."}}
    r = ollama_gw("qwen3:8b").chat(MSGS)
    assert FakeClient.last["think"] is False
    assert r.text == "The answer." and r.prompt_tokens == 0 and r.prefill_ms == 0.0
    ollama_gw("llama3.1:8b").chat(MSGS)
    assert "think" not in FakeClient.last  # ordinary models are untouched


def test_draft_logprobs_returns_top_logprobs_per_token(ollama_gw):
    FakeClient.reply = {
        "message": {"content": "Yes"},
        "logprobs": [
            {"top_logprobs": [{"logprob": -0.1}, {"logprob": -2.3}]},
            {"logprob": -0.5},  # no alternatives: falls back to the token's own logprob
        ],
    }
    text, per_token, wall = ollama_gw("llama3.1:8b").draft_logprobs(MSGS, max_tokens=4, top=2)
    assert text == "Yes" and per_token == [[-0.1, -2.3], [-0.5]] and wall >= 0
    assert FakeClient.last["options"]["num_predict"] == 4
    assert FakeClient.last["options"]["temperature"] == 0.0
    assert FakeClient.last["logprobs"] is True and FakeClient.last["top_logprobs"] == 2
    FakeClient.reply = {"message": {"content": "x"}}  # a server without logprob support
    assert ollama_gw("m").draft_logprobs(MSGS)[1] == []
