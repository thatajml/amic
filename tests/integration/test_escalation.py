"""Verify-and-escalate end to end (brief P6 / M7): SKIP answers are checked and retried at most
once with memory; nothing else is touched."""

from __future__ import annotations

import uuid

import fakeredis
import pytest

from amic.app import build_arm, build_components, make_context
from amic.contracts.config import load_settings
from amic.contracts.models import Route
from amic.generation.gateway.base import FakeLLM, LLMResult
from amic.memory.embedder import HashEmbedder
from amic.telemetry.store import InMemoryTelemetry
from tests.conftest import T0

HISTORY = [("I keep getting IndexError in my list loop", "range(len(xs)) stops at len-1.")]


class GateLLM:
    """Scripted Tier-2/verifier model: always answers `needs_memory`."""

    model = "gate-fake"

    def __init__(self, needs: bool) -> None:
        self.needs, self.calls = needs, 0

    def chat(self, messages, **kw) -> LLMResult:
        self.calls += 1
        txt = '{"needs_memory": ' + ("true" if self.needs else "false") + "}"
        return LLMResult(text=txt, prompt_tokens=60, completion_tokens=6, total_ms=1.0)


class FabricatingLLM(FakeLLM):
    """Always invents shared history, so its answers fail the lexical check every time."""

    def chat(self, messages, **kw) -> LLMResult:
        res = super().chat(messages, **kw)
        res.text = "Last time you struggled with loops. " + res.text
        return res


def _comps(chroma_client, *, mode="both", needs=True, tutor=None, escalate=True):
    s = load_settings(
        chroma_collection=f"t_{uuid.uuid4().hex[:10]}",
        metrics_db_url="sqlite://",
        ablation={"use_escalation": escalate},
        verify={"mode": mode, "escalation_route": "SHALLOW"},
    )
    gate = GateLLM(needs)
    c = build_components(
        s,
        embedder=HashEmbedder(),
        llm=tutor or FakeLLM(),
        gate_llm=gate,
        redis=fakeredis.FakeRedis(),
        chroma_client=chroma_client,
        telemetry=InMemoryTelemetry(),
    )
    return c, gate


def _seed(c, tenant="alice"):
    for i, (q, a) in enumerate(HISTORY):
        c.writer.write_turn(tenant, f"{tenant}-old", f"{tenant}-h{i}", q, a, T0)


def _ask(c, arm_name, tenant="alice", query="Why did my list loop crash?"):
    arm = build_arm(arm_name, c)
    return arm.answer(make_context(c, tenant, f"s-{uuid.uuid4().hex[:6]}", query, arm_name, now=T0))


def test_failed_verify_escalates_once_to_shallow(chroma_client):
    c, gate = _comps(chroma_client, needs=True)
    _seed(c)
    r = _ask(c, "never")
    assert "memory=yes" in r.answer  # the retry saw memory
    assert r.decision.route == Route.SHALLOW and r.decision.escalated_from == Route.SKIP
    assert r.items
    t = c.telemetry.traces[-1]
    assert t.route == Route.SKIP and t.escalated  # the trace keeps the gate's decision
    assert t.latency.verify > 0 and t.latency.retrieval == 0
    # the retry is timed with the gateway's model time (FakeLLM reports 1 ms), never ~0
    assert t.latency.escalation >= 1.0 and t.latency.total >= t.latency.escalation
    assert len(c.llm.calls) == 2 and gate.calls == 1


def test_passing_verify_keeps_the_memoryless_answer(chroma_client):
    c, gate = _comps(chroma_client, needs=False)
    _seed(c)
    r = _ask(c, "never")
    assert "memory=no" in r.answer and r.decision.escalated_from is None
    t = c.telemetry.traces[-1]
    assert not t.escalated and t.latency.escalation == 0 and t.latency.verify > 0
    assert len(c.llm.calls) == 1 and gate.calls == 1


def test_never_more_than_one_retry(chroma_client):
    # every answer fails the lexical check, including the retry: still exactly one retry
    c, gate = _comps(chroma_client, mode="lexical", tutor=FabricatingLLM())
    _seed(c)
    r = _ask(c, "never")
    assert r.decision.escalated_from == Route.SKIP
    assert len(c.llm.calls) == 2 and gate.calls == 0


def test_no_verify_when_memory_was_used(chroma_client):
    c, gate = _comps(chroma_client, needs=True)
    _seed(c)
    _ask(c, "always")
    assert gate.calls == 0 and len(c.llm.calls) == 1
    assert c.telemetry.traces[-1].latency.verify == 0


def test_no_escalation_without_long_term_memory(chroma_client):
    c, gate = _comps(chroma_client, needs=True)  # nothing seeded for this learner
    r = _ask(c, "never", tenant="newbie")
    assert r.decision.escalated_from is None and gate.calls == 0 and len(c.llm.calls) == 1


def test_escalation_off_by_default(chroma_client):
    c, gate = _comps(chroma_client, needs=True, escalate=False)
    _seed(c)
    r = _ask(c, "never")
    assert c.verifier is None and r.decision.escalated_from is None and gate.calls == 0


@pytest.mark.parametrize("mode", ["lexical", "llm", "both"])
def test_every_mode_builds(chroma_client, mode):
    c, _ = _comps(chroma_client, mode=mode)
    assert c.verifier is not None and c.verifier.mode == mode
