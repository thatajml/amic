"""Day-2 DoD: baselines end to end, M0 visibility, tenant isolation, telemetry, P2/P5."""

from __future__ import annotations

from datetime import timedelta

import pytest

from amic.app import build_arm, make_context
from amic.contracts.models import MemoryTier, ResolvedBy, Route
from tests.conftest import T0

HISTORY = [
    (
        "I keep getting IndexError with my list loop",
        "Use range(len(xs)) carefully, it stops at len-1.",
    ),
    ("What does append do to a list", "append adds one element to the end of the list."),
]


def _ask(c, arm_name, tenant, session, query, **kw):
    arm = build_arm(arm_name, c)
    return arm.answer(make_context(c, tenant, session, query, arm_name, now=T0, **kw))


def test_always_retrieves_and_injects_memory(comps, seed_history):
    seed_history(comps, "alice", HISTORY)
    r = _ask(comps, "always", "alice", "s1", "Why did my list loop give an IndexError?")
    assert r.decision.route == Route.DEEP and r.decision.resolved_by == ResolvedBy.FIXED
    assert r.items and all(i.tier == MemoryTier.M1 for i in r.items)
    assert "memory=yes" in r.answer
    t = comps.telemetry.traces[-1]
    assert t.arm == "always" and t.n_items == len(r.items) and t.memory_tokens > 0
    assert t.latency.embed > 0 and t.latency.retrieval > 0 and t.latency.total > 0


def test_never_has_zero_store_contact_and_no_embedding(comps, seed_history):
    seed_history(comps, "alice", HISTORY)
    calls_before = comps.retrieval.calls
    r = _ask(comps, "never", "alice", "s1", "Why did my list loop give an IndexError?")
    assert r.decision.route == Route.SKIP and r.items == []
    assert comps.retrieval.calls == calls_before
    t = comps.telemetry.traces[-1]
    assert t.latency.embed == 0 and t.latency.retrieval == 0 and t.memory_tokens == 0
    assert "memory=no" in r.answer


def test_m0_visibility_of_turn_n(comps):
    """Turn N's content reaches turn N+1 via M0, with no long-term store involved."""
    _ask(comps, "never", "bob", "s1", "My variable is called zebra_count")
    _ask(comps, "never", "bob", "s1", "What was my variable called?")
    last_prompt = comps.llm.calls[-1]
    assert any("zebra_count" in m["content"] for m in last_prompt[1:-1])
    assert comps.retrieval.calls == 0


def test_tenant_isolation(comps, seed_history):
    seed_history(comps, "alice", [("my secret project is about parrots", "Nice, parrots!")])
    seed_history(comps, "bob", [("I like loops", "Loops are great.")])
    r = _ask(comps, "always", "bob", "s9", "tell me about my parrots project")
    assert all("parrot" not in i.text for i in r.items)
    assert all(i.id.startswith("bob-") for i in r.items)


def test_long_term_write_is_deferred_not_on_response_path(comps):
    arm = build_arm("always", comps)
    ctx = make_context(comps, "carol", "s1", "explain tuples", "always", now=T0)
    resp, job = arm.answer_deferred(ctx)
    assert comps.episodic.count("carol") == 0  # nothing written yet
    job()
    assert comps.episodic.count("carol") == 1
    assert resp.trace_id == ctx.request_id


def test_one_trace_per_request(comps):
    for i in range(3):
        _ask(comps, "never", "dan", "s1", f"question {i}")
    assert len(comps.telemetry.traces) == 3
    assert len({t.request_id for t in comps.telemetry.traces}) == 3


def test_oracle_follows_label(comps, seed_history):
    seed_history(comps, "erin", HISTORY)
    skip = _ask(comps, "oracle", "erin", "s1", "what is a tuple", oracle_label=False)
    deep = _ask(comps, "oracle", "erin", "s1", "what did I get wrong", oracle_label=True)
    shallow = _ask(
        comps,
        "oracle",
        "erin",
        "s1",
        "my list error",
        oracle_label=True,
        oracle_depth=Route.SHALLOW,
    )
    assert skip.decision.route == Route.SKIP
    assert deep.decision.route == Route.DEEP
    assert shallow.decision.route == Route.SHALLOW and len(shallow.items) <= 3
    with pytest.raises(ValueError):
        _ask(comps, "oracle", "erin", "s1", "no label")


def test_shallow_vs_deep_budgets(comps, seed_history):
    pairs = [(f"question about loops number {i}", f"answer about loops {i}") for i in range(20)]
    seed_history(comps, "fay", pairs)
    comps.settings.routes.always_route = "SHALLOW"
    s = _ask(comps, "always", "fay", "s1", "loops question")
    comps.settings.routes.always_route = "DEEP"
    d = _ask(comps, "always", "fay", "s2", "loops question")
    assert len(s.items) <= comps.settings.routes.shallow_k < len(d.items)


def test_session_touch_tracks_previous_session(comps):
    later = T0 + timedelta(days=2)
    comps.sessions.touch("gus", "s1", T0)
    prev = comps.sessions.touch("gus", "s2", later)
    assert prev["last_session_id"] == "s1"
    meta = comps.sessions.tenant_meta("gus")
    assert meta["prev_session_end"] == T0.timestamp() and meta["n_sessions"] == 2


@pytest.mark.parametrize("backend", ["redis", "sqlite"])
def test_session_backends_are_equivalent(backend, tmp_path):
    import fakeredis

    from amic.contracts.models import Turn
    from amic.service.session import SessionManager, SqliteSessionManager

    sm = (
        SessionManager(fakeredis.FakeRedis(), m0_turns=2)
        if backend == "redis"
        else SqliteSessionManager(str(tmp_path / "s.sqlite"), m0_turns=2)
    )
    later = T0 + timedelta(days=2)
    assert sm.touch("u", "s1", T0) == {}
    sm.touch("u", "s1", T0 + timedelta(minutes=30))
    prev = sm.touch("u", "s2", later)
    assert prev["last_session_id"] == "s1"
    meta = sm.tenant_meta("u")
    assert meta["n_sessions"] == 2 and meta["last_session_id"] == "s2"
    assert meta["prev_session_end"] == (T0 + timedelta(minutes=30)).timestamp()
    for i in range(5):
        sm.append("s2", Turn(role="user", text=f"t{i}", ts=later))
    assert [t.text for t in sm.get_turns("s2")] == ["t3", "t4"]
    sm.erase_tenant("u")
    assert sm.tenant_meta("u") == {} and sm.get_turns("s2") == []
