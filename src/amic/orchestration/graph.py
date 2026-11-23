"""AgentOrchestrator (L3): one LangGraph state machine shared by every arm.

    decide ──(SKIP/WRITE)──────────────► generate ─► verify ─► record ─► END
       └────(SHALLOW/DEEP)─► retrieve ─┘    ▲           │
                                ▲           │           └─(failed, once)─► escalate ─┐
                                └───────────┴────────────────────────────────────────┘

Arms differ ONLY in `decide`, so every comparison isolates the gating decision.
`verify` is a no-op unless `ablation.use_escalation` is on and the answer was generated without
memory (route SKIP) for a learner who has memory; a failed check retries ONCE with the
`verify.escalation_route` memory (P6: a wrong skip must be recoverable, bounded to one retry).
The long-term write is returned as a deferred job and is never timed on the response path (P5).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from amic.contracts.models import (
    AgentResponse,
    DecisionTrace,
    LatencySplit,
    MemoryDecision,
    MemoryItem,
    RequestContext,
    ResolvedBy,
    Route,
    Turn,
)
from amic.generation.gateway.base import LLMResult

if TYPE_CHECKING:  # avoids an import cycle with amic.app
    from amic.app import Components

WriteJob = Callable[[], None]


class RunState(TypedDict, total=False):
    ctx: RequestContext
    q_vec: np.ndarray | None
    decision: MemoryDecision
    features: dict[str, float]
    items: list[MemoryItem]
    memory_block: str
    memory_tokens: int
    llm: LLMResult
    answer: str
    lat: dict[str, float]
    t_start: float
    # verify-and-escalate
    verify_failed: bool
    verify_reason: str
    escalated: bool
    first_llm: LLMResult  # the discarded memory-less generation of an escalated request


def _ms(t: float) -> float:
    return (time.perf_counter() - t) * 1000


class PipelineArm:
    """Base arm. Subclasses implement `decide`. Implements IPipelineArm."""

    name = "base"

    def __init__(self, comps: Components) -> None:
        self.c = comps
        g = StateGraph(RunState)
        g.add_node("decide", self._decide_node)
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("generate", self._generate_node)
        g.add_node("verify", self._verify_node)
        g.add_node("escalate", self._escalate_node)
        g.add_node("record", self._record_node)
        g.add_edge(START, "decide")
        g.add_conditional_edges(
            "decide", self._after_decide, {"retrieve": "retrieve", "generate": "generate"}
        )
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", "verify")
        g.add_conditional_edges(
            "verify", self._after_verify, {"escalate": "escalate", "record": "record"}
        )
        g.add_edge("escalate", "retrieve")
        g.add_edge("record", END)
        self._graph = g.compile()

    # ----------------------------------------------------------------- to override
    def decide(self, state: RunState) -> dict[str, Any]:
        """Return a state update containing at least `decision` (may add q_vec/features/lat)."""
        raise NotImplementedError

    # ----------------------------------------------------------------- helpers
    def embed(self, state: RunState) -> dict[str, Any]:
        """Embed the query once (P2). Returns a state update (q_vec + lat.embed)."""
        if state.get("q_vec") is not None:
            return {}
        t = time.perf_counter()
        v = self.c.embedder.encode(state["ctx"].query)
        lat = {**state.get("lat", {}), "embed": _ms(t)}
        return {"q_vec": v, "lat": lat}

    def _should_verify(self, state: RunState) -> bool:
        if not self.c.settings.ablation.use_escalation or state.get("escalated"):
            return False
        if self.c.verifier is None:
            return False
        d = state["decision"]
        if d.route != Route.SKIP:
            return False
        if d.resolved_by == ResolvedBy.TIER0 and d.rationale.startswith("R1"):
            return False  # no long-term memory exists: escalating cannot help
        feats = state.get("features") or {}
        if "s3_has_memory" in feats:  # amic arm: known from the sketch, set even if S3 is ablated
            return feats["s3_has_memory"] > 0
        return self.c.sketches.get(state["ctx"].tenant_id) is not None

    # ----------------------------------------------------------------- nodes
    def _decide_node(self, state: RunState) -> dict[str, Any]:
        return self.decide(state)

    @staticmethod
    def _after_decide(state: RunState) -> str:
        return "retrieve" if state["decision"].route in (Route.SHALLOW, Route.DEEP) else "generate"

    def _retrieve_node(self, state: RunState) -> dict[str, Any]:
        upd = self.embed(state)
        q_vec = upd.get("q_vec", state.get("q_vec"))
        lat = upd.get("lat", state.get("lat", {}))
        d = state["decision"]
        t = time.perf_counter()
        items = self.c.retrieval.retrieve(d.plan, q_vec, state["ctx"].tenant_id)
        block, used, mem_tok = self.c.assembler.assemble(items, d.token_budget)
        key = "esc_retrieval" if state.get("escalated") else "retrieval"
        return {
            "q_vec": q_vec,
            "items": used,
            "memory_block": block,
            "memory_tokens": mem_tok,
            "lat": {**lat, key: _ms(t)},
        }

    def _generate_node(self, state: RunState) -> dict[str, Any]:
        ctx = state["ctx"]
        msgs = self.c.prompts.build(
            ctx.query, ctx.m0_turns, state.get("memory_block", ""), now=ctx.now
        )
        t = time.perf_counter()
        res = self.c.llm.chat(msgs, seed=self.c.settings.models.seed)
        lat = dict(state.get("lat", {}))
        if state.get("escalated"):  # the retry: first-pass timings stay as they were
            wall = _ms(t)
            # A gateway that serves cached results (experiments only) reports the first-run
            # model time in total_ms; count that, and add the difference to the request total,
            # so a cached retry is not timed at ~0 ms. Without a cache this is a no-op.
            model = max(wall, res.total_ms)
            lat["esc_generation"] = model
            lat["esc_cache_correction"] = model - wall
            return {"llm": res, "first_llm": state["llm"], "answer": res.text.strip(), "lat": lat}
        lat.update(generation=_ms(t), prefill=res.prefill_ms, decode=res.decode_ms)
        return {"llm": res, "answer": res.text.strip(), "lat": lat}

    def _verify_node(self, state: RunState) -> dict[str, Any]:
        if not self._should_verify(state):
            return {}
        ctx = state["ctx"]
        res = self.c.verifier.check(ctx.query, state["answer"], ctx.m0_turns)  # type: ignore[union-attr]
        lat = dict(state.get("lat", {}))
        lat["verify"] = lat.get("verify", 0.0) + res.ms
        return {"lat": lat, "verify_failed": res.needs_memory, "verify_reason": res.reason}

    @staticmethod
    def _after_verify(state: RunState) -> str:
        failed = state.get("verify_failed", False) and not state.get("escalated", False)
        return "escalate" if failed else "record"

    def _escalate_node(self, state: RunState) -> dict[str, Any]:
        d0 = state["decision"]
        route = Route(self.c.settings.verify.escalation_route)
        plan = self.c.planner.plan(route)
        d = d0.model_copy(
            update={
                "route": route,
                "plan": plan,
                "token_budget": plan.token_budget,
                "escalated_from": d0.route,
                "rationale": f"{d0.rationale}; verify failed ({state.get('verify_reason', '')})"
                f" -> retried with {route.value} memory",
            }
        )
        return {"decision": d, "escalated": True}

    def _record_node(self, state: RunState) -> dict[str, Any]:
        ctx = state["ctx"]
        now = ctx.now or datetime.now(UTC)
        # M0 is updated synchronously: turn N must be visible to turn N+1 (brief section 7)
        self.c.sessions.append(
            ctx.session_id,
            Turn(role="user", text=ctx.query, ts=now),
            Turn(role="assistant", text=state["answer"], ts=now),
        )
        lat = dict(state.get("lat", {}))
        lat["total"] = _ms(state["t_start"]) + lat.pop("esc_cache_correction", 0.0)
        escalated = bool(state.get("escalated"))
        esc_ms = lat.pop("esc_retrieval", 0.0) + lat.pop("esc_generation", 0.0)
        if escalated:
            lat["escalation"] = esc_ms
        d = state["decision"]
        res, first = state["llm"], state.get("first_llm")
        trace = DecisionTrace(
            request_id=ctx.request_id,
            ts=now,
            tenant_id=ctx.tenant_id,
            session_id=ctx.session_id,
            arm=self.name,
            config_hash=self.c.config_hash,
            query=ctx.query,
            route=d.escalated_from or d.route,  # the gate's decision; `escalated` marks a retry
            resolved_by=d.resolved_by,
            p=d.p,
            tau_low=d.tau_low,
            tau_high=d.tau_high,
            latency=LatencySplit(**lat),
            # tokens actually processed: both generations of an escalated request
            prompt_tokens=res.prompt_tokens + (first.prompt_tokens if first else 0),
            completion_tokens=res.completion_tokens + (first.completion_tokens if first else 0),
            memory_tokens=state.get("memory_tokens", 0),
            n_items=len(state.get("items", [])),
            escalated=escalated,
            features=state.get("features", {}),
            oracle_label=ctx.oracle_label,
        )
        self.c.telemetry.record(trace)
        return {"lat": lat}

    # ----------------------------------------------------------------- public
    def answer_deferred(self, ctx: RequestContext) -> tuple[AgentResponse, WriteJob]:
        state: RunState = {"ctx": ctx, "lat": {}, "t_start": time.perf_counter()}
        out = self._graph.invoke(state)
        resp = AgentResponse(
            answer=out["answer"],
            decision=out["decision"],  # final route; `escalated_from` holds the gate's route
            items=out.get("items", []),
            trace_id=ctx.request_id,
        )
        now = ctx.now or datetime.now(UTC)

        def job() -> None:
            self.c.writer.write_turn(
                ctx.tenant_id, ctx.session_id, ctx.request_id, ctx.query, resp.answer, now
            )

        return resp, job

    def answer(self, ctx: RequestContext) -> AgentResponse:
        """IPipelineArm. Runs the long-term write inline AFTER the timed path completes."""
        resp, job = self.answer_deferred(ctx)
        job()
        return resp


def new_request_id() -> str:
    return uuid.uuid4().hex
