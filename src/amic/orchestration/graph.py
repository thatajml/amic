"""AgentOrchestrator (L3): one LangGraph state machine shared by every arm.

    decide ──(SKIP/WRITE)──────────────► generate ─► record ─► END
       └────(SHALLOW/DEEP)─► retrieve ─┘

Arms differ ONLY in `decide`, so every comparison isolates the gating decision.
The long-term write is returned as a deferred job and is never timed on the response path (P5).
The optional shadow-baseline measurement (telemetry.shadow_baseline) runs in that same job,
before the write, so it never touches the response path or a SKIP's zero-retrieval guarantee.
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
    model: str
    answer: str
    lat: dict[str, float]
    t_start: float


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
        g.add_node("record", self._record_node)
        g.add_edge(START, "decide")
        g.add_conditional_edges(
            "decide", self._after_decide, {"retrieve": "retrieve", "generate": "generate"}
        )
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", "record")
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
        return {
            "q_vec": q_vec,
            "items": used,
            "memory_block": block,
            "memory_tokens": mem_tok,
            "lat": {**lat, "retrieval": _ms(t)},
        }

    def _generate_node(self, state: RunState) -> dict[str, Any]:
        ctx = state["ctx"]
        msgs = self.c.prompts.build(ctx.query, ctx.m0_turns, state.get("memory_block", ""))
        llm = self.c.llm_for(ctx.model)
        t = time.perf_counter()
        res = llm.chat(msgs, seed=self.c.settings.models.seed)
        return {
            "llm": res,
            "model": llm.model,
            "answer": res.text.strip(),
            "lat": {
                **state.get("lat", {}),
                "generation": _ms(t),
                "prefill": res.prefill_ms,
                "decode": res.decode_ms,
            },
        }

    def _record_node(self, state: RunState) -> dict[str, Any]:
        ctx = state["ctx"]
        now = ctx.now or datetime.now(UTC)
        # M0 is updated synchronously: turn N must be visible to turn N+1 (brief section 7)
        self.c.sessions.append(
            ctx.session_id,
            Turn(role="user", text=ctx.query, ts=now),
            Turn(role="assistant", text=state["answer"], ts=now),
        )
        lat = {**state.get("lat", {}), "total": _ms(state["t_start"])}
        d = state["decision"]
        res = state["llm"]
        mem_tok = state.get("memory_tokens", 0)
        trace = DecisionTrace(
            request_id=ctx.request_id,
            ts=datetime.now(UTC),  # wall clock: when it was served (ctx.now may be simulated)
            tenant_id=ctx.tenant_id,
            session_id=ctx.session_id,
            arm=self.name,
            config_hash=self.c.config_hash,
            query=ctx.query,
            route=d.route,
            resolved_by=d.resolved_by,
            p=d.p,
            tau_low=d.tau_low,
            tau_high=d.tau_high,
            latency=LatencySplit(**lat),
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            memory_tokens=mem_tok,
            n_items=len(state.get("items", [])),
            features=state.get("features", {}),
            oracle_label=ctx.oracle_label,
            model=state.get("model", ""),
            baseline_memory_tokens=mem_tok if d.route == self._baseline_route() else None,
        )
        self.c.telemetry.record(trace)
        return {"lat": lat}

    # ----------------------------------------------------------------- shadow baseline
    def _baseline_route(self) -> Route:
        return Route(self.c.settings.routes.always_route)

    def shadow_baseline_tokens(self, ctx: RequestContext, q_vec: np.ndarray | None) -> int:
        """Memory tokens the always-retrieve plan would inject for `ctx` (no generation)."""
        if q_vec is None:
            q_vec = self.c.embedder.encode(ctx.query)
        plan = self.c.planner.plan(self._baseline_route())
        items = self.c.retrieval.retrieve(plan, q_vec, ctx.tenant_id)
        return self.c.assembler.assemble(items, plan.token_budget)[2]

    # ----------------------------------------------------------------- public
    def answer_deferred(
        self, ctx: RequestContext, *, write: bool = True
    ) -> tuple[AgentResponse, WriteJob]:
        """Answer now; return the deferred job (shadow baseline if enabled, then the write).

        `write=False` keeps memory frozen (experiments, traffic simulation).
        """
        state: RunState = {"ctx": ctx, "lat": {}, "t_start": time.perf_counter()}
        out = self._graph.invoke(state)
        resp = AgentResponse(
            answer=out["answer"],
            decision=out["decision"],
            items=out.get("items", []),
            trace_id=ctx.request_id,
        )
        now = ctx.now or datetime.now(UTC)

        shadow = (
            self.c.settings.telemetry.shadow_baseline
            and out["decision"].route != self._baseline_route()
            and hasattr(self.c.telemetry, "annotate")
        )

        def job() -> None:
            if shadow:  # before the write, so the baseline sees the same memory the gate saw
                tok = self.shadow_baseline_tokens(ctx, out.get("q_vec"))
                self.c.telemetry.annotate(ctx.request_id, baseline_memory_tokens=tok)  # type: ignore[attr-defined]
            if write:
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
