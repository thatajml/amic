"""FastAPI service (L2). Run: uvicorn amic.service.api:app --reload"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from amic.app import Components, build_arm, build_components, make_context
from amic.contracts.models import AgentResponse, DecisionTrace
from amic.orchestration.graph import PipelineArm
from amic.service.shadow import ShadowRunner


class ChatRequest(BaseModel):
    tenant_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=4000)
    arm: Literal["always", "never", "amic"] | None = None  # oracle needs labels: not exposed


def create_app(comps: Components | None = None) -> FastAPI:
    state: dict[str, object] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state["comps"] = comps or build_components()
        state["arms"] = {}
        yield

    app = FastAPI(title="AMIC tutor", lifespan=lifespan)

    def _comps() -> Components:
        return state["comps"]  # type: ignore[return-value]

    def _arm(name: str) -> PipelineArm:
        arms: dict[str, PipelineArm] = state["arms"]  # type: ignore[assignment]
        if name not in arms:
            arms[name] = build_arm(name, _comps())  # type: ignore[arg-type]
        return arms[name]

    def _shadow() -> ShadowRunner | None:
        c = _comps()
        if not c.settings.shadow.enabled:
            return None
        if "shadow" not in state:
            state["shadow"] = ShadowRunner(c, c.settings.shadow.arm, c.settings.shadow.log_path)
        return state["shadow"]  # type: ignore[return-value]

    @app.get("/health")
    def health() -> dict[str, str]:
        c = _comps()
        return {"status": "ok", "config_hash": c.config_hash, "default_arm": c.settings.arm}

    @app.post("/chat", response_model=AgentResponse)
    def chat(req: ChatRequest, bg: BackgroundTasks) -> AgentResponse:
        c = _comps()
        arm_name = req.arm or c.settings.arm
        try:
            arm = _arm(arm_name)
        except (ImportError, KeyError) as e:
            raise HTTPException(400, f"arm '{arm_name}' unavailable: {e}") from e
        ctx = make_context(c, req.tenant_id, req.session_id, req.query, arm_name)  # type: ignore[arg-type]
        t = time.perf_counter()
        resp, write_job = arm.answer_deferred(ctx)
        served_ms = (time.perf_counter() - t) * 1000
        shadow = _shadow()
        if shadow is not None and arm_name != shadow.arm_name:
            # before the write job, so the shadow never sees this turn in long-term memory
            bg.add_task(shadow.run, ctx, resp, served_ms)
        bg.add_task(write_job)  # after the response is sent (P5)
        return resp

    @app.get("/traces", response_model=list[DecisionTrace])
    def get_traces(limit: int = 50, arm: str | None = None) -> list[DecisionTrace]:
        tel = _comps().telemetry
        if not hasattr(tel, "fetch"):
            raise HTTPException(501, "telemetry backend cannot be queried")
        return tel.fetch(limit=limit, arm=arm)  # type: ignore[no-any-return]

    @app.get("/learners/{tenant_id}")
    def learner(tenant_id: str) -> dict[str, object]:
        """What long-term memory holds for one learner (counts only; the Educator Console shows
        the content). Lets clients and tests observe the async consolidation."""
        c = _comps()
        prof = c.profiles.get(tenant_id) if c.profiles is not None else None
        return {
            "episodes": c.episodic.count(tenant_id),
            "facts": c.semantic.count(tenant_id) if c.semantic is not None else 0,
            "profile": prof is not None,
            "mem_version": c.sessions.mem_version(tenant_id),
        }

    @app.delete("/learners/{tenant_id}")
    def erase(tenant_id: str) -> dict[str, int]:
        """Per-learner erasure (synchronous by design, plan R6)."""
        from amic.writepath.forgetting import ForgettingManager

        c = _comps()
        fm = ForgettingManager(
            c.settings.forgetting, c.episodic, c.semantic, c.profiles, c.sessions
        )
        # M1, M2, M3 and M0; the new memory version drops the M4 sketch and cached decisions
        return fm.erase(tenant_id)

    return app


app = create_app()
