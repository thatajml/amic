"""FastAPI service (L2). Run: uvicorn amic.service.api:app --reload"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from amic.app import Components, build_arm, build_components, make_context
from amic.contracts.models import AgentResponse, DecisionTrace
from amic.orchestration.graph import PipelineArm


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
        resp, write_job = arm.answer_deferred(ctx)
        bg.add_task(write_job)  # after the response is sent (P5)
        return resp

    @app.get("/traces", response_model=list[DecisionTrace])
    def get_traces(limit: int = 50, arm: str | None = None) -> list[DecisionTrace]:
        tel = _comps().telemetry
        if not hasattr(tel, "fetch"):
            raise HTTPException(501, "telemetry backend cannot be queried")
        return tel.fetch(limit=limit, arm=arm)  # type: ignore[no-any-return]

    @app.delete("/learners/{tenant_id}")
    def erase(tenant_id: str) -> dict[str, int]:
        """Per-learner erasure (synchronous by design, plan R6)."""
        c = _comps()
        n = c.episodic.delete_tenant(tenant_id)
        c.sessions.erase_tenant(tenant_id)
        return {"deleted_records": n}

    return app


app = create_app()
