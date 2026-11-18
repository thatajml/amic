"""Composition root: builds every component from Settings. Tests inject fakes via overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from amic.contracts.config import Settings
from amic.contracts.interfaces import TelemetryCollector
from amic.contracts.models import ArmName, RequestContext
from amic.control.route_planner import RoutePlanner
from amic.generation.gateway.base import LLMGateway
from amic.generation.prompt import PromptBuilder
from amic.memory.assembler import ContextAssembler
from amic.memory.embedder import Embedder
from amic.memory.retrieval import ChromaRetrievalEngine
from amic.memory.sketch import EpisodicSketchStore
from amic.memory.stores.episodic import EpisodicStore, make_chroma_client
from amic.orchestration.graph import PipelineArm, new_request_id
from amic.service.session import SessionManager
from amic.writepath.writer import MemoryWriter


@dataclass
class Components:
    settings: Settings
    embedder: Embedder
    episodic: EpisodicStore
    retrieval: ChromaRetrievalEngine
    assembler: ContextAssembler
    prompts: PromptBuilder
    llm: LLMGateway
    sessions: SessionManager
    telemetry: TelemetryCollector
    writer: MemoryWriter
    planner: RoutePlanner
    sketches: EpisodicSketchStore
    config_hash: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def build_components(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    llm: LLMGateway | None = None,
    redis: Any = None,
    chroma_client: Any = None,
    telemetry: TelemetryCollector | None = None,
) -> Components:
    s = settings or Settings()
    if embedder is None:
        from amic.memory.embedder import STEmbedder

        embedder = STEmbedder(s.models.embedding, s.models.embedding_device)
    if llm is None:
        from amic.generation.gateway.ollama import OllamaGateway

        m = s.models
        llm = OllamaGateway(m.tutor, m.ollama_base_url, m.temperature, m.seed, m.max_answer_tokens)
    if redis is None:
        import redis as redis_lib

        redis = redis_lib.Redis.from_url(s.redis_url)
    if chroma_client is None:
        chroma_client = make_chroma_client(
            s.chroma_client, s.chroma_path, s.chroma_host, s.chroma_port
        )
    if telemetry is None:
        from amic.telemetry.store import SqliteTelemetry

        telemetry = SqliteTelemetry(s.metrics_db_url)

    episodic = EpisodicStore(chroma_client, s.chroma_collection, s.chroma_mode)
    sketches = EpisodicSketchStore(episodic)
    writer = MemoryWriter(embedder, episodic)
    writer.on_written.append(sketches.invalidate)  # M4 rebuilt lazily after each write
    return Components(
        settings=s,
        embedder=embedder,
        episodic=episodic,
        retrieval=ChromaRetrievalEngine(episodic, s.routes),
        assembler=ContextAssembler(),
        prompts=PromptBuilder(),
        llm=llm,
        sessions=SessionManager(redis, s.m0_turns),
        telemetry=telemetry,
        writer=writer,
        planner=RoutePlanner(s.routes),
        sketches=sketches,
        config_hash=s.config_hash(),
    )


def build_arm(name: ArmName, comps: Components) -> PipelineArm:
    from amic.arms.fixed import AlwaysArm, NeverArm, OracleArm

    arms: dict[str, type[PipelineArm]] = {
        "always": AlwaysArm,
        "never": NeverArm,
        "oracle": OracleArm,
    }
    if name == "amic":
        from amic.arms.amic import AmicArm  # Day 5

        return AmicArm(comps)
    return arms[name](comps)


def make_context(
    comps: Components,
    tenant_id: str,
    session_id: str,
    query: str,
    arm: ArmName,
    *,
    now: datetime | None = None,
    request_id: str | None = None,
    **extra: Any,
) -> RequestContext:
    """L2 duty: resolve the tenant, load M0, record session activity, build the context."""
    m0 = comps.sessions.get_turns(session_id)
    comps.sessions.touch(tenant_id, session_id, now or datetime.now(UTC))
    return RequestContext(
        request_id=request_id or new_request_id(),
        tenant_id=tenant_id,
        session_id=session_id,
        query=query,
        m0_turns=m0,
        arm=arm,
        now=now,
        **extra,
    )
