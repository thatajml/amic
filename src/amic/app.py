"""Composition root: builds every component from Settings. Tests inject fakes via overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from amic.contracts.config import Settings
from amic.contracts.interfaces import TelemetryCollector
from amic.contracts.models import ArmName, RequestContext
from amic.control.cache import DecisionCache
from amic.control.route_planner import RoutePlanner
from amic.generation.gateway.base import LLMGateway
from amic.generation.prompt import PromptBuilder
from amic.generation.verifier import ResponseVerifier
from amic.memory.assembler import ContextAssembler
from amic.memory.embedder import Embedder
from amic.memory.retrieval import ChromaRetrievalEngine
from amic.memory.sketch import EpisodicSketchStore
from amic.memory.stores.episodic import EpisodicStore, make_chroma_client
from amic.memory.stores.profile import ProfileStore, RedisKV, SqliteKV
from amic.memory.stores.semantic import SemanticStore
from amic.orchestration.graph import PipelineArm, new_request_id
from amic.service.session import SessionManager, SqliteSessionManager
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
    sessions: SessionManager | SqliteSessionManager
    telemetry: TelemetryCollector
    writer: MemoryWriter
    planner: RoutePlanner
    sketches: EpisodicSketchStore
    config_hash: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
    verifier: ResponseVerifier | None = None  # set when ablation.use_escalation is on
    gate_llm: LLMGateway | None = None  # Tier 2 / verifier model (models.tier2)
    semantic: SemanticStore | None = None  # M2 facts
    decision_cache: DecisionCache | None = None  # set when ablation.use_cache is on
    write_queue: Any = None  # WriteIntentQueue: where answered turns go (see build_queue)
    profiles: ProfileStore | None = None  # M3 learner profiles


def build_queue(comps: Components, for_worker: bool = False) -> Any:
    """The WriteIntentQueue from settings.writepath.queue. `inline` consolidates in-process right
    after the response (the pre-queue behaviour); `sqlite` / `redis` only enqueue, and the
    ConsolidationWorker process consumes."""
    from amic.writepath.queue import InlineQueue, RedisStreamQueue, SqliteWriteQueue
    from amic.writepath.worker import build_worker

    w = comps.settings.writepath
    if w.queue == "inline":
        return InlineQueue(build_worker(comps).process)
    if w.queue == "sqlite":
        return SqliteWriteQueue(w.queue_db)
    redis = comps.extras.get("redis")
    if redis is None:
        raise RuntimeError("writepath.queue=redis needs the Redis session backend")
    consumer = "worker" if for_worker else "api"
    return RedisStreamQueue(redis, w.stream, w.group, consumer)


def gate_llm_from_settings(s: Settings, timeout_s: float | None = None) -> LLMGateway:
    """The Tier-2 / verifier model: deterministic, short JSON answers, same context as the tutor
    (Ollama would otherwise reload the shared model on every switch)."""
    from amic.generation.gateway.ollama import OllamaGateway

    m = s.models
    return OllamaGateway(
        m.tier2,
        m.ollama_base_url,
        temperature=0.0,
        seed=0,
        max_tokens=s.verify.llm_max_tokens,
        num_ctx=m.num_ctx,
        timeout_s=timeout_s,
    )


def build_components(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    llm: LLMGateway | None = None,
    gate_llm: LLMGateway | None = None,
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
        llm = OllamaGateway(
            m.tutor,
            m.ollama_base_url,
            m.temperature,
            m.seed,
            m.max_answer_tokens,
            num_ctx=m.num_ctx,
        )
    if redis is None and s.session_backend == "redis":
        import redis as redis_lib
        from redis.backoff import ExponentialBackoff
        from redis.retry import Retry

        redis = redis_lib.Redis.from_url(
            s.redis_url,
            socket_timeout=5,
            socket_connect_timeout=5,
            health_check_interval=30,
            retry=Retry(ExponentialBackoff(cap=2.0, base=0.1), retries=5),
            retry_on_timeout=True,
        )
    if chroma_client is None:
        chroma_client = make_chroma_client(
            s.chroma_client, s.chroma_path, s.chroma_host, s.chroma_port
        )
    if telemetry is None:
        from amic.telemetry.store import SqliteTelemetry

        telemetry = SqliteTelemetry(s.metrics_db_url)

    sessions: SessionManager | SqliteSessionManager = (
        SessionManager(redis, s.m0_turns)
        if redis is not None
        else SqliteSessionManager(s.session_db, s.m0_turns)
    )
    episodic = EpisodicStore(chroma_client, s.chroma_collection, s.chroma_mode)
    # M4 sketch: rebuilt lazily whenever the learner's memory version changes (any process)
    sketches = EpisodicSketchStore(episodic, version_fn=sessions.mem_version)
    writer = MemoryWriter(embedder, episodic)

    def bump(tenant_id: str) -> None:
        sessions.bump_mem_version(tenant_id)

    writer.on_written.append(bump)  # invalidates M4 + the decision cache in every process
    writer.on_written.append(sketches.invalidate)

    verifier = None
    if s.ablation.use_escalation:
        if gate_llm is None and s.verify.mode != "lexical":
            gate_llm = gate_llm_from_settings(s, timeout_s=10.0)
        verifier = ResponseVerifier(s.verify.mode, gate_llm, s.verify.llm_max_tokens)
    semantic = SemanticStore(chroma_client, s.chroma_collection, s.chroma_mode)
    profiles = ProfileStore(
        RedisKV(redis)
        if redis is not None
        else SqliteKV(":memory:" if s.session_db == ":memory:" else s.profile.db)
    )
    reranker = None
    if s.routes.deep_rerank:
        from amic.memory.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker(s.routes.rerank_model, s.routes.rerank_device)
    comps = Components(
        settings=s,
        embedder=embedder,
        episodic=episodic,
        retrieval=ChromaRetrievalEngine(
            episodic, s.routes, semantic, profiles=profiles, embedder=embedder, reranker=reranker
        ),
        assembler=ContextAssembler(),
        prompts=PromptBuilder.for_profile(s.prompt_profile),
        llm=llm,
        sessions=sessions,
        telemetry=telemetry,
        writer=writer,
        planner=RoutePlanner(s.routes),
        sketches=sketches,
        config_hash=s.config_hash(),
        verifier=verifier,
        gate_llm=gate_llm,
        semantic=semantic,
        profiles=profiles,
        decision_cache=(
            DecisionCache(s.cache.nbits, s.cache.ttl_s) if s.ablation.use_cache else None
        ),
    )
    if redis is not None:
        comps.extras["redis"] = redis
    comps.write_queue = build_queue(comps)
    return comps


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
