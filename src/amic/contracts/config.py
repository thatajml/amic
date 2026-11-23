"""Config schema (plan section 4). YAML file + environment overrides (prefix AMIC_, nested '__')."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GateConfig(BaseModel):
    lam: float = Field(4.0, gt=0, description="C(false skip) / C(false retrieve)")
    tau_mode: Literal["cost", "fixed"] = "cost"
    tau_fixed: float = 0.5
    band_low_offset: float = 0.08
    band_high_offset: float = 0.15
    tier1_model: Literal["logreg", "lgbm"] = "logreg"
    calibration: Literal["platt", "isotonic"] = "isotonic"
    tier1_artifact: str = "models/tier1.joblib"
    abstain_fallback: Literal["SHALLOW", "DEEP"] = "SHALLOW"
    use_band: bool = True  # with Tier 2 off: in-band p -> abstain_fallback; False = hard tau
    tier2_max_rate: float = 0.12  # hard cap: share of ALL gate decisions that may reach Tier 2
    tier2_timeout_ms: float = 800.0  # slower verdicts fail open to abstain_fallback
    distill_path: str | None = "data/distill/tier2.jsonl"  # Tier-2 verdicts for Tier-1 retrains


class AblationFlags(BaseModel):
    tiers: Literal["t0", "t0_t1", "full"] = "t0_t1"
    use_sketch_probe: bool = True
    use_cache: bool = False
    use_escalation: bool = False


class RouteConfig(BaseModel):
    shallow_k: int = 3
    deep_k: int = 12
    deep_fetch_k: int = 30  # candidates before MMR
    deep_mmr: bool = True
    always_route: Literal["SHALLOW", "DEEP"] = "DEEP"
    shallow_token_budget: int = 400
    deep_token_budget: int = 1000
    # Which stores each route reads. Brief: SHALLOW = profile + top M2 facts, DEEP = M1 + M2.
    # Defaults keep the first results' M1-only routes; configs/ablations/m2.yaml switches.
    shallow_stores: list[Literal["M1", "M2"]] = Field(default_factory=lambda: ["M1"])
    deep_stores: list[Literal["M1", "M2"]] = Field(default_factory=lambda: ["M1"])
    m2_k: int = 5  # facts added to a route that also reads M1


class WritePathConfig(BaseModel):
    """Async write path (L7). `inline` consolidates right after the response (no worker);
    `sqlite` / `redis` only enqueue, and `python -m amic.writepath.worker` consolidates."""

    queue: Literal["inline", "sqlite", "redis"] = "inline"
    queue_db: str = "data/writequeue.sqlite"
    stream: str = "amic:writes"
    group: str = "consolidators"
    extract_facts: bool = False  # LLM fact extraction into M2 (needs the gate model)
    salience_min_words: int = 4


class CacheConfig(BaseModel):
    """DecisionCache (brief 6.4), on with ablation.use_cache."""

    nbits: int = 16  # LSH bits of the query vector
    ttl_s: float = 900.0


class VerifyConfig(BaseModel):
    """ResponseVerifier + bounded escalation (brief P6 / M7); on with ablation.use_escalation.

    Only answers generated WITHOUT memory (route SKIP) are verified; a failed check retries once
    with `escalation_route` memory, never more.
    """

    mode: Literal["lexical", "llm", "both"] = "both"
    escalation_route: Literal["SHALLOW", "DEEP"] = "SHALLOW"
    llm_max_tokens: int = 24


class ModelConfig(BaseModel):
    embedding: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    tutor: str = "llama3.1:8b"
    # Tier 2 and the LLM verifier. The brief asks for a small model (llama3.2:3b), but on the
    # 8 GB dev GPU Ollama cannot keep it resident next to the 8B tutor (measured 2026-11-23:
    # 5.28 GB + 2.21 GB, one is evicted per load), so every Tier-2 call would pay a model swap
    # of seconds. The resident tutor model is used instead (flagged deviation).
    tier2: str = "llama3.1:8b"
    # Context window for every Ollama call. None = Ollama's default. Tier 2 and the verifier use
    # the same value as the tutor, or Ollama reloads the shared model on every switch.
    num_ctx: int | None = None
    ollama_base_url: str = "http://localhost:11434"
    # gemini-2.5-flash is closed to new users; 3.8-flash took ~71 s/call (2026-11-18 test);
    # 3.5-flash-lite answers in 3-7 s and is far stronger than the 8B tutor it judges.
    judge: str = "gemini-3.5-flash-lite"
    judge_reasoning: str | None = "low"
    judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    temperature: float = 0.0
    seed: int = 0
    max_answer_tokens: int = 256


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AMIC_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )

    arm: Literal["always", "never", "amic", "oracle"] = "amic"
    gate: GateConfig = Field(default_factory=GateConfig)
    ablation: AblationFlags = Field(default_factory=AblationFlags)
    routes: RouteConfig = Field(default_factory=RouteConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    verify: VerifyConfig = Field(default_factory=VerifyConfig)
    writepath: WritePathConfig = Field(default_factory=WritePathConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    session_backend: Literal["redis", "sqlite"] = "redis"  # sqlite: offline runs, no Docker
    session_db: str = "data/sessions.sqlite"
    redis_url: str = "redis://localhost:6379/0"
    chroma_client: Literal["embedded", "http"] = "embedded"  # see docs/results/probe.md
    chroma_path: str = "data/chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_mode: Literal["single", "per_tenant"] = "single"
    chroma_collection: str = "amic_memory"
    metrics_db_url: str = "sqlite:///./data/metrics.db"
    m0_turns: int = 6
    # "tutor" (education study) or "assistant" (LongMemEval: general assistant, dated prompt)
    prompt_profile: Literal["tutor", "assistant"] = "tutor"

    def config_hash(self) -> str:
        """Stable hash of everything that affects behaviour; stored on every trace."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _merge(data: dict[str, Any], more: dict[str, Any]) -> dict[str, Any]:
    """Top-level keys replace; nested sections (gate, routes, ...) merge key by key."""
    for key, val in more.items():
        if isinstance(val, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **val}
        else:
            data[key] = val
    return data


def load_settings(
    path: str | Path | Sequence[str | Path] | None = None, **overrides: Any
) -> Settings:
    """Precedence (highest first): explicit overrides > YAML file(s) > AMIC_* env / .env > defaults.

    Several YAML files merge in order (later ones win), e.g. a dataset config plus an ablation.
    Experiment configs (YAML) must be reproducible, so they outrank the environment;
    env is meant for hosts and secrets that YAML files leave unset.
    """
    data: dict[str, Any] = {}
    paths = [] if path is None else [path] if isinstance(path, str | Path) else list(path)
    for p in paths:
        _merge(data, yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {})
    return Settings(**_merge(data, overrides))
