"""Config schema (plan section 4). YAML file + environment overrides (prefix AMIC_, nested '__')."""

from __future__ import annotations

import hashlib
import json
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
    tier2_max_rate: float = 0.12
    tier2_timeout_ms: float = 800.0


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


class ModelConfig(BaseModel):
    embedding: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    tutor: str = "llama3.1:8b"
    tier2: str = "llama3.2:3b"
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
    redis_url: str = "redis://localhost:6379/0"
    chroma_client: Literal["embedded", "http"] = "embedded"  # see docs/results/probe.md
    chroma_path: str = "data/chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_mode: Literal["single", "per_tenant"] = "single"
    chroma_collection: str = "amic_memory"
    metrics_db_url: str = "sqlite:///./data/metrics.db"
    m0_turns: int = 6

    def config_hash(self) -> str:
        """Stable hash of everything that affects behaviour; stored on every trace."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_settings(path: str | Path | None = None, **overrides: Any) -> Settings:
    """Precedence (highest first): explicit overrides > YAML file > AMIC_* env / .env > defaults.

    Experiment configs (YAML) must be reproducible, so they outrank the environment;
    env is meant for hosts and secrets that YAML files leave unset.
    """
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **val}
        else:
            data[key] = val
    return Settings(**data)
