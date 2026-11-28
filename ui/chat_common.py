"""Shared pieces of the chat UI (entry `ui/chat.py`, pages in `ui/app_pages/`)."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from amic.chatui import USERS_COLLECTION, UserRegistry, make_gateway

DASHBOARD_URL = os.environ.get("AMIC_DASHBOARD_URL", "http://localhost:8501")
EDUCATOR_URL = os.environ.get("AMIC_EDUCATOR_URL", "http://localhost:8502")


@st.cache_resource
def app_settings():  # noqa: ANN201
    from amic.contracts.config import load_settings

    return load_settings(None)


def users_collection() -> str:
    """Where accounts' memory lives. Locally: their own `amic_users` collection, apart from the
    frozen `edu_v1`. Against a shared Chroma server (Docker) the worker and the Educator Console
    read the stack's collection, so accounts must use that one or the worker's writes would never
    be read back; `edu_v1` is not on that server, so nothing seeded can be touched."""
    s = app_settings()
    return s.chroma_collection if s.chroma_client == "http" else USERS_COLLECTION


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embedder(name: str, device: str):  # noqa: ANN201
    from amic.memory.embedder import STEmbedder

    return STEmbedder(name, device)


@st.cache_resource(show_spinner="Starting AMIC...")
def get_stack(
    provider: str,
    model: str,
    base_url: str,
    key_env: str,
    api_key: str,
    lam: float,
    temperature: float,
    collection: str,
):  # noqa: ANN201
    from amic.app import build_arm, build_components
    from amic.contracts.config import load_settings
    from amic.telemetry.store import InMemoryTelemetry

    s = load_settings(None, chroma_collection=collection, gate={"lam": lam})
    llm = make_gateway(
        provider,
        model,
        base_url=base_url,
        key_env=key_env,
        api_key=api_key,
        temperature=temperature,
        max_tokens=s.models.max_answer_tokens,
    )
    tel = InMemoryTelemetry()
    comps = build_components(
        s,
        embedder=get_embedder(s.models.embedding, s.models.embedding_device),
        llm=llm,
        telemetry=tel,
    )
    arms = {name: build_arm(name, comps) for name in ("amic", "always", "never")}  # type: ignore[arg-type]
    return comps, arms, tel


@st.cache_resource
def get_registry() -> UserRegistry:
    return UserRegistry(os.environ.get("AMIC_CHAT_USERS_DB", "data/chat_users.sqlite"))


def learner_options() -> dict[str, str]:
    """Label -> learner id of the seeded learners (data/edu/learners.jsonl). Empty when the
    seeded memory is not reachable: no file, or a shared Chroma server (Docker) that does not
    hold the frozen `edu_v1` collection."""
    opts: dict[str, str] = {}
    if app_settings().chroma_client == "http" or not Path("data/edu/learners.jsonl").exists():
        return opts
    from amic.eval.datasets import edu

    for lr in edu.load_learners():
        opts[f"{lr.learner_id} · {lr.level} · {lr.goal[:40]}"] = lr.learner_id
    return opts
