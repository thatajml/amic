"""Support code for the chat UI (`ui/chat.py`): LLM provider presets, gateway construction and
the per-turn decision summary. Kept out of the Streamlit file so it can be unit-tested.

Any LLM works as the tutor: a local Ollama model, or any OpenAI-compatible chat API. AMIC itself
does not depend on which model answers; it only decides what memory goes into the prompt.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from amic.contracts.models import AgentResponse, DecisionTrace
from amic.generation.gateway.base import LLMGateway


@dataclass(frozen=True)
class Preset:
    base_url: str
    key_env: str  # env var (or .env entry) holding the API key; "" = no key needed
    default_model: str
    reasoning: str | None = None  # reasoning_effort sent to the API, if the model supports it


OLLAMA = "Ollama (local)"
CUSTOM = "Custom OpenAI-compatible"
PRESETS: dict[str, Preset] = {
    "Gemini": Preset(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-3.5-flash-lite",
        "low",
    ),
    "OpenAI": Preset("https://api.openai.com/v1", "OPENAI_API_KEY", ""),
    "OpenRouter": Preset("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", ""),
    "Groq": Preset("https://api.groq.com/openai/v1", "GROQ_API_KEY", ""),
    "LM Studio / vLLM (local)": Preset("http://localhost:1234/v1", "", ""),
    CUSTOM: Preset("", "", ""),
}
PROVIDERS = [OLLAMA, *PRESETS]


def list_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Models installed in the local Ollama server; [] if it is not reachable."""
    import httpx

    try:
        r = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except (httpx.HTTPError, ValueError, KeyError):
        return []


def make_gateway(
    provider: str,
    model: str,
    *,
    base_url: str = "",
    key_env: str = "",
    api_key: str = "",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> LLMGateway:
    """Build the tutor gateway for the chosen provider. The key comes from `api_key` if given
    (typed into the UI, never stored), else from `key_env` in the environment / .env."""
    if not model:
        raise ValueError("choose a model")
    if provider == OLLAMA:
        from amic.generation.gateway.ollama import OllamaGateway

        return OllamaGateway(
            model, base_url or "http://localhost:11434", temperature, 0, max_tokens
        )
    from dotenv import load_dotenv

    from amic.generation.gateway.openai_compat import OpenAICompatGateway

    load_dotenv(".env")
    preset = PRESETS.get(provider, PRESETS[CUSTOM])
    url = base_url or preset.base_url
    if not url:
        raise ValueError("enter the API base URL")
    reasoning = preset.reasoning
    return OpenAICompatGateway(
        model=model,
        base_url=url,
        # local OpenAI-compatible servers accept any key; the SDK still needs a non-empty one
        api_key=api_key or (None if (key_env or preset.key_env) else "not-needed"),
        api_key_env=key_env or preset.key_env or "OPENAI_API_KEY",
        temperature=temperature,
        # reasoning tokens count against the cap, so leave room for the answer itself
        max_tokens=max(max_tokens, 1024) if reasoning else max_tokens,
        reasoning_effort=reasoning,
    )


def find_trace(traces: list[DecisionTrace], resp: AgentResponse) -> DecisionTrace | None:
    for t in reversed(traces):
        if t.request_id == resp.trace_id:
            return t
    return traces[-1] if traces else None


def decision_summary(resp: AgentResponse, tr: DecisionTrace | None) -> dict[str, Any]:
    """Everything the decision panel shows for one turn, as plain data."""
    d = resp.decision
    out: dict[str, Any] = {
        "route": d.route.value,
        "resolved_by": d.resolved_by.value,
        "p": d.p,
        "band": (d.tau_low, d.tau_high) if d.tau_low is not None else None,
        "rationale": d.rationale,
        "items": [
            {"score": round(it.score, 3), "text": it.text, "ts": it.ts.isoformat()}
            for it in resp.items
        ],
    }
    if tr is not None:
        lat = tr.latency
        out |= {
            "signals": sorted(
                ((k, round(v, 3)) for k, v in tr.features.items() if v and not k.startswith("x_")),
                key=lambda kv: -abs(kv[1]),
            )[:8],
            "latency_ms": {
                "embed": round(lat.embed, 1),
                "gate": round(lat.gate, 2),
                "retrieval": round(lat.retrieval, 1),
                "generation": round(lat.generation, 0),
                "total": round(lat.total, 0),
            },
            "prompt_tokens": tr.prompt_tokens,
            "memory_tokens": tr.memory_tokens,
            "completion_tokens": tr.completion_tokens,
        }
    return out


# ------------------------------------------------------------------------------------ accounts
MIN_PIN = 4
USERS_COLLECTION = "amic_users"  # Chroma collection for registered users (never the frozen edu_v1)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:20] or "user"


def _hash_pin(pin: str, salt: bytes) -> bytes:
    return hashlib.scrypt(pin.encode(), salt=salt, n=2**14, r=8, p=1)


class UserRegistry:
    """Name + PIN accounts for the chat UI. A name is unique (case-insensitive); the PIN is stored
    only as a salted scrypt hash. No recovery: a forgotten PIN means a new account."""

    def __init__(self, path: str = "data/chat_users.sqlite") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY,"
            " display_name TEXT NOT NULL, name_key TEXT NOT NULL UNIQUE,"
            " pin_salt BLOB NOT NULL, pin_hash BLOB NOT NULL, created REAL NOT NULL)"
        )
        self._db.commit()

    def register(self, name: str, pin: str) -> str:
        """Create an account and return its learner id. Raises ValueError with a user-facing
        message for an empty name, a short PIN or a name that is taken."""
        name = " ".join(name.split())
        if not name:
            raise ValueError("Enter a name.")
        if len(pin) < MIN_PIN:
            raise ValueError(f"The PIN needs at least {MIN_PIN} characters.")
        salt = secrets.token_bytes(16)
        uid = f"u-{_slug(name)}-{secrets.token_hex(3)}"
        try:
            with self._db:
                self._db.execute(
                    "INSERT INTO users VALUES (?,?,?,?,?,?)",
                    (uid, name, name.casefold(), salt, _hash_pin(pin, salt), time.time()),
                )
        except sqlite3.IntegrityError:
            raise ValueError("That name is taken. Sign in, or pick another name.") from None
        return uid

    def login(self, name: str, pin: str) -> tuple[str, str] | None:
        """(learner id, display name) when name and PIN match, else None."""
        row = self._db.execute(
            "SELECT user_id, display_name, pin_salt, pin_hash FROM users WHERE name_key = ?",
            (" ".join(name.split()).casefold(),),
        ).fetchone()
        salt = row[2] if row else bytes(16)  # same work either way: no timing hint on the name
        ok = hmac.compare_digest(_hash_pin(pin, salt), row[3] if row else bytes(64))
        return (row[0], row[1]) if row and ok else None


def save_name_to_profile(profiles: Any, user_id: str, name: str) -> None:
    """Create the learner's M3 profile with their name, so it exists before the first turn."""
    from amic.memory.stores.profile import LearnerProfile

    prof = profiles.get_or_new(user_id)
    if not isinstance(prof, LearnerProfile):  # pragma: no cover
        raise TypeError("profile store returned an unexpected object")
    prof.notes.setdefault("personal", {})["name"] = f"Name is {name}"
    profiles.put(prof)
