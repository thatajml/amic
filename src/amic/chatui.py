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
import unicodedata
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
    return None  # never another request's (possibly another user's) trace


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
MAX_NAME = 40
MAX_FAILS = 5  # wrong PINs in a row before the account is locked
LOCK_SECONDS = 300
USERS_COLLECTION = "amic_users"  # Chroma collection for registered users (never the frozen edu_v1)
_ALPHABETS = ("LATIN", "CYRILLIC", "GREEK")  # a name may use only one of these


class LoginLocked(Exception):
    """Too many wrong PINs for this name; `seconds` until the next try is allowed."""

    def __init__(self, seconds: int) -> None:
        super().__init__(f"locked for {seconds}s")
        self.seconds = seconds


def normalize_name(name: str) -> str:
    """The form a name is stored and compared in: Unicode NFKC (so an accent typed as one or two
    characters is the same name), invisible format/control characters removed, whitespace
    collapsed. Compared case-insensitively via casefold()."""
    s = unicodedata.normalize("NFKC", name)
    s = "".join(c for c in s if unicodedata.category(c) not in ("Cc", "Cf") or c.isspace())
    return " ".join(s.split())


def _alphabets(name: str) -> set[str]:
    out = set()
    for c in name:
        if c.isalpha():
            first = unicodedata.name(c, "").split(" ")[0]
            if first in _ALPHABETS:
                out.add(first)
    return out


def plain(text: str) -> str:
    """Escape Markdown so a user-chosen name is shown as typed (no links, bold, colours)."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|~$<>:@&^])", r"\\\1", text)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:20] or "user"


def _hash_pin(pin: str, salt: bytes) -> bytes:
    return hashlib.scrypt(pin.encode(), salt=salt, n=2**14, r=8, p=1)


def _pin_ok(pin: str) -> str:
    """The PIN as stored: surrounding spaces dropped (a pasted trailing space is not part of it);
    ValueError if what is left is shorter than MIN_PIN."""
    p = pin.strip()
    if len(p) < MIN_PIN:
        raise ValueError(f"The PIN needs at least {MIN_PIN} characters (spaces do not count).")
    return p


class UserRegistry:
    """Name + PIN accounts for the chat UI. A name is unique (case-insensitive, Unicode-
    normalised); the PIN is stored only as a salted scrypt hash. After MAX_FAILS wrong PINs in a
    row an account is locked for LOCK_SECONDS. A forgotten PIN is reset by staff (chatui_admin)."""

    def __init__(self, path: str = "data/chat_users.sqlite") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._clock = time.time  # tests replace it
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY,"
            " display_name TEXT NOT NULL, name_key TEXT NOT NULL UNIQUE,"
            " pin_salt BLOB NOT NULL, pin_hash BLOB NOT NULL, created REAL NOT NULL)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS attempts (name_key TEXT PRIMARY KEY,"
            " fails INTEGER NOT NULL, locked_until REAL NOT NULL, updated REAL NOT NULL)"
        )
        self._db.commit()

    @staticmethod
    def _key(name: str) -> str:
        return normalize_name(name).casefold()

    def register(self, name: str, pin: str) -> str:
        """Create an account and return its learner id. Raises ValueError with a user-facing
        message for a bad name or PIN, or a name that is taken."""
        name = normalize_name(name)
        if not name:
            raise ValueError("Enter a name.")
        if len(name) > MAX_NAME:
            raise ValueError(f"The name can have at most {MAX_NAME} characters.")
        if len(_alphabets(name)) > 1:
            raise ValueError(
                "Use one alphabet in a name (do not mix Latin, Cyrillic and Greek letters)."
            )
        pin = _pin_ok(pin)
        salt = secrets.token_bytes(16)
        uid = f"u-{_slug(name)}-{secrets.token_hex(3)}"
        try:
            with self._db:
                self._db.execute(
                    "INSERT INTO users VALUES (?,?,?,?,?,?)",
                    (uid, name, self._key(name), salt, _hash_pin(pin, salt), self._clock()),
                )
        except sqlite3.IntegrityError:
            raise ValueError("That name is taken. Sign in, or pick another name.") from None
        return uid

    def login(self, name: str, pin: str) -> tuple[str, str] | None:
        """(learner id, display name) when name and PIN match, else None. Raises LoginLocked
        while the name is locked after too many wrong PINs (the PIN is not even checked then).
        Surrounding spaces in the PIN are ignored; a PIN stored with spaces still works."""
        key = self._key(name)
        now = self._clock()
        row = self._db.execute(
            "SELECT fails, locked_until FROM attempts WHERE name_key = ?", (key,)
        ).fetchone()
        if row and row[1] > now:
            raise LoginLocked(int(row[1] - now) + 1)
        acct = self._db.execute(
            "SELECT user_id, display_name, pin_salt, pin_hash FROM users WHERE name_key = ?",
            (key,),
        ).fetchone()
        salt = acct[2] if acct else bytes(16)  # same work either way: no timing hint on the name
        stored = acct[3] if acct else bytes(64)
        ok = False
        for cand in dict.fromkeys((pin.strip(), pin)):  # trimmed first, then as typed (legacy)
            ok |= hmac.compare_digest(_hash_pin(cand, salt), stored) and acct is not None
        with self._db:
            if ok:
                self._db.execute("DELETE FROM attempts WHERE name_key = ?", (key,))
            else:
                fails = (row[0] if row else 0) + 1
                locked = now + LOCK_SECONDS if fails >= MAX_FAILS else 0.0
                self._db.execute(
                    "INSERT OR REPLACE INTO attempts VALUES (?,?,?,?)",
                    (key, 0 if locked else fails, locked, now),
                )
                self._db.execute(  # keep the table small: forget day-old, unlocked entries
                    "DELETE FROM attempts WHERE updated < ? AND locked_until < ?",
                    (now - 86400, now),
                )
        return (acct[0], acct[1]) if ok else None

    def unlock(self, name: str) -> None:
        with self._db:
            self._db.execute("DELETE FROM attempts WHERE name_key = ?", (self._key(name),))

    def set_pin(self, name: str, pin: str) -> str:
        """Staff reset: give an existing account a new PIN (and unlock it). The learner id, and
        so the memory and profile, is kept. Returns the display name; ValueError for an unknown
        name or a short PIN."""
        pin = _pin_ok(pin)
        salt = secrets.token_bytes(16)
        with self._db:
            cur = self._db.execute(
                "UPDATE users SET pin_salt = ?, pin_hash = ? WHERE name_key = ?",
                (salt, _hash_pin(pin, salt), self._key(name)),
            )
        if cur.rowcount == 0:
            raise ValueError(f"No account named {name!r}.")
        self.unlock(name)
        return self._db.execute(  # type: ignore[no-any-return]
            "SELECT display_name FROM users WHERE name_key = ?", (self._key(name),)
        ).fetchone()[0]

    def list_users(self) -> list[tuple[str, str, float]]:
        """(learner id, display name, created) of every account, oldest first."""
        return list(
            self._db.execute("SELECT user_id, display_name, created FROM users ORDER BY created")
        )


def save_name_to_profile(profiles: Any, user_id: str, name: str) -> None:
    """Create the learner's M3 profile with their name, so it exists before the first turn."""
    from amic.memory.stores.profile import LearnerProfile

    prof = profiles.get_or_new(user_id)
    if not isinstance(prof, LearnerProfile):  # pragma: no cover
        raise TypeError("profile store returned an unexpected object")
    prof.notes.setdefault("personal", {})["name"] = f"Name is {name}"
    profiles.put(prof)


class TeeTelemetry:
    """Telemetry for the chat UI: every trace goes to the shared store the dashboard reads AND to a
    per-process list the decision panel reads (`find_trace`). `annotate` (the shadow baseline)
    is forwarded to both so the dashboard sees the saved-tokens figure."""

    def __init__(self, memory: Any, shared: Any) -> None:
        self._memory = memory
        self._shared = shared
        self.traces = memory.traces

    KEEP = 200  # decision-panel traces kept in memory (the shared store keeps everything)

    def record(self, trace: DecisionTrace) -> None:
        self._memory.record(trace)
        if len(self.traces) > 2 * self.KEEP:
            del self.traces[: -self.KEEP]  # same list object the panel reads
        self._shared.record(trace)

    def annotate(self, request_id: str, **fields: Any) -> None:
        self._memory.annotate(request_id, **fields)
        self._shared.annotate(request_id, **fields)

    def fetch(self, *args: Any, **kwargs: Any) -> list[DecisionTrace]:
        return self._shared.fetch(*args, **kwargs)  # type: ignore[no-any-return]
