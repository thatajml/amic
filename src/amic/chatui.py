"""Support code for the chat UI (`ui/chat.py`): LLM provider presets, gateway construction and
the per-turn decision summary. Kept out of the Streamlit file so it can be unit-tested.

Any LLM works as the tutor: a local Ollama model, or any OpenAI-compatible chat API. AMIC itself
does not depend on which model answers; it only decides what memory goes into the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
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
