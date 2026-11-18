"""LLMGateway: the only way any layer talks to a language model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_ms: float
    prefill_ms: float = 0.0  # prompt processing; 0 when the provider does not report it
    decode_ms: float = 0.0
    raw: dict[str, object] = field(default_factory=dict)


class LLMGateway(Protocol):
    model: str

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult: ...


class FakeLLM:
    """Deterministic stand-in for tests: echoes the question and whether memory was supplied."""

    model = "fake"

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        self.calls.append(messages)
        question = messages[-1]["content"]
        has_mem = "Learner memory (from earlier sessions)" in messages[0]["content"]
        text = f"[fake answer | memory={'yes' if has_mem else 'no'}] {question}"
        prompt_tokens = sum(len(m["content"]) for m in messages) // 4
        return LLMResult(text=text, prompt_tokens=prompt_tokens, completion_tokens=12, total_ms=1.0)
