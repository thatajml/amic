"""PromptBuilder. Identical template for every arm, so arms differ only in the memory block."""

from __future__ import annotations

from amic.contracts.models import Turn
from amic.generation.gateway.base import Message

SYSTEM = (
    "You are a patient, encouraging Python tutor for beginner programmers. "
    "Answer the student's latest message concisely (at most about 150 words). "
    "Use a short code example when it helps. "
    "If a 'Learner memory' section is present, use it only where it is relevant to personalise "
    "the answer (for example, refer back to an earlier mistake). Never invent past interactions "
    "that are not in the memory or the conversation."
)


class PromptBuilder:
    def __init__(self, system: str = SYSTEM) -> None:
        self.system = system

    def build(self, query: str, m0_turns: list[Turn], memory_block: str = "") -> list[Message]:
        system = self.system
        if memory_block:
            system += f"\n\nLearner memory (from earlier sessions):\n{memory_block}"
        msgs: list[Message] = [{"role": "system", "content": system}]
        msgs += [{"role": t.role, "content": t.text} for t in m0_turns]
        msgs.append({"role": "user", "content": query})
        return msgs
