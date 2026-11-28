"""PromptBuilder. Identical template for every arm, so arms differ only in the memory block.

Two profiles: `tutor` (the education study; byte-identical to the first results, so the
generation cache still matches) and `assistant` (LongMemEval: a general assistant that is told
the current date, since many questions are about time).
"""

from __future__ import annotations

from datetime import datetime

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
SYSTEM_ASSISTANT = (
    "You are a helpful personal assistant. Answer the user's latest message concisely. "
    "If a 'Memory' section is present, it holds dated excerpts from your earlier conversations "
    "with this user; use it to answer questions about the user and their past. If the answer "
    "depends on earlier conversations and is in neither the memory nor the current "
    "conversation, say that you do not know instead of guessing."
)
# Chat UI: the tutor prompt plus a guard against inventing facts about the learner (a model with no
# memory otherwise makes up "your project is ..."). A separate profile so `tutor` stays
# byte-identical for the experiments and their generation cache.
SYSTEM_CHAT = SYSTEM + (
    " If the student asks about themselves, their goals, projects or earlier work and neither the "
    "memory nor the conversation says, tell them you do not know yet and ask them; never guess."
)
PROFILES: dict[str, tuple[str, str, bool]] = {
    # profile -> (system prompt, memory header, show the current date)
    "tutor": (SYSTEM, "Learner memory (from earlier sessions)", False),
    "tutor_chat": (SYSTEM_CHAT, "Learner memory (from earlier sessions)", False),
    "assistant": (SYSTEM_ASSISTANT, "Memory (excerpts from earlier conversations)", True),
}


class PromptBuilder:
    def __init__(
        self,
        system: str = SYSTEM,
        memory_header: str = "Learner memory (from earlier sessions)",
        show_date: bool = False,
    ) -> None:
        self.system = system
        self.memory_header = memory_header
        self.show_date = show_date

    @classmethod
    def for_profile(cls, profile: str) -> PromptBuilder:
        return cls(*PROFILES[profile])

    def build(
        self,
        query: str,
        m0_turns: list[Turn],
        memory_block: str = "",
        now: datetime | None = None,
    ) -> list[Message]:
        system = self.system
        if self.show_date and now is not None:
            system += f"\n\nCurrent date: {now:%Y/%m/%d (%a) %H:%M}"
        if memory_block:
            system += f"\n\n{self.memory_header}:\n{memory_block}"
        msgs: list[Message] = [{"role": "system", "content": system}]
        msgs += [{"role": t.role, "content": t.text} for t in m0_turns]
        msgs.append({"role": "user", "content": query})
        return msgs
