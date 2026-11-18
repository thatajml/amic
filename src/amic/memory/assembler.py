"""ContextAssembler: turns retrieved MemoryItems into a prompt block within a token budget."""

from __future__ import annotations

from amic.contracts.models import MemoryItem


def estimate_tokens(text: str) -> int:
    """~4 chars per token for English/code. Real prompt counts come from the LLM."""
    return max(1, (len(text) + 3) // 4)


class ContextAssembler:
    def assemble(
        self, items: list[MemoryItem], token_budget: int
    ) -> tuple[str, list[MemoryItem], int]:
        """Return (block, items actually used, estimated tokens). Highest score first."""
        if not items or token_budget <= 0:
            return "", [], 0
        lines: list[str] = []
        used: list[MemoryItem] = []
        total = 0
        for it in sorted(items, key=lambda x: x.score, reverse=True):
            line = f"- [{it.ts:%Y-%m-%d}] {it.text}"
            cost = estimate_tokens(line)
            if total + cost > token_budget:
                continue
            lines.append(line)
            used.append(it)
            total += cost
        # chronological order reads better for the model
        order = sorted(range(len(used)), key=lambda i: used[i].ts)
        return "\n".join(lines[i] for i in order), [used[i] for i in order], total
