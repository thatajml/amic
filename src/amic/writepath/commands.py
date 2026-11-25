"""Explicit memory commands (Tier 0 R4 -> route WRITE): `/remember <text>` and `/forget <text>`.

The response path never writes (P5): it answers with a fixed acknowledgement (no LLM call) and
enqueues a WriteIntent with op "remember" / "forget"; the ConsolidationWorker applies it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CMD = re.compile(r"^\s*/(remember|forget)\b[\s:,-]*(.*)$", re.I | re.S)


@dataclass(frozen=True)
class Command:
    op: str  # "remember" | "forget"
    text: str  # payload, may be empty


def parse_command(query: str) -> Command | None:
    m = _CMD.match(query)
    if not m:
        return None
    return Command(m.group(1).lower(), m.group(2).strip())


def acknowledgement(cmd: Command) -> str:
    if not cmd.text:
        return (
            f"Tell me what to {cmd.op} after the command, for example "
            f"`/{cmd.op} I prefer short worked examples`."
        )
    if cmd.op == "remember":
        return f'Noted. I will remember: "{cmd.text}".'
    return f'Understood. I will forget what I stored about: "{cmd.text}".'
