"""ResponseVerifier (L6, brief P6 / M7): was skipping memory actually safe for this answer?

Runs only on answers generated WITHOUT long-term memory (route SKIP). Two checks:

- lexical (microseconds): the answer claims shared history the current session does not
  contain ("last time", "in our previous session", "remember when you"), or it defers because it
  lacks the learner's history ("I don't have information about your previous sessions",
  "could you remind me which project").
- llm (one call to the Tier-2 model): would the reply have been clearly better with the student's
  notes from earlier sessions?

A failed check makes the pipeline retry ONCE with memory (orchestration/graph.py). A verifier
error keeps the answer (the gate's decision stands); it never blocks the response.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Literal

from amic.contracts.models import Turn
from amic.generation.gateway.base import LLMGateway


def _rx(*alts: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.I)


# Claims about earlier SESSIONS: wrong in an answer that had no memory, whatever M0 says.
PAST_SESSION = _rx(
    r"last time",
    r"(?:in|during|from) (?:our|your|the) (?:last|previous|earlier|past|prior) (?:session|lesson|class|conversation|chat|meeting)s?",
    r"(?:previous|earlier|past|prior|last) (?:session|lesson|class)s?",
    r"remember (?:when|how) (?:you|we)",
    r"as (?:we|i) (?:discussed|covered|talked about|went over|saw|practised|practiced) (?:before|previously|last|earlier)",
    r"(?:we|you) (?:have )?(?:discussed|covered|talked about|went over|worked on|looked at) (?:this|that|it) (?:before|previously)",
    r"(?:you(?:'ve| have)?|we(?:'ve| have)?) (?:already )?(?:struggled|had trouble|got stuck|made (?:this|that|the same) mistake)(?: with| on)? (?:this|that|it )?(?:before|previously|last)",
)
# Claims about the student's history that M0 could support: wrong only with an empty M0.
STUDENT_HISTORY = _rx(
    r"you (?:struggled|had trouble|got stuck|were stuck) (?:with|on)",
    r"your (?:earlier|previous|past) (?:mistake|error|bug|code|question|attempt)s?",
    r"you (?:mentioned|told me|said) (?:before|earlier|previously)",
)
# The answer admits it lacks the learner's history or asks them to restate it.
DEFERRAL = _rx(
    r"(?:don't|do not|can't|cannot) (?:have|see|access|recall|remember|retain)(?: any)?(?: access to)?(?: the)? (?:information|record|records|details|history|memory|context|notes)? ?(?:about|of|from|on)? ?(?:your|our) (?:previous|past|earlier|last|prior)",
    r"(?:don't|do not|can't|cannot) (?:remember|recall) (?:our|your|what we|what you)",
    r"(?:don't|do not) have (?:any )?(?:memory|record|access) (?:of|to) (?:our|your|past|previous)",
    r"(?:could|can) you (?:remind me|tell me again|share again|refresh my memory)",
    r"which (?:bug|error|project|exercise|problem|code|mistake|topic|one) (?:do|are|did|were) you (?:mean|referring|talking)",
    r"(?:not sure|unsure) (?:what|which) .{0,40}(?:you're|you are|you were) referring",
    r"without (?:more )?(?:context|information|details) (?:about|on) (?:your|our) (?:previous|past|earlier)",
)

LLM_PROMPT = """A tutor answered a beginner Python student WITHOUT access to the notes from the
student's previous tutoring sessions (it only saw the current session).

=== CURRENT SESSION SO FAR ===
{m0}

=== STUDENT'S MESSAGE ===
{query}

=== TUTOR'S REPLY ===
{answer}

Would this reply have been clearly better if the tutor had the student's notes from previous
sessions? Answer true if the message refers to earlier sessions or to the student's own past
work, mistakes, progress or preferences, or if the reply guesses or invents facts about the
student. Answer false if the message is self-contained and the reply answers it well.
Return JSON only: {{"needs_memory": true | false}}"""

Mode = Literal["lexical", "llm", "both"]


@dataclass(frozen=True)
class VerifyResult:
    needs_memory: bool
    reason: str  # pass | lexical:{deferral,past-session,history} | llm | llm-error
    lexical_ms: float = 0.0
    llm_ms: float = 0.0

    @property
    def ms(self) -> float:
        return self.lexical_ms + self.llm_ms


def lexical_check(answer: str, m0_turns: list[Turn]) -> str | None:
    """The failing pattern's name, or None when the answer looks safe without memory.

    Deferral is checked first: "I can't see your previous sessions" is an honest deferral,
    not an invented past session, even though it mentions one.
    """
    if DEFERRAL.search(answer):
        return "deferral"
    if PAST_SESSION.search(answer):
        return "past-session"
    if not m0_turns and STUDENT_HISTORY.search(answer):
        return "history"
    return None


def parse_needs_memory(raw: str) -> bool:
    """Parse the LLM's JSON verdict; raises ValueError when it is not a usable boolean."""
    j = json.loads(raw)
    if isinstance(j, list) and len(j) == 1 and isinstance(j[0], dict):
        j = j[0]
    v = j["needs_memory"] if isinstance(j, dict) else j
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in {"true", "yes", "false", "no"}:
        return v.strip().lower() in {"true", "yes"}
    raise ValueError(f"not a boolean verdict: {v!r}")


class ResponseVerifier:
    def __init__(
        self, mode: Mode = "both", llm: LLMGateway | None = None, max_tokens: int = 24
    ) -> None:
        if mode != "lexical" and llm is None:
            raise ValueError(f"verify mode {mode!r} needs an LLM")
        self.mode = mode
        self.llm = llm
        self.max_tokens = max_tokens
        self.calls = 0  # LLM calls made (observable in tests and reports)

    def check(self, query: str, answer: str, m0_turns: list[Turn]) -> VerifyResult:
        t = time.perf_counter()
        hit = lexical_check(answer, m0_turns) if self.mode != "llm" else None
        lex_ms = (time.perf_counter() - t) * 1000
        if hit is not None:
            return VerifyResult(True, f"lexical:{hit}", lex_ms)
        if self.mode == "lexical" or self.llm is None:
            return VerifyResult(False, "pass", lex_ms)
        m0 = "\n".join(f"{t.role.upper()}: {t.text}" for t in m0_turns) or "(new session)"
        prompt = LLM_PROMPT.format(m0=m0, query=query, answer=answer)
        t = time.perf_counter()
        self.calls += 1
        try:
            raw = self.llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0.0,
                seed=0,
                json_mode=True,
            ).text
            needs = parse_needs_memory(raw)
        except Exception:  # noqa: BLE001 - a broken verifier must never break the response
            return VerifyResult(False, "llm-error", lex_ms, (time.perf_counter() - t) * 1000)
        llm_ms = (time.perf_counter() - t) * 1000
        return VerifyResult(needs, "llm" if needs else "pass", lex_ms, llm_ms)
