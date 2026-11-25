from __future__ import annotations

from datetime import UTC, datetime

import pytest

from amic.contracts.models import Turn
from amic.generation.gateway.base import LLMResult
from amic.generation.verifier import ResponseVerifier, lexical_check, parse_needs_memory

NOW = datetime(2026, 11, 1, tzinfo=UTC)
M0 = [Turn(role="user", text="I struggled with this loop just now", ts=NOW)]


@pytest.mark.parametrize(
    ("answer", "m0", "expected"),
    [
        ("Last time you mixed up append and extend.", [], "past-session"),
        ("In our previous session we covered while loops.", [], "past-session"),
        ("Remember when you got an IndexError with range?", [], "past-session"),
        ("As we discussed before, strings are immutable.", [], "past-session"),
        ("You struggled with for loops, so here is a simpler example.", [], "history"),
        ("You struggled with for loops, so here is a simpler example.", M0, None),
        (
            "I don't have information about your previous sessions, but here is the idea.",
            [],
            "deferral",
        ),
        ("I don't have any record of our previous lessons.", [], "deferral"),
        ("Could you remind me which project you are working on?", [], "deferral"),
        ("Which bug do you mean? Paste the code and I'll look.", [], "deferral"),
        # clean answers to self-contained questions
        ("A for loop repeats code for each item: for x in [1, 2]: print(x)", [], None),
        ("Remember to convert input() to int before comparing it to a number.", [], None),
        ("As we can see, range(3) stops at 2, not 3.", [], None),
        ("Use .get() to avoid a KeyError when a key may be missing.", [], None),
    ],
)
def test_lexical_check(answer, m0, expected):
    assert lexical_check(answer, m0) == expected


def test_parse_needs_memory():
    assert parse_needs_memory('{"needs_memory": true}') is True
    assert parse_needs_memory('[{"needs_memory": "no"}]') is False
    with pytest.raises((KeyError, ValueError)):
        parse_needs_memory('{"other": 1}')
    with pytest.raises(ValueError):
        parse_needs_memory('{"needs_memory": "maybe"}')


class ScriptedLLM:
    model = "scripted"

    def __init__(self, text: str = '{"needs_memory": true}', fail: bool = False) -> None:
        self.text, self.fail, self.calls = text, fail, 0

    def chat(self, messages, **kw) -> LLMResult:
        self.calls += 1
        if self.fail:
            raise TimeoutError("model down")
        return LLMResult(text=self.text, prompt_tokens=50, completion_tokens=5, total_ms=1.0)


def test_lexical_hit_skips_the_llm():
    llm = ScriptedLLM('{"needs_memory": false}')
    r = ResponseVerifier("both", llm).check("q", "Last time you forgot the colon.", [])
    assert r.needs_memory and r.reason == "lexical:past-session" and llm.calls == 0


def test_llm_decides_when_lexical_passes():
    llm = ScriptedLLM('{"needs_memory": true}')
    r = ResponseVerifier("both", llm).check("What should I practise?", "Try a loop exercise.", [])
    assert r.needs_memory and r.reason == "llm" and llm.calls == 1


def test_lexical_mode_never_calls_an_llm():
    r = ResponseVerifier("lexical").check("q", "A clean generic answer.", [])
    assert not r.needs_memory and r.reason == "pass"


def test_llm_error_keeps_the_answer():
    r = ResponseVerifier("llm", ScriptedLLM(fail=True)).check("q", "answer", [])
    assert not r.needs_memory and r.reason == "llm-error"


def test_llm_modes_need_an_llm():
    with pytest.raises(ValueError):
        ResponseVerifier("both", None)
