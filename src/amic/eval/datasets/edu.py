"""Synthetic intro-Python tutoring dataset (plan Day 3).

Design:
- Personas are sampled in CODE from curated lists (seeded RNG), so diversity and the ground
  truth (which misconceptions happened in which session) are controlled, not left to the LLM.
- The LLM only writes the dialogue and the probe queries.
- Probe `category` is the *intended* need for memory. It is NOT the label: labels come from
  the counterfactual judge (scripts/label_oracle.py).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ------------------------------------------------------------------------- curriculum
CURRICULUM = [
    "variables and types",
    "strings",
    "input and type conversion",
    "conditionals",
    "while loops",
    "for loops and range",
    "lists",
    "dictionaries",
    "functions and return values",
    "variable scope",
    "exceptions and try/except",
    "file reading and writing",
    "list comprehensions",
    "reading tracebacks and debugging",
]

MISCONCEPTIONS: list[tuple[str, str]] = [
    ("for loops and range", "thinks range(n) includes n, causing an off-by-one IndexError"),
    ("lists", "confuses append with extend and ends up with nested lists"),
    ("lists", "removes items from a list while looping over it and skips elements"),
    ("functions and return values", "uses print instead of return, then gets None"),
    ("variables and types", "concatenates a str and an int without converting (TypeError)"),
    ("input and type conversion", "forgets input() returns a string and compares it to a number"),
    ("conditionals", "writes `if x == 1 or 2` expecting it to test both values"),
    ("conditionals", "uses = instead of == inside an if condition"),
    ("strings", "tries to change one character of a string by index (strings are immutable)"),
    ("dictionaries", "reads a missing key and gets KeyError instead of using .get()"),
    ("variable scope", "expects assigning a variable inside a function to change the global"),
    ("while loops", "forgets to update the loop variable and creates an infinite loop"),
    ("exceptions and try/except", "wraps everything in a bare except that hides real errors"),
    ("file reading and writing", "opens a file in 'w' mode and accidentally erases its content"),
    ("functions and return values", "uses a mutable default argument like def f(items=[])"),
    ("list comprehensions", "puts the if-filter in the wrong place in a comprehension"),
]

GOALS = [
    "pass an intro programming course exam",
    "automate spreadsheet chores at work",
    "build a small text-based adventure game",
    "analyse football statistics",
    "prepare for a data science bootcamp",
    "write scripts to organise photo files",
    "help their child with coding homework",
    "build a personal budgeting tool",
]
PREFERENCES = [
    "short worked examples",
    "everyday analogies",
    "step-by-step explanations",
    "small exercises to try",
    "seeing the broken code first, then the fix",
    "minimal jargon",
]
PROJECTS = [
    "a grade calculator",
    "a recipe scaler",
    "a football league table",
    "a to-do list app",
    "a quiz game",
    "an expense tracker",
    "a word counter for essays",
    "a weather data logger",
    "a library book tracker",
    "a workout log",
]

# category -> (count per learner, intended memory need)
PROBE_MIX: dict[str, tuple[int, bool]] = {
    "general_new": (3, False),  # concept question on a topic NOT in the history
    "general_covered": (3, False),  # self-contained question on a topic the learner studied
    "lexical_trap": (2, False),  # memory-sounding words, but self-contained
    "explicit_recall": (4, True),  # directly asks about earlier sessions
    "implicit_personal": (4, True),  # needs history, no explicit memory words
    "personalize": (2, True),  # recommendation/exercise that should fit the learner
    "m0_followup": (2, False),  # refers to the CURRENT session (answerable from M0)
}
PROBES_PER_LEARNER = sum(n for n, _ in PROBE_MIX.values())


# ------------------------------------------------------------------------- schema
class DTurn(BaseModel):
    role: str  # "student" | "tutor"
    text: str


class DSession(BaseModel):
    session_id: str
    start: datetime
    concepts: list[str]
    misconceptions: list[str] = Field(default_factory=list)
    turns: list[DTurn]


class Learner(BaseModel):
    learner_id: str
    level: str
    goal: str
    preference: str
    project: str
    sessions: list[DSession]
    facts: list[str] = Field(default_factory=list)  # memorable facts, for judge + probes
    probe_time: datetime  # when probes are asked (a new session after a gap)


class Probe(BaseModel):
    probe_id: str
    learner_id: str
    category: str
    intended_memory: bool
    query: str
    m0_context: list[DTurn] = Field(default_factory=list)
    relevant_facts: list[int] = Field(default_factory=list)
    ts: datetime


@dataclass
class Persona:
    learner_id: str
    level: str
    goal: str
    preference: str
    project: str
    plan: list[dict[str, Any]]  # per session: concepts, misconceptions, start
    probe_time: datetime


# ------------------------------------------------------------------------- personas
BASE_DATE = datetime(2026, 9, 1, 18, 0, tzinfo=UTC)


def sample_personas(n: int, seed: int = 7) -> list[Persona]:
    rng = random.Random(seed)
    out: list[Persona] = []
    for i in range(n):
        n_sessions = rng.randint(3, 5)
        start_idx = rng.randint(0, 4)
        concepts = CURRICULUM[start_idx : start_idx + n_sessions * 2] or CURRICULUM[:4]
        mis = rng.sample(MISCONCEPTIONS, k=rng.randint(2, 3))
        t = BASE_DATE + timedelta(days=rng.randint(0, 20), hours=rng.randint(-3, 3))
        plan: list[dict[str, Any]] = []
        for s in range(n_sessions):
            cs = concepts[s * 2 : s * 2 + 2] or [rng.choice(CURRICULUM)]
            plan.append({"concepts": cs, "misconceptions": [], "start": t})
            t += timedelta(days=rng.randint(2, 7), hours=rng.randint(-2, 2))
        for m_concept, m_text in mis:  # place each misconception in a session
            plan[rng.randrange(n_sessions)]["misconceptions"].append(f"{m_concept}: {m_text}")
        probe_time = plan[-1]["start"] + timedelta(days=rng.randint(1, 14), hours=1)
        out.append(
            Persona(
                learner_id=f"edu-{i + 1:03d}",
                level=rng.choice(["complete beginner", "beginner with a little experience"]),
                goal=rng.choice(GOALS),
                preference=rng.choice(PREFERENCES),
                project=rng.choice(PROJECTS),
                plan=plan,
                probe_time=probe_time,
            )
        )
    return out


# ------------------------------------------------------------------------- prompts
def history_prompt(p: Persona) -> str:
    sessions = "\n".join(
        f"Session {i + 1} ({s['start']:%Y-%m-%d}): concepts = {', '.join(s['concepts'])}; "
        f"mistakes the student makes in this session = {s['misconceptions'] or 'none'}"
        for i, s in enumerate(p.plan)
    )
    return f"""Write realistic past tutoring sessions between a Python TUTOR and a STUDENT.

Student: {p.level}; goal: {p.goal}; prefers {p.preference}; personal project: {p.project}.

Session plan:
{sessions}

Rules:
- Each session has 3-4 exchanges (student turn then tutor turn), 6-8 turns total.
- The student writes casually, sometimes pastes a short code snippet or error, and mentions
  their project naturally in at least two sessions. They say what their goal is at least once.
- The student actually makes the listed mistakes in the listed session; the tutor diagnoses them.
- Tutor turns: helpful, under 90 words, adapted to the student's preference.
- Vary wording between sessions. No names for either person.
Also list 6-10 short "facts" a good tutor should remember about this student (project, goal,
preference, each mistake with its session number, anything else personal they mentioned).

Return JSON only:
{{"sessions": [{{"session": 1, "turns": [{{"role": "student", "text": "..."}},
{{"role": "tutor", "text": "..."}}]}}], "facts": ["...", "..."]}}"""


_CATEGORY_RULES = {
    "general_new": "a standalone Python question about a topic from {new_topics} that the "
    "student has NOT studied; answerable perfectly without knowing the student.",
    "general_covered": "a standalone, self-contained Python question about a topic the student "
    "DID study; a generic answer is fully adequate. Do not reference the past.",
    "lexical_trap": "contains words like 'my', 'again', 'remember', 'last time' or 'before', but "
    "is fully self-contained and needs no knowledge of past sessions (e.g. quoting something "
    "their teacher or a book said, or 'my code' with the code included in the message).",
    "explicit_recall": "explicitly asks about earlier sessions (what they got wrong, what was "
    "explained, what they said about their project). Cannot be answered without memory.",
    "implicit_personal": "needs knowledge of the student's history to answer well but uses NO "
    "explicit memory words (e.g. 'the same bug is back in the league table thing', 'is that "
    "why it printed None?' without context). Must be ambiguous without memory.",
    "personalize": "asks for a next step, exercise or study plan; the best answer is tailored "
    "to their goal, project, level and past mistakes.",
    "m0_followup": "a follow-up that refers to the CURRENT conversation. Also write "
    "`m0_context`: 2-4 turns (student/tutor alternating, starting with student) from today's "
    "session; the query refers back to them (e.g. 'what did you mean by that last part?').",
}


def probes_prompt(p: Persona, learner: Learner) -> str:
    studied = sorted({c for s in p.plan for c in s["concepts"]})
    new_topics = [c for c in CURRICULUM if c not in studied]
    facts = "\n".join(f"[{i}] {f}" for i, f in enumerate(learner.facts))
    rules = "\n".join(
        f'- "{cat}" x{n}: ' + _CATEGORY_RULES[cat].format(new_topics=new_topics)
        for cat, (n, _) in PROBE_MIX.items()
    )
    return f"""A student is starting a NEW tutoring session with a Python tutor.

Student: {p.level}; goal: {p.goal}; prefers {p.preference}; project: {p.project}.
Topics studied before: {studied}
Facts from earlier sessions:
{facts}

Write exactly {PROBES_PER_LEARNER} messages the student could send now, in these categories:
{rules}

Rules: casual student voice; vary length (5-40 words) and phrasing; refer to time the way a
student would ('last week', 'the first time', 'a while ago'), never 'session N'; never copy the category
name or these instructions into a message; no names. For each, list `relevant_facts` = indices of
facts that a good answer would use ([] if none).

Return JSON only:
{{"probes": [{{"category": "...", "query": "...", "relevant_facts": [0], "m0_context":
[{{"role": "student", "text": "..."}}, {{"role": "tutor", "text": "..."}}]}}]}}
(`m0_context` only for m0_followup; omit or [] otherwise.)"""


# ------------------------------------------------------------------------- parsing
def parse_history(p: Persona, raw: dict[str, Any]) -> Learner:
    sessions_raw = raw["sessions"]
    if len(sessions_raw) != len(p.plan):
        raise ValueError(f"expected {len(p.plan)} sessions, got {len(sessions_raw)}")
    sessions: list[DSession] = []
    for i, (sr, plan) in enumerate(zip(sessions_raw, p.plan, strict=True)):
        turns = [DTurn(role=t["role"].lower(), text=t["text"].strip()) for t in sr["turns"]]
        if len(turns) < 4 or turns[0].role != "student":
            raise ValueError(f"session {i + 1}: bad turns")
        sessions.append(
            DSession(
                session_id=f"{p.learner_id}-s{i + 1}",
                start=plan["start"],
                concepts=plan["concepts"],
                misconceptions=plan["misconceptions"],
                turns=turns,
            )
        )
    facts = [str(f).strip() for f in raw.get("facts", []) if str(f).strip()]
    if len(facts) < 4:
        raise ValueError("too few facts")
    return Learner(
        learner_id=p.learner_id,
        level=p.level,
        goal=p.goal,
        preference=p.preference,
        project=p.project,
        sessions=sessions,
        facts=facts,
        probe_time=p.probe_time,
    )


def parse_probes(learner: Learner, raw: dict[str, Any]) -> list[Probe]:
    items = raw["probes"]
    counts: dict[str, int] = {}
    out: list[Probe] = []
    for it in items:
        cat = it.get("category", "")
        if cat not in PROBE_MIX or not str(it.get("query", "")).strip():
            continue
        counts[cat] = counts.get(cat, 0) + 1
        m0 = [DTurn(role=t["role"].lower(), text=t["text"]) for t in it.get("m0_context") or []]
        if cat == "m0_followup" and len(m0) < 2:
            continue
        if cat != "m0_followup":
            m0 = []
        out.append(
            Probe(
                probe_id=f"{learner.learner_id}-p{len(out) + 1:02d}",
                learner_id=learner.learner_id,
                category=cat,
                intended_memory=PROBE_MIX[cat][1],
                query=str(it["query"]).strip(),
                m0_context=m0,
                relevant_facts=[
                    int(x) for x in it.get("relevant_facts") or [] if str(x).lstrip("-").isdigit()
                ],
                ts=learner.probe_time,
            )
        )
    for cat, (n, _) in PROBE_MIX.items():
        got = sum(1 for p in out if p.category == cat)
        if got < max(1, n - 1):
            raise ValueError(f"category {cat}: {got} < {n}")
    return out


# ------------------------------------------------------------------------- io
DATA_DIR = Path("data/edu")


def save_jsonl(path: Path, rows: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")


def load_learners(path: Path = DATA_DIR / "learners.jsonl") -> list[Learner]:
    with path.open(encoding="utf-8") as f:
        return [Learner.model_validate_json(line) for line in f if line.strip()]


def load_probes(path: Path = DATA_DIR / "probes.jsonl") -> list[Probe]:
    with path.open(encoding="utf-8") as f:
        return [Probe.model_validate_json(line) for line in f if line.strip()]


def history_transcript(learner: Learner, max_chars: int = 9000) -> str:
    """Compact transcript for the judge (reference only; never shown to the tutor)."""
    parts = []
    for i, s in enumerate(learner.sessions):
        parts.append(f"## Session {i + 1} ({s.start:%Y-%m-%d})")
        parts += [f"{t.role.upper()}: {t.text}" for t in s.turns]
    txt = "\n".join(parts)
    return txt if len(txt) <= max_chars else txt[:max_chars] + "\n[...]"


def split_of(learner_ids: list[str], seed: int = 13) -> dict[str, str]:
    """60/20/20 train/val/test split BY LEARNER (no learner appears in two splits)."""
    ids = sorted(learner_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    a, b = int(0.6 * n), int(0.8 * n)
    return {i: ("train" if k < a else "val" if k < b else "test") for k, i in enumerate(ids)}


# ------------------------------------------------------------------------- seeding
def exchanges(session: DSession) -> list[tuple[str, str]]:
    """Pair each student turn with the following tutor turn."""
    out: list[tuple[str, str]] = []
    pending: str | None = None
    for t in session.turns:
        if t.role == "student":
            if pending is not None:
                out.append((pending, ""))
            pending = t.text
        elif pending is not None:
            out.append((pending, t.text))
            pending = None
    if pending is not None:
        out.append((pending, ""))
    return out


def seed_learner(comps: Any, learner: Learner, write_memory: bool = True) -> int:
    """Write the learner's past sessions into M1 + session metadata, with simulated time.
    write_memory=False re-seeds only the session metadata (e.g. into a new session backend)."""
    n = 0
    for s in learner.sessions:
        comps.sessions.touch(learner.learner_id, s.session_id, s.start)
        for j, (q, a) in enumerate(exchanges(s) if write_memory else []):
            ts = s.start + timedelta(minutes=3 * j)
            comps.writer.write_turn(
                learner.learner_id, s.session_id, f"{s.session_id}-x{j + 1}", q, a, ts
            )
            n += 1
        comps.sessions.touch(learner.learner_id, s.session_id, s.start + timedelta(minutes=30))
    return n


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)
