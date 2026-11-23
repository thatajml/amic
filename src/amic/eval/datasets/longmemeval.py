"""LongMemEval_S (Wu et al., ICLR 2025), cleaned release (MIT licence), as an external check.

500 questions; each has its OWN chat history (~48 sessions, ~490 turns), so each question is one
tenant, `lme-<question_id>`, with one M1 record per haystack turn. Gold answers make answer
quality an objective accuracy (scripts/judge_longmemeval.py) instead of a pairwise preference.

Labels (decision metrics):
- answerable questions need memory (memory-required);
- the 30 abstention questions (`question_id` ends with `_abs`: the asked-about fact was never
  mentioned) are a WEAK memory-not-required analogue, reported separately like LoCoMo cat. 5.
Depth: single-session-* -> SHALLOW; multi-session, temporal-reasoning, knowledge-update -> DEEP.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from amic.eval.datasets.edu import Probe

PATH = Path("data/longmemeval/longmemeval_s_cleaned.json")
DIR = PATH.parent
URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
    "longmemeval_s_cleaned.json"
)
MAX_CHARS = 2000  # stored record text; the assembler's token budget bounds the prompt anyway
TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]


def parse_date(s: str) -> datetime:
    """'2023/05/30 (Tue) 23:40' -> aware datetime (UTC)."""
    day, _, rest = s.partition(" (")
    hm = rest.partition(") ")[2] or "00:00"
    return datetime.strptime(f"{day} {hm}", "%Y/%m/%d %H:%M").replace(tzinfo=UTC)


def load(path: Path = PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def tenant_of(q: dict[str, Any]) -> str:
    return f"lme-{q['question_id']}"


def is_abstention(q: dict[str, Any]) -> bool:
    return str(q["question_id"]).endswith("_abs")


def gold(q: dict[str, Any]) -> str:
    return str(q["answer"])  # 32 answers are integers in the file


def records(q: dict[str, Any]) -> Iterator[tuple[str, str, datetime, str]]:
    """(record_id, text, ts, session_id) for every haystack turn, in time order."""
    tenant = tenant_of(q)
    # session ids can repeat within one haystack, so ids carry the session's position k
    for k, (sid, date, turns) in enumerate(
        zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"], strict=True)
    ):
        start = parse_date(date)
        for j, t in enumerate(turns):
            text = f"{str(t['role']).capitalize()}: {t['content']}"
            yield (
                f"{tenant}-s{k}-{sid}-{j}",
                text[:MAX_CHARS],
                start + timedelta(minutes=j),
                f"{tenant}-{sid}",
            )


def session_meta(q: dict[str, Any]) -> dict[str, Any]:
    """S2 session metadata exactly as seeding leaves it in the session backend: one session per
    haystack session, last activity = the end of the latest session (start + 1 min per turn)."""
    ends = [
        parse_date(d) + timedelta(minutes=len(turns))
        for d, turns in zip(q["haystack_dates"], q["haystack_sessions"], strict=True)
    ]
    return {
        "n_sessions": float(len(ends)),
        "last_seen": max(ends).timestamp() if ends else 0.0,
        "last_session_id": "haystack",
    }


def probe(q: dict[str, Any]) -> Probe:
    t = tenant_of(q)
    return Probe(
        probe_id=t,
        learner_id=t,
        category=q["question_type"] + ("_abs" if is_abstention(q) else ""),
        intended_memory=not is_abstention(q),
        query=q["question"],
        ts=parse_date(q["question_date"]),
    )


def depth_of(q: dict[str, Any]) -> str | None:
    if is_abstention(q):
        return None
    return "SHALLOW" if q["question_type"].startswith("single-session") else "DEEP"


def labels_frame(qs: list[dict[str, Any]]) -> Any:
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "probe_id": tenant_of(q),
                "learner_id": tenant_of(q),
                "split": "longmemeval",
                "category": q["question_type"],
                "abstention": is_abstention(q),
                "intended_memory": not is_abstention(q),
                "query": q["question"],
                "has_m0": False,
                "label_required": not is_abstention(q),
                "depth": depth_of(q),
            }
            for q in qs
        ]
    )


def stratified_subset(qs: list[dict[str, Any]], per_type: int = 20, seed: int = 5) -> list[str]:
    """Live-QA subset: every abstention question + `per_type` answerable ones per question type.

    With the cleaned release: 30 + 6 x 20 = 150 questions.
    """
    rng = random.Random(seed)
    out = [q["question_id"] for q in qs if is_abstention(q)]
    for t in TYPES:
        pool = sorted(
            q["question_id"] for q in qs if q["question_type"] == t and not is_abstention(q)
        )
        out += rng.sample(pool, min(per_type, len(pool)))
    return sorted(out)
