from __future__ import annotations

import json
from pathlib import Path

import pytest

from amic.eval.escalation import replay_escalation


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    r = tmp_path / "runs"
    _write(
        r / "amic" / "pairwise_vs_always.jsonl",
        [
            {"probe_id": "p1", "seed": 1, "route": "SKIP", "result": "loss"},  # lexical hit
            {"probe_id": "p2", "seed": 1, "route": "SKIP", "result": "loss"},  # llm hit
            {"probe_id": "p3", "seed": 1, "route": "SKIP", "result": "win"},  # passes both
            {"probe_id": "p4", "seed": 1, "route": "SHALLOW", "result": "tie"},  # not verified
        ],
    )
    _write(
        r / "amic" / "verify_replay.jsonl",
        [
            {"probe_id": "p1", "seed": 1, "lexical": "past-session", "llm": False},
            {"probe_id": "p2", "seed": 1, "lexical": None, "llm": True},
            {"probe_id": "p3", "seed": 1, "lexical": None, "llm": False},
        ],
    )
    _write(
        r / "always_shallow" / "pairwise_vs_always.jsonl",
        [
            {"probe_id": f"p{i}", "seed": 1, "route": "SHALLOW", "result": "win"}
            for i in (1, 2, 3, 4)
        ],
    )
    return r


@pytest.mark.parametrize(
    ("mode", "escalated", "results"),
    [
        ("lexical", ["p1"], ["win", "loss", "win", "tie"]),
        ("llm", ["p2"], ["loss", "win", "win", "tie"]),
        ("both", ["p1", "p2"], ["win", "win", "win", "tie"]),
    ],
)
def test_replay_takes_the_shallow_verdict_for_escalated_rows(runs, mode, escalated, results):
    m = replay_escalation("amic", mode, runs_dir=runs)
    assert sorted(m[m.escalated].probe_id) == escalated
    assert list(m.sort_values("probe_id").result) == results


def test_missing_inputs_give_none(runs):
    assert replay_escalation("never", "both", runs_dir=runs) is None
