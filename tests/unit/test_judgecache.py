from __future__ import annotations

import json
from pathlib import Path

from amic.eval.judgecache import PairwiseCache, self_consistency


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _runs(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    _write(
        runs / "always" / "answers.jsonl",
        [
            {"probe_id": "p1", "seed": 1, "answer": "base one"},
            {"probe_id": "p2", "seed": 1, "answer": "base two"},
        ],
    )
    _write(
        runs / "amic" / "answers.jsonl",
        [
            {"probe_id": "p1", "seed": 1, "answer": "answer A"},
            {"probe_id": "p2", "seed": 1, "answer": "answer B"},
        ],
    )
    _write(
        runs / "amic" / "pairwise_vs_always.jsonl",
        [
            {"probe_id": "p1", "seed": 1, "result": "win", "identical": False, "fab_run": False},
            {"probe_id": "p2", "seed": 1, "result": "loss", "identical": False, "fab_run": True},
        ],
    )
    _write(runs / "other" / "answers.jsonl", [{"probe_id": "p1", "seed": 1, "answer": "answer A"}])
    _write(
        runs / "other" / "pairwise_vs_always.jsonl",
        [{"probe_id": "p1", "seed": 1, "result": "loss", "identical": False}],
    )
    return runs


def test_reuse_follows_the_source_order(tmp_path):
    runs = _runs(tmp_path)
    c = PairwiseCache.load(["amic", "other"], runs_dir=runs)
    src, v = c.get("p1", 1, "answer A", "base one")
    assert src == "amic" and v["result"] == "win"
    src, v = PairwiseCache.load(["other", "amic"], runs_dir=runs).get(
        "p1", 1, " answer A\n", "base one"
    )  # whitespace-insensitive, like the identical-answer check
    assert src == "other" and v["result"] == "loss"


def test_no_reuse_for_a_different_answer_seed_or_baseline(tmp_path):
    c = PairwiseCache.load(["amic"], runs_dir=_runs(tmp_path))
    assert c.get("p1", 1, "a new answer", "base one") is None
    assert c.get("p1", 2, "answer A", "base one") is None
    assert c.get("p1", 1, "answer A", "another baseline answer") is None


def test_missing_sources_are_ignored(tmp_path):
    c = PairwiseCache.load(["nope"], runs_dir=_runs(tmp_path))
    assert len(c) == 0


def test_self_consistency_counts_pairs_judged_twice(tmp_path):
    n, agree = self_consistency(["amic", "other"], runs_dir=_runs(tmp_path))
    assert n == 1 and agree == 0.0
