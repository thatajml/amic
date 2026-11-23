"""Reuse pairwise verdicts for byte-identical answer pairs (PLAN section 9).

Arms share one deterministic pipeline, so a new run often produces exactly the answer an
earlier run produced for the same probe and seed (an escalated SKIP is the always-SHALLOW
answer; an unescalated one is the original run's answer). Judging that identical pair again
would only add judge noise to paired comparisons and cost an API call.

Key: (probe_id, seed, sha256(run answer), sha256(baseline answer)). Sources are searched IN THE
ORDER GIVEN, so a run compared with its parent (amic_esc vs amic) reuses the parent's own
verdicts and the paired difference reflects only the rows that changed. Reused rows are marked.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _h(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def pair_key(probe_id: str, seed: int, run_answer: str, base_answer: str) -> str:
    return f"{probe_id}|{seed}|{_h(run_answer)}|{_h(base_answer)}"


def _answers(path: Path) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            r = json.loads(line)
            out[(r["probe_id"], int(r["seed"]))] = r["answer"]
    return out


class PairwiseCache:
    def __init__(self) -> None:
        self._by_source: list[tuple[str, dict[str, dict[str, Any]]]] = []

    @classmethod
    def load(
        cls, sources: list[str], baseline: str = "always", runs_dir: Path = Path("runs")
    ) -> PairwiseCache:
        cache = cls()
        base_path = runs_dir / baseline / "answers.jsonl"
        if not base_path.exists():
            return cache
        base = _answers(base_path)
        for run in sources:
            pw_path = runs_dir / run / f"pairwise_vs_{baseline}.jsonl"
            ans_path = runs_dir / run / "answers.jsonl"
            if not (pw_path.exists() and ans_path.exists()):
                continue
            mine = _answers(ans_path)
            table: dict[str, dict[str, Any]] = {}
            for line in pw_path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                r = json.loads(line)
                k = (r["probe_id"], int(r["seed"]))
                if k not in mine or k not in base:
                    continue
                table[pair_key(k[0], k[1], mine[k], base[k])] = {
                    "result": r["result"],
                    "identical": r.get("identical", False),
                    "fab_run": r.get("fab_run"),
                    "fab_base": r.get("fab_base"),
                }
            cache._by_source.append((run, table))
        return cache

    def get(
        self, probe_id: str, seed: int, run_answer: str, base_answer: str
    ) -> tuple[str, dict[str, Any]] | None:
        """(source run, stored verdict) from the first source that judged this exact pair."""
        k = pair_key(probe_id, seed, run_answer, base_answer)
        for run, table in self._by_source:
            if k in table:
                return run, table[k]
        return None

    def __len__(self) -> int:
        return sum(len(t) for _, t in self._by_source)


def self_consistency(
    sources: list[str], baseline: str = "always", runs_dir: Path = Path("runs")
) -> tuple[int, float]:
    """Judge self-consistency: identical (answer, baseline) pairs that were judged more than once
    (by different runs, in independently randomised A/B order). Returns (n pairs, agreement)."""
    cache = PairwiseCache.load(sources, baseline, runs_dir)
    seen: dict[str, list[str]] = {}
    for _, table in cache._by_source:
        for k, v in table.items():
            if not v["identical"]:
                seen.setdefault(k, []).append(v["result"])
    multi = [v for v in seen.values() if len(v) > 1]
    if not multi:
        return 0, float("nan")
    agree = sum(len(set(v)) == 1 for v in multi) / len(multi)
    return len(multi), agree
