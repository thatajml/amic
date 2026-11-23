"""Pairwise LLM-as-judge: each run vs the `always` baseline, per probe and seed (brief 9.2).

    python scripts/judge_pairwise.py --runs amic never oracle amic_no_s3

Identical answers are a tie without a judge call. Order is randomised per item. The judge
sees the learner's real history (to judge personalisation and fabricated memories).
Output: runs/<run>/pairwise_vs_always.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from amic.contracts.config import Settings
from amic.eval.datasets import edu
from amic.generation.gateway.openai_compat import judge_from_settings

PROMPT = """You are comparing two replies from a Python tutor to a beginner student. You have
privileged access to the student's REAL history; the tutor may or may not have seen it.

=== STUDENT HISTORY (ground truth) ===
{history}

=== KEY FACTS ===
{facts}

=== CURRENT SESSION SO FAR ===
{m0}

=== STUDENT'S NEW MESSAGE ===
{query}

=== REPLY A ===
{a}

=== REPLY B ===
{b}

Which reply is better for THIS student? Consider correctness, whether it addresses what the
student actually meant (use the history to interpret references), and appropriate use of their
history. A reply that asserts something false about the student's past is seriously worse.
Ignore length and style differences. Say "tie" if they are about equally good.
Return JSON only: {{"winner": "A" | "B" | "tie", "fabricated_A": bool, "fabricated_B": bool}}"""


def load_answers(run: str) -> dict[tuple[str, int], dict]:
    path = Path("runs") / run / "answers.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            r = json.loads(line)
            out[(r["probe_id"], r["seed"])] = r
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--baseline", default="always")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--wait-for-quota", action="store_true", help="sleep through daily-cap resets")
    a = ap.parse_args()
    from amic.eval.quota import run_with_quota

    llm = judge_from_settings(Settings().models)
    learners = {lr.learner_id: lr for lr in edu.load_learners()}
    probes = {p.probe_id: p for p in edu.load_probes()}
    base = load_answers(a.baseline)

    jobs: list[tuple[str, tuple[str, int]]] = []
    judges: dict[str, Callable[[tuple[str, int]], dict]] = {}
    sinks: dict[str, Callable[[dict], None]] = {}
    files: list[TextIO] = []
    for run in a.runs:
        mine = load_answers(run)
        out_path = Path("runs") / run / f"pairwise_vs_{a.baseline}.jsonl"
        done = set()
        if out_path.exists():
            done = {
                (r["probe_id"], r["seed"])
                for r in map(json.loads, out_path.read_text(encoding="utf-8").splitlines())
                if r
            }
        keys = [k for k in mine if k in base and k not in done]
        print(f"[{run}] {len(keys)} comparisons", flush=True)

        def judge_one(k: tuple[str, int], mine: dict = mine, run: str = run) -> dict:
            pid, seed = k
            x, b = mine[k]["answer"], base[k]["answer"]
            row = {
                "probe_id": pid,
                "seed": seed,
                "route": mine[k]["route"],
                "base_route": base[k]["route"],
            }
            if x.strip() == b.strip():
                return {
                    **row,
                    "result": "tie",
                    "identical": True,
                    "fab_run": None,
                    "fab_base": None,
                }
            p = probes[pid]
            lr = learners[p.learner_id]
            run_is_a = random.Random(f"{run}|{pid}|{seed}").random() < 0.5
            ra, rb = (x, b) if run_is_a else (b, x)
            m0 = "\n".join(f"{t.role.upper()}: {t.text}" for t in p.m0_context) or "(new session)"
            prompt = PROMPT.format(
                history=edu.history_transcript(lr),
                facts="\n".join(f"- {f}" for f in lr.facts),
                m0=m0,
                query=p.query,
                a=ra,
                b=rb,
            )
            raw = ""
            for _ in range(3):
                try:
                    raw = llm.chat(
                        [{"role": "user", "content": prompt}], json_mode=True, max_tokens=1024
                    ).text
                    j = json.loads(raw)
                    if isinstance(j, list) and len(j) == 1 and isinstance(j[0], dict):
                        j = j[0]  # the judge occasionally wraps its object in a list
                    w = str(j["winner"]).strip().upper()
                    if w not in {"A", "B", "TIE"}:
                        raise ValueError(w)
                    break
                except (ValueError, KeyError, TypeError):
                    continue
            else:
                raise RuntimeError(f"{run} {pid}: unparseable: {raw[:300]!r}")
            result = "tie" if w == "TIE" else ("win" if (w == "A") == run_is_a else "loss")
            fa, fb = bool(j.get("fabricated_A")), bool(j.get("fabricated_B"))
            return {
                **row,
                "result": result,
                "identical": False,
                "fab_run": fa if run_is_a else fb,
                "fab_base": fb if run_is_a else fa,
            }

        f = out_path.open("a", encoding="utf-8")
        files.append(f)
        judges[run] = judge_one

        def sink(row: dict, f: TextIO = f) -> None:
            f.write(json.dumps(row) + "\n")
            f.flush()

        sinks[run] = sink
        jobs += [(run, k) for k in keys]

    # priority under a daily cap: seed 1 for every run first (a complete one-seed table),
    # then seed 2, then seed 3
    order = {r: i for i, r in enumerate(a.runs)}
    jobs.sort(key=lambda j: (j[1][1], order[j[0]], j[1][0]))
    lock = threading.Lock()

    def do(job: tuple[str, tuple[str, int]]) -> tuple[str, dict]:
        run, k = job
        return run, judges[run](k)

    def write(res: tuple[str, dict]) -> None:
        with lock:
            sinks[res[0]](res[1])

    try:
        run_with_quota(jobs, do, write, workers=a.workers, wait=a.wait_for_quota, label="pairwise")
    finally:
        for fh in files:
            fh.close()


if __name__ == "__main__":
    main()
