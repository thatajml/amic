"""Grade LongMemEval answers against the gold answer (pre-registered L2, docs/PLAN.md section 9).

    python scripts/judge_longmemeval.py --runs lme_always lme_always_shallow lme_never lme_amic lme_amic_esc

Yes/no correctness per answer, with per-question-type rules that follow the official
LongMemEval protocol (src/evaluation/evaluate_qa.py; rules restated here in our own words):
temporal questions forgive off-by-one day counts, knowledge-update answers may mention the old
value as long as the updated one is given, preference questions are graded against a rubric,
and abstention questions are correct only when the reply says the information is not available.
Judge: the project's Gemini judge (the official script uses GPT-4o; reported as a difference).

Identical answers to the same question are graded once (data/longmemeval/judge_cache.jsonl).
Output: runs/<run>/lme_judged.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from pathlib import Path

from amic.contracts.config import Settings
from amic.generation.gateway.openai_compat import judge_from_settings

DIR = Path("data/longmemeval")
CACHE = DIR / "judge_cache.jsonl"

RULES = {
    "default": "Answer yes if the response contains the correct answer. An equivalent answer, or "
    "one that contains all the intermediate steps needed to reach it, also counts. Answer no if "
    "the response gives only part of the answer or a different answer.",
    "temporal-reasoning": "Answer yes if the response contains the correct answer. An equivalent "
    "answer, or one that contains all the intermediate steps needed to reach it, also counts. Do "
    "not penalise an off-by-one difference in a count of days, weeks or months. Answer no if the "
    "response gives only part of the answer or a different answer.",
    "knowledge-update": "The user's information changed over time; the correct answer is the "
    "UPDATED one. Answer yes if the response gives the updated answer, even if it also mentions "
    "the earlier information. Answer no otherwise.",
    "single-session-preference": "The 'correct answer' below is a rubric describing the reply the "
    "user would want, given preferences they stated earlier. Answer yes if the response recalls "
    "and uses the user's personal information correctly (it need not cover every rubric point). "
    "Answer no if it ignores or gets their preferences wrong.",
    "abstention": "This question cannot be answered from the conversation history: the asked-about "
    "information was never mentioned. Answer yes if the response says it does not know or that "
    "the information is not available (it may explain why). Answer no if it gives an answer.",
}

PROMPT = """You are grading an assistant's reply to a question about the user's past conversations.

Grading rule: {rule}

Question: {question}
Correct answer: {answer}
Assistant's reply: {reply}

Return JSON only: {{"correct": true | false}}"""


def _key(probe_id: str, reply: str) -> str:
    return f"{probe_id}|{hashlib.sha256(reply.strip().encode()).hexdigest()[:16]}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--wait-for-quota", action="store_true")
    a = ap.parse_args()
    from amic.eval.quota import run_with_quota

    gold = {
        r["probe_id"]: r
        for r in map(json.loads, (DIR / "gold.jsonl").read_text(encoding="utf-8").splitlines())
    }
    cache: dict[str, bool] = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                cache[r["key"]] = r["correct"]
    llm = judge_from_settings(Settings().models)
    lock = threading.Lock()
    fcache = CACHE.open("a", encoding="utf-8")

    def grade(pid: str, reply: str) -> tuple[bool, bool]:
        """(correct, reused)"""
        k = _key(pid, reply)
        with lock:
            if k in cache:
                return cache[k], True
        g = gold[pid]
        if g["abstention"]:
            rule = RULES["abstention"]
        else:
            rule = RULES.get(g["question_type"], RULES["default"])
        prompt = PROMPT.format(rule=rule, question=g["question"], answer=g["answer"], reply=reply)
        for _ in range(3):
            raw = llm.chat(
                [{"role": "user", "content": prompt}], json_mode=True, max_tokens=256
            ).text
            try:
                j = json.loads(raw)
                if isinstance(j, list) and len(j) == 1:
                    j = j[0]
                ok = j["correct"]
                if isinstance(ok, str):
                    ok = ok.strip().lower() in {"true", "yes"}
                ok = bool(ok)
                break
            except (ValueError, KeyError, TypeError):
                continue
        else:
            raise RuntimeError(f"{pid}: unparseable judge output {raw[:200]!r}")
        with lock:
            cache[k] = ok
            fcache.write(json.dumps({"key": k, "correct": ok}) + "\n")
            fcache.flush()
        return ok, False

    for run in a.runs:
        ans = [
            json.loads(x)
            for x in (Path("runs") / run / "answers.jsonl").read_text(encoding="utf-8").splitlines()
            if x
        ]
        out_path = Path("runs") / run / "lme_judged.jsonl"
        done = set()
        if out_path.exists():
            done = {
                (r["probe_id"], r["seed"])
                for r in map(json.loads, out_path.read_text(encoding="utf-8").splitlines())
                if r
            }
        todo = [r for r in ans if (r["probe_id"], r["seed"]) not in done]
        print(f"[{run}] {len(todo)} answers to grade", flush=True)
        fout = out_path.open("a", encoding="utf-8")

        def do(r: dict) -> dict:
            ok, reused = grade(r["probe_id"], r["answer"])
            return {
                "probe_id": r["probe_id"],
                "seed": r["seed"],
                "route": r["route"],
                "final_route": r.get("final_route", r["route"]),
                "correct": ok,
                "reused": reused,
            }

        def sink(row: dict, fout=fout) -> None:
            with lock:
                fout.write(json.dumps(row) + "\n")
                fout.flush()

        run_with_quota(todo, do, sink, workers=a.workers, wait=a.wait_for_quota, label=run)
        fout.close()
    fcache.close()


if __name__ == "__main__":
    main()
