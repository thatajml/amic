"""Generate the synthetic intro-Python tutoring dataset with the Gemini judge model.

    python scripts/gen_edu_dataset.py --learners 40          # generate (resumable)
    python scripts/gen_edu_dataset.py --seed-memory          # write histories into M1

Outputs data/edu/learners.jsonl, data/edu/probes.jsonl; raw LLM output cached in data/edu/raw/.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from amic.contracts.config import Settings
from amic.eval.datasets import edu
from amic.generation.gateway.openai_compat import judge_from_settings

RAW = edu.DATA_DIR / "raw"


def _ask_json(llm, prompt: str, cache: Path, attempt: int) -> dict:
    if cache.exists() and attempt == 0:
        return json.loads(cache.read_text(encoding="utf-8"))
    r = llm.chat(
        [{"role": "user", "content": prompt}],
        json_mode=True,
        max_tokens=8000,
        temperature=0.8 + 0.1 * attempt,
    )
    data = json.loads(r.text)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def build_one(llm, p: edu.Persona, tries: int = 3):
    last: Exception | None = None
    for a in range(tries):
        try:
            learner = edu.parse_history(
                p, _ask_json(llm, edu.history_prompt(p), RAW / f"{p.learner_id}.history.json", a)
            )
            break
        except Exception as e:  # noqa: BLE001 - retry any malformed output
            last = e
    else:
        raise RuntimeError(f"{p.learner_id} history failed: {last}")
    for a in range(tries):
        try:
            raw = _ask_json(
                llm, edu.probes_prompt(p, learner), RAW / f"{p.learner_id}.probes.json", a
            )
            return learner, edu.parse_probes(learner, raw)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"{p.learner_id} probes failed: {last}")


def generate(n: int, workers: int) -> None:
    llm = judge_from_settings(Settings().models)
    personas = edu.sample_personas(n)
    learners, probes, failed = [], [], []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(build_one, llm, p): p for p in personas}
        for f in as_completed(futs):
            p = futs[f]
            try:
                lr, pr = f.result()
                learners.append(lr)
                probes += pr
                print(f"ok   {p.learner_id}: {len(lr.sessions)} sessions, {len(pr)} probes")
            except Exception as e:  # noqa: BLE001
                failed.append(p.learner_id)
                print(f"FAIL {p.learner_id}: {e}")
    learners.sort(key=lambda x: x.learner_id)
    probes.sort(key=lambda x: x.probe_id)
    edu.save_jsonl(edu.DATA_DIR / "learners.jsonl", learners)
    edu.save_jsonl(edu.DATA_DIR / "probes.jsonl", probes)
    cats: dict[str, int] = {}
    for pr in probes:
        cats[pr.category] = cats.get(pr.category, 0) + 1
    print(f"\n{len(learners)} learners, {len(probes)} probes, failed={failed}\n{cats}")


def seed_memory(collection: str) -> None:
    from amic.app import build_components

    s = Settings(chroma_collection=collection)
    c = build_components(s, llm=object())  # type: ignore[arg-type]  # no LLM needed
    total = 0
    for lr in edu.load_learners():
        c.episodic.delete_tenant(lr.learner_id)  # idempotent re-seed
        c.sessions.erase_tenant(lr.learner_id)
        total += edu.seed_learner(c, lr)
    print(f"seeded {total} M1 records into collection '{collection}'")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed-memory", action="store_true")
    ap.add_argument("--collection", default="edu_v1")
    a = ap.parse_args()
    if a.seed_memory:
        seed_memory(a.collection)
    else:
        generate(a.learners, a.workers)


if __name__ == "__main__":
    main()
