"""Counterfactual oracle labelling (brief 9.5, plan T3.3). Every stage is resumable.

    python scripts/label_oracle.py generate   # GPU: SKIP / SHALLOW / DEEP answer per probe
    python scripts/label_oracle.py judge      # Gemini scores the 3 answers per probe
    python scripts/label_oracle.py label      # -> data/edu/labels.parquet (+ summary)

Label rule (primary, "necessity"): memory-required iff
    max(needs_met_SHALLOW, needs_met_DEEP) - needs_met_SKIP >= MARGIN (1.0)
`needs_met` explicitly does NOT penalise a correct generic reply to a self-contained question.
Rubric v1 labelled on `overall`, which the judge inflated for personalisation (general
questions: +2.0 "personal", +0.0 "correct"); that variant is kept as `label_personal`.
Depth (required only): SHALLOW iff score_SHALLOW >= score_DEEP - 0.5 (cheapest sufficient route).
Each probe is judged twice with different answer orders; scores are averaged.
Generation runs against FROZEN seeded memory: long-term writes are dropped (Day-2 finding).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from amic.contracts.config import Settings
from amic.contracts.models import RequestContext, Route, Turn
from amic.eval.datasets import edu

GEN = edu.DATA_DIR / "generations.jsonl"
JUDG = edu.DATA_DIR / "judgments.jsonl"
LABELS = edu.DATA_DIR / "labels.parquet"
ROUTES = [Route.SKIP, Route.SHALLOW, Route.DEEP]
MARGIN = 1.0


def _done_keys(path: Path, key_fn) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {key_fn(json.loads(line)) for line in f if line.strip()}


def m0_turns(p: edu.Probe) -> list[Turn]:
    role = {"student": "user", "tutor": "assistant"}
    return [Turn(role=role.get(t.role, "user"), text=t.text, ts=p.ts) for t in p.m0_context]  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- generate
def generate(collection: str, limit: int | None) -> None:
    from amic.app import build_components
    from amic.arms.fixed import OracleArm
    from amic.telemetry.store import InMemoryTelemetry

    tel = InMemoryTelemetry()
    c = build_components(Settings(chroma_collection=collection), telemetry=tel)
    arm = OracleArm(c)
    c.llm.chat([{"role": "user", "content": "hi"}], max_tokens=1)  # warm up (Day-2 finding)
    done = _done_keys(GEN, lambda r: f"{r['probe_id']}|{r['route']}")
    probes = edu.load_probes()[:limit]
    todo = [(p, r) for p in probes for r in ROUTES if f"{p.probe_id}|{r.value}" not in done]
    print(f"{len(todo)} generations to run ({len(done)} already done)")
    with GEN.open("a", encoding="utf-8") as out:
        for i, (p, route) in enumerate(todo, 1):
            ctx = RequestContext(
                request_id=f"{p.probe_id}-{route.value.lower()}-{uuid.uuid4().hex[:6]}",
                tenant_id=p.learner_id,
                session_id=f"label-{p.probe_id}-{route.value.lower()}",
                query=p.query,
                m0_turns=m0_turns(p),
                arm="oracle",
                now=p.ts,
                oracle_label=route != Route.SKIP,
                oracle_depth=route if route != Route.SKIP else None,
            )
            resp, _write_job = arm.answer_deferred(ctx)  # write dropped: memory stays frozen
            trace = tel.traces.pop()
            row = {
                "probe_id": p.probe_id,
                "route": route.value,
                "answer": resp.answer,
                "items": [it.id for it in resp.items],
                "trace": trace.model_dump(mode="json"),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if i % 30 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  last={trace.latency.total:.0f}ms", flush=True)


# ----------------------------------------------------------------------------- judge
JUDGE_PROMPT = """You are grading replies from a Python tutor to a beginner student. You have
privileged access to the student's REAL history; the tutor may or may not have seen it.

=== STUDENT HISTORY (ground truth) ===
{history}

=== KEY FACTS ===
{facts}

=== CURRENT SESSION SO FAR ===
{m0}

=== STUDENT'S NEW MESSAGE ===
{query}

{replies}

Score EACH reply independently on a 1-5 scale:
- "correct": the Python content is technically correct.
- "helpful": it addresses what the student actually meant. Use the history to interpret
  references such as "that bug", "last time", "my project".
- "personal": it uses the student's real history/goals where the message depends on them
  (3 = history not needed and reply is fine; lower if it ignores history the message depends
  on; higher if well tailored).
- "needs_met": how completely the reply meets what this student actually needs right now.
  If the message is self-contained (answerable without knowing this student), a correct, clear
  generic reply scores 5: do NOT lower it for missing personalisation. Lower it only when the
  reply misses or misreads something that depends on the student's history or today's
  conversation, or states false things about the student's past.
- "fabricated": true if the reply asserts anything about the student's past (earlier mistakes,
  projects, what was discussed) that is NOT supported by the history or current session.
- "overall": overall quality of the reply for THIS student. A fabricated claim about the
  student's past must lower "overall" by at least 2.
Judge content, not length or style. Return JSON only:
{{"A": {{"correct": n, "helpful": n, "personal": n, "needs_met": n, "fabricated": bool, "overall": n}}, "B": {{...}}, "C": {{...}}}}"""


def _judge_prompt(lr: edu.Learner, p: edu.Probe, answers: dict[str, str], perm: list[str]) -> str:
    facts = "\n".join(f"- {f}" for f in lr.facts)
    m0 = (
        "\n".join(f"{t.role.upper()}: {t.text}" for t in p.m0_context)
        if p.m0_context
        else "(new session; no earlier messages today)"
    )
    replies = "\n\n".join(
        f"=== REPLY {letter} ===\n{answers[route]}"
        for letter, route in zip("ABC", perm, strict=True)
    )
    return JUDGE_PROMPT.format(
        history=edu.history_transcript(lr), facts=facts, m0=m0, query=p.query, replies=replies
    )


def judge(workers: int, repeat_frac: float) -> None:
    from amic.generation.gateway.openai_compat import judge_from_settings

    llm = judge_from_settings(Settings().models)
    learners = {lr.learner_id: lr for lr in edu.load_learners()}
    probes = {p.probe_id: p for p in edu.load_probes()}
    gens: dict[str, dict[str, str]] = defaultdict(dict)
    with GEN.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gens[r["probe_id"]][r["route"]] = r["answer"]
    done = _done_keys(JUDG, lambda r: f"{r['probe_id']}|{r['rep']}")
    rng = random.Random(42)
    jobs = []
    for pid in sorted(gens):
        if len(gens[pid]) < 3:
            continue
        reps = [0] + ([1] if rng.random() < repeat_frac else [])
        jobs += [(pid, rep) for rep in reps if f"{pid}|{rep}" not in done]
    print(f"{len(jobs)} judgments to run ({len(done)} done)")
    lock = threading.Lock()

    def run(pid: str, rep: int) -> dict:
        perm = [r.value for r in ROUTES]
        random.Random(f"{pid}|{rep}").shuffle(perm)
        prompt = _judge_prompt(learners[probes[pid].learner_id], probes[pid], gens[pid], perm)
        last = None
        for _ in range(3):
            res = llm.chat([{"role": "user", "content": prompt}], json_mode=True, max_tokens=1500)
            try:
                raw = json.loads(res.text)
                scores = {
                    route: _clean(raw[letter]) for letter, route in zip("ABC", perm, strict=True)
                }
                return {"probe_id": pid, "rep": rep, "perm": perm, "scores": scores}
            except (KeyError, ValueError, TypeError) as e:
                last = e
        raise RuntimeError(f"{pid}: unparseable judge output: {last}")

    n = 0
    with JUDG.open("a", encoding="utf-8") as out, ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run, pid, rep) for pid, rep in jobs]
        for fut in as_completed(futs):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                print("FAIL", e)
                continue
            with lock:
                out.write(json.dumps(row) + "\n")
                out.flush()
                n += 1
                if n % 25 == 0 or n == len(jobs):
                    print(f"  {n}/{len(jobs)}", flush=True)


def _clean(d: dict) -> dict:
    out = {k: float(d[k]) for k in ("correct", "helpful", "personal", "needs_met", "overall")}
    if not all(1 <= v <= 5 for v in out.values()):
        raise ValueError("score out of range")
    fab = d["fabricated"]
    out["fabricated"] = float(fab if isinstance(fab, bool) else str(fab).lower() == "true")
    return out


# ----------------------------------------------------------------------------- label
def decide_label(s_skip: float, s_shallow: float, s_deep: float) -> tuple[bool, str | None]:
    required = max(s_shallow, s_deep) - s_skip >= MARGIN
    if not required:
        return False, None
    return True, ("SHALLOW" if s_shallow >= s_deep - 0.5 else "DEEP")


def split_of(learner_ids: list[str], seed: int = 13) -> dict[str, str]:
    ids = sorted(learner_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    a, b = int(0.6 * n), int(0.8 * n)
    return {i: ("train" if k < a else "val" if k < b else "test") for k, i in enumerate(ids)}


def label() -> None:
    import pandas as pd

    probes = {p.probe_id: p for p in edu.load_probes()}
    per: dict[str, list[dict]] = defaultdict(list)
    with JUDG.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            per[r["probe_id"]].append(r)
    gen_tok: dict[tuple[str, str], dict] = {}
    with GEN.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gen_tok[(r["probe_id"], r["route"])] = r["trace"]
    splits = split_of(list({p.learner_id for p in probes.values()}))
    rows, consistency = [], []
    for pid, js in per.items():
        p = probes[pid]
        mean = {
            route: {
                k: sum(j["scores"][route][k] for j in js) / len(js) for k in js[0]["scores"][route]
            }
            for route in ("SKIP", "SHALLOW", "DEEP")
        }
        req, depth = decide_label(
            mean["SKIP"]["needs_met"], mean["SHALLOW"]["needs_met"], mean["DEEP"]["needs_met"]
        )
        if req and depth is not None:  # depth: cheapest route that is not worse overall
            depth = (
                "SHALLOW"
                if mean["SHALLOW"]["needs_met"] >= mean["DEEP"]["needs_met"] - 0.5
                else "DEEP"
            )
        personal, _ = decide_label(
            mean["SKIP"]["overall"], mean["SHALLOW"]["overall"], mean["DEEP"]["overall"]
        )
        if len(js) > 1:
            ls = [
                decide_label(*(j["scores"][r]["overall"] for r in ("SKIP", "SHALLOW", "DEEP")))[0]
                for j in js
            ]
            consistency.append(ls[0] == ls[1])
        row = {
            "probe_id": pid,
            "learner_id": p.learner_id,
            "split": splits[p.learner_id],
            "category": p.category,
            "intended_memory": p.intended_memory,
            "query": p.query,
            "has_m0": bool(p.m0_context),
            "label_required": req,
            "label_personal": personal,
            "depth": depth,
            "n_judgments": len(js),
        }
        for route in ("SKIP", "SHALLOW", "DEEP"):
            k = route.lower()
            row[f"score_{k}"] = mean[route]["needs_met"]
            row[f"overall_{k}"] = mean[route]["overall"]
            row[f"fab_{k}"] = mean[route]["fabricated"]
            t = gen_tok.get((pid, route), {})
            row[f"prompt_tokens_{k}"] = t.get("prompt_tokens")
            row[f"memory_tokens_{k}"] = t.get("memory_tokens")
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("probe_id")
    df.to_parquet(LABELS, index=False)
    print(f"{len(df)} labelled probes -> {LABELS}")
    print(
        f"memory-required: {df.label_required.mean():.1%}  (personalisation-inclusive: {df.label_personal.mean():.1%})   depth: {df.depth.value_counts().to_dict()}"
    )
    print("\nrequired-rate by category (intended need in brackets):")
    g = df.groupby(["category", "intended_memory"])[["label_required", "label_personal"]].mean()
    g["n"] = df.groupby(["category", "intended_memory"]).size()
    print(g.to_string(float_format=lambda x: f"{x:.2f}"))
    print(
        f"\nfabrication rate: SKIP {df.fab_skip.mean():.1%}  SHALLOW {df.fab_shallow.mean():.1%}  DEEP {df.fab_deep.mean():.1%}"
    )
    if consistency:
        print(
            f"judge self-consistency (re-shuffled, n={len(consistency)}): {sum(consistency) / len(consistency):.1%}"
        )
    print(df.split.value_counts().to_dict())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["generate", "judge", "label"])
    ap.add_argument("--collection", default="edu_v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repeat-frac", type=float, default=1.0, help="share judged twice (mean used)")
    a = ap.parse_args()
    if a.stage == "generate":
        generate(a.collection, a.limit)
    elif a.stage == "judge":
        judge(a.workers, a.repeat_frac)
    else:
        label()


if __name__ == "__main__":
    main()
