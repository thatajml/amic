"""ExperimentRunner (L8): run arms LIVE over a labelled split against frozen memory.

    python -m amic.eval.runner --arms always never oracle amic --seeds 1 2 3
    python -m amic.eval.runner --arms amic --config configs/ablations/no_s3.yaml --name amic_no_s3 \
        --tier1 models/tier1_no_s3.joblib

- Memory is frozen: long-term writes are dropped (Day-2 finding).
- Tutor sampling: temperature 0.7, one fixed seed per repetition (variance across seeds).
- Tutor calls go through CachingGateway (identical prompt+seed => identical output; first-run
  timings preserved). Gate, embedding and retrieval are always measured live.
- Traces -> data/experiments.db; answers -> runs/<name>/answers.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from amic.app import build_components, gate_llm_from_settings
from amic.contracts.config import Settings, load_settings
from amic.contracts.models import RequestContext, Route, Turn
from amic.eval.datasets import edu
from amic.generation.gateway.cache import CachingGateway
from amic.generation.gateway.ollama import OllamaGateway

EXP_DB = "sqlite:///./data/experiments.db"


def _m0(p: edu.Probe) -> list[Turn]:
    role = {"student": "user", "tutor": "assistant"}
    return [Turn(role=role.get(t.role, "user"), text=t.text, ts=p.ts) for t in p.m0_context]  # type: ignore[arg-type]


def _round_robin(probes: list[edu.Probe]) -> list[edu.Probe]:
    """Interleave learners so consecutive requests never share a memory-block prefix.

    Ollama keeps the previous prompt's KV cache; back-to-back queries from one learner that
    retrieve the same memory would get their memory-block prefill almost free, understating
    what retrieval costs. Real multi-user traffic does not line up like that.
    """
    by: dict[str, list[edu.Probe]] = {}
    for p in probes:
        by.setdefault(p.learner_id, []).append(p)
    queues = [by[k] for k in sorted(by)]
    out: list[edu.Probe] = []
    for i in range(max((len(q) for q in queues), default=0)):
        out += [q[i] for q in queues if i < len(q)]
    return out


DATASETS = {
    # name -> (data dir, Chroma collection)
    "edu": (edu.DATA_DIR, "edu_v1"),
    "longmemeval": (Path("data/longmemeval"), "lme_v1"),
}


def _probes(dataset: str, split: str) -> list[edu.Probe]:
    if dataset == "edu":
        all_probes = edu.load_probes()
        splits = edu.split_of(list({p.learner_id for p in all_probes}))
        return _round_robin([p for p in all_probes if splits[p.learner_id] == split])
    d = DATASETS[dataset][0]
    subset = set(json.loads((d / "subset.json").read_text(encoding="utf-8")))
    # one probe per tenant: the live-QA subset (scripts/prepare_longmemeval.py)
    return [
        p for p in edu.load_probes(d / "probes.jsonl") if p.probe_id.removeprefix("lme-") in subset
    ]


def run(
    arm_name: str,
    run_name: str,
    seeds: list[int],
    split: str,
    config: list[str] | None,
    tier1_path: str | None,
    temperature: float,
    limit: int | None,
    dataset: str = "edu",
) -> None:
    from amic.app import build_arm
    from amic.arms.amic import AmicArm
    from amic.control.tier1 import Tier1Model

    data_dir, collection = DATASETS[dataset]
    # labels are needed only by the oracle arm; the split is by learner and label-independent
    lab_path = data_dir / "labels.parquet"
    labels = pd.read_parquet(lab_path).set_index("probe_id") if lab_path.exists() else None
    if arm_name == "oracle" and labels is None:
        raise SystemExit(f"the oracle arm needs {lab_path}")
    probes = _probes(dataset, split)[:limit]
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    answers = out_dir / "answers.jsonl"
    done = set()
    if answers.exists():
        done = {
            (r["probe_id"], r["seed"])
            for r in (json.loads(x) for x in answers.read_text(encoding="utf-8").splitlines() if x)
        }
    hashes = {}
    with answers.open("a", encoding="utf-8") as fout:
        for seed in seeds:
            overrides = {
                "chroma_collection": collection,
                "metrics_db_url": EXP_DB,
                "models": {"temperature": temperature, "seed": seed},
                "gate": {"distill_path": f"data/distill/{run_name}.jsonl"},  # Tier-2 verdicts
            }
            s: Settings = load_settings(config, **overrides)
            if tier1_path:
                s.gate.tier1_artifact = tier1_path
            m = s.models
            llm = CachingGateway(
                OllamaGateway(
                    m.tutor,
                    m.ollama_base_url,
                    m.temperature,
                    m.seed,
                    m.max_answer_tokens,
                    num_ctx=m.num_ctx,
                )
            )
            c = build_components(s, llm=llm)
            arm = (
                AmicArm(
                    c, Tier1Model.load(s.gate.tier1_artifact) if s.ablation.tiers != "t0" else None
                )
                if arm_name == "amic"
                else build_arm(arm_name, c)  # type: ignore[arg-type]
            )
            arm.name = run_name  # traces are tagged with the run name (arm or ablation)
            for lid in {p.learner_id for p in probes}:  # built by the write path in deployment,
                c.sketches.get(lid)  # so never inside the timed gate
            hashes[seed] = c.config_hash
            llm.chat([{"role": "user", "content": "hi"}], max_tokens=1, seed=seed)  # warm-up
            if c.gate_llm is not None or s.ablation.tiers == "full":
                # tutor calls may all be cache hits: load the gate model now, without Tier 2's
                # 800 ms HTTP timeout (a cold load takes seconds)
                gate_llm_from_settings(s).chat([{"role": "user", "content": "hi"}], max_tokens=1)
            todo = [p for p in probes if (p.probe_id, seed) not in done]
            print(f"[{run_name}] seed={seed}: {len(todo)} probes", flush=True)
            for i, p in enumerate(todo, 1):
                lab = (
                    labels.loc[p.probe_id]
                    if labels is not None and p.probe_id in labels.index
                    else None
                )
                ctx = RequestContext(
                    request_id=f"{run_name}-s{seed}-{p.probe_id}",
                    tenant_id=p.learner_id,
                    session_id=f"exp-{run_name}-s{seed}-{p.probe_id}",
                    query=p.query,
                    m0_turns=_m0(p),
                    arm=arm_name,  # type: ignore[arg-type]
                    now=p.ts,
                    oracle_label=bool(lab.label_required) if lab is not None else None,
                    oracle_depth=(
                        Route(lab.depth) if lab is not None and isinstance(lab.depth, str) else None
                    ),
                )
                resp, _drop_write = arm.answer_deferred(ctx)
                d = resp.decision
                fout.write(
                    json.dumps(
                        {
                            "probe_id": p.probe_id,
                            "seed": seed,
                            "run": run_name,
                            "route": (d.escalated_from or d.route).value,  # the gate's decision
                            "final_route": d.route.value,  # differs only after an escalation
                            "escalated": d.escalated_from is not None,
                            "resolved_by": d.resolved_by.value,
                            "p": resp.decision.p,
                            "answer": resp.answer,
                            "trace_id": resp.trace_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fout.flush()
                if i % 25 == 0 or i == len(todo):
                    print(
                        f"  {i}/{len(todo)} cache hits={llm.hits} misses={llm.misses}", flush=True
                    )
    meta = out_dir / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "arm": arm_name,
                "config": config,
                "tier1": tier1_path,
                "split": split,
                "seeds": seeds,
                "config_hash": hashes,
                "temperature": temperature,
            },
            indent=2,
        )
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["always", "never", "oracle", "amic"])
    ap.add_argument("--name", default=None, help="run name (single arm only); default = arm")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--split", default="test", help="edu only")
    ap.add_argument("--config", nargs="*", default=None, help="YAML file(s), merged in order")
    ap.add_argument("--tier1", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="edu")
    a = ap.parse_args()
    for arm in a.arms:
        run(
            arm,
            a.name or arm,
            a.seeds,
            a.split,
            a.config,
            a.tier1,
            a.temperature,
            a.limit,
            a.dataset,
        )


if __name__ == "__main__":
    main()
