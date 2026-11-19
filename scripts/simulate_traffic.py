"""Replay education-set questions through the pipeline at a steady rate, to drive the dashboard.

    python scripts/simulate_traffic.py                         # amic arm, default tutor, Ollama
    python scripts/simulate_traffic.py --models llama3.1:8b llama3.2:3b --rate 0.5
    python scripts/simulate_traffic.py --fake-llm --rate 5     # no Ollama: for UI work

Runs in-process against the frozen `edu_v1` memory (long-term writes are dropped, so the
experiment collection is never modified), with the shadow baseline on, and records traces into
settings.metrics_db_url (default data/metrics.db), which the dashboard reads.
Probes are reused from the dataset, so each learner's history and session context are real.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from amic.app import build_arm, build_components, new_request_id
from amic.contracts.config import Settings
from amic.contracts.models import RequestContext, Turn
from amic.eval.datasets import edu
from amic.generation.gateway.base import FakeLLM, LLMGateway


def _m0(p: edu.Probe) -> list[Turn]:
    role = {"student": "user", "tutor": "assistant"}
    return [Turn(role=role.get(t.role, "user"), text=t.text, ts=p.ts) for t in p.m0_context]  # type: ignore[arg-type]


def _fake(name: str) -> LLMGateway:
    llm = FakeLLM()
    llm.model = name  # type: ignore[misc]
    return llm


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", default="amic", choices=["amic", "always", "never"])
    ap.add_argument("--models", nargs="+", default=None, help="tutor models, used round-robin")
    ap.add_argument("--rate", type=float, default=1.0, help="requests per second (upper bound)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N requests")
    ap.add_argument("--fake-llm", action="store_true", help="deterministic stand-in, no Ollama")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    base = Settings()
    models = a.models or [base.models.tutor]
    s = Settings(
        session_backend="sqlite",
        chroma_collection="edu_v1",
        telemetry={"shadow_baseline": True},
        models={**base.models.model_dump(), "tutor": models[0], "available": models[1:]},
    )
    if a.fake_llm:
        c = build_components(s, llm=_fake(models[0]), llm_factory=_fake)
    else:
        c = build_components(s)
    arm = build_arm(a.arm, c)  # type: ignore[arg-type]

    probes = edu.load_probes()
    random.Random(a.seed).shuffle(probes)
    total = a.limit or len(probes)
    print(f"{a.arm} arm, models={models}, {a.rate}/s -> {s.metrics_db_url}  (Ctrl+C to stop)")
    gap = 1.0 / a.rate if a.rate > 0 else 0.0
    try:
        for i in range(total):
            t0 = time.perf_counter()
            p = probes[i % len(probes)]
            ctx = RequestContext(
                request_id=new_request_id(),
                tenant_id=p.learner_id,
                session_id=f"sim-{p.probe_id}",
                query=p.query,
                m0_turns=_m0(p),
                arm=a.arm,  # type: ignore[arg-type]
                now=p.ts,  # the learner's own clock, so session signals match the dataset
                model=models[i % len(models)],
            )
            resp, job = arm.answer_deferred(ctx, write=False)
            job()  # shadow baseline only: memory stays frozen
            d = resp.decision
            print(
                f"{i + 1:>5}  {ctx.model:<14} {d.route.value:<7} by {d.resolved_by.value:<6} "
                f"{p.learner_id}  {p.query[:60]}",
                flush=True,
            )
            time.sleep(max(0.0, gap - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
