"""Performance budgets on the dev machine (pre-registered PB1; brief section 11).

    python scripts/bench_gate.py        # -> runs/perf_budgets.json

Real Tier-1 artifact (models/tier1.joblib), real feature vectors from the education set:
Tier 0 alone, Tier 0 + Tier 1 (controller, band, route planning), feature extraction against a
sketch built from 10^2 / 10^3 / 10^4 records, sketch build time, and SHALLOW / DEEP retrieval
(M1, Chroma embedded) at the same memory sizes. 2000 / 500 / 200 repetitions after warm-up.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from amic.contracts.config import Settings
from amic.contracts.models import RequestContext, Route
from amic.control.controller import Controller
from amic.control.route_planner import RoutePlanner
from amic.control.signals.bus import FEATURES, Signals, compute_features
from amic.control.tier0 import Tier0
from amic.control.tier1 import Tier1Model
from amic.eval.replay import load_dataset
from amic.memory.retrieval import ChromaRetrievalEngine
from amic.memory.sketch import build_sketch
from amic.memory.stores.episodic import EpisodicStore, make_chroma_client


def timed(fn, n: int) -> tuple[float, float]:
    for _ in range(min(50, n)):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), float(np.quantile(ts, 0.95))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    s = Settings()
    df = load_dataset("edu")
    rec = df[df.tier0_rule == "abstain"].iloc[0]
    feats = {k: float(rec[k]) for k in FEATURES}
    sig = Signals(np.zeros(384, np.float32), feats, None, False, 0.0, 0.0)
    ctx = RequestContext(
        request_id="r", tenant_id="t", session_id="s", query=str(rec.query), arm="amic"
    )
    rows = []
    p50, p95 = timed(lambda: Tier0().decide(sig), 2000)
    rows.append({"what": "Tier 0", "records": "", "p50 ms": p50, "p95 ms": p95, "budget ms": 0.1})
    ctl = Controller(s, RoutePlanner(s.routes), Tier1Model.load(s.gate.tier1_artifact))
    p50, p95 = timed(lambda: ctl.decide_signals(sig, ctx), 2000)
    rows.append(
        {"what": "Tier 0 + Tier 1", "records": "", "p50 ms": p50, "p95 ms": p95, "budget ms": 5.0}
    )
    rng = np.random.default_rng(0)
    q = rng.standard_normal(384).astype(np.float32)
    q /= np.linalg.norm(q)
    client = make_chroma_client("ephemeral")
    for n in (100, 1_000, 10_000):
        v = rng.standard_normal((n, 384)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        texts = [
            f"Student asked: question {i} about lists, loops and dictionaries" for i in range(n)
        ]
        t = time.perf_counter()
        sk = build_sketch(v, texts)
        build_ms = (time.perf_counter() - t) * 1000
        p50, p95 = timed(lambda sk=sk: compute_features(ctx, q, sk, {}, True), 500)
        rows.append(
            {
                "what": "signals vs sketch (S1-S4)",
                "records": n,
                "p50 ms": p50,
                "p95 ms": p95,
                "budget ms": None,
            }
        )
        rows.append(
            {
                "what": "sketch build (write path)",
                "records": n,
                "p50 ms": build_ms,
                "p95 ms": build_ms,
                "budget ms": None,
            }
        )
        ep = EpisodicStore(client, f"bench_{uuid.uuid4().hex[:6]}")
        now = datetime.now(UTC)
        ep.add_many("t", [f"r{i}" for i in range(n)], texts, v, [now] * n)
        eng = ChromaRetrievalEngine(ep, s.routes)
        for route in (Route.SHALLOW, Route.DEEP):
            plan = RoutePlanner(s.routes).plan(route)
            p50, p95 = timed(lambda plan=plan, eng=eng: eng.retrieve(plan, q, "t"), 200)
            rows.append(
                {
                    "what": f"{route.value} retrieval (M1)",
                    "records": n,
                    "p50 ms": p50,
                    "p95 ms": p95,
                    "budget ms": None,
                }
            )
    for r in rows:
        print(r)
    Path("runs/perf_budgets.json").write_text(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
