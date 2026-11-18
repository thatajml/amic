"""Compute the gate FeatureVector + Tier-0 verdict for every probe (offline, read-only).

    python scripts/extract_features.py            # -> data/{edu,locomo}/features.parquet

These rows feed Tier-1 training and the offline replay of every ablation (plan section 1).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from amic.app import build_components
from amic.contracts.config import Settings
from amic.contracts.models import RequestContext
from amic.control.signals.bus import FEATURES, SignalBus
from amic.control.tier0 import Tier0
from amic.eval.datasets import edu, locomo
from amic.memory.sketch import EpisodicSketchStore

sys.path.insert(0, str(Path(__file__).parent))
from label_oracle import m0_turns  # noqa: E402


def run(dataset: str, collection: str, probes: list[edu.Probe], out: Path) -> pd.DataFrame:
    c = build_components(Settings(chroma_collection=collection), llm=object())  # type: ignore[arg-type]
    sketches = EpisodicSketchStore(c.episodic)
    bus = SignalBus(c.embedder, sketches, c.sessions)
    tier0 = Tier0()
    rows = []
    for p in probes:
        ctx = RequestContext(
            request_id=p.probe_id,
            tenant_id=p.learner_id,
            session_id=f"probe-{p.probe_id}",
            query=p.query,
            m0_turns=m0_turns(p),
            arm="amic",
            now=p.ts,
        )
        sketches.get(p.learner_id)  # build outside the timed region (built by the write path)
        sig = bus.extract(ctx)
        t = time.perf_counter()
        t0 = tier0.decide(sig)
        t0_ms = (time.perf_counter() - t) * 1000
        rows.append(
            {
                "probe_id": p.probe_id,
                **{k: sig.features[k] for k in FEATURES},
                "tier0_route": t0.route.value if t0.route else None,
                "tier0_rule": t0.rule.split(":")[0],
                "embed_ms": sig.embed_ms,
                "feat_ms": sig.feat_ms,
                "tier0_ms": t0_ms,
            }
        )
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[{dataset}] {len(df)} rows -> {out}")
    print(f"  tier0: {df.tier0_rule.value_counts().to_dict()}")
    print(
        f"  ms p50: embed {df.embed_ms.median():.2f}  features {df.feat_ms.median():.3f}"
        f"  tier0 {df.tier0_ms.median():.4f}"
    )
    return df


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["edu", "locomo"], default=None)
    a = ap.parse_args()
    if a.only in (None, "edu"):
        run("edu", "edu_v1", edu.load_probes(), edu.DATA_DIR / "features.parquet")
    if a.only in (None, "locomo"):
        ps = edu.load_probes(locomo.PATH.parent / "probes.jsonl")
        run("locomo", "locomo_v1", ps, locomo.PATH.parent / "features.parquet")


if __name__ == "__main__":
    main()
