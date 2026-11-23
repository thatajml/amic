"""Decision-cache replay on the education test split (pre-registered C1).

    python scripts/replay_cache.py           # -> runs/cache_replay.json

The gate (T0+T1, base config) is replayed over the test probes in the live runs' order with the
DecisionCache in front of Tier 1 (memory frozen, so the version never changes). Reported:
- natural stream (each probe once): hit rate, and wrong hits (cached route != fresh route);
- repeat stream (each probe asked twice in a row): hit rate on the second ask;
- false sharing: share of DISTINCT queries of one learner that land in an occupied bucket, at
  8, 12 and 16 LSH bits (the cost side of a coarser key).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from amic.contracts.config import Settings
from amic.contracts.models import RequestContext
from amic.control.cache import DecisionCache
from amic.control.controller import Controller
from amic.control.route_planner import RoutePlanner
from amic.control.signals.bus import FEATURES, Signals
from amic.control.tier1 import Tier1Model
from amic.eval.datasets import edu
from amic.eval.replay import load_dataset
from amic.eval.runner import _round_robin
from amic.memory.embedder import STEmbedder


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    s = Settings()
    df = load_dataset("edu").set_index("probe_id")
    probes = edu.load_probes()
    split = edu.split_of(list({p.learner_id for p in probes}))
    test = _round_robin([p for p in probes if split[p.learner_id] == "test"])
    emb = STEmbedder(s.models.embedding, "cpu")
    qv = {p.probe_id: emb.encode(p.query) for p in test}
    ctl = Controller(s, RoutePlanner(s.routes), Tier1Model.load(s.gate.tier1_artifact))

    def fresh(p: edu.Probe) -> tuple[str, str]:
        feats = {k: float(df.loc[p.probe_id, k]) for k in FEATURES}
        sig = Signals(qv[p.probe_id], feats, None, False, 0.0, 0.0)
        ctx = RequestContext(
            request_id=p.probe_id, tenant_id=p.learner_id, session_id="r", query=p.query, arm="amic"
        )
        d, _ = ctl.decide_signals(sig, ctx)
        return d.route.value, d.resolved_by.value

    def stream(order: list[edu.Probe], nbits: int) -> dict[str, float]:
        cache = DecisionCache(nbits=nbits)
        hits = wrong = eligible = 0
        for p in order:
            route, by = fresh(p)
            if by == "tier0":  # never cached (deviation R4)
                continue
            eligible += 1
            m0 = float(df.loc[p.probe_id, "s2_m0_turns"]) > 0
            k = cache.key(p.learner_id, 0, qv[p.probe_id], m0)
            got = cache.get(k)
            if got is not None:
                hits += 1
                wrong += got.route.value != route
            else:
                from amic.contracts.models import MemoryDecision, Route

                cache.put(k, MemoryDecision(route=Route(route), resolved_by=by))  # type: ignore[arg-type]
        return {"eligible": eligible, "hits": hits, "wrong_hits": wrong}

    out: dict[str, object] = {"n_test_probes": len(test)}
    for nbits in (8, 12, 16):
        nat = stream(test, nbits)
        rep = stream([p for p in test for _ in (0, 1)], nbits)
        # false sharing: distinct queries of one learner that share a bucket with an earlier one
        cache = DecisionCache(nbits=nbits)
        shared = total = 0
        by: dict[str, set[int]] = {}
        for p in test:
            b = cache.bucket(qv[p.probe_id])
            seen = by.setdefault(p.learner_id, set())
            shared += b in seen
            total += 1
            seen.add(b)
        out[f"bits_{nbits}"] = {
            "natural": nat,
            "repeat": rep,
            "false_sharing_rate": shared / total,
        }
        print(nbits, out[f"bits_{nbits}"], flush=True)
    Path("runs/cache_replay.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
