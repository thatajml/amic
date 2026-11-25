"""Prepare LongMemEval_S (cleaned) for AMIC: gate features for all 500 questions, plus frozen
memory for the live-QA subset (pre-registered L1/L2, docs/PLAN.md section 9).

    python scripts/prepare_longmemeval.py --device cuda

1. features: every question's haystack is embedded (GPU recommended: ~247k turns); the M4 sketch
   is built straight from the vectors and the 24-d FeatureVector + Tier-0 verdict are computed
   with the same code as the live bus (no Chroma needed) -> data/longmemeval/features.parquet,
   labels.parquet, probes.jsonl (all 500), gold.jsonl.
2. subset: every abstention question + 20 answerable per question type (150) -> subset.json;
   their haystacks are written to Chroma collection `lme_v1`, and their session metadata to the
   configured session backend, for the live runs.

Download first (MIT licence): curl -L -o data/longmemeval/longmemeval_s_cleaned.json <URL>
(the URL is in amic/eval/datasets/longmemeval.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta

import pandas as pd

from amic.app import build_components
from amic.contracts.config import Settings
from amic.contracts.models import RequestContext
from amic.control.signals.bus import COMMAND, FEATURES, Signals, compute_features
from amic.control.tier0 import Tier0
from amic.eval.datasets import edu
from amic.eval.datasets import longmemeval as lme
from amic.memory.embedder import STEmbedder
from amic.memory.sketch import build_sketch


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--collection", default="lme_v1")
    ap.add_argument("--per-type", type=int, default=20)
    ap.add_argument("--skip-features", action="store_true")
    ap.add_argument("--skip-seed", action="store_true")
    a = ap.parse_args()

    qs = lme.load()
    s = Settings()
    emb = STEmbedder(s.models.embedding, a.device)
    subset = set(lme.stratified_subset(qs, a.per_type))
    (lme.DIR / "subset.json").write_text(json.dumps(sorted(subset), indent=1), encoding="utf-8")
    edu.save_jsonl(lme.DIR / "probes.jsonl", [lme.probe(q) for q in qs])
    lme.labels_frame(qs).to_parquet(lme.DIR / "labels.parquet", index=False)
    with (lme.DIR / "gold.jsonl").open("w", encoding="utf-8") as f:
        for q in qs:
            f.write(
                json.dumps(
                    {
                        "probe_id": lme.tenant_of(q),
                        "question_type": q["question_type"],
                        "abstention": lme.is_abstention(q),
                        "question": q["question"],
                        "answer": lme.gold(q),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"{len(qs)} questions; live subset {len(subset)}", flush=True)

    c = None
    if not a.skip_seed:
        c = build_components(Settings(chroma_collection=a.collection), llm=object(), embedder=emb)  # type: ignore[arg-type]
    tier0 = Tier0()
    rows = []
    t_start = time.perf_counter()
    for i, q in enumerate(qs, 1):
        recs = list(lme.records(q))
        texts = [r[1] for r in recs]
        vecs = emb.encode_many(texts, batch_size=256)
        p = lme.probe(q)
        tenant = p.learner_id
        if not a.skip_features:
            sketch = build_sketch(vecs, texts)
            ctx = RequestContext(
                request_id=tenant,
                tenant_id=tenant,
                session_id=f"q-{tenant}",
                query=p.query,
                arm="amic",
                now=p.ts,
            )
            meta = lme.session_meta(q)
            q_vec = emb.encode(p.query)
            feats = compute_features(ctx, q_vec, sketch, meta)
            sig = Signals(q_vec, feats, sketch, bool(COMMAND.match(p.query)), 0.0, 0.0)
            t0 = tier0.decide(sig)
            rows.append(
                {
                    "probe_id": tenant,
                    **{k: feats[k] for k in FEATURES},
                    "tier0_route": t0.route.value if t0.route else None,
                    "tier0_rule": t0.rule.split(":")[0],
                    "n_records": len(recs),
                }
            )
        if c is not None and q["question_id"] in subset:
            c.episodic.delete_tenant(tenant)
            c.sessions.erase_tenant(tenant)
            c.episodic.add_many(
                tenant,
                [r[0] for r in recs],
                texts,
                vecs,
                [r[2] for r in recs],
                [{"session_id": r[3], "kind": "dialogue"} for r in recs],
            )
            for sid, date, turns in zip(
                q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"], strict=True
            ):
                start = lme.parse_date(date)
                c.sessions.touch(tenant, f"{tenant}-{sid}", start)
                c.sessions.touch(tenant, f"{tenant}-{sid}", start + timedelta(minutes=len(turns)))
            c.sessions.bump_mem_version(tenant)  # a direct store write: invalidate caches
        if i % 25 == 0 or i == len(qs):
            print(f"  {i}/{len(qs)}  {time.perf_counter() - t_start:.0f}s", flush=True)
    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(lme.DIR / "features.parquet", index=False)
        print("tier0:", df.tier0_rule.value_counts().to_dict())


if __name__ == "__main__":
    main()
