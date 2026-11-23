"""Record Tier-2 verdicts for every probe that reaches Tier 1 (Tier 0 abstains), so any band,
cap and timeout can be replayed offline (pre-registered T1, docs/PLAN.md section 9).

    python scripts/replay_tier2.py                 # val + test + train -> data/edu/tier2_verdicts.jsonl

Uses the live adjudicator's prompt (control/tier2.py) with the learner's real sketch summary and
the probe's session context, at temperature 0. The HTTP timeout is generous here: the measured
latency is stored and the replay applies `gate.tier2_timeout_ms` to it. Train-split verdicts
also go to the distillation buffer data/distill/replay_train.jsonl (secondary analysis).
Resumable: probes already recorded for this model and prompt version are skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from amic.app import build_components, gate_llm_from_settings
from amic.contracts.config import Settings
from amic.contracts.models import RequestContext
from amic.control.distill import DistillationBuffer
from amic.control.signals.bus import FEATURES, Signals
from amic.control.tier2 import PROMPT_VERSION, Tier2Adjudicator
from amic.eval.datasets import edu
from amic.eval.replay import load_dataset
from amic.memory.sketch import EpisodicSketchStore

sys.path.insert(0, str(Path(__file__).parent))
from label_oracle import m0_turns  # noqa: E402

OUT = edu.DATA_DIR / "tier2_verdicts.jsonl"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    s = Settings(chroma_collection="edu_v1", session_backend="sqlite")
    c = build_components(s, llm=object(), embedder=object())  # type: ignore[arg-type]
    sketches = EpisodicSketchStore(c.episodic)
    llm = gate_llm_from_settings(s, timeout_s=30.0)
    llm.chat([{"role": "user", "content": "hi"}], max_tokens=1)  # load the model
    probes = {p.probe_id: p for p in edu.load_probes()}

    df = load_dataset("edu")
    df = df[df.tier0_rule == "abstain"]
    order = {"val": 0, "test": 1, "train": 2}
    df = df.sort_values(by=["split", "probe_id"], key=lambda col: col.map(order).fillna(col))

    done: set[str] = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                if r["model"] == llm.model and r["prompt_version"] == PROMPT_VERSION:
                    done.add(r["probe_id"])
    todo = df[~df.probe_id.isin(done)]
    print(f"{len(todo)} probes to adjudicate ({len(done)} already recorded)", flush=True)

    train_buf = DistillationBuffer(Path("data/distill/replay_train.jsonl"))
    adj = Tier2Adjudicator(llm, timeout_ms=1e9, max_rate=1.0)  # no cap, no timeout: record all
    with OUT.open("a", encoding="utf-8") as out:
        for i, rec in enumerate(todo.to_dict("records"), 1):
            p = probes[rec["probe_id"]]
            feats = {k: float(rec[k]) for k in FEATURES}
            sig = Signals(np.zeros(1), feats, sketches.get(p.learner_id), False, 0.0, 0.0)
            ctx = RequestContext(
                request_id=p.probe_id,
                tenant_id=p.learner_id,
                session_id=f"t2-{p.probe_id}",
                query=p.query,
                m0_turns=m0_turns(p),
                arm="amic",
                now=p.ts,
            )
            adj.buffer = train_buf if rec["split"] == "train" else None
            res = adj.adjudicate(sig, ctx, p=float("nan"))
            out.write(
                json.dumps(
                    {
                        "probe_id": p.probe_id,
                        "split": rec["split"],
                        "model": llm.model,
                        "prompt_version": PROMPT_VERSION,
                        "route": res.route.value if res.route else None,
                        "status": res.status,
                        "ms": res.ms,
                    }
                )
                + "\n"
            )
            out.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)
    rows = pd.read_json(OUT, lines=True)
    print(rows.groupby("split").route.value_counts().unstack(fill_value=0))
    print(f"latency ms p50={rows.ms.median():.0f} p95={rows.ms.quantile(0.95):.0f}")


if __name__ == "__main__":
    main()
