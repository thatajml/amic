"""S5 uncertainty features for every education probe (pre-registered S1).

    python scripts/extract_s5.py              # -> data/edu/s5.parquet (resumable)

Per probe: the SKIP prompt (no memory, the probe's M0 turns, its timestamp) is sent to the tutor
model for a 16-token greedy draft with the top-5 logprobs per token; features = mean and max
token entropy (amic.control.signals.uncertainty). Also records the draft time (the signal's
cost on the gate path).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from amic.contracts.config import Settings
from amic.control.signals.uncertainty import UncertaintySignal
from amic.eval.datasets import edu
from amic.eval.runner import _m0
from amic.generation.gateway.ollama import OllamaGateway
from amic.generation.prompt import PromptBuilder

OUT = Path("data/edu/s5.parquet")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    s = Settings()
    m = s.models
    llm = OllamaGateway(m.tutor, m.ollama_base_url, 0.0, 0, 16, num_ctx=m.num_ctx)
    sig = UncertaintySignal(llm, PromptBuilder.for_profile("tutor"), s.gate.s5_tokens)
    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["probe_id"])
    seen = set(done.probe_id)
    rows = done.to_dict("records")
    probes = [p for p in edu.load_probes() if p.probe_id not in seen]
    for i, p in enumerate(probes, 1):
        f = sig.features(p.query, _m0(p), p.ts)
        rows.append({"probe_id": p.probe_id, **f, "s5_ms": sig.last_ms})
        if i % 50 == 0 or i == len(probes):
            pd.DataFrame(rows).to_parquet(OUT, index=False)
            print(f"{i}/{len(probes)}", flush=True)
    d = pd.DataFrame(rows)
    print(d.describe().T[["mean", "50%", "max"]])


if __name__ == "__main__":
    main()
