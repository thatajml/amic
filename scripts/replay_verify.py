"""Offline replay of the verify policy (pre-registered E4, docs/PLAN.md section 9).

Every SKIP answer of the given runs is checked lexically AND by the LLM verifier, so escalation
can be replayed for each verifier mode (lexical / llm / both) without regenerating anything: an
escalated answer is byte for byte the always-SHALLOW answer for that probe and seed (same
prompt, same seed, generation cache), whose pairwise verdict already exists.

    python scripts/replay_verify.py --runs amic amic_lam1 never   # -> runs/<run>/verify_replay.jsonl

LLM verdicts are cached in data/edu/verify_verdicts.jsonl by (model, query, answer, M0), so runs
that share a SKIP answer share one call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from amic.app import gate_llm_from_settings
from amic.contracts.config import Settings
from amic.eval.datasets import edu
from amic.generation.verifier import ResponseVerifier, lexical_check

sys.path.insert(0, str(Path(__file__).parent))
from label_oracle import m0_turns  # noqa: E402

CACHE = edu.DATA_DIR / "verify_verdicts.jsonl"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()

    s = Settings()
    gate = gate_llm_from_settings(s, timeout_s=60.0)
    verifier = ResponseVerifier("llm", gate, s.verify.llm_max_tokens)
    gate.chat([{"role": "user", "content": "hi"}], max_tokens=1)  # load the model
    probes = {p.probe_id: p for p in edu.load_probes()}

    cache: dict[str, dict] = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                cache[r["key"]] = r
    fcache = CACHE.open("a", encoding="utf-8")

    for run in a.runs:
        rows = [
            json.loads(x)
            for x in (Path("runs") / run / "answers.jsonl").read_text(encoding="utf-8").splitlines()
            if x
        ]
        skips = [r for r in rows if r["route"] == "SKIP"]
        out_path = Path("runs") / run / "verify_replay.jsonl"
        n_new = 0
        with out_path.open("w", encoding="utf-8") as out:
            for r in skips:
                p = probes[r["probe_id"]]
                m0 = m0_turns(p)
                blob = json.dumps(
                    [gate.model, p.query, r["answer"].strip(), [t.text for t in m0]],
                    ensure_ascii=False,
                )
                key = hashlib.sha256(blob.encode()).hexdigest()[:24]
                if key not in cache:
                    res = verifier.check(p.query, r["answer"], m0)
                    cache[key] = {
                        "key": key,
                        "llm": None if res.reason == "llm-error" else res.needs_memory,
                        "llm_ms": res.llm_ms,
                    }
                    fcache.write(json.dumps(cache[key]) + "\n")
                    fcache.flush()
                    n_new += 1
                v = cache[key]
                out.write(
                    json.dumps(
                        {
                            "probe_id": r["probe_id"],
                            "seed": r["seed"],
                            "lexical": lexical_check(r["answer"], m0),
                            "llm": v["llm"],
                            "llm_ms": v["llm_ms"],
                        }
                    )
                    + "\n"
                )
        print(f"[{run}] {len(skips)} SKIP answers checked ({n_new} new LLM calls) -> {out_path}")
    fcache.close()


if __name__ == "__main__":
    main()
