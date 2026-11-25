"""Summarise a shadow-mode log (data/shadow.jsonl): the served arm vs the shadow arm.

    python scripts/shadow_report.py [--log data/shadow.jsonl]

Per request both arms saw the same context; reported: route agreement, the served arm's routes,
memory tokens and latency vs the shadow's (the savings the gate would claim in production), and
how often the served arm skipped memory where the shadow used some. Answer quality needs a judge:
the rows carry both answers, ready for scripts/judge_pairwise.py-style comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="data/shadow.jsonl")
    a = ap.parse_args()
    rows = [json.loads(x) for x in Path(a.log).read_text(encoding="utf-8").splitlines() if x]
    if not rows:
        print("empty log")
        return
    sv = [r["served"] for r in rows]
    sh = [r["shadow"] for r in rows]

    def m(xs: list[dict], k: str) -> float:
        return float(np.mean([x[k] for x in xs]))

    routes: dict[str, int] = {}
    for x in sv:
        routes[x["route"]] = routes.get(x["route"], 0) + 1
    out = {
        "n": len(rows),
        "served_arm": rows[0]["served_arm"],
        "shadow_arm": rows[0]["shadow_arm"],
        "served_routes": {k: v / len(rows) for k, v in sorted(routes.items())},
        "route_agreement": float(
            np.mean([a["route"] == b["route"] for a, b in zip(sv, sh, strict=True)])
        ),
        "skipped_where_shadow_retrieved": float(
            np.mean([a["n_items"] == 0 and b["n_items"] > 0 for a, b in zip(sv, sh, strict=True)])
        ),
        "memory_tokens": {
            "served": m(sv, "memory_tokens_est"),
            "shadow": m(sh, "memory_tokens_est"),
        },
        "latency_ms_p50": {
            "served": float(np.median([x["latency_ms"] for x in sv])),
            "shadow": float(np.median([x["latency_ms"] for x in sh])),
        },
    }
    out["memory_token_saving"] = 1 - out["memory_tokens"]["served"] / max(
        1e-9, out["memory_tokens"]["shadow"]
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
