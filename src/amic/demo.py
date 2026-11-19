"""CLI demo with a per-turn decision panel (plan T7.2).

    python -m amic.demo --script                 # 3 canned turns: cold start, general, recall
    python -m amic.demo --learner edu-001        # interactive chat as a seeded learner
    python -m amic.demo --arm always             # compare with a baseline arm

Uses the real stack (Ollama, Redis, embedded Chroma) and the seeded `edu_v1` memory.
In demo mode long-term writes are ON (it is a real conversation), except with --frozen.
"""

from __future__ import annotations

import argparse
import sys
import uuid

from amic.app import build_arm, build_components, make_context
from amic.contracts.config import load_settings
from amic.contracts.models import AgentResponse, DecisionTrace

SCRIPT = [
    ("demo-newcomer", "What is a Python list?"),  # R1: no memory -> SKIP
    ("edu-001", "What's the difference between a list and a tuple?"),  # general -> Tier 1
    ("edu-001", "What did I get wrong last time with files?"),  # R3 explicit recall -> DEEP
]


def panel(resp: AgentResponse, tr: DecisionTrace) -> str:
    d = resp.decision
    L = tr.latency
    band = f"[{d.tau_low:.2f}, {d.tau_high:.2f}]" if d.tau_low is not None else "-"
    p = f"{d.p:.3f}" if d.p is not None else "-"
    top = sorted(
        ((k, v) for k, v in tr.features.items() if v and not k.startswith("x_")),
        key=lambda kv: -abs(kv[1]),
    )[:6]
    lines = [
        "┌─ decision " + "─" * 58,
        f"│ route {d.route.value:<8} by {d.resolved_by.value:<9} p={p}  band={band}",
        f"│ why   {d.rationale}",
    ]
    if top:
        lines.append("│ signals " + ", ".join(f"{k}={v:.2f}" for k, v in top))
    for it in resp.items[:3]:
        lines.append(f"│ memory [{it.score:.2f}] {it.text[:70].replace(chr(10), ' ')}")
    if len(resp.items) > 3:
        lines.append(f"│ ... {len(resp.items) - 3} more items")
    lines += [
        f"│ ms    embed {L.embed:.1f} | gate {L.gate:.2f} | retrieval {L.retrieval:.1f} | "
        f"generation {L.generation:.0f} | total {L.total:.0f}",
        f"│ tok   prompt {tr.prompt_tokens} (memory ~{tr.memory_tokens})"
        f" | completion {tr.completion_tokens}",
        "└" + "─" * 69,
    ]
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="amic", choices=["amic", "always", "never"])
    ap.add_argument("--learner", default="edu-001")
    ap.add_argument("--script", action="store_true")
    ap.add_argument("--frozen", action="store_true", help="do not write new turns to memory")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    from amic.telemetry.store import InMemoryTelemetry

    tel = InMemoryTelemetry()
    s = load_settings(a.config, chroma_collection="edu_v1")
    c = build_components(s, telemetry=tel)
    arm = build_arm(a.arm, c)  # type: ignore[arg-type]
    session = f"demo-{uuid.uuid4().hex[:6]}"
    print(f"AMIC demo · arm={a.arm} · config={c.config_hash} · type 'quit' to exit\n")

    def turn(learner: str, q: str) -> None:
        if a.script:  # typed turns are already on screen
            print(f"\n[{learner}] > {q}")
        print("  thinking...", end="\r", flush=True)
        ctx = make_context(c, learner, f"{session}-{learner}", q, a.arm)  # type: ignore[arg-type]
        resp, job = arm.answer_deferred(ctx)
        print(" " * 14, end="\r")
        print(f"\n{resp.answer}\n")
        print(panel(resp, tel.traces[-1]))
        if not a.frozen and learner != "demo-newcomer":
            job()

    if a.script:
        for learner, q in SCRIPT:
            turn(learner, q)
        return
    while True:
        try:
            q = input(f"\n[{a.learner}] > ").strip()
        except EOFError:
            break
        if q.lower() in {"quit", "exit"}:
            break
        if q:
            turn(a.learner, q)


if __name__ == "__main__":
    main()
