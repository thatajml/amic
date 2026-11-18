"""End-to-end smoke test against the real stack (Redis, MiniLM, embedded Chroma, Ollama).

python scripts/smoke_e2e.py
"""

from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime, timedelta

from amic.app import build_arm, build_components, make_context
from amic.contracts.config import Settings


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    run = uuid.uuid4().hex[:6]
    s = Settings(chroma_collection=f"smoke_{run}", metrics_db_url="sqlite:///./data/smoke.db")
    c = build_components(s)
    learner = f"smoke-{run}"
    past = datetime.now(UTC) - timedelta(days=3)
    c.writer.write_turn(
        learner,
        "old",
        "h1",
        "Why does my for loop over range(len(nums)+1) crash?",
        "range(len(nums)+1) goes one past the last index, so nums[i] raises IndexError.",
        past,
    )
    q = "I got that same kind of error again, what was my mistake last time?"
    for arm_name in ["never", "always"]:
        ctx = make_context(c, learner, f"{arm_name}-{run}", q, arm_name)  # type: ignore[arg-type]
        r = build_arm(arm_name, c).answer(ctx)  # type: ignore[arg-type]
        t = c.telemetry.fetch(limit=1)[0]  # type: ignore[attr-defined]
        L = t.latency
        print(f"\n=== {arm_name}: route={r.decision.route.value} items={len(r.items)}")
        print(
            f"tokens prompt={t.prompt_tokens} memory~{t.memory_tokens} completion={t.completion_tokens}"
            f" | ms embed={L.embed:.1f} retrieval={L.retrieval:.1f} gen={L.generation:.0f}"
            f" total={L.total:.0f}"
        )
        print(r.answer[:500])
    c.episodic.delete_tenant(learner)


if __name__ == "__main__":
    main()
