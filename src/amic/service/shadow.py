"""Shadow mode (brief section 12, staged rollout: local -> GPU/hosted -> shadow vs `always`).

With `shadow.enabled`, every /chat request is replayed after the response is sent through the
shadow arm (default `always`), on the same context (same M0, same time). The shadow run never
reaches the user and never writes: its M0 append and long-term write are dropped. One JSONL row
per request compares the served and shadow decision, memory, latency and answer;
`scripts/shadow_report.py` summarises the log. Its trace is recorded with arm `shadow-<arm>`.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from amic.contracts.models import AgentResponse, RequestContext
from amic.memory.assembler import estimate_tokens


class _NoWriteSessions:
    """Session manager proxy for the shadow run: reads pass through, M0 appends are dropped."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def append(self, *_: Any, **__: Any) -> None:
        return None

    def touch(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def summary(resp: AgentResponse, ms: float) -> dict[str, Any]:
    d = resp.decision
    return {
        "route": (d.escalated_from or d.route).value,
        "final_route": d.route.value,
        "resolved_by": d.resolved_by.value,
        "n_items": len(resp.items),
        "memory_tokens_est": sum(estimate_tokens(i.text) for i in resp.items),
        "latency_ms": round(ms, 1),
        "answer": resp.answer,
    }


class ShadowRunner:
    def __init__(self, comps: Any, arm_name: str, log_path: str) -> None:
        from amic.app import build_arm

        self.arm_name = arm_name
        shadow_comps = dataclasses.replace(comps, sessions=_NoWriteSessions(comps.sessions))
        self.arm = build_arm(arm_name, shadow_comps)  # type: ignore[arg-type]
        self.arm.name = f"shadow-{arm_name}"
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.rows = 0

    def run(self, ctx: RequestContext, served: AgentResponse, served_ms: float) -> dict[str, Any]:
        sctx = ctx.model_copy(update={"request_id": f"{ctx.request_id}-shadow"})
        t = time.perf_counter()
        resp, _drop_write = self.arm.answer_deferred(sctx)  # never written (P5 + shadow rule)
        row = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "request_id": ctx.request_id,
            "tenant_id": ctx.tenant_id,
            "query": ctx.query,
            "served_arm": ctx.arm,
            "shadow_arm": self.arm_name,
            "served": summary(served, served_ms),
            "shadow": summary(resp, (time.perf_counter() - t) * 1000),
        }
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            self.rows += 1
        return row
