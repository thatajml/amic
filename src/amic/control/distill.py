"""DistillationBuffer (brief 6.2): every Tier-2 verdict becomes a training row for Tier 1.

Rows are appended to a JSONL file (features, Tier-1 p, Tier-2 route). `scripts/train_tier1.py
--distill PATH` adds them to Tier 1's training set as weak labels (memory needed iff the
verdict is not SKIP), so the cheap tier learns from the expensive one over time.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from amic.contracts.models import RequestContext, Route


class DistillationBuffer:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.rows = 0

    def append(
        self, ctx: RequestContext, features: dict[str, float], p: float, route: Route, model: str
    ) -> None:
        row = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "request_id": ctx.request_id,
            "tenant_id": ctx.tenant_id,
            "p": p,
            "route": route.value,
            "model": model,
            "features": features,
        }
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            self.rows += 1


def load_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]
