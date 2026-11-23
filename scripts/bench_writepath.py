"""Write-path benchmark (pre-registered W1): what the response path pays per turn, and how long
until a turn is visible in long-term memory.

    python scripts/bench_writepath.py --n 30 [--extract]     # -> runs/writepath_bench.json

On a scratch Chroma collection (ephemeral client), with the real embedder, for n turns taken from
the education histories:
- inline: the old path, the write done synchronously (embed + M1 upsert [+ fact extraction]);
- enqueue: the new response-path cost, one insert into the SQLite write queue;
- visibility: enqueue -> processed by a ConsolidationWorker thread (queue wait + consolidation).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from amic.app import build_components, gate_llm_from_settings
from amic.contracts.config import Settings
from amic.contracts.models import WriteIntent
from amic.eval.datasets import edu
from amic.memory.stores.episodic import make_chroma_client
from amic.memory.stores.semantic import SemanticStore
from amic.telemetry.store import InMemoryTelemetry
from amic.writepath.extract import ConflictResolver, FactExtractor
from amic.writepath.queue import SqliteWriteQueue
from amic.writepath.worker import ConsolidationWorker


def pct(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, float)
    return {"n": len(a), "p50": float(np.median(a)), "p95": float(np.quantile(a, 0.95))}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--extract", action="store_true")
    a = ap.parse_args()
    s = Settings(
        session_backend="sqlite",
        session_db=":memory:",
        chroma_collection=f"bench_{uuid.uuid4().hex[:6]}",
    )
    client = make_chroma_client("ephemeral")
    c = build_components(s, llm=object(), chroma_client=client, telemetry=InMemoryTelemetry())  # type: ignore[arg-type]
    turns = [
        (lr.learner_id, sess.session_id, q, ans)
        for lr in edu.load_learners()
        for sess in lr.sessions
        for q, ans in edu.exchanges(sess)
    ][: a.n]
    extractor = resolver = None
    if a.extract:
        llm = gate_llm_from_settings(s, timeout_s=180.0)
        llm.chat([{"role": "user", "content": "hi"}], max_tokens=1)
        extractor = FactExtractor(llm)
        resolver = ConflictResolver(SemanticStore(client, s.chroma_collection), c.embedder)
    worker = ConsolidationWorker(c.writer, c.sessions, extractor, resolver)
    now = datetime.now(UTC)

    def intent(i: int, t: tuple[str, str, str, str], tag: str) -> WriteIntent:
        lid, sid, q, ans = t
        return WriteIntent(
            tenant_id=f"{tag}-{lid}",
            session_id=sid,
            record_id=f"{tag}-{i}",
            query=q,
            answer=ans,
            ts=now,
        )

    inline = []
    for i, t in enumerate(turns):  # old path: the whole consolidation on the response path
        t0 = time.perf_counter()
        worker.process(intent(i, t, "inline"))
        inline.append((time.perf_counter() - t0) * 1000)

    q = SqliteWriteQueue(str(Path("data") / f"bench_queue_{uuid.uuid4().hex[:6]}.sqlite"))
    enq, sent = [], {}
    for i, t in enumerate(turns):
        it = intent(i, t, "queued")
        t0 = time.perf_counter()
        q.enqueue(it)
        enq.append((time.perf_counter() - t0) * 1000)
        sent[it.record_id] = time.perf_counter()
    visible: list[float] = []
    stop = threading.Event()

    def run() -> None:
        while not stop.is_set():
            for msg_id, it in q.read(count=1, block_ms=200):
                worker.process(it)
                q.ack(msg_id)
                visible.append((time.perf_counter() - sent[it.record_id]) * 1000)

    th = threading.Thread(target=run)
    th.start()
    while len(visible) < len(turns):
        time.sleep(0.05)
    stop.set()
    th.join()
    out = {
        "n": len(turns),
        "extract_facts": a.extract,
        "inline_write_ms": pct(inline),
        "enqueue_ms": pct(enq),
        "enqueue_to_visible_ms": pct(visible),
        "facts": worker.stats.facts,
    }
    p = Path("runs/writepath_bench.json")
    prev = json.loads(p.read_text()) if p.exists() else {}
    prev["extract" if a.extract else "no_extract"] = out
    p.write_text(json.dumps(prev, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
