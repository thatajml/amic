"""ConsolidationWorker (L7): turns queued WriteIntents into durable memory, off the hot path.

Per intent: M1 upsert (the raw turn; bumps the learner's memory version, which invalidates the
M4 sketch and the decision cache in every process) -> salience filter -> LLM fact extraction ->
conflict resolution into M2 -> version bump if M2 changed. Acknowledged only after success, so a
crash re-delivers the intent (at-least-once; the M1 upsert is idempotent by record id).

    python -m amic.writepath.worker              # backend from settings.writepath.queue
    python -m amic.writepath.worker --once       # drain what is queued, then exit
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from amic.contracts.models import WriteIntent
from amic.writepath.extract import ConflictResolver, FactExtractor, is_salient
from amic.writepath.queue import WriteIntentQueue


@dataclass
class WorkerStats:
    processed: int = 0
    extracted: int = 0
    facts: dict[str, int] = field(
        default_factory=lambda: {"insert": 0, "update": 0, "refresh": 0, "skip": 0}
    )
    failed: int = 0
    ms: list[float] = field(default_factory=list)


class ConsolidationWorker:
    def __init__(
        self,
        writer: Any,
        sessions: Any,
        extractor: FactExtractor | None = None,
        resolver: ConflictResolver | None = None,
        min_words: int = 4,
    ) -> None:
        self.writer = writer
        self.sessions = sessions
        self.extractor = extractor
        self.resolver = resolver
        self.min_words = min_words
        self.stats = WorkerStats()

    def process(self, it: WriteIntent) -> None:
        t = time.perf_counter()
        self.writer.write_turn(
            it.tenant_id, it.session_id, it.record_id, it.query, it.answer, it.ts, it.metadata
        )
        if (
            self.extractor is not None
            and self.resolver is not None
            and is_salient(it.query, it.answer, self.min_words)
        ):
            facts = self.extractor.extract([(it.query, it.answer)], it.ts)
            self.stats.extracted += 1
            counts = self.resolver.resolve(it.tenant_id, facts, it.ts, source=it.record_id)
            for k, v in counts.items():
                self.stats.facts[k] += v
            if counts["insert"] or counts["update"]:
                self.sessions.bump_mem_version(it.tenant_id)
        self.stats.processed += 1
        self.stats.ms.append((time.perf_counter() - t) * 1000)

    def run(self, queue: WriteIntentQueue, once: bool = False, batch: int = 10) -> WorkerStats:
        while True:
            msgs = queue.read(count=batch, block_ms=200 if once else 2000)
            if not msgs and once:
                return self.stats
            for msg_id, intent in msgs:
                try:
                    self.process(intent)
                    queue.ack(msg_id)
                except Exception as e:  # noqa: BLE001 - leave unacked: it will be re-delivered
                    self.stats.failed += 1
                    print(f"[worker] {intent.record_id} failed: {e}", file=sys.stderr, flush=True)


def build_worker(comps: Any) -> ConsolidationWorker:
    """Worker over the app's components; fact extraction when writepath.extract_facts is on."""
    s = comps.settings
    extractor = resolver = None
    if s.writepath.extract_facts and comps.semantic is not None:
        from amic.app import gate_llm_from_settings

        # the resident model (shared context size); FactExtractor sets its own max_tokens
        llm = (
            comps.extras.get("extract_llm")
            or comps.gate_llm  # injected gate model (tests, a shared gateway)
            or gate_llm_from_settings(s, timeout_s=120.0)
        )
        extractor = FactExtractor(llm)
        resolver = ConflictResolver(comps.semantic, comps.embedder)
    return ConsolidationWorker(
        comps.writer, comps.sessions, extractor, resolver, s.writepath.salience_min_words
    )


def main() -> None:
    from amic.app import build_components, build_queue
    from amic.contracts.config import load_settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=None)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    s = load_settings(a.config)
    comps = build_components(s, llm=object())  # type: ignore[arg-type] - the worker never chats
    queue = build_queue(comps, for_worker=True)
    w = build_worker(comps)
    print(
        f"[worker] queue={s.writepath.queue} extract_facts={s.writepath.extract_facts}", flush=True
    )
    st = w.run(queue, once=a.once)
    print(f"[worker] processed={st.processed} facts={st.facts} failed={st.failed}")


if __name__ == "__main__":
    main()
