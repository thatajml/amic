"""ConsolidationWorker (L7): turns queued WriteIntents into durable memory, off the hot path.

Per turn (brief section 7): M1 upsert (the raw turn; bumps the learner's memory version, which
invalidates the M4 sketch and the decision cache in every process) -> salience filter -> LLM fact
extraction -> conflict resolution into M2 -> EWMA update of the M3 profile -> version bump if M2
or M3 changed. Explicit commands (route WRITE):
- `remember`: the note goes to M1 (kind "remember") and, as a confidence-1 fact, to M2 and M3;
- `forget`: M1 records and M2 facts (current or closed) that match the text (substring, or
  cosine >= forget_sim) are hard-deleted, and matching M3 notes/misconceptions removed.
Intents from before a learner's erasure (the `erased_at` tombstone) are dropped, and an
erasure that overtakes an intent mid-consolidation purges what it wrote, so a queued turn can
never resurrect an erased learner. Acknowledged only after success, so a crash re-delivers the
intent (at-least-once; the M1 upsert is idempotent by record id).

    python -m amic.writepath.worker              # backend from settings.writepath.queue
    python -m amic.writepath.worker --once       # drain what is queued, then exit
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from amic.contracts.models import WriteIntent
from amic.writepath.extract import ConflictResolver, ExtractedFact, FactExtractor, is_salient
from amic.writepath.queue import WriteIntentQueue


@dataclass
class WorkerStats:
    processed: int = 0
    extracted: int = 0
    extract_failed: int = 0  # extraction raised (Ollama timeout/error): the turn is still stored
    facts: dict[str, int] = field(
        default_factory=lambda: {"insert": 0, "update": 0, "refresh": 0, "skip": 0}
    )
    profile_updates: int = 0
    remembered: int = 0
    forgotten: dict[str, int] = field(default_factory=lambda: {"M1": 0, "M2": 0, "M3": 0})
    failed: int = 0
    dropped_erased: int = 0  # intents from before the learner's erasure
    ms: list[float] = field(default_factory=list)


def _slug(text: str, n: int = 5) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", text.lower())[:n]) or "note"


class ConsolidationWorker:
    def __init__(
        self,
        writer: Any,
        sessions: Any,
        extractor: FactExtractor | None = None,
        resolver: ConflictResolver | None = None,
        min_words: int = 4,
        *,
        profiles: Any = None,
        profile_alpha: float = 0.3,
        profile_max_notes: int = 3,
        episodic: Any = None,
        semantic: Any = None,
        embedder: Any = None,
        forget_sim: float = 0.55,
    ) -> None:
        self.writer = writer
        self.sessions = sessions
        self.extractor = extractor
        self.resolver = resolver
        self.min_words = min_words
        self.profiles = profiles
        self.profile_alpha = profile_alpha
        self.profile_max_notes = profile_max_notes
        self.episodic = episodic
        self.semantic = semantic
        self.embedder = embedder
        self.forget_sim = forget_sim
        self.stats = WorkerStats()

    # ------------------------------------------------------------------ dispatch
    def _erased_at(self, tenant_id: str) -> float:
        fn = getattr(self.sessions, "erased_at", None)
        return float(fn(tenant_id)) if fn is not None else 0.0

    def process(self, it: WriteIntent) -> None:
        t = time.perf_counter()
        if it.ts.timestamp() <= self._erased_at(it.tenant_id):
            self.stats.dropped_erased += 1  # said before the learner's erasure: never stored
            return
        if it.op == "remember":
            self._remember(it)
        elif it.op == "forget":
            self._forget(it)
        else:
            self._turn(it)
        if it.ts.timestamp() <= self._erased_at(it.tenant_id):
            self._purge(it)  # erased while this intent was being consolidated
        self.stats.processed += 1
        self.stats.ms.append((time.perf_counter() - t) * 1000)

    def _turn(self, it: WriteIntent) -> None:
        self.writer.write_turn(
            it.tenant_id, it.session_id, it.record_id, it.query, it.answer, it.ts, it.metadata
        )
        facts: list[ExtractedFact] = []
        changed = False
        if (
            self.extractor is not None
            and self.resolver is not None
            and is_salient(it.query, it.answer, self.min_words)
        ):
            try:
                facts = self.extractor.extract([(it.query, it.answer)], it.ts)
            except Exception as e:  # noqa: BLE001 - a slow/broken extractor must not lose the turn
                self.stats.extract_failed += 1
                print(f"[worker] fact extraction failed for {it.record_id}: {e}", file=sys.stderr)
                facts = []
            else:
                self.stats.extracted += 1
            counts = self.resolver.resolve(it.tenant_id, facts, it.ts, source=it.record_id)
            for k, v in counts.items():
                self.stats.facts[k] += v
            changed = bool(counts["insert"] or counts["update"])
        changed |= self._update_profile(it, facts)
        if changed:
            self.sessions.bump_mem_version(it.tenant_id)

    def _update_profile(self, it: WriteIntent, facts: list[ExtractedFact]) -> bool:
        if self.profiles is None:
            return False
        prof = self.profiles.get_or_new(it.tenant_id)
        if prof.observe_turn(it.query, it.ts, facts, self.profile_alpha, self.profile_max_notes):
            self.profiles.put(prof)
            self.stats.profile_updates += 1
            return True
        return False

    def _purge(self, it: WriteIntent) -> None:
        """Undo what this intent wrote after an erasure that overtook it."""
        tid = it.tenant_id
        if self.episodic is not None:
            self.episodic.delete_ids(tid, [it.record_id])
        if self.semantic is not None:
            ids = [
                f for f, _, m, _ in self.semantic.records(tid) if m.get("source") == it.record_id
            ]
            self.semantic.delete_ids(tid, ids)
        if self.profiles is not None:
            prof = self.profiles.get(tid)
            if prof is not None and prof.updated <= it.ts.timestamp() + 1e-6:
                self.profiles.delete(tid)  # built only from pre-erasure input
        self.sessions.bump_mem_version(tid)
        self.stats.dropped_erased += 1

    # ------------------------------------------------------------------ commands
    def _remember(self, it: WriteIntent) -> None:
        note = it.query.strip()
        if not note:
            return
        self.writer.write_record(
            it.tenant_id,
            it.session_id,
            it.record_id,
            f"Student asked me to remember: {note}",
            it.ts,
            it.metadata,
            kind="remember",
        )
        fact = ExtractedFact(f"remember:{_slug(note)}", "personal", note, 1.0)
        if self.resolver is not None:
            self.resolver.resolve(it.tenant_id, [fact], it.ts, source=it.record_id)
        self._update_profile(it.model_copy(update={"query": ""}), [fact])
        self.sessions.bump_mem_version(it.tenant_id)
        self.stats.remembered += 1

    def _matches(self, needle: str, vec: np.ndarray | None, text: str, emb: Any) -> bool:
        if needle.lower() in text.lower():
            return True
        if vec is None or emb is None or len(emb) == 0:
            return False
        return float(np.asarray(emb, dtype=np.float32) @ vec) >= self.forget_sim

    def _forget(self, it: WriteIntent) -> None:
        needle = it.query.strip()
        if not needle:
            return
        vec = self.embedder.encode(needle) if self.embedder is not None else None
        tid = it.tenant_id
        if self.episodic is not None:
            ids = [
                rid
                for rid, text, _, emb in self.episodic.records(tid)
                if self._matches(needle, vec, text, emb)
            ]
            self.stats.forgotten["M1"] += self.episodic.delete_ids(tid, ids)
        if self.semantic is not None:
            ids = [
                fid
                for fid, text, _, emb in self.semantic.records(tid)
                if self._matches(needle, vec, text, emb)
            ]
            self.stats.forgotten["M2"] += self.semantic.delete_ids(tid, ids)
        if self.profiles is not None and (prof := self.profiles.get(tid)) is not None:
            n = 0
            for bucket in [prof.misconceptions, *prof.notes.values()]:
                for k in [k for k, v in bucket.items() if self._match_text(needle, vec, v)]:
                    del bucket[k]
                    n += 1
            if n:
                self.profiles.put(prof)
            self.stats.forgotten["M3"] += n
        self.sessions.bump_mem_version(tid)

    def _match_text(self, needle: str, vec: np.ndarray | None, text: str) -> bool:
        emb = self.embedder.encode(text) if (vec is not None and self.embedder) else None
        return self._matches(needle, vec, text, emb)

    # ------------------------------------------------------------------ loop
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
            if msgs and not once:  # visible in `docker compose logs worker`
                st = self.stats
                print(
                    f"[worker] processed={st.processed} extracted={st.extracted} "
                    f"extract_failed={st.extract_failed} facts={st.facts} failed={st.failed}",
                    flush=True,
                )


def build_worker(comps: Any) -> ConsolidationWorker:
    """Worker over the app's components; fact extraction when writepath.extract_facts is on."""
    s = comps.settings
    extractor = None
    resolver = (
        ConflictResolver(comps.semantic, comps.embedder) if comps.semantic is not None else None
    )
    if s.writepath.extract_facts and comps.semantic is not None:
        from amic.app import gate_llm_from_settings

        # the resident model (shared context size); FactExtractor sets its own max_tokens
        llm = (
            comps.extras.get("extract_llm")
            or comps.gate_llm  # injected gate model (tests, a shared gateway)
            or gate_llm_from_settings(s, timeout_s=120.0)
        )
        extractor = FactExtractor(llm)
    return ConsolidationWorker(
        comps.writer,
        comps.sessions,
        extractor,
        resolver,
        s.writepath.salience_min_words,
        profiles=comps.profiles,
        profile_alpha=s.profile.ewma_alpha,
        profile_max_notes=s.profile.max_notes,
        episodic=comps.episodic,
        semantic=comps.semantic,
        embedder=comps.embedder,
        forget_sim=s.forgetting.forget_sim,
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
    print(
        f"[worker] processed={st.processed} extracted={st.extracted} "
        f"extract_failed={st.extract_failed} facts={st.facts} failed={st.failed}"
    )


if __name__ == "__main__":
    main()
