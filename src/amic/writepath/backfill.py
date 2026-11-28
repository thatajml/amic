"""Back-fill M2 facts (and the M3 profile) from the turns already stored in M1.

    python -m amic.writepath.backfill                       # every learner
    python -m amic.writepath.backfill --learner u-ala-37fb56
    docker compose run --rm worker python -m amic.writepath.backfill

For learners who chatted while fact extraction was off. It reads each learner's stored turns
(`Student asked: ...\\nTutor answered: ...`, see writer.record_text), extracts facts with the same
FactExtractor/ConflictResolver the worker uses (one LLM call per stored session), in time order so
later facts can supersede earlier ones, and feeds them to the M3 profile. Idempotent: a turn
that already produced a fact (the fact's `source` is the turn's record id) is not processed
again, so it can be re-run safely.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from amic.writepath.extract import ConflictResolver, FactExtractor, is_salient

_TURN = re.compile(r"^Student asked: (.*?)\nTutor answered: (.*)$", re.S)


def parse_turn(text: str) -> tuple[str, str] | None:
    """(query, answer) from a stored turn record, else None (a /remember note, etc.)."""
    m = _TURN.match(text)
    return (m.group(1), m.group(2)) if m else None


def backfill_learner(
    tenant_id: str,
    episodic: Any,
    semantic: Any,
    extractor: FactExtractor,
    resolver: ConflictResolver,
    profiles: Any = None,
    sessions: Any = None,
    min_words: int = 4,
    alpha: float = 0.3,
    max_notes: int = 3,
) -> dict[str, int]:
    """Extract facts for one learner's not-yet-processed turns; returns the resolver counts."""
    records = episodic.records(tenant_id)
    ts_of = {rid: float(meta.get("ts", 0.0)) for rid, _, meta, _ in records}
    sess_of = {rid: str(meta.get("session_id", "")) for rid, _, meta, _ in records}
    # a fact's `source` is the last turn of the batch it came from: everything in that session up
    # to that turn is already processed
    done_upto: dict[str, float] = defaultdict(float)
    for _, _, meta in semantic.all(tenant_id):
        src = str(meta.get("source", ""))
        if src in ts_of:
            done_upto[sess_of[src]] = max(done_upto[sess_of[src]], ts_of[src])
    by_session: dict[str, list[tuple[float, str, str, str]]] = defaultdict(list)
    for rid, text, meta, _ in records:
        turn = parse_turn(text) if meta.get("kind", "turn") == "turn" else None
        sid = sess_of[rid]
        if turn is None or ts_of[rid] <= done_upto[sid] or not is_salient(*turn, min_words):
            continue
        by_session[sid].append((ts_of[rid], rid, turn[0], turn[1]))
    totals = {"insert": 0, "update": 0, "refresh": 0, "skip": 0, "sessions": 0}
    for _, turns in sorted(by_session.items(), key=lambda kv: min(t[0] for t in kv[1])):
        turns.sort()
        when = datetime.fromtimestamp(turns[-1][0], UTC)
        facts = extractor.extract([(q, a) for _, _, q, a in turns], when)
        counts = resolver.resolve(tenant_id, facts, when, source=turns[-1][1])
        for k, v in counts.items():
            totals[k] += v
        totals["sessions"] += 1
        # a session that yielded no fact leaves no marker and is simply tried again next run
        if profiles is not None:
            prof = profiles.get_or_new(tenant_id)
            changed = False
            for ts, _, q, _ in turns:
                changed |= prof.observe_turn(
                    q, datetime.fromtimestamp(ts, UTC), facts, alpha, max_notes
                )
                facts = []  # the batch's facts are credited once, to its first turn
            if changed:
                profiles.put(prof)
        if sessions is not None and (counts["insert"] or counts["update"]):
            sessions.bump_mem_version(tenant_id)
    return totals


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--learner", help="one learner id (default: every learner)")
    ap.add_argument("--config", nargs="*", default=None)
    a = ap.parse_args()

    from amic.app import build_components, gate_llm_from_settings
    from amic.contracts.config import load_settings

    s = load_settings(a.config)
    comps = build_components(s, llm=object())  # type: ignore[arg-type] - never chats
    if comps.semantic is None:
        raise SystemExit("this configuration has no semantic store (M2)")
    extractor = FactExtractor(gate_llm_from_settings(s, timeout_s=180.0))
    resolver = ConflictResolver(comps.semantic, comps.embedder)
    ids = [a.learner] if a.learner else comps.episodic.tenants()
    print(f"[backfill] {len(ids)} learner(s), collection={s.chroma_collection}", flush=True)
    for tid in ids:
        r = backfill_learner(
            tid,
            comps.episodic,
            comps.semantic,
            extractor,
            resolver,
            comps.profiles,
            comps.sessions,
            s.writepath.salience_min_words,
            s.profile.ewma_alpha,
            s.profile.max_notes,
        )
        print(f"  {tid}: {r}", flush=True)
    print(f"[backfill] extraction calls: {extractor.calls}")


if __name__ == "__main__":
    main()
