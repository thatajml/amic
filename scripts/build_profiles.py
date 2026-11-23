"""Build M3 learner profiles for the seeded education learners (no LLM, ~seconds).

    python scripts/build_profiles.py            # -> settings.profile.db (data/profiles.sqlite)

Replays what the consolidation worker would have done, in time order, per session: each past
student turn (lexical evidence + concept exposure), then the facts extracted from that session
(scripts/extract_facts.py, collection edu_v1_facts; superseded facts included, so a fixed
misconception is first recorded and then closed by its progress fact). Prints a summary and an
example profile.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from amic.app import build_components
from amic.contracts.config import Settings
from amic.eval.datasets import edu
from amic.telemetry.store import InMemoryTelemetry
from amic.writepath.extract import ExtractedFact


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    s = Settings(chroma_collection="edu_v1")
    c = build_components(s, llm=object(), telemetry=InMemoryTelemetry())  # type: ignore[arg-type]
    assert c.profiles is not None and c.semantic is not None
    n_tok, n_facts = [], 0
    learners = edu.load_learners()
    for lr in learners:
        by_session: dict[str, list[tuple[float, ExtractedFact]]] = defaultdict(list)
        for _, text, meta in c.semantic.all(lr.learner_id):
            f = ExtractedFact(str(meta.get("key", "")), str(meta.get("kind", "other")), text, 0.9)
            by_session[str(meta.get("source", ""))].append((float(meta["valid_from"]), f))
        c.profiles.delete(lr.learner_id)
        prof = c.profiles.get_or_new(lr.learner_id)
        a = s.profile
        for sess in sorted(lr.sessions, key=lambda x: x.start):
            for j, (q, _) in enumerate(edu.exchanges(sess)):
                prof.observe_turn(q, sess.start + timedelta(minutes=3 * j), (), a.ewma_alpha)
            facts = [f for _, f in sorted(by_session.get(sess.session_id, []), key=lambda t: t[0])]
            n_facts += len(facts)
            end = sess.start + timedelta(minutes=30)
            prof.observe_turn("", end, facts, a.ewma_alpha, a.max_notes)
        if not prof.updated:
            prof.updated = datetime.now(UTC).timestamp()
        c.profiles.put(prof)
        n_tok.append(len(prof.render(s.routes.profile_tokens)) / 4)
    summary = {
        "profiles": len(learners),
        "facts_replayed": n_facts,
        "alpha": s.profile.ewma_alpha,
        "block_tokens_mean": sum(n_tok) / len(n_tok),
        "block_tokens_max": max(n_tok),
        "budget": s.routes.profile_tokens,
    }
    Path("runs/profiles_build.json").write_text(json.dumps(summary, indent=2))
    print(
        f"profiles={len(learners)} facts_replayed={n_facts} "
        f"block_tokens mean={sum(n_tok) / len(n_tok):.0f} max={max(n_tok):.0f} -> {s.profile.db}"
    )
    print("example:", c.profiles.get(learners[0].learner_id).render(s.routes.profile_tokens))  # type: ignore[union-attr]


if __name__ == "__main__":
    main()
