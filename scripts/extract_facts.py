"""Build M2 (semantic facts) for the seeded education learners (pre-registered F1, V1).

    python scripts/extract_facts.py            # -> Chroma collection edu_v1_facts

Runs the consolidation steps the async worker runs (salience -> LLM fact extraction ->
conflict resolution with validity intervals), one call per past SESSION (a batch of its salient
exchanges) instead of one per turn, in time order, so later sessions can update earlier facts.
Resumable: sessions already processed are listed in data/edu/facts_done.jsonl.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import timedelta

from amic.app import build_components, gate_llm_from_settings
from amic.contracts.config import Settings
from amic.eval.datasets import edu
from amic.memory.stores.semantic import SemanticStore
from amic.writepath.extract import ConflictResolver, FactExtractor, is_salient

DONE = edu.DATA_DIR / "facts_done.jsonl"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    # num_ctx 8192: same context as the LongMemEval runs, so Ollama shares one loaded model
    s = Settings(chroma_collection="edu_v1", models={"num_ctx": 8192})
    c = build_components(s, llm=object())  # type: ignore[arg-type] - embedder + Chroma only
    store = SemanticStore(c.episodic._client, "edu_v1")
    llm = gate_llm_from_settings(s, timeout_s=180.0)
    llm.chat([{"role": "user", "content": "hi"}], max_tokens=1)
    extractor = FactExtractor(llm)
    resolver = ConflictResolver(store, c.embedder)
    done = set()
    if DONE.exists():
        done = {
            json.loads(x)["session_id"] for x in DONE.read_text(encoding="utf-8").splitlines() if x
        }
    totals = {"insert": 0, "update": 0, "refresh": 0, "skip": 0}
    t0 = time.perf_counter()
    with DONE.open("a", encoding="utf-8") as fdone:
        learners = edu.load_learners()
        for li, lr in enumerate(learners, 1):
            for sess in sorted(lr.sessions, key=lambda x: x.start):
                if sess.session_id in done:
                    continue
                ex = [(q, a) for q, a in edu.exchanges(sess) if is_salient(q, a)]
                facts = extractor.extract(ex, sess.start) if ex else []
                when = sess.start + timedelta(minutes=30)  # facts become valid at session end
                counts = resolver.resolve(lr.learner_id, facts, when, source=sess.session_id)
                for k, v in counts.items():
                    totals[k] += v
                fdone.write(
                    json.dumps({"session_id": sess.session_id, "facts": len(facts), **counts})
                    + "\n"
                )
                fdone.flush()
            c.sessions.bump_mem_version(lr.learner_id)
            print(
                f"  {li}/{len(learners)} learners  {totals}  {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    n = sum(store.count(lr.learner_id) for lr in edu.load_learners())
    print(f"current facts: {n} (all learners); extraction calls: {extractor.calls}")


if __name__ == "__main__":
    main()
