"""Re-seed ONLY the per-learner session metadata (S2 signals) into the configured backend.

python scripts/seed_sessions.py            # -> data/sessions.sqlite (edu + LoCoMo)
"""

from __future__ import annotations

from amic.app import build_components
from amic.contracts.config import Settings
from amic.eval.datasets import edu, locomo


def main() -> None:
    s = Settings(session_backend="sqlite")
    c = build_components(s, llm=object(), embedder=object())  # type: ignore[arg-type]
    for lr in edu.load_learners():
        c.sessions.erase_tenant(lr.learner_id)
        edu.seed_learner(c, lr, write_memory=False)
    locomo.seed(c, locomo.load(), write_memory=False)
    print("session metadata seeded:", c.sessions.tenant_meta("edu-001"))


if __name__ == "__main__":
    main()
