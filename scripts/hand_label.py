"""Blind human labelling of a stratified sample, for judge agreement (Cohen's kappa).

Run in YOUR terminal (it is interactive):   python scripts/hand_label.py
Resumable; answers go to data/edu/human_labels.csv. The judge's labels are never shown.
Then:                                        python scripts/hand_label.py --score
"""

from __future__ import annotations

import argparse
import csv
import random
import sys

from amic.eval.datasets import edu

OUT = edu.DATA_DIR / "human_labels.csv"

QUESTION = (
    "Would a tutor need this learner's PAST SESSIONS (not today's conversation) to answer this\n"
    "message well? Use the rule: would an answer without that memory be clearly worse?\n"
    "  [y] yes, memory needed   [n] no   [s] skip   [q] save & quit"
)


def sample(n: int, seed: int = 99) -> list[edu.Probe]:
    by_cat: dict[str, list[edu.Probe]] = {}
    for p in edu.load_probes():
        by_cat.setdefault(p.category, []).append(p)
    rng = random.Random(seed)
    per = max(1, n // len(by_cat))
    out = [p for ps in by_cat.values() for p in rng.sample(ps, min(per, len(ps)))]
    rng.shuffle(out)  # mixed order: don't label category by category
    return out[:n]


def show(lr: edu.Learner, p: edu.Probe, i: int, n: int) -> None:
    print("\n" + "=" * 78)
    print(f"[{i}/{n}] learner {lr.learner_id}: goal={lr.goal}; project={lr.project}")
    for s_i, s in enumerate(lr.sessions, 1):
        print(f"  past session {s_i} ({s.start:%b %d}): {', '.join(s.concepts)}")
        for m in s.misconceptions:
            print(f"      mistake: {m}")
    print("  remembered facts:", *(f"    - {f}" for f in lr.facts), sep="\n")
    if p.m0_context:
        print("  TODAY so far:")
        for t in p.m0_context:
            print(f"    {t.role}: {t.text[:240]}")
    print(f"\n  >>> STUDENT NOW: {p.query}\n")


def label(n: int) -> None:
    learners = {lr.learner_id: lr for lr in edu.load_learners()}
    done = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            done = {r["probe_id"] for r in csv.DictReader(f)}
    items = [p for p in sample(n) if p.probe_id not in done]
    new = not OUT.exists()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["probe_id", "human_required"])
        for i, p in enumerate(items, len(done) + 1):
            show(learners[p.learner_id], p, i, n)
            while True:
                a = input(QUESTION + "\n  > ").strip().lower()
                if a in {"y", "n", "s", "q"}:
                    break
            if a == "q":
                break
            if a == "s":
                continue
            w.writerow([p.probe_id, int(a == "y")])
            f.flush()
    print(f"saved to {OUT}")


def score() -> None:
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score

    h = pd.read_csv(OUT)
    lab = pd.read_parquet(edu.DATA_DIR / "labels.parquet")[
        ["probe_id", "label_required", "category"]
    ]
    m = h.merge(lab, on="probe_id")
    k = cohen_kappa_score(m.human_required, m.label_required.astype(int))
    agree = (m.human_required == m.label_required.astype(int)).mean()
    print(f"n={len(m)}  agreement={agree:.1%}  Cohen's kappa={k:.3f}")
    print(
        m.groupby("category")
        .apply(lambda g: (g.human_required == g.label_required.astype(int)).mean())
        .to_string()
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    score() if a.score else label(a.n)


if __name__ == "__main__":
    main()
