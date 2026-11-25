"""CI regression check on gate metrics (brief section 11).

    python scripts/gate_regression.py            # check against tests/fixtures/gate_baseline.json
    python scripts/gate_regression.py --update   # rewrite the baseline (after an intended change)
    python scripts/gate_regression.py --make-fixture   # refresh the fixture from data/edu

The fixture (tests/fixtures/gate_eval.parquet) holds the synthetic education set's gate features
and labels, no text. The check trains Tier 1 exactly as scripts/train_tier1.py does (train split,
isotonic calibration on val), replays Tier 0 + Tier 1 at lambda = 4 on the test split and fails
(exit 1) if test AUC drops, the expected cost rises, or the Tier-0 rule shares move by more than
the tolerances in the baseline file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from amic.contracts.config import Settings
from amic.control.signals.bus import FEATURES
from amic.control.tier1 import train_tier1
from amic.eval.replay import decision_metrics, replay

FIX = Path("tests/fixtures/gate_eval.parquet")
BASE = Path("tests/fixtures/gate_baseline.json")
COLS = ["probe_id", "learner_id", "split", "category", "label_required", "depth", "tier0_rule"]
TOL = {"auc": 0.02, "cost": 0.02, "retrieval_rate": 0.05, "share_tier0": 0.02}


def measure(df: pd.DataFrame) -> dict[str, float]:
    seen = df[df.tier0_rule == "abstain"]
    parts = {s: seen[seen.split == s] for s in ("train", "val", "test")}
    X = {s: d[FEATURES].to_numpy(np.float32) for s, d in parts.items()}
    y = {s: d.label_required.astype(int).to_numpy() for s, d in parts.items()}
    model = train_tier1(
        X["train"],
        y["train"],
        X["val"],
        y["val"],
        FEATURES,
        depth_tr=(parts["train"].depth == "DEEP").astype(float).to_numpy(),
    )
    auc = float(roc_auc_score(y["test"], model.predict_many(X["test"])[0]))
    dec = replay(df[df.split == "test"], Settings(), model)
    m = decision_metrics(dec, 4.0)
    return {"auc": auc, **{k: m[k] for k in ("cost", "retrieval_rate", "share_tier0")}}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--make-fixture", action="store_true")
    a = ap.parse_args()
    if a.make_fixture:
        from amic.eval.replay import load_dataset

        FIX.parent.mkdir(parents=True, exist_ok=True)
        load_dataset("edu")[COLS + FEATURES].to_parquet(FIX, index=False)
        print(f"wrote {FIX}")
    got = measure(pd.read_parquet(FIX))
    if a.update or not BASE.exists():
        BASE.write_text(json.dumps({"metrics": got, "tolerance": TOL}, indent=2))
        print("baseline written:", json.dumps(got))
        return
    base = json.loads(BASE.read_text())
    ref, tol = base["metrics"], base["tolerance"]
    bad = []
    if got["auc"] < ref["auc"] - tol["auc"]:
        bad.append(f"AUC {got['auc']:.3f} < {ref['auc']:.3f} - {tol['auc']}")
    if got["cost"] > ref["cost"] + tol["cost"]:
        bad.append(f"cost {got['cost']:.3f} > {ref['cost']:.3f} + {tol['cost']}")
    for k in ("retrieval_rate", "share_tier0"):
        if abs(got[k] - ref[k]) > tol[k]:
            bad.append(f"{k} {got[k]:.3f} vs {ref[k]:.3f} (+/- {tol[k]})")
    print("gate metrics:", json.dumps({k: round(v, 4) for k, v in got.items()}))
    if bad:
        print("REGRESSION:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print("no regression vs", BASE)


if __name__ == "__main__":
    main()
