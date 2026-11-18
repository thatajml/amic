"""Train + calibrate the Tier-1 gate (plan T5.1) on the education set, split by learner.

    python scripts/train_tier1.py                         # main model -> models/tier1.joblib
    python scripts/train_tier1.py --no-s3                 # S3 ablation -> models/tier1_no_s3.joblib
    python scripts/train_tier1.py --kind lgbm             # comparison

Trains on rows Tier 0 abstains on (the traffic Tier 1 actually sees), calibrates on `val`,
reports discrimination + calibration on `test`. Writes runs/tier1_<name>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from amic.control.signals.bus import FEATURES, S3_FEATURES
from amic.control.tier1 import train_tier1
from amic.eval.replay import load_dataset


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[list[dict], float]:
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            conf, acc = float(p[m].mean()), float(y[m].mean())
            rows.append({"bin": i, "n": int(m.sum()), "mean_p": conf, "frac_pos": acc})
            ece += m.sum() / len(p) * abs(conf - acc)
    return rows, float(ece)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["logreg", "lgbm"], default="logreg")
    ap.add_argument("--calibration", choices=["isotonic", "platt"], default="isotonic")
    ap.add_argument(
        "--label", default="label_required", choices=["label_required", "label_personal"]
    )
    ap.add_argument("--no-s3", action="store_true")
    ap.add_argument("--all-rows", action="store_true", help="train on Tier-0-resolved rows too")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    feats = [f for f in FEATURES if not (a.no_s3 and f in S3_FEATURES and f != "s3_has_memory")]
    name = a.name or "tier1" + ("_no_s3" if a.no_s3 else "") + (
        "" if a.kind == "logreg" else f"_{a.kind}"
    ) + ("" if a.label == "label_required" else "_personal")
    df = load_dataset("edu")
    if not a.all_rows:
        df = df[df.tier0_rule == "abstain"]
    parts = {s: df[df.split == s] for s in ("train", "val", "test")}
    X = {s: d[feats].to_numpy(np.float32) for s, d in parts.items()}
    y = {s: d[a.label].astype(int).to_numpy() for s, d in parts.items()}
    depth = (parts["train"].depth == "DEEP").astype(int).to_numpy()

    model = train_tier1(
        X["train"],
        y["train"],
        X["val"],
        y["val"],
        feats,
        kind=a.kind,
        calibration=a.calibration,
        depth_tr=depth,
    )
    model.meta = {
        "label": a.label,
        "rows": "abstain" if not a.all_rows else "all",
        "n_train": len(X["train"]),
    }
    out = Path("models") / f"{name}.joblib"
    model.save(out)

    report: dict = {
        "name": name,
        "features": feats,
        "kind": a.kind,
        "calibration": a.calibration,
        "label": a.label,
    }
    for s in ("val", "test"):
        p, _ = model.predict_many(X[s])
        rel, ece = reliability(y[s], p)
        report[s] = {
            "n": int(len(p)),
            "pos_rate": float(y[s].mean()),
            "auc": float(roc_auc_score(y[s], p)) if 0 < y[s].sum() < len(y[s]) else float("nan"),
            "brier": float(brier_score_loss(y[s], p)),
            "ece": ece,
            "reliability": rel,
        }
    if not a.no_s3 and a.kind == "logreg":
        lr = model.gate.estimator.estimator.named_steps["logisticregression"]  # frozen base
        report["coef"] = dict(
            sorted(
                zip(feats, lr.coef_[0].round(3).tolist(), strict=True), key=lambda kv: -abs(kv[1])
            )
        )
    Path("runs").mkdir(exist_ok=True)
    Path(f"runs/{name}.json").write_text(json.dumps(report, indent=2))
    t = report["test"]
    print(
        f"{name}: train={len(X['train'])} val={len(X['val'])} test={t['n']} (pos {t['pos_rate']:.0%})"
    )
    print(f"  test AUC={t['auc']:.3f}  Brier={t['brier']:.3f}  ECE={t['ece']:.3f}  -> {out}")
    if "coef" in report:
        print("  top features:", list(report["coef"].items())[:8])


if __name__ == "__main__":
    main()
