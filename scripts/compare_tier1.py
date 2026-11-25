"""Tier-1 variants vs the base Tier 1, offline on the education test split (pre-registered S1,
DS1; docs/PLAN.md section 10).

    python scripts/compare_tier1.py tier1_distill05 tier1_distill10 tier1_distill_only [tier1_s5]
        # -> runs/tier1_variants.json

For each variant: test AUC on the rows Tier 1 sees (Tier 0 abstains) and the paired AUC
difference vs the base; the full gate replayed (Tier 0 + Tier 1, base config, lambda = 4) on every
test row: expected cost, retrieval and false-skip rate, and the paired per-row cost difference.
CIs: cluster bootstrap by learner and by probe (section 9 decision rule: both must exclude 0).
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
from amic.control.signals.uncertainty import S5_FEATURES
from amic.control.tier1 import Tier1Model
from amic.eval.replay import RETRIEVE, decision_metrics, load_dataset, replay
from amic.eval.stats import mean_ci, stat_ci


def _auc(d: pd.DataFrame, col: str) -> float:
    return float(roc_auc_score(d.y, d[col])) if 0 < d.y.sum() < len(d) else float("nan")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--base", default="tier1")
    ap.add_argument("--lam", type=float, default=4.0)
    a = ap.parse_args()
    df = load_dataset("edu")
    s5 = Path("data/edu/s5.parquet")
    if s5.exists():
        df = df.merge(pd.read_parquet(s5), on="probe_id", how="left")
        df[S5_FEATURES] = df[S5_FEATURES].fillna(0.0)
    test = df[df.split == "test"].reset_index(drop=True)
    seen = test[test.tier0_rule == "abstain"]
    s = Settings()
    s.gate.lam = a.lam
    models = {n: Tier1Model.load(f"models/{n}.joblib") for n in [a.base, *a.variants]}

    def probs(m: Tier1Model, d: pd.DataFrame) -> np.ndarray:
        return m.predict_many(d[m.features].to_numpy(np.float32))[0]

    def row_cost(dec: pd.DataFrame) -> np.ndarray:
        y = dec.label_required.astype(bool).to_numpy()
        r = dec.route.isin(RETRIEVE).to_numpy()
        return a.lam * (y & ~r) + (~y & r)

    base_dec = replay(test, s, models[a.base])
    base_cost = row_cost(base_dec)
    out: dict = {"base": a.base, "lam": a.lam, "n_test": len(test), "n_tier1_rows": len(seen)}
    out[a.base] = {
        "auc": _auc(
            seen.assign(y=seen.label_required.astype(int), p=probs(models[a.base], seen)), "p"
        ),
        **decision_metrics(base_dec, a.lam),
    }
    for v in a.variants:
        f = seen.assign(
            y=seen.label_required.astype(int),
            pa=probs(models[v], seen),
            pb=probs(models[a.base], seen),
        )
        row: dict = {"meta": models[v].meta, "auc": _auc(f, "pa")}
        for unit, col in (("learner", "learner_id"), ("probe", "probe_id")):
            row[f"auc_diff_ci_{unit}"] = stat_ci(f, col, lambda d: _auc(d, "pa") - _auc(d, "pb"))
        dec = replay(test, s, models[v])
        row.update(decision_metrics(dec, a.lam))
        diff = row_cost(dec) - base_cost
        for unit, col in (("learner", "learner_id"), ("probe", "probe_id")):
            row[f"cost_diff_ci_{unit}"] = mean_ci(diff, test[col])
        out[v] = row
        lo_l, hi_l = row["auc_diff_ci_learner"][1:]
        lo_p, hi_p = row["auc_diff_ci_probe"][1:]
        print(
            f"{v}: AUC {row['auc']:.3f} (diff {row['auc_diff_ci_probe'][0]:+.3f}; "
            f"learner [{lo_l:+.3f}, {hi_l:+.3f}], probe [{lo_p:+.3f}, {hi_p:+.3f}]) | "
            f"cost {row['cost']:.3f} vs {out[a.base]['cost']:.3f} "
            f"(diff {row['cost_diff_ci_probe'][0]:+.3f}; learner "
            f"[{row['cost_diff_ci_learner'][1]:+.3f}, {row['cost_diff_ci_learner'][2]:+.3f}], "
            f"probe [{row['cost_diff_ci_probe'][1]:+.3f}, {row['cost_diff_ci_probe'][2]:+.3f}]) "
            f"retrieval {row['retrieval_rate']:.1%}"
        )
    p = Path("runs/tier1_variants.json")
    prev = json.loads(p.read_text()) if p.exists() else {}
    prev.update(out)
    p.write_text(json.dumps(prev, indent=2, default=float))


if __name__ == "__main__":
    main()
