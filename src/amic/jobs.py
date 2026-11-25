"""Scheduled jobs (brief section 12): Tier-1 distillation retrain, threshold recalibration,
memory decay/compaction. Never on the response path.

    python -m amic.jobs retrain      [--dry-run]   # Tier 1 + distilled rows, promotion gate
    python -m amic.jobs recalibrate  [--dry-run]   # band sized to the Tier-2 target on validation
    python -m amic.jobs forget       [--dry-run]   # ForgettingManager over every learner
    python -m amic.jobs nightly      [--every 24]  # all three, in that order (optionally looping)

Retrain: a candidate Tier 1 is trained on the labelled set (train split) plus every Tier-2
verdict in `jobs.distill_glob` (weight `jobs.distill_weight`; rows of validation/test learners
are dropped), and promoted to `jobs.live_artifact` only if, on validation, its expected cost at
lambda is not higher and its AUC not lower by more than `jobs.max_auc_drop` than the current
live model. The experiment artifact (models/tier1.joblib) is never touched: a deployment points
`gate.tier1_artifact` at the live artifact. Every run appends one line to `jobs.log_path`.

Recalibrate: the band [tau_low, tau_high] around tau* is re-sized on validation for the live
model so Tier 2 sees at most `gate.tier2_max_rate` of traffic; written to `jobs.thresholds_file`
(set `gate.thresholds_file` to the same path to use it).
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from amic.contracts.config import Settings, load_settings


def _log(s: Settings, job: str, out: dict[str, Any]) -> None:
    p = Path(s.jobs.log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "job": job, **out}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=float) + "\n")


def _labelled(s: Settings) -> Any:
    from amic.eval.replay import load_dataset

    df = load_dataset(s.jobs.dataset)
    return df[df.tier0_rule == "abstain"]  # the rows Tier 1 actually sees


def _val_metrics(model: Any, val: Any, lam: float) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    from amic.control.thresholds import expected_cost, tau_star

    p = model.predict_many(val[model.features].to_numpy(np.float32))[0]
    y = val.label_required.astype(int).to_numpy()
    return {
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "cost": expected_cost(y, p > tau_star(lam), lam),  # hard-tau cost on validation
    }


# ---------------------------------------------------------------------- retrain
def retrain(s: Settings, dry_run: bool = False) -> dict[str, Any]:
    from amic.control.distill import load_rows
    from amic.control.tier1 import Tier1Model, train_tier1

    df = _labelled(s)
    cur_path = Path(s.jobs.live_artifact)
    current = Tier1Model.load(cur_path if cur_path.exists() else s.gate.tier1_artifact)
    feats = current.features
    tr, val = df[df.split == "train"], df[df.split == "val"]
    held_out = set(df[df.split != "train"].learner_id)
    rows = [r for path in sorted(glob.glob(s.jobs.distill_glob)) for r in load_rows(path)]
    rows = [
        r
        for r in rows
        if r.get("tenant_id") not in held_out and r.get("route") in ("SKIP", "SHALLOW", "DEEP")
    ]
    X = tr[feats].to_numpy(np.float32)
    y = tr.label_required.astype(int).to_numpy()
    w = np.ones(len(y))
    depth = (tr.depth == "DEEP").astype(float).to_numpy()
    if rows:
        Xd = np.array([[float(r["features"].get(f, 0.0)) for f in feats] for r in rows])
        X = np.vstack([X, Xd])
        y = np.concatenate([y, [int(r["route"] != "SKIP") for r in rows]])
        w = np.concatenate([w, np.full(len(rows), s.jobs.distill_weight)])
        depth = np.concatenate([depth, np.full(len(rows), np.nan)])
    cand = train_tier1(
        X,
        y,
        val[feats].to_numpy(np.float32),
        val.label_required.astype(int).to_numpy(),
        feats,
        kind=current.kind,  # type: ignore[arg-type]
        calibration=current.calibration,  # type: ignore[arg-type]
        depth_tr=depth,
        sample_weight=w,
    )
    cand.meta = {"job": "retrain", "n_distill": len(rows), "distill_weight": s.jobs.distill_weight}
    m_cur, m_cand = _val_metrics(current, val, s.gate.lam), _val_metrics(cand, val, s.gate.lam)
    promote = (
        m_cand["cost"] <= m_cur["cost"] + 1e-9
        and m_cand["auc"] >= m_cur["auc"] - s.jobs.max_auc_drop
    )
    out = {
        "n_distill": len(rows),
        "current": m_cur,
        "candidate": m_cand,
        "promoted": bool(promote and not dry_run),
        "would_promote": bool(promote),
        "target": str(cur_path),
    }
    if promote and not dry_run:
        if cur_path.exists():
            shutil.copy2(cur_path, cur_path.with_suffix(".prev.joblib"))
        cand.save(cur_path)
    _log(s, "retrain", out)
    return out


# ---------------------------------------------------------------------- recalibrate
def recalibrate(s: Settings, dry_run: bool = False) -> dict[str, Any]:
    from amic.control.thresholds import tau_star, widest_band_under
    from amic.control.tier1 import Tier1Model
    from amic.eval.replay import load_dataset

    path = Path(s.jobs.live_artifact)
    model = Tier1Model.load(path if path.exists() else s.gate.tier1_artifact)
    val_all = load_dataset(s.jobs.dataset).query("split == 'val'")
    val = val_all[val_all.tier0_rule == "abstain"]
    p = model.predict_many(val[model.features].to_numpy(np.float32))[0]
    tau = tau_star(s.gate.lam)
    lo, hi = widest_band_under(p, tau, s.gate.tier2_max_rate * len(val_all))
    share = float(((np.round(p, 4) >= lo) & (np.round(p, 4) <= hi)).sum() / len(val_all))
    rec = {
        "lam": s.gate.lam,
        "tau": tau,
        "band_low_offset": round(tau - lo, 4),
        "band_high_offset": round(hi - tau, 4),
        "val_share": share,
        "model": str(path if path.exists() else s.gate.tier1_artifact),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if not dry_run:
        Path(s.jobs.thresholds_file).parent.mkdir(parents=True, exist_ok=True)
        Path(s.jobs.thresholds_file).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    _log(s, "recalibrate", {**rec, "written": not dry_run})
    return rec


# ---------------------------------------------------------------------- forget
def forget(s: Settings, dry_run: bool = False, comps: Any = None) -> dict[str, Any]:
    from amic.writepath.forgetting import ForgettingManager

    if comps is None:
        from amic.app import build_components

        comps = build_components(s, llm=object())  # type: ignore[arg-type]
    fm = ForgettingManager(
        s.forgetting, comps.episodic, comps.semantic, comps.profiles, comps.sessions
    )
    c = fm.run(dry_run=dry_run)
    out = {k: v for k, v in vars(c).items() if k != "per_tenant"} | {"dry_run": dry_run}
    _log(s, "forget", out)
    return out


JOBS = {"retrain": retrain, "recalibrate": recalibrate, "forget": forget}


def nightly(s: Settings, dry_run: bool = False) -> dict[str, Any]:
    return {name: fn(s, dry_run) for name, fn in JOBS.items()}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=[*JOBS, "nightly"])
    ap.add_argument("--config", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--every", type=float, default=None, help="repeat every N hours")
    a = ap.parse_args()
    s = load_settings(a.config)
    fn = nightly if a.job == "nightly" else JOBS[a.job]
    while True:
        print(json.dumps(fn(s, a.dry_run), indent=2, default=float), flush=True)
        if a.every is None:
            return
        time.sleep(a.every * 3600)


if __name__ == "__main__":
    main()
