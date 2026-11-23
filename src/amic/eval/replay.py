"""Offline replay: run any controller configuration over stored FeatureVectors (no LLM, no
embedding). Every decision metric, ablation and lambda sweep is computed from this.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from amic.contracts.config import Settings
from amic.contracts.models import RequestContext, Route
from amic.control.controller import Controller
from amic.control.route_planner import RoutePlanner
from amic.control.signals.bus import FEATURES, S3_FEATURES, Signals
from amic.control.thresholds import expected_cost
from amic.control.tier1 import Tier1Model
from amic.control.tier2 import _Tier2Base

RETRIEVE = {Route.SHALLOW.value, Route.DEEP.value}


def load_dataset(name: str) -> pd.DataFrame:
    """Features joined with labels. name: 'edu' or 'locomo'."""
    base = f"data/{name}"
    f = pd.read_parquet(f"{base}/features.parquet")
    lab = pd.read_parquet(f"{base}/labels.parquet")
    return lab.merge(f, on="probe_id", how="inner")


def s3_ablated(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in S3_FEATURES:
        if c != "s3_has_memory":
            out[c] = 0.0
    return out


def replay(
    df: pd.DataFrame,
    settings: Settings,
    tier1: Tier1Model | None,
    tier2: _Tier2Base | None = None,
) -> pd.DataFrame:
    """Decisions for every row, in row order. `tier2` (ReplayTier2) serves recorded Tier-2
    verdicts keyed by probe_id when settings.ablation.tiers == "full"."""
    if not settings.ablation.use_sketch_probe:
        df = s3_ablated(df)
    ctl = Controller(settings, RoutePlanner(settings.routes), tier1, tier2=tier2)
    rows: list[dict[str, Any]] = []
    for rec in df[["probe_id", *FEATURES]].to_dict("records"):
        feats = {k: float(rec[k]) for k in FEATURES}
        sig = Signals(np.zeros(1), feats, None, False, 0.0, 0.0)
        ctx = RequestContext(
            request_id=rec["probe_id"], tenant_id="", session_id="replay", query="", arm="amic"
        )
        d, tm = ctl.decide_signals(sig, ctx)
        rows.append(
            {
                "probe_id": rec["probe_id"],
                "route": d.route.value,
                "resolved_by": d.resolved_by.value,
                "p": d.p,
                "tier2_status": tm.tier2_status,
                "tier2_ms": tm.tier2,
            }
        )
    return df[["probe_id", "label_required", "depth", "category", "split"]].merge(
        pd.DataFrame(rows), on="probe_id"
    )


def fixed_policy(df: pd.DataFrame, name: str, always_route: str = "DEEP") -> pd.DataFrame:
    """Decisions of the baseline arms, for the same table."""
    out = df[["probe_id", "label_required", "depth", "category", "split"]].copy()
    if name == "always":
        out["route"] = always_route
    elif name == "never":
        out["route"] = "SKIP"
    elif name == "oracle":
        out["route"] = np.where(out.label_required, out.depth.fillna("DEEP"), "SKIP")
    else:
        raise ValueError(name)
    out["resolved_by"] = "fixed"
    out["p"] = np.nan
    return out


def decision_metrics(dec: pd.DataFrame, lam: float) -> dict[str, float]:
    y = dec.label_required.astype(bool).to_numpy()
    r = dec.route.isin(RETRIEVE).to_numpy()
    tp = int((y & r).sum())
    fp = int((~y & r).sum())
    fn = int((y & ~r).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    req = dec[dec.label_required.astype(bool) & dec.route.isin(RETRIEVE)]
    depth_ok = (req.route == req.depth.fillna("DEEP")).mean() if len(req) else float("nan")
    shares = dec.resolved_by.value_counts(normalize=True).to_dict()
    route_share = dec.route.value_counts(normalize=True).to_dict()
    return {
        "n": float(len(dec)),
        "retrieval_rate": float(r.mean()),
        "deep_rate": float(route_share.get("DEEP", 0.0)),
        "shallow_rate": float(route_share.get("SHALLOW", 0.0)),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "false_skip_rate": fn / max(1, int(y.sum())),  # of memory-required queries
        "false_retrieve_rate": fp / max(1, int((~y).sum())),
        "cost": expected_cost(y, r, lam),
        "depth_acc": float(depth_ok),
        "share_tier0": float(shares.get("tier0", 0.0)),
        "share_tier1": float(shares.get("tier1", 0.0)),
        "share_failopen": float(shares.get("failopen", 0.0)),
    }


def savings_captured(
    m_amic: dict[str, float], m_always: dict[str, float], m_oracle: dict[str, float]
) -> float:
    """Share of the achievable retrieval reduction (always -> oracle) that AMIC captures."""
    achievable = m_always["retrieval_rate"] - m_oracle["retrieval_rate"]
    if achievable <= 0:
        return float("nan")
    return (m_always["retrieval_rate"] - m_amic["retrieval_rate"]) / achievable
