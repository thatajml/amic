"""Bootstrap confidence intervals for the report. Deterministic (fixed seed) so tables regenerate.

Resampling is by CLUSTER, never by row: one probe appears once per tutor seed, and one learner
contributes many probes, so rows are not independent. The education test split holds only 8
learners, so learner-clustered intervals are approximate (a bootstrap over few clusters tends
to be too narrow); they are reported next to probe-clustered ones, not instead of them.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

B_MEAN = 10_000  # resamples for means (vectorised)
B_STAT = 2_000  # resamples for statistics that need a full recomputation (AUC, kappa)
SEED = 0
SCORE = {"win": 1.0, "tie": 0.0, "loss": -1.0}


def learner_of(probe_id: str) -> str:
    """'edu-009-p08' -> 'edu-009'. Ids without a probe suffix are their own learner."""
    head, _, tail = probe_id.rpartition("-")
    return head if head and tail[:1] == "p" and tail[1:].isdigit() else probe_id


def mean_ci(
    values: np.ndarray | pd.Series,
    clusters: np.ndarray | pd.Series,
    *,
    b: int = B_MEAN,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(mean, lo, hi): percentile CI of the row mean, resampling whole clusters."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    sums = np.bincount(inv, weights=v)
    counts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), (b, len(sums)))
    est = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(est, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def stat_ci(
    df: pd.DataFrame,
    cluster_col: str,
    fn: Callable[[pd.DataFrame], float],
    *,
    b: int = B_STAT,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(stat, lo, hi) for any statistic of a frame, resampling whole clusters.

    Resamples on which `fn` is undefined (e.g. AUC with one class) are dropped.
    """
    point = float(fn(df))
    groups = [g for _, g in df.groupby(cluster_col, sort=True)]
    rng = np.random.default_rng(seed)
    est = []
    for _ in range(b):
        pick = rng.integers(0, len(groups), len(groups))
        try:
            v = float(fn(pd.concat([groups[i] for i in pick], ignore_index=True)))
        except ValueError:
            continue
        if np.isfinite(v):
            est.append(v)
    if not est:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(est, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def pairwise_scores(pw: pd.DataFrame) -> pd.DataFrame:
    """Pairwise rows -> score (+1 win, 0 tie, -1 loss) with probe and learner cluster ids."""
    out = pw[["probe_id", "seed"]].copy()
    out["score"] = pw.result.map(SCORE).astype(float)
    out["learner"] = out.probe_id.map(learner_of)
    return out


def net_win_ci(pw: pd.DataFrame, unit: str = "learner") -> tuple[float, float, float]:
    """Net win rate (wins - losses) vs the baseline, CI clustered by 'probe' or 'learner'."""
    s = pairwise_scores(pw)
    return mean_ci(s.score, s.probe_id if unit == "probe" else s.learner)


def paired_diff_ci(
    pw_a: pd.DataFrame, pw_b: pd.DataFrame, unit: str = "learner"
) -> tuple[float, float, float, int]:
    """Net-win difference a - b on the same (probe, seed) pairs: (diff, lo, hi, n_pairs)."""
    a, b = pairwise_scores(pw_a), pairwise_scores(pw_b)
    m = a.merge(b, on=["probe_id", "seed", "learner"], suffixes=("_a", "_b"))
    d = m.score_a - m.score_b
    mean, lo, hi = mean_ci(d, m.probe_id if unit == "probe" else m.learner)
    return mean, lo, hi, len(m)


def fmt_ci(mean: float, lo: float, hi: float, pct: bool = True, signed: bool = True) -> str:
    """'+6.8% [-14.6%, +1.2%]' (pct) or '+0.081 [+0.020, +0.150]'."""
    if not np.isfinite(mean):
        return ""
    sign = "+" if signed else ""
    if pct:
        return f"{100 * mean:{sign}.1f}% [{100 * lo:{sign}.1f}%, {100 * hi:{sign}.1f}%]"
    return f"{mean:{sign}.3f} [{lo:{sign}.3f}, {hi:{sign}.3f}]"
