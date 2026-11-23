"""ThresholdPolicy (P4): thresholds follow from costs, not argmax.

lambda = C(false skip) / C(false retrieve). Minimising expected cost, retrieve iff
    p * C(false skip) > (1 - p) * C(false retrieve)   <=>   p > tau* = 1 / (1 + lambda)
With lambda = 4, tau* = 0.2. The abstention band [tau_low, tau_high] sits around tau*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from amic.contracts.config import GateConfig

Verdict = Literal["skip", "retrieve", "abstain"]


def load_thresholds(path: str | None, lam: float) -> dict | None:
    """Recalibrated band offsets, used only if the file exists and was made for this lambda."""
    if not path or not Path(path).exists():
        return None
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    return rec if abs(float(rec.get("lam", -1)) - lam) < 1e-9 else None


def tau_star(lam: float) -> float:
    if lam <= 0:
        raise ValueError("lambda must be > 0")
    return 1.0 / (1.0 + lam)


@dataclass(frozen=True)
class ThresholdPolicy:
    tau: float
    tau_low: float
    tau_high: float

    @classmethod
    def from_config(cls, g: GateConfig) -> ThresholdPolicy:
        t = tau_star(g.lam) if g.tau_mode == "cost" else g.tau_fixed
        lo_off, hi_off = g.band_low_offset, g.band_high_offset
        rec = load_thresholds(g.thresholds_file, g.lam)
        if rec is not None:  # written by the recalibration job (python -m amic.jobs recalibrate)
            lo_off, hi_off = rec["band_low_offset"], rec["band_high_offset"]
        return cls(tau=t, tau_low=max(0.0, t - lo_off), tau_high=min(1.0, t + hi_off))

    def verdict(self, p: float, band: bool = True) -> Verdict:
        """band=False: hard decision at tau (used when there is no abstention tier)."""
        if not band:
            return "retrieve" if p > self.tau else "skip"
        if p < self.tau_low:
            return "skip"
        if p > self.tau_high:
            return "retrieve"
        return "abstain"


def expected_cost(y: np.ndarray, retrieve: np.ndarray, lam: float) -> float:
    """Mean cost per query in units of C(false retrieve): FS costs lam, FR costs 1."""
    y = y.astype(bool)
    retrieve = retrieve.astype(bool)
    false_skip = (y & ~retrieve).sum()
    false_retrieve = (~y & retrieve).sum()
    return float((lam * false_skip + false_retrieve) / max(1, len(y)))


def band_for_rate(
    p: np.ndarray, tau: float, target_rate: float, max_offset: float = 0.5
) -> tuple[float, float]:
    """Symmetric-in-rank band around tau that abstains on ~target_rate of validation traffic."""
    if target_rate <= 0:
        return tau, tau
    lo_off, hi_off = 0.0, 0.0
    step = 0.005
    while (lo_off < max_offset or hi_off < max_offset) and (
        ((p >= tau - lo_off) & (p <= tau + hi_off)).mean() < target_rate
    ):
        # widen the side with fewer points inside, so the band grows in probability mass
        below = ((p >= tau - lo_off - step) & (p < tau - lo_off)).sum()
        above = ((p > tau + hi_off) & (p <= tau + hi_off + step)).sum()
        if (below <= above and lo_off < min(tau, max_offset)) or hi_off >= max_offset:
            lo_off = min(tau, lo_off + step)
        else:
            hi_off += step
    return max(0.0, tau - lo_off), min(1.0, tau + hi_off)


def widest_band_under(p: np.ndarray, tau: float, max_count: float) -> tuple[float, float]:
    """Band [lo, hi] containing tau, spanning whole p levels, holding the most rows <= max_count.

    Ties go to the narrower band. Edges sit halfway to the next level outside the band.
    """
    # 4 decimals: merges float noise (one isotonic level came out as 0.339622 AND 0.339623)
    pr = np.round(p, 4)
    levels = np.unique(pr)
    below = [x for x in levels if x <= tau]
    above = [x for x in levels if x >= tau]
    best: tuple[int, float, float, float] | None = None  # (count, -width, lo_level, hi_level)
    for lo_l in [tau, *below]:
        for hi_l in [tau, *above]:
            if lo_l > hi_l:
                continue
            n = int(((pr >= lo_l) & (pr <= hi_l)).sum())
            if n <= max_count and (best is None or (n, -(hi_l - lo_l)) > best[:2]):
                best = (n, -(hi_l - lo_l), lo_l, hi_l)
    assert best is not None  # [tau, tau] always qualifies unless max_count < rows at tau
    _, _, lo_l, hi_l = best
    nxt_below = [x for x in levels if x < lo_l]
    nxt_above = [x for x in levels if x > hi_l]
    lo = (lo_l + nxt_below[-1]) / 2 if nxt_below else 0.0
    hi = (hi_l + nxt_above[0]) / 2 if nxt_above else 1.0
    return float(min(lo, tau)), float(max(hi, tau))
