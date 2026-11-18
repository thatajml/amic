"""ThresholdPolicy (P4): thresholds follow from costs, not argmax.

lambda = C(false skip) / C(false retrieve). Minimising expected cost, retrieve iff
    p * C(false skip) > (1 - p) * C(false retrieve)   <=>   p > tau* = 1 / (1 + lambda)
With lambda = 4, tau* = 0.2. The abstention band [tau_low, tau_high] sits around tau*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from amic.contracts.config import GateConfig

Verdict = Literal["skip", "retrieve", "abstain"]


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
        return cls(
            tau=t,
            tau_low=max(0.0, t - g.band_low_offset),
            tau_high=min(1.0, t + g.band_high_offset),
        )

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
