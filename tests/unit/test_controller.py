from __future__ import annotations

import numpy as np
import pytest

from amic.app import build_arm, make_context
from amic.contracts.config import GateConfig, Settings, load_settings
from amic.contracts.models import ResolvedBy, Route
from amic.control.controller import Controller
from amic.control.route_planner import RoutePlanner
from amic.control.signals.bus import FEATURES, S3_FEATURES, Signals
from amic.control.thresholds import ThresholdPolicy, band_for_rate, expected_cost, tau_star
from amic.control.tier1 import Tier1Model, train_tier1
from tests.conftest import T0


# ----------------------------------------------------------------------------- thresholds
def test_tau_star_from_lambda():
    assert tau_star(4.0) == pytest.approx(0.2)
    assert tau_star(1.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        tau_star(0)


def test_policy_band_and_fixed_mode():
    p = ThresholdPolicy.from_config(GateConfig(lam=4.0, band_low_offset=0.05, band_high_offset=0.1))
    assert (p.tau, p.tau_low, p.tau_high) == pytest.approx((0.2, 0.15, 0.3))
    assert [p.verdict(x) for x in (0.1, 0.2, 0.31)] == ["skip", "abstain", "retrieve"]
    assert p.verdict(0.21, band=False) == "retrieve" and p.verdict(0.19, band=False) == "skip"
    f = ThresholdPolicy.from_config(GateConfig(tau_mode="fixed", tau_fixed=0.5))
    assert f.tau == 0.5


def test_expected_cost_weights_false_skips():
    y = np.array([1, 1, 0, 0])
    assert expected_cost(y, np.array([0, 1, 0, 0]), lam=4) == pytest.approx(4 / 4)
    assert expected_cost(y, np.array([1, 1, 1, 0]), lam=4) == pytest.approx(1 / 4)


def test_band_for_rate_hits_target():
    p = np.random.default_rng(0).uniform(0, 1, 2000)
    lo, hi = band_for_rate(p, 0.2, 0.12)
    rate = ((p >= lo) & (p <= hi)).mean()
    assert lo < 0.2 < hi and 0.11 <= rate <= 0.14


# ----------------------------------------------------------------------------- tier 1
def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES))).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0.3).astype(int)
    depth = (X[:, 1] > 0).astype(int)
    return X, y, depth


@pytest.mark.parametrize("kind", ["logreg", "lgbm"])
def test_tier1_trains_calibrates_and_round_trips(tmp_path, kind):
    X, y, d = _toy()
    m = train_tier1(X[:300], y[:300], X[300:], y[300:], FEATURES, kind=kind, depth_tr=d[:300])
    p, dp = m.predict_many(X[300:])
    assert ((p >= 0) & (p <= 1)).all() and m.depth is not None
    auc_like = p[y[300:] == 1].mean() - p[y[300:] == 0].mean()
    assert auc_like > 0.3
    m.save(tmp_path / "t1.joblib")
    m2 = Tier1Model.load(tmp_path / "t1.joblib")
    f = dict(zip(FEATURES, X[0].tolist(), strict=True))
    assert m2.predict(f) == pytest.approx(m.predict(f))


# ----------------------------------------------------------------------------- controller
class StubT1:
    def __init__(self, p, d=0.2, features=FEATURES):
        self.p, self.d, self.features = p, d, features

    def predict(self, feats):
        return self.p, self.d


def _sig(**over):
    f = dict.fromkeys(FEATURES, 0.0)
    f["s3_has_memory"] = 1.0
    f.update(over)
    return Signals(np.zeros(4), f, None, False, 0.0, 0.0)


def _ctl(p=0.5, d=0.2, **gate):
    s = Settings(gate=GateConfig(lam=4.0, band_low_offset=0.05, band_high_offset=0.1, **gate))
    return Controller(s, RoutePlanner(s.routes), StubT1(p, d))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("p", "d", "route", "by"),
    [
        (0.05, 0.9, Route.SKIP, ResolvedBy.TIER1),
        (0.9, 0.2, Route.SHALLOW, ResolvedBy.TIER1),
        (0.9, 0.8, Route.DEEP, ResolvedBy.TIER1),
        (0.2, 0.9, Route.SHALLOW, ResolvedBy.FAILOPEN),  # in band -> fail open, never SKIP
    ],
)
def test_controller_tier1_routes(p, d, route, by):
    dec, tm = _ctl(p, d).decide_signals(_sig())
    assert dec.route == route and dec.resolved_by == by and dec.p == p
    assert dec.tau_low == pytest.approx(0.15) and tm.tier1 >= 0


def test_hard_tau_without_band():
    dec, _ = _ctl(0.21, use_band=False).decide_signals(_sig())
    assert dec.route == Route.SHALLOW and dec.resolved_by == ResolvedBy.TIER1


def test_tier0_wins_before_tier1():
    dec, tm = _ctl(0.01).decide_signals(_sig(s1_explicit_ref=1.0))
    assert dec.route == Route.DEEP and dec.resolved_by == ResolvedBy.TIER0 and tm.tier1 == 0.0
    dec, _ = _ctl(0.99).decide_signals(_sig(s3_has_memory=0.0))
    assert dec.route == Route.SKIP and dec.resolved_by == ResolvedBy.TIER0


def test_tier0_only_ablation_fails_open():
    s = load_settings(ablation={"tiers": "t0"})
    dec, _ = Controller(s, RoutePlanner(s.routes), None).decide_signals(_sig())
    assert dec.route == Route.SHALLOW and dec.resolved_by == ResolvedBy.FAILOPEN


def test_s3_ablation_rejects_model_using_s3():
    s = load_settings(ablation={"use_sketch_probe": False})
    with pytest.raises(ValueError, match="S3 ablation"):
        Controller(s, RoutePlanner(s.routes), StubT1(0.5))  # type: ignore[arg-type]
    ok = [f for f in FEATURES if f not in S3_FEATURES or f == "s3_has_memory"]
    Controller(s, RoutePlanner(s.routes), StubT1(0.5, features=ok))  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- amic arm e2e
def test_amic_arm_end_to_end(comps, seed_history):
    X, y, d = _toy()
    t1 = train_tier1(X[:300], y[:300], X[300:], y[300:], FEATURES, depth_tr=d[:300])
    from amic.arms.amic import AmicArm

    arm = AmicArm(comps, tier1=t1)
    cold = arm.answer(make_context(comps, "newbie", "s1", "what is a list?", "amic", now=T0))
    assert cold.decision.route == Route.SKIP and cold.decision.rationale.startswith("R1")
    seed_history(comps, "amy", [("my loop crashed with IndexError", "range goes too far")])
    comps.sketches.invalidate("amy")
    r = arm.answer(
        make_context(comps, "amy", "s2", "what did I get wrong last time?", "amic", now=T0)
    )
    assert r.decision.route == Route.DEEP and r.items
    t = comps.telemetry.traces[-1]
    assert t.arm == "amic" and t.features["s1_explicit_ref"] > 0
    assert t.latency.embed > 0 and t.latency.tier0 > 0 and t.latency.retrieval > 0


def test_write_invalidates_sketch(comps):
    assert comps.sketches.get("zed") is None
    arm = build_arm("always", comps)
    arm.answer(make_context(comps, "zed", "s1", "loops please", "always", now=T0))
    assert comps.sketches.get("zed") is not None  # hook invalidated the cached None
