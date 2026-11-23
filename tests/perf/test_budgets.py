"""Performance budgets (brief section 11; pre-registered PB1).

Brief budgets: Tier 0 ~0.1 ms, Tier 0+1 path <= 5 ms, Tier 2 <= 10-15 % of traffic, memory
growth 10^2 -> 10^4 records. CI machines are noisy and shared, so the asserted bounds are loose
multiples of the budgets; `scripts/bench_gate.py` measures the real numbers on the dev machine
for the report. Run alone with `pytest tests/perf -q`.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

from amic.contracts.config import Settings
from amic.contracts.models import RequestContext
from amic.control.controller import Controller
from amic.control.route_planner import RoutePlanner
from amic.control.signals.bus import FEATURES, Signals, compute_features
from amic.control.tier0 import Tier0
from amic.control.tier1 import train_tier1
from amic.control.tier2 import RateCap
from amic.memory.sketch import build_sketch

pytestmark = pytest.mark.perf


def _feats(rng: np.random.Generator) -> dict[str, float]:
    f = {k: float(rng.random()) for k in FEATURES}
    f["s3_has_memory"] = 1.0
    f["s1_explicit_ref"] = f["s1_followup"] = 0.0
    return f


def _p50_ms(fn, n: int = 2000) -> float:
    for _ in range(50):  # warm-up
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


@pytest.fixture(scope="module")
def tier1():
    rng = np.random.default_rng(0)
    X = rng.random((400, len(FEATURES))).astype(np.float32)
    y = (X[:, 0] + rng.random(400) * 0.5 > 0.7).astype(int)
    return train_tier1(X[:300], y[:300], X[300:], y[300:], FEATURES, depth_tr=y[:300] * 0.0)


def test_tier0_is_about_a_tenth_of_a_millisecond():
    rng = np.random.default_rng(1)
    sig = Signals(np.zeros(384, np.float32), _feats(rng), None, False, 0.0, 0.0)
    t0 = Tier0()
    assert _p50_ms(lambda: t0.decide(sig)) < 0.1 * 5  # budget 0.1 ms; 5x slack for CI


def test_tier0_plus_tier1_path_under_five_ms(tier1):
    rng = np.random.default_rng(2)
    s = Settings()
    ctl = Controller(s, RoutePlanner(s.routes), tier1)
    sig = Signals(np.zeros(384, np.float32), _feats(rng), None, False, 0.0, 0.0)
    ctx = RequestContext(request_id="r", tenant_id="t", session_id="s", query="q", arm="amic")
    assert _p50_ms(lambda: ctl.decide_signals(sig, ctx), n=500) < 5.0


def test_feature_extraction_stays_flat_as_memory_grows():
    """Memory growth 10^2 -> 10^4 records: the gate reads only the sketch (k <= 8 centroids +
    a Bloom filter), so signal cost must not grow with the number of records."""
    rng = np.random.default_rng(3)
    ctx = RequestContext(
        request_id="r",
        tenant_id="t",
        session_id="s",
        query="why does my list loop skip items",
        arm="amic",
    )
    q = rng.standard_normal(384).astype(np.float32)
    q /= np.linalg.norm(q)
    cost = {}
    for n in (100, 1_000, 10_000):
        v = rng.standard_normal((n, 384)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        sk = build_sketch(v, [f"record {i} about lists and loops" for i in range(n)])
        assert len(sk.centroids) <= 8  # bounded sketch size
        cost[n] = _p50_ms(lambda sk=sk: compute_features(ctx, q, sk, {}, True), n=300)
    assert cost[10_000] < 1.0 and cost[10_000] < 3 * cost[100] + 0.05


def test_tier2_never_exceeds_its_traffic_cap():
    """Worst case: EVERY decision wants Tier 2; the cap still holds it to 12 % of traffic."""
    cap = RateCap(max_rate=0.12, window=200)
    fired = 0
    for _ in range(5000):
        ok = cap.allow()
        cap.observe(ok)
        fired += ok
    assert fired / 5000 <= 0.12 + 1e-9 and fired > 0
