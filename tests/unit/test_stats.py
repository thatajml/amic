from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amic.eval.stats import (
    fmt_ci,
    learner_of,
    mean_ci,
    net_win_ci,
    paired_diff_ci,
    stat_ci,
)


def test_learner_of():
    assert learner_of("edu-009-p08") == "edu-009"
    assert learner_of("edu-009-p8") == "edu-009"
    assert learner_of("locomo-conv-26-q0001") == "locomo-conv-26-q0001"
    assert learner_of("lme-e47becba_abs") == "lme-e47becba_abs"


def test_mean_ci_is_deterministic_and_brackets_the_mean():
    rng = np.random.default_rng(1)
    v = rng.normal(0.3, 1.0, 400)
    c = np.repeat(np.arange(40), 10)
    m1, lo1, hi1 = mean_ci(v, c)
    m2, lo2, hi2 = mean_ci(v, c)
    assert (m1, lo1, hi1) == (m2, lo2, hi2)
    assert lo1 < m1 < hi1
    assert m1 == pytest.approx(v.mean())


def test_clustering_widens_the_interval_for_correlated_rows():
    # 10 clusters, each row repeated 20 times: the effective n is 10, not 200
    base = np.random.default_rng(2).normal(0, 1, 10)
    v = np.repeat(base, 20)
    clusters = np.repeat(np.arange(10), 20)
    _, lo_c, hi_c = mean_ci(v, clusters)
    _, lo_r, hi_r = mean_ci(v, np.arange(200))
    assert (hi_c - lo_c) > 2 * (hi_r - lo_r)


def test_constant_values_give_a_zero_width_interval():
    m, lo, hi = mean_ci(np.full(30, 0.5), np.arange(30) % 3)
    assert m == lo == hi == 0.5


def _pw(results: list[str], probes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"probe_id": probes, "seed": 1, "result": results})


def test_net_win_and_paired_difference():
    probes = [f"edu-00{i // 4}-p{i % 4}" for i in range(16)]
    a = _pw(["win"] * 8 + ["tie"] * 4 + ["loss"] * 4, probes)
    b = _pw(["win"] * 4 + ["tie"] * 4 + ["loss"] * 8, probes)
    net, lo, hi = net_win_ci(a)
    assert net == pytest.approx(0.25) and lo <= net <= hi
    d, dlo, dhi, n = paired_diff_ci(a, b)
    assert n == 16 and d == pytest.approx(0.5) and dlo <= d <= dhi
    # identical runs: zero difference, zero width
    d0, lo0, hi0, _ = paired_diff_ci(a, a)
    assert d0 == lo0 == hi0 == 0.0


def test_stat_ci_skips_undefined_resamples():
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame(
        {
            "g": np.repeat(np.arange(12), 5),
            "y": np.tile([0, 1, 0, 1, 1], 12),
            "p": np.tile([0.1, 0.9, 0.3, 0.7, 0.6], 12),
        }
    )
    auc, lo, hi = stat_ci(df, "g", lambda d: roc_auc_score(d.y, d.p), b=200)
    assert auc == pytest.approx(1.0) and lo <= auc <= hi


def test_fmt_ci():
    assert fmt_ci(-0.068, -0.146, 0.012) == "-6.8% [-14.6%, +1.2%]"
    assert fmt_ci(0.081, 0.02, 0.15, pct=False) == "+0.081 [+0.020, +0.150]"
    assert fmt_ci(float("nan"), 0, 0) == ""
