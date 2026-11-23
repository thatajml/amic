"""Scheduled jobs (retrain with a promotion gate, recalibration, forgetting) and the S5
uncertainty signal, on a synthetic labelled set (no data/ in CI)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from amic import jobs
from amic.contracts.config import Settings
from amic.contracts.models import RequestContext
from amic.control.signals.bus import FEATURES, SignalBus
from amic.control.signals.uncertainty import UncertaintySignal, s5_features, token_entropy
from amic.control.thresholds import ThresholdPolicy
from amic.control.tier1 import Tier1Model, train_tier1


@pytest.fixture
def labelled(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(rng.random((n, len(FEATURES))), columns=FEATURES)
    df["label_required"] = (df[FEATURES[0]] + rng.random(n) * 0.6 > 0.8).astype(bool)
    df["depth"] = np.where(df.label_required, "SHALLOW", None)
    df["learner_id"] = [f"L{i % 20}" for i in range(n)]
    df["split"] = df.learner_id.map(
        lambda x: "train" if int(x[1:]) < 12 else "val" if int(x[1:]) < 16 else "test"
    )
    df["tier0_rule"] = "abstain"
    df["probe_id"] = [f"p{i}" for i in range(n)]
    monkeypatch.setattr("amic.eval.replay.load_dataset", lambda name: df)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    base = train_tier1(
        tr[FEATURES].to_numpy(np.float32),
        tr.label_required.astype(int).to_numpy(),
        va[FEATURES].to_numpy(np.float32),
        va.label_required.astype(int).to_numpy(),
        FEATURES,
    )
    base.save(tmp_path / "tier1.joblib")
    # distilled verdicts: one from a train learner, one from a test learner (must be dropped)
    rows = [
        {"tenant_id": "L1", "route": "SHALLOW", "features": dict.fromkeys(FEATURES, 0.9)},
        {"tenant_id": "L19", "route": "SKIP", "features": dict.fromkeys(FEATURES, 0.1)},
    ]
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    s = Settings()
    s.gate.tier1_artifact = str(tmp_path / "tier1.joblib")
    s.jobs.distill_glob = str(tmp_path / "*.jsonl")
    s.jobs.live_artifact = str(tmp_path / "tier1_live.joblib")
    s.jobs.thresholds_file = str(tmp_path / "thr.json")
    s.jobs.log_path = str(tmp_path / "jobs.jsonl")
    s.jobs.max_auc_drop = 1.0  # always promote unless the cost is worse
    return s, tmp_path


def test_retrain_uses_only_train_learner_verdicts_and_never_touches_the_experiment_model(labelled):
    s, tmp = labelled
    before = (tmp / "tier1.joblib").read_bytes()
    out = jobs.retrain(s, dry_run=True)
    assert out["n_distill"] == 1 and not out["promoted"]
    assert not (tmp / "tier1_live.joblib").exists()
    out = jobs.retrain(s)
    assert out["promoted"] == out["would_promote"]
    if out["promoted"]:
        assert Tier1Model.load(tmp / "tier1_live.joblib").meta["n_distill"] == 1
    assert (tmp / "tier1.joblib").read_bytes() == before
    log = [json.loads(x) for x in (tmp / "jobs.jsonl").read_text().splitlines()]
    assert [r["job"] for r in log] == ["retrain", "retrain"]


def test_recalibrate_writes_a_band_the_gate_then_uses(labelled):
    s, tmp = labelled
    rec = jobs.recalibrate(s)
    assert rec["val_share"] <= s.gate.tier2_max_rate + 1e-9
    s.gate.thresholds_file = s.jobs.thresholds_file
    pol = ThresholdPolicy.from_config(s.gate)
    assert pol.tau_low == pytest.approx(max(0.0, 0.2 - rec["band_low_offset"]))
    assert pol.tau_high == pytest.approx(0.2 + rec["band_high_offset"])
    s.gate.lam = 1.0  # a file made for another lambda is ignored
    assert ThresholdPolicy.from_config(s.gate).tau_low == pytest.approx(0.5 - 0.08)


def test_forget_job_runs_over_components(labelled, comps):
    s, _ = labelled
    comps.writer.write_turn(
        "ann", "s", "r1", "how do lists work", "...", pd.Timestamp("2026-01-01", tz="UTC")
    )
    out = jobs.forget(s, dry_run=True, comps=comps)
    assert out["tenants"] >= 1 and out["dry_run"]


# ----------------------------------------------------------------------------- S5
def test_token_entropy():
    assert token_entropy([0.0]) == 0.0
    assert token_entropy([math.log(0.5), math.log(0.5)]) == pytest.approx(math.log(2))
    f = s5_features([[0.0], [math.log(0.5), math.log(0.5)]])
    assert f["s5_entropy_max"] == pytest.approx(math.log(2))
    assert f["s5_entropy_mean"] == pytest.approx(math.log(2) / 2)
    assert s5_features([]) == {"s5_entropy_mean": 0.0, "s5_entropy_max": 0.0}


class _DraftLLM:
    def __init__(self):
        self.calls = 0

    def draft_logprobs(self, messages, max_tokens, top):
        self.calls += 1
        return "draft", [[math.log(0.5), math.log(0.5)]] * 3, 12.0


def test_signal_bus_adds_s5_only_where_tier1_can_run(comps, seed_history):
    from amic.generation.prompt import PromptBuilder

    llm = _DraftLLM()
    bus = SignalBus(
        comps.embedder,
        comps.sketches,
        comps.sessions,
        uncertainty=UncertaintySignal(llm, PromptBuilder()),
    )
    ctx = RequestContext(
        request_id="r", tenant_id="new", session_id="s", query="what is a list?", arm="amic"
    )
    assert "s5_entropy_mean" not in bus.extract(ctx).features and llm.calls == 0  # R1: no memory
    seed_history(comps, "old", [("I struggle with lists", "ok")])
    ctx = ctx.model_copy(update={"tenant_id": "old"})
    f = bus.extract(ctx).features
    assert f["s5_entropy_mean"] == pytest.approx(math.log(2)) and llm.calls == 1
    cmd = ctx.model_copy(update={"query": "/remember x"})
    assert "s5_entropy_mean" not in bus.extract(cmd).features  # R4 resolves first
