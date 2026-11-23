"""One command -> every results table and figure for the paper (plan T6.2-T6.4).

    python -m amic.eval.report            # writes docs/results/main.md (+ figures)

Sections whose inputs do not exist yet are skipped with a note, so this can run at any stage.
All numbers are measurements from this repository's runs; nothing is typed in by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from amic.contracts.config import Settings, load_settings
from amic.control.thresholds import ThresholdPolicy, tau_star
from amic.control.tier1 import Tier1Model
from amic.eval.replay import (
    RETRIEVE,
    decision_metrics,
    fixed_policy,
    load_dataset,
    replay,
    savings_captured,
)
from amic.eval.stats import fmt_ci, mean_ci, net_win_ci, paired_diff_ci, stat_ci

OUT = Path("docs/results")
LAMBDAS = [1.0, 2.0, 4.0, 8.0, 16.0]
CI_NOTE = (
    "95% CIs are percentile bootstraps that resample whole clusters (fixed seed). The test split "
    "has only 8 learners, so learner-clustered intervals are approximate and probably too narrow."
)
# Paired comparisons on the same (probe, seed) pairs; rows appear once both runs are judged.
# The second block is the pre-registered set for the components added after the first results.
PAIRS: list[tuple[str, str]] = [
    ("amic", "always_shallow"),
    ("amic", "amic_no_s3"),
    ("amic_lam1", "amic"),
    ("oracle", "amic"),
    ("amic_esc", "amic"),
    ("amic_lam1_esc", "amic_lam1"),
    ("never_esc", "never"),
    ("amic_esc", "never_esc"),
    ("amic_full", "amic"),
    ("always_m2", "always_shallow"),
    ("amic_v2", "always_shallow"),
    ("amic_v2", "amic"),
]


def _fmt(df: pd.DataFrame, pct: list[str] = (), dec: dict[str, int] | None = None) -> str:
    d = df.copy()
    for c in pct:
        if c in d:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{100 * v:.1f}%")
    for c, n in (dec or {}).items():
        if c in d:
            d[c] = d[c].map(lambda v, n=n: "" if pd.isna(v) else f"{v:.{n}f}")
    return d.to_markdown(index=False)


# ----------------------------------------------------------------------------- dataset
def section_dataset() -> str:
    p = Path("data/edu/labels.parquet")
    if not p.exists():
        return "## Dataset\n\n_labels not built yet_\n"
    df = pd.read_parquet(p)
    by = (
        df.groupby("category")
        .agg(
            n=("probe_id", "size"),
            required=("label_required", "mean"),
            personal=("label_personal", "mean"),
        )
        .reset_index()
    )
    lines = [
        "## 1. Dataset and labels",
        "",
        f"Education set: {df.learner_id.nunique()} synthetic learners, {len(df)} probes; split by learner "
        f"({df.split.value_counts().to_dict()}). Counterfactual label: memory-required iff the best "
        "memory-backed answer meets the student's need better than the no-memory answer by >= 1 point (judge `needs_met`, 1-5, mean of 2 judgments; a correct generic reply to a self-contained question is not penalised).",
        "",
        f"Memory-required: **{df.label_required.mean():.1%}** (personalisation-inclusive variant {df.label_personal.mean():.1%}). "
        f"Depth among required: {df.depth.value_counts().to_dict()}. "
        f"Fabricated-history rate of the no-memory answer: {df.fab_skip.mean():.1%} "
        f"(SHALLOW {df.fab_shallow.mean():.1%}, DEEP {df.fab_deep.mean():.1%}).",
        "",
        _fmt(by, pct=["required", "personal"]),
    ]
    h = Path("data/edu/human_labels.csv")
    if h.exists():
        from sklearn.metrics import cohen_kappa_score

        m = pd.read_csv(h).merge(
            df[["probe_id", "label_required", "intended_memory"]], on="probe_id"
        )
        if len(m) >= 10:
            judge = m.label_required.astype(int)
            k = cohen_kappa_score(m.human_required, judge)
            _, k_lo, k_hi = stat_ci(
                m.assign(judge=judge),
                "probe_id",
                lambda d: cohen_kappa_score(d.human_required, d.judge),
            )
            agree = (m.human_required == judge).mean()
            k_hi = cohen_kappa_score(m.human_required, m.intended_memory.astype(int))
            k_ji = cohen_kappa_score(df.intended_memory.astype(int), df.label_required.astype(int))
            lines += [
                "",
                f"Human vs judge labels (n={len(m)}): agreement {agree:.1%}, Cohen's kappa **{k:.3f}** "
                f"(95% CI {k_lo:.3f} to {k_hi:.3f}, bootstrap over probes). "
                f"The judge marks memory as required more often than the human annotator "
                f"({judge.mean():.0%} vs {m.human_required.mean():.0%} on the same probes). "
                f"For reference, kappa human vs the generator's intended category = {k_hi:.3f}; "
                f"judge vs intended (all {len(df)} probes) = {k_ji:.3f}. "
                "Label validity is therefore moderate at best and bounds every downstream metric.",
            ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- decisions
VARIANTS: list[tuple[str, dict[str, Any], str]] = [
    ("amic (T0+T1, band->SHALLOW)", {}, "models/tier1.joblib"),
    ("amic, hard tau (no band)", {"gate": {"use_band": False}}, "models/tier1.joblib"),
    ("ablation: Tier 0 only", {"ablation": {"tiers": "t0"}}, ""),
    (
        "ablation: no S3 sketch probe",
        {"ablation": {"use_sketch_probe": False}},
        "models/tier1_no_s3.joblib",
    ),
    (
        "ablation: tau fixed 0.5",
        {"gate": {"tau_mode": "fixed", "tau_fixed": 0.5}},
        "models/tier1.joblib",
    ),
    ("variant: LightGBM Tier 1", {}, "models/tier1_lgbm.joblib"),
]


def probe_costs(dec: pd.DataFrame, lam: float) -> pd.Series:
    """Per-probe decision cost (false skip = lam, false retrieve = 1), indexed by probe_id."""
    y = dec.label_required.astype(bool).to_numpy()
    r = dec.route.isin(RETRIEVE).to_numpy()
    cost = np.where(y & ~r, lam, np.where(~y & r, 1.0, 0.0))
    return pd.Series(cost, index=dec.probe_id.to_numpy())


def _decision_rows(df: pd.DataFrame, lam: float) -> pd.DataFrame:
    """Decision metrics per policy, plus its cost minus `always`'s (learner-clustered CI)."""
    learners = df.set_index("probe_id").learner_id
    decs = {n: fixed_policy(df, n) for n in ("never", "always", "oracle")}
    base = {n: decision_metrics(d, lam) for n, d in decs.items()}
    always_cost = probe_costs(decs["always"], lam)

    def vs_always(dec: pd.DataFrame) -> str:
        diff = probe_costs(dec, lam) - always_cost
        return fmt_ci(*mean_ci(diff.to_numpy(), learners.reindex(diff.index).to_numpy()), pct=False)

    rows = [{"policy": n, **base[n], "cost_vs_always": vs_always(decs[n])} for n in decs]
    for name, over, model in VARIANTS:
        if model and not Path(model).exists():
            continue
        s = load_settings(**over)
        s.gate.lam = lam
        t1 = Tier1Model.load(model) if model else None
        dec = replay(df, s, t1)
        m = decision_metrics(dec, lam)
        m["savings_captured"] = savings_captured(m, base["always"], base["oracle"])
        rows.append({"policy": name, **m, "cost_vs_always": vs_always(dec)})
    return pd.DataFrame(rows)


TIER2_VERDICTS = Path("data/edu/tier2_verdicts.jsonl")


def section_tier2() -> str:
    """Tier 2 (pre-registered T1): full cascade vs the same band failing open, on the test split."""
    from amic.control.tier2 import ReplayTier2

    full_cfg = Path("configs/ablations/full.yaml")
    if not (TIER2_VERDICTS.exists() and full_cfg.exists() and Path("models/tier1.joblib").exists()):
        return ""
    s_full = load_settings(full_cfg)
    lam = s_full.gate.lam
    pol = ThresholdPolicy.from_config(s_full.gate)
    t1 = Tier1Model.load("models/tier1.joblib")
    df = load_dataset("edu")
    test = df[df.split == "test"].reset_index(drop=True)
    learners = test.set_index("probe_id").learner_id
    model = s_full.models.tier2
    s_open = load_settings(full_cfg, ablation={"tiers": "t0_t1"})  # same band, no Tier 2
    s_hard = load_settings(gate={"use_band": False})
    policies = {
        "full cascade (Tier 2 in the band)": replay(
            test,
            s_full,
            t1,
            ReplayTier2.from_jsonl(
                str(TIER2_VERDICTS),
                model,
                s_full.gate.tier2_max_rate,
                s_full.gate.tier2_timeout_ms,
            ),
        ),
        "same band, fail open to SHALLOW (no Tier 2)": replay(test, s_open, t1),
        "hard tau (no band)": replay(test, s_hard, t1),
        "amic as run (T0+T1, original band -> SHALLOW)": replay(test, load_settings(), t1),
    }
    always_cost = probe_costs(fixed_policy(test, "always"), lam)
    rows = []
    for name, dec in policies.items():
        m = decision_metrics(dec, lam)
        diff = probe_costs(dec, lam) - always_cost
        rows.append(
            {
                "policy": name,
                "retrieval_rate": m["retrieval_rate"],
                "false_skip_rate": m["false_skip_rate"],
                "cost": m["cost"],
                "cost_vs_always (learner CI)": fmt_ci(
                    *mean_ci(diff.to_numpy(), learners.reindex(diff.index).to_numpy()), pct=False
                ),
                "share_tier2": (dec.resolved_by == "tier2").mean(),
                "share_failopen": (dec.resolved_by == "failopen").mean(),
            }
        )
    full = policies["full cascade (Tier 2 in the band)"]
    opened = policies["same band, fail open to SHALLOW (no Tier 2)"]
    d = probe_costs(full, lam) - probe_costs(opened, lam)
    d_ci = fmt_ci(*mean_ci(d.to_numpy(), learners.reindex(d.index).to_numpy()), pct=False)
    # Tier-2 verdict quality inside the band, against the labels
    band = full[full.tier2_status != ""]  # replay rows carry label_required already
    v = pd.read_json(TIER2_VERDICTS, lines=True)
    v = v[(v.model == model) & v.split.isin(["val", "test"])]
    ok = v[v.status == "ok"]
    lines = [
        "## 10. Tier 2: LLM adjudicator in the uncertain band (extension; pre-registered T1)",
        "",
        f"Band [{pol.tau_low:.3f}, {pol.tau_high:.3f}] around tau* = {pol.tau:.2f}, from "
        "`configs/ablations/full.yaml` (sized on the validation "
        "split by `scripts/tune_band.py` to the widest band holding at most 12% of traffic: "
        "isotonic calibration leaves only a few distinct p values, so the band holds less). "
        f"Tier 2 = `{model}` (the brief's 3B model cannot stay resident next to the tutor on the "
        "8 GB GPU), constrained JSON, 800 ms timeout and 12% traffic cap, both failing open to "
        "SHALLOW. Verdicts were recorded once per probe (`scripts/replay_tier2.py`) and replayed. "
        f"Test split, lambda = {lam:g}.",
        "",
        _fmt(
            pd.DataFrame(rows),
            pct=["retrieval_rate", "false_skip_rate", "share_tier2", "share_failopen"],
            dec={"cost": 3},
        ),
        "",
        f"Full cascade minus the same band failing open (cost per query, learner CI): {d_ci}. "
        f"Tier 2 decided {len(band)} test probes; of those, Tier 2 retrieved on "
        f"{(band.route != 'SKIP').mean():.0%} and the label says memory was required on "
        f"{band.label_required.mean():.0%}. Verdict latency (val + test, n = {len(v)}): "
        f"p50 {v.ms.median():.0f} ms, p95 {v.ms.quantile(0.95):.0f} ms; "
        f"{(ok.ms > s_full.gate.tier2_timeout_ms).mean():.1%} would exceed the timeout; "
        f"routes {ok.route.value_counts(normalize=True).round(3).to_dict()}.",
    ]
    return "\n".join(lines) + "\n"


LME_RUNS = ["lme_always", "lme_always_shallow", "lme_never", "lme_amic", "lme_amic_esc"]
LME_PAIRS = [
    ("lme_amic", "lme_always"),
    ("lme_amic", "lme_always_shallow"),
    ("lme_amic_esc", "lme_amic"),
    ("lme_always_shallow", "lme_always"),
    ("lme_never", "lme_always"),
]


def section_longmemeval() -> str:
    """LongMemEval_S (pre-registered L1 decisions, L2 gold-answer QA accuracy)."""
    d = Path("data/longmemeval")
    if not (d / "features.parquet").exists() or not Path("models/tier1.joblib").exists():
        return ""
    lam = Settings().gate.lam
    df = load_dataset("longmemeval")
    ans = df[df.label_required.astype(bool)]
    dec_rows = _decision_rows(df, lam)
    abst = df[~df.label_required.astype(bool)].probe_id
    t1 = Tier1Model.load("models/tier1.joblib")
    amic_dec = replay(df, load_settings(), t1)
    by_type = (
        amic_dec.merge(df[["probe_id", "category"]], on="probe_id", suffixes=("", "_c"))
        .assign(retrieve=lambda x: x.route.isin(list(RETRIEVE)))
        .groupby("category")
        .agg(n=("probe_id", "size"), amic_retrieval_rate=("retrieve", "mean"))
        .reset_index()
    )
    lines = [
        "## 11. LongMemEval_S, cleaned (external check; pre-registered L1, L2)",
        "",
        f"{len(df)} questions, each with its own ~48-session history ({len(ans)} answerable = "
        f"memory-required; {len(abst)} abstention questions = weak memory-not-required analogue). "
        "Tier 1 is the education-trained model, unchanged; S4 concept features are Python-specific, "
        "so this measures transfer. Decisions are replayed from features computed with the live "
        "bus code (`scripts/prepare_longmemeval.py`).",
        "",
        _fmt(
            dec_rows[
                ["policy", "retrieval_rate", "recall", "false_skip_rate", "cost", "cost_vs_always"]
            ],
            pct=["retrieval_rate", "recall", "false_skip_rate"],
            dec={"cost": 3},
        ),
        "",
        "AMIC (T0+T1) retrieval rate by question type:",
        "",
        _fmt(by_type, pct=["amic_retrieval_rate"]),
    ]
    qa = []
    judged: dict[str, pd.DataFrame] = {}
    gold = pd.read_json(d / "gold.jsonl", lines=True)[["probe_id", "question_type", "abstention"]]
    for run in LME_RUNS:
        p = Path("runs") / run / "lme_judged.jsonl"
        if not p.exists():
            continue
        j = pd.read_json(p, lines=True).merge(gold, on="probe_id")
        j["score"] = j.correct.astype(float)
        judged[run] = j
        a, b = j[~j.abstention], j[j.abstention]
        row: dict[str, Any] = {
            "run": run,
            "n": len(j),
            "accuracy (answerable, 95% CI)": fmt_ci(*mean_ci(a.score, a.probe_id), signed=False),
            "abstention accuracy": f"{b.score.mean():.1%} (n={len(b)})" if len(b) else "",
            "retrieval_rate": j.final_route.isin(list(RETRIEVE)).mean(),
        }
        for t, g in a.groupby("question_type"):
            row[t] = g.score.mean()
        qa.append(row)
    if qa:
        tr = _traces("lme-")
        q = pd.DataFrame(qa)
        if tr is not None and not tr.empty:
            q["memory_tokens"] = q.run.map(tr.groupby("arm").memory_tokens.mean())
        type_cols = [c for c in q.columns if c in lme_types()]
        lines += [
            "",
            "### Gold-answer QA accuracy (live, stratified subset, temperature 0)",
            "",
            "Every abstention question plus 20 answerable questions per type; one run per arm at "
            "temperature 0; graded yes/no against the gold answer by the Gemini judge with "
            "per-type rules following the official protocol (`scripts/judge_longmemeval.py`). "
            "CIs: bootstrap over questions.",
            "",
            _fmt(q, pct=["retrieval_rate", *type_cols], dec={"memory_tokens": 0}),
        ]
        pr = []
        for a_run, b_run in LME_PAIRS:
            if a_run in judged and b_run in judged:
                m = judged[a_run].merge(
                    judged[b_run], on=["probe_id", "seed"], suffixes=("_a", "_b")
                )
                m = m[~m.abstention_a]
                diff = m.score_a - m.score_b
                pr.append(
                    {
                        "comparison": f"{a_run} − {b_run}",
                        "questions": len(m),
                        "accuracy diff (95% CI)": fmt_ci(*mean_ci(diff, m.probe_id)),
                    }
                )
        if pr:
            lines += [
                "",
                "Paired accuracy differences (answerable questions):",
                "",
                _fmt(pd.DataFrame(pr)),
            ]
    return "\n".join(lines) + "\n"


def lme_types() -> list[str]:
    from amic.eval.datasets.longmemeval import TYPES

    return TYPES


def section_decisions() -> tuple[str, pd.DataFrame | None]:
    if not Path("data/edu/labels.parquet").exists() or not Path("models/tier1.joblib").exists():
        return "## 2. Decision quality\n\n_needs labels + models/tier1.joblib_\n", None
    lam = Settings().gate.lam
    df = load_dataset("edu")
    test = df[df.split == "test"]
    t = _decision_rows(test, lam)
    cols = [
        "policy",
        "retrieval_rate",
        "recall",
        "false_skip_rate",
        "precision",
        "f1",
        "cost",
        "cost_vs_always",
        "savings_captured",
        "share_tier0",
        "share_tier1",
        "share_failopen",
        "depth_acc",
    ]
    pct = [
        "retrieval_rate",
        "recall",
        "false_skip_rate",
        "precision",
        "savings_captured",
        "share_tier0",
        "share_tier1",
        "share_failopen",
        "depth_acc",
    ]
    lines = [
        "## 2. Decision quality (offline replay, education TEST split)",
        "",
        f"lambda = {lam:g} (tau* = {tau_star(lam):.2f}); n = {len(test)} probes from held-out learners. "
        "`cost` = expected cost per query in units of one false retrieve (a false skip costs lambda). "
        "`cost_vs_always` = the policy's cost minus `always`'s on the same probes, with a 95% CI "
        "clustered by learner (positive = worse than retrieving every time). "
        "`savings_captured` = share of the always->oracle retrieval reduction achieved. " + CI_NOTE,
        "",
        _fmt(t[cols], pct=pct, dec={"f1": 3, "cost": 3}),
    ]
    lp = Path("data/locomo/features.parquet")
    if lp.exists() and Path("data/locomo/labels.parquet").exists():
        loc = load_dataset("locomo")
        lt = _decision_rows(loc, lam)
        lines += [
            "",
            f"### LoCoMo (external check, n = {len(loc)}; Tier 1 trained on the education set only)",
            "",
            "Category 5 (adversarial) is used as a weak memory-not-required analogue; reported "
            "separately and never pooled with the education results.",
            "",
            _fmt(
                lt[
                    [
                        "policy",
                        "retrieval_rate",
                        "recall",
                        "false_skip_rate",
                        "precision",
                        "cost",
                        "cost_vs_always",
                    ]
                ],
                pct=["retrieval_rate", "recall", "false_skip_rate", "precision"],
                dec={"cost": 3},
            ),
        ]
    return "\n".join(lines) + "\n", t


def section_lambda_sweep() -> str:
    if not Path("models/tier1.joblib").exists():
        return ""
    df = load_dataset("edu")
    test = df[df.split == "test"]
    t1 = Tier1Model.load("models/tier1.joblib")
    rows = []
    for lam in LAMBDAS:
        for mode in ("cost", "fixed"):
            s = load_settings(gate={"lam": lam, "tau_mode": mode, "tau_fixed": 0.5})
            m = decision_metrics(replay(test, s, t1), lam)
            rows.append(
                {
                    "lambda": lam,
                    "tau": "1/(1+lambda)" if mode == "cost" else "0.5 (argmax)",
                    "tau_value": tau_star(lam) if mode == "cost" else 0.5,
                    "retrieval_rate": m["retrieval_rate"],
                    "false_skip_rate": m["false_skip_rate"],
                    "cost": m["cost"],
                }
            )
    d = pd.DataFrame(rows)
    _plot_sweep(d)
    return (
        "\n".join(
            [
                "## 3. Cost-sensitive thresholding: lambda sweep (test split)",
                "",
                "Same calibrated Tier-1 model; only the threshold rule changes. Lower cost is better.",
                "",
                _fmt(d, pct=["retrieval_rate", "false_skip_rate"], dec={"tau_value": 3, "cost": 3}),
                "",
                "![lambda sweep](fig_lambda_sweep.png)",
            ]
        )
        + "\n"
    )


def _plot_sweep(d: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for mode, g in d.groupby("tau"):
        ax.plot(g["lambda"], g["cost"], marker="o", label=f"tau = {mode}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("lambda = C(false skip) / C(false retrieve)")
    ax.set_ylabel("expected cost / query")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_lambda_sweep.png", dpi=150)
    plt.close(fig)


def section_calibration() -> str:
    p = Path("runs/tier1.json")
    if not p.exists():
        return ""
    from sklearn.metrics import roc_auc_score

    r = json.loads(p.read_text())
    # Per-row test predictions (the same rows train_tier1.py evaluates) for AUC intervals.
    df = load_dataset("edu")
    test = df[(df.split == "test") & (df.tier0_rule == "abstain")].reset_index(drop=True)
    frame = test[["probe_id", "learner_id", "label_required"]].copy()
    names = [n for n in ("tier1", "tier1_no_s3", "tier1_lgbm") if Path(f"runs/{n}.json").exists()]
    for name in names:
        model = Tier1Model.load(f"models/{name}.joblib")
        frame[f"p_{name}"] = model.predict_many(test[model.features].to_numpy(np.float32))[0]

    def auc(d: pd.DataFrame, name: str) -> float:
        return float(roc_auc_score(d.label_required.astype(int), d[f"p_{name}"]))

    rows = []
    for name in names:
        j = json.loads(Path(f"runs/{name}.json").read_text())["test"]
        _, lo, hi = stat_ci(frame, "learner_id", lambda d, n=name: auc(d, n))
        rows.append(
            {
                "model": name,
                "n_test": j["n"],
                "auc": j["auc"],
                "auc_95ci_learner": f"[{lo:.3f}, {hi:.3f}]",
                "brier": j["brier"],
                "ece": j["ece"],
            }
        )
    diff_line = ""
    if {"tier1", "tier1_no_s3"} <= set(names):
        d, lo, hi = stat_ci(frame, "learner_id", lambda x: auc(x, "tier1") - auc(x, "tier1_no_s3"))
        diff_line = (
            f"AUC with the S3 sketch probe minus without it: {d:+.3f} "
            f"(95% CI {lo:+.3f} to {hi:+.3f}, clustered by learner)."
        )
    rel = pd.DataFrame(r["test"]["reliability"])
    _plot_reliability(rel)
    coef = list(r.get("coef", {}).items())[:8]
    return (
        "\n".join(
            [
                "## 4. Tier-1 calibration (test split, rows Tier 0 abstains on)",
                "",
                _fmt(pd.DataFrame(rows), dec={"auc": 3, "brier": 3, "ece": 3}),
                "",
                diff_line,
                "",
                "![reliability](fig_reliability.png)",
                "",
                "Largest standardized LogReg coefficients: "
                + ", ".join(f"`{k}` {v:+.2f}" for k, v in coef),
            ]
        )
        + "\n"
    )


def _plot_reliability(rel: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.plot(rel.mean_p, rel.frac_pos, marker="o")
    for _, r in rel.iterrows():
        ax.annotate(
            str(int(r.n)),
            (r.mean_p, r.frac_pos),
            fontsize=7,
            xytext=(3, -8),
            textcoords="offset points",
        )
    ax.set_xlabel("predicted p")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "fig_reliability.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- live runs
def _traces(prefix: str = "edu-") -> pd.DataFrame | None:
    """Experiment traces of one dataset (by learner-id prefix: 'edu-' or 'lme-')."""
    tr = _all_traces()
    if tr is None:
        return None
    return tr[tr.tenant_id.str.startswith(prefix)].reset_index(drop=True)


def _all_traces() -> pd.DataFrame | None:
    db = Path("data/experiments.db")
    if not db.exists():
        return None
    from sqlalchemy import create_engine

    return pd.read_sql_table("traces", create_engine(f"sqlite:///{db}"))


def _uncached_prefill_ms_per_token() -> float | None:
    """Least-squares slope of prefill ms vs prompt tokens from the Day-1 probe (runs/probe.json).

    The probe put a nonce first in every prompt, so its prefill was never served from Ollama's
    prompt cache; live-run prefill was (identical memory blocks repeat per learner).
    """
    p = Path("runs/probe.json")
    if not p.exists():
        return None
    gen = json.loads(p.read_text())["generation"].values()
    x = np.array([g["prompt_tokens"] for g in gen], float)
    y = np.array([g["prefill_ms"] for g in gen], float)
    return float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else None


def section_efficiency() -> tuple[str, pd.DataFrame | None]:
    tr = _traces()
    if tr is None or tr.empty:
        return "## 5. Efficiency (live runs)\n\n_no experiment traces yet_\n", None
    tr["lat_gate"] = tr.lat_signals + tr.lat_cache + tr.lat_tier0 + tr.lat_tier1 + tr.lat_tier2
    tr["escalated"] = tr.escalated.fillna(False).astype(bool)
    for c in ("lat_escalation", "lat_verify"):  # columns added after the first runs
        tr[c] = tr[c].fillna(0.0) if c in tr else 0.0
    # an escalated SKIP did retrieve (on the retry); `route` keeps the gate's decision
    tr["retrieved"] = tr.route.isin(["SHALLOW", "DEEP"]) | tr.escalated
    # Cache hits (prompt+seed already generated by another arm) return in ~0 ms wall time but
    # carry their first-run prefill/decode. Rebuild the total with the model time the call took
    # when it was actually generated; cache misses are unchanged (generation >= prefill+decode).
    gen_first = np.maximum(tr.lat_generation, tr.lat_prefill + tr.lat_decode)
    tr["lat_total_raw"] = tr.lat_total
    tr["lat_total"] = tr.lat_total - tr.lat_generation + gen_first
    rate = _uncached_prefill_ms_per_token()
    g = tr.groupby("arm")
    t = pd.DataFrame(
        {
            "n": g.size(),
            "retrieval_rate": g.retrieved.mean(),
            "prompt_tokens": g.prompt_tokens.mean(),
            "memory_tokens": g.memory_tokens.mean(),
            "embed_p50": g.lat_embed.median(),
            "gate_p50": g.lat_gate.median(),
            "gate_p95": g.lat_gate.quantile(0.95),
            "retrieval_p50": g.lat_retrieval.median(),
            "prefill_mean": g.lat_prefill.mean(),
            "total_p50": g.lat_total.median(),
            "total_p95": g.lat_total.quantile(0.95),
        }
    ).reset_index()
    if "always" in set(tr.arm):
        t["mem_tok_saved_vs_always"] = tr[tr.arm == "always"].memory_tokens.mean() - t.memory_tokens
    if rate is not None:
        t["mem_prefill_uncached_est"] = t.memory_tokens * rate
    note = [
        "",
        "**Prefill caveat.** Measured prefill is nearly flat across arms (the `always` arm sends "
        "~6x the prompt tokens of `never` at similar prefill) because Ollama reuses cached prompt "
        "prefixes: each synthetic learner has only a handful of history records, so DEEP retrieval "
        "returns an almost identical memory block on every query from that learner. Real "
        "deployments with larger, more varied memories would cache less. "
        + (
            f"`mem_prefill_uncached_est` = memory tokens x {rate:.3f} ms/token, the uncached "
            "prefill rate measured by the Day-1 probe (nonce-first prompts, `runs/probe.json`). "
            "Token counts are unaffected by caching and are the primary efficiency metric."
            if rate is not None
            else ""
        ),
    ]
    return "\n".join(
        [
            "## 5. Efficiency (live runs, test split, all seeds)",
            "",
            "Milliseconds. `gate` = signals + Tier 0 + Tier 1 (excludes the query embedding, which "
            "`always` pays too). Tutor calls are served from a prompt-keyed cache when an identical "
            "prompt+seed was already generated; cached entries keep their first-run prefill/decode "
            "timings, and `total` adds those back in place of the ~0 ms cache lookup.",
            "",
            _fmt(
                t,
                pct=["retrieval_rate"],
                dec={c: 1 for c in t.columns if c not in ("arm", "n", "retrieval_rate")},
            ),
            *note,
        ]
    ) + "\n", tr


def _breakeven_one(tr: pd.DataFrame, arm: str, rate: float | None) -> tuple[str, pd.DataFrame]:
    a, b = tr[tr.arm == arm], tr[tr.arm == "always"]
    r = a.retrieved.mean()
    c_feat = a.lat_signals.mean()
    c_t0 = a.lat_tier0.mean()
    p_t1 = (a.lat_tier1 > 0).mean()
    c_t1 = a.loc[a.lat_tier1 > 0, "lat_tier1"].mean() if p_t1 > 0 else 0.0
    # per-query means of terms paid only on some queries = P(term) x C(term)
    t2 = a.lat_tier2.mean()  # P(T2) x C_T2
    cache = a.lat_cache.mean()
    ver = a.lat_verify.mean()  # P(SKIP verified) x C_ver
    p_esc = a.escalated.mean()
    esc = a.lat_escalation.mean()  # P(esc) x (C_ret + C_gen)
    lhs = c_feat + c_t0 + p_t1 * (c_t1 or 0.0) + t2 + cache + ver + esc
    # Savings per query vs always: vector-store time not spent, plus prefill of memory tokens not
    # injected (counts both skips and SHALLOW-instead-of-DEEP).
    store_saved = b.lat_retrieval.mean() - a.lat_retrieval.mean()
    tok_saved = b.memory_tokens.mean() - a.memory_tokens.mean()
    prefill_saved_meas = b.lat_prefill.mean() - a.lat_prefill.mean()
    rhs_meas = store_saved + prefill_saved_meas
    rows = [
        {"term": "C_feat (signals, excl. embedding)", "value": c_feat},
        {"term": "C_T0", "value": c_t0},
        {"term": "P(T1)", "value": p_t1},
        {"term": "C_T1 (when fired)", "value": c_t1},
        {"term": "P(T2) x C_T2 (Tier-2 ms per query)", "value": t2},
        {"term": "decision-cache ms per query", "value": cache},
        {"term": "verify ms per query (P(SKIP) x C_ver)", "value": ver},
        {"term": "P(esc)", "value": p_esc},
        {"term": "P(esc) x (C_ret + C_gen) (escalation ms per query)", "value": esc},
        {"term": "r (retrieval rate, escalations included)", "value": r},
        {"term": "memory tokens saved / query (vs always)", "value": tok_saved},
        {"term": "store ms saved / query", "value": store_saved},
        {"term": "prefill ms saved / query (measured, prompt-cached)", "value": prefill_saved_meas},
        {"term": "LHS: gate + verify + escalation overhead per query (ms)", "value": lhs},
        {"term": "RHS measured: store + prefill saved (ms)", "value": rhs_meas},
    ]
    verdicts = [
        f"measured: {lhs:.3f} ms overhead vs {rhs_meas:.1f} ms saved "
        f"({'holds' if lhs < rhs_meas else 'does NOT hold'})"
    ]
    if rate is not None:
        rhs_unc = store_saved + tok_saved * rate
        rows.append({"term": "RHS uncached estimate: store + tokens x rate (ms)", "value": rhs_unc})
        verdicts.append(
            f"uncached estimate: {rhs_unc:.1f} ms saved "
            f"({'holds' if lhs < rhs_unc else 'does NOT hold'})"
        )
    summary = (
        f"**`{arm}`: " + "; ".join(verdicts) + ".** "
        f"End-to-end p50 total {a.lat_total.median():.0f} ms vs always {b.lat_total.median():.0f} ms; "
        f"mean prompt tokens {a.prompt_tokens.mean():.0f} vs {b.prompt_tokens.mean():.0f}."
    )
    return summary, pd.DataFrame(rows)


def section_breakeven(tr: pd.DataFrame | None) -> str:
    if tr is None or "always" not in set(tr.arm):
        return ""
    gated = sorted(a for a in set(tr.arm) if a.startswith("amic") or a.endswith("_esc"))
    if not gated:
        return ""
    rate = _uncached_prefill_ms_per_token()
    lines = [
        "## 6. Break-even inequality (measured)",
        "",
        "C_emb cancels (both arms embed the query once; plan R1). Every overhead term is measured "
        "from the traces: terms paid on only some queries (Tier 2, verification, escalation) are "
        "averaged over all queries, i.e. P(term) x C(term). An escalation's retry (retrieval + "
        "regeneration) counts in full on the overhead side. Savings are per query against "
        "`always` on the same probes and seeds. See the prefill caveat in section 5.",
    ]
    for arm in gated:
        summary, rows = _breakeven_one(tr, arm, rate)
        lines += ["", f"### {arm}", "", _fmt(rows, dec={"value": 4}), "", summary]
    return "\n".join(lines) + "\n"


def _pairwise(run: str) -> pd.DataFrame | None:
    p = Path("runs") / run / "pairwise_vs_always.jsonl"
    if not p.exists():
        return None
    pw = pd.read_json(p, lines=True)
    return None if pw.empty else pw


def _paired_rows() -> pd.DataFrame | None:
    rows = []
    for a, b in PAIRS:
        pa, pb = _pairwise(a), _pairwise(b)
        if pa is None or pb is None:
            continue
        d, lo, hi, n = paired_diff_ci(pa, pb, "learner")
        _, plo, phi, _ = paired_diff_ci(pa, pb, "probe")
        rows.append(
            {
                "comparison": f"{a} − {b}",
                "pairs": n,
                "diff (95% CI, probe)": fmt_ci(d, plo, phi),
                "diff (95% CI, learner)": fmt_ci(d, lo, hi),
                # conservative: both clusterings must agree (8 learners undercover on their own)
                "both CIs exclude 0": (
                    "yes" if (lo > 0 and plo > 0) or (hi < 0 and phi < 0) else "no"
                ),
            }
        )
    return pd.DataFrame(rows) if rows else None


def section_quality() -> str:
    rows = []
    for d in sorted(Path("runs").glob("*/pairwise_vs_always.jsonl")):
        pw = pd.read_json(d, lines=True)
        if pw.empty:
            continue
        per_seed = pw.groupby("seed").result.value_counts(normalize=True).unstack(fill_value=0)
        for c in ("win", "tie", "loss"):
            if c not in per_seed:
                per_seed[c] = 0.0
        judged = pw[~pw.identical]
        rows.append(
            {
                "run": d.parent.name,
                "n": len(pw),
                "seeds": pw.seed.nunique(),
                "win": f"{per_seed.win.mean():.1%} ± {per_seed.win.std(ddof=1) if len(per_seed) > 1 else 0:.1%}",
                "tie": f"{per_seed.tie.mean():.1%} ± {per_seed.tie.std(ddof=1) if len(per_seed) > 1 else 0:.1%}",
                "loss": f"{per_seed.loss.mean():.1%} ± {per_seed.loss.std(ddof=1) if len(per_seed) > 1 else 0:.1%}",
                "net (95% CI, probe)": fmt_ci(*net_win_ci(pw, "probe")),
                "net (95% CI, learner)": fmt_ci(*net_win_ci(pw, "learner")),
                "identical_to_always": f"{pw.identical.mean():.1%}",
                "fabricated (run)": f"{judged.fab_run.mean():.1%}" if len(judged) else "",
                "fabricated (always)": f"{judged.fab_base.mean():.1%}" if len(judged) else "",
            }
        )
    if not rows:
        return ""
    by_route = []
    for d in sorted(Path("runs").glob("*/pairwise_vs_always.jsonl")):
        pw = pd.read_json(d, lines=True)
        if pw.empty or "route" not in pw:
            continue
        for route, g in pw.groupby("route"):
            vc = g.result.value_counts(normalize=True)
            by_route.append(
                {
                    "run": d.parent.name,
                    "route chosen": route,
                    "n": len(g),
                    "win": vc.get("win", 0.0),
                    "tie": vc.get("tie", 0.0),
                    "loss": vc.get("loss", 0.0),
                    "identical_to_always": g.identical.mean(),
                }
            )
    lines = [
        "## 7. Response quality: pairwise LLM judge vs `always` (test split)",
        "",
        "Mean ± sd across tutor sampling seeds (T = 0.7). Identical answers count as ties without a "
        "judge call. Fabrication rates are over judged (non-identical) pairs. `net` = win rate "
        "minus loss rate (all seeds pooled) with 95% CIs clustered by probe (the 3 seeds of a "
        "probe move together) and by learner. " + CI_NOTE,
        "",
        pd.DataFrame(rows).to_markdown(index=False),
    ]
    paired = _paired_rows()
    if paired is not None:
        lines += [
            "",
            "### Paired comparisons (same probes and seeds)",
            "",
            "`diff` = net win of the first run minus net win of the second, on the (probe, seed) "
            "pairs both runs answered. A difference counts as real only when BOTH its probe- and "
            "learner-clustered CIs exclude 0 (decision rule fixed before the extension runs; "
            "docs/PLAN.md).",
            "",
            paired.to_markdown(index=False),
        ]
    if by_route:
        lines += [
            "",
            "### By the route each run chose (all seeds pooled)",
            "",
            "`always` routes every query DEEP, so a DEEP choice reproduces its answer (tie). The "
            "SHALLOW and SKIP rows are where the runs differ from `always`.",
            "",
            _fmt(
                pd.DataFrame(by_route),
                pct=["win", "tie", "loss", "identical_to_always"],
            ),
        ]
    return "\n".join(lines) + "\n"


ESC_RUNS = [("amic_esc", "amic"), ("amic_lam1_esc", "amic_lam1"), ("never_esc", "never")]


def section_escalation(tr: pd.DataFrame | None) -> str:
    """Verify-and-escalate (P6 / M7): live runs (E1-E3) and the verifier-mode replay (E4)."""
    from amic.eval.escalation import MODES, replay_escalation
    from amic.eval.judgecache import self_consistency

    lab = pd.read_parquet("data/edu/labels.parquet").set_index("probe_id").label_required
    live = []
    for run, base in ESC_RUNS:
        p = Path("runs") / run / "answers.jsonl"
        if not p.exists():
            continue
        a = pd.read_json(p, lines=True)
        esc = a.escalated.astype(bool)
        skip = a.route == "SKIP"
        need = a.probe_id.map(lab).astype(bool)
        row: dict[str, Any] = {
            "run": run,
            "n": len(a),
            "skip_rate": skip.mean(),
            "escalation_rate": esc.mean(),
            "escalated_of_skips": esc[skip].mean() if skip.any() else float("nan"),
            # of escalated answers, how many the oracle label says needed memory
            "esc_precision_vs_label": need[esc].mean() if esc.any() else float("nan"),
            # of SKIPs the label says needed memory, how many were rescued
            "esc_recall_vs_label": esc[skip & need].mean() if (skip & need).any() else float("nan"),
        }
        if tr is not None and run in set(tr.arm):
            g = tr[tr.arm == run]
            row["memory_tokens"] = g.memory_tokens.mean()
            row["verify_ms_p50 (SKIPs)"] = g[g.route == "SKIP"].lat_verify.median()
            row["retry_ms_p50"] = g[g.escalated].lat_escalation.median() if g.escalated.any() else 0
        pw, pb = _pairwise(run), _pairwise(base)
        if pw is not None:
            row["net_vs_always (95% CI, learner)"] = fmt_ci(*net_win_ci(pw, "learner"))
        if pw is not None and pb is not None:
            d, lo, hi, _ = paired_diff_ci(pw, pb, "learner")
            row["diff vs base (learner CI)"] = f"vs {base}: " + fmt_ci(d, lo, hi)
        live.append(row)

    replay_rows = []
    for base in ("amic", "amic_lam1", "never"):
        pb = _pairwise(base)
        for mode in MODES:
            m = replay_escalation(base, mode)
            if m is None or pb is None:
                continue
            d, lo, hi, _ = paired_diff_ci(m, pb, "learner")
            _, plo, phi, _ = paired_diff_ci(m, pb, "probe")
            replay_rows.append(
                {
                    "base run": base,
                    "verifier": mode,
                    "escalation_rate": m.escalated.mean(),
                    "net_vs_always": fmt_ci(*net_win_ci(m, "learner")),
                    "gain vs base (probe CI)": fmt_ci(d, plo, phi),
                    "gain vs base (learner CI)": fmt_ci(d, lo, hi),
                }
            )
    if not live and not replay_rows:
        return ""
    n_sc, agree = self_consistency(["amic", "amic_lam1", "amic_no_s3", "oracle", "always_shallow"])
    lines = [
        "## 9. Verify-and-escalate (extension; pre-registered E1-E4)",
        "",
        "Answers generated without memory (SKIP) are checked by the ResponseVerifier (lexical "
        "patterns for invented history or deferral, then the LLM verifier); a failed check retries "
        "ONCE with SHALLOW memory. Escalated answers are byte-identical to the always-SHALLOW answer "
        "for the same probe and seed (same prompt and seed; generation cache), and identical answer "
        "pairs reuse the earlier pairwise verdict (docs/PLAN.md section 9), so differences against "
        "the base run come only from the escalated rows. " + CI_NOTE,
    ]
    if live:
        lines += [
            "",
            "### Live runs",
            "",
            _fmt(
                pd.DataFrame(live),
                pct=[
                    "skip_rate",
                    "escalation_rate",
                    "escalated_of_skips",
                    "esc_precision_vs_label",
                    "esc_recall_vs_label",
                ],
                dec={"memory_tokens": 0, "verify_ms_p50 (SKIPs)": 1, "retry_ms_p50": 1},
            ),
        ]
    if replay_rows:
        lines += [
            "",
            "### Verifier mode (offline replay of the same answers, E4)",
            "",
            _fmt(pd.DataFrame(replay_rows), pct=["escalation_rate"]),
        ]
    if n_sc:
        lines += [
            "",
            f"Judge self-consistency: {n_sc} identical answer pairs were judged twice by "
            f"different runs (independent A/B order); the two verdicts agree on {agree:.1%}. "
            "This is the noise floor of a single pairwise judgment.",
        ]
    return "\n".join(lines) + "\n"


def section_memory_system() -> str:
    """M2 facts, the async write path and the decision cache (pre-registered F1 stats, W1, C1)."""
    lines: list[str] = []
    done = Path("data/edu/facts_done.jsonl")
    if done.exists():
        f = pd.read_json(done, lines=True)
        lines += [
            "### M2 fact extraction (seeded education histories)",
            "",
            f"{len(f)} past sessions consolidated by `scripts/extract_facts.py` (one LLM call per "
            f"session, salient exchanges only). Facts extracted: {int(f['facts'].sum())} "
            f"({f['facts'].mean():.1f} per session). Resolution: inserted {int(f['insert'].sum())}, "
            f"superseded an older fact {int(f['update'].sum())}, refreshed {int(f['refresh'].sum())}, "
            f"skipped as duplicates {int(f['skip'].sum())}. Superseded facts are closed (validity "
            "interval ended), never deleted; routes read current facts only.",
            "",
        ]
    wp = Path("runs/writepath_bench.json")
    if wp.exists():
        j = json.loads(wp.read_text())
        rows = []
        for name, r in j.items():
            for m in ("inline_write_ms", "enqueue_ms", "enqueue_to_visible_ms"):
                rows.append(
                    {
                        "variant": name,
                        "measure": m,
                        "n": r[m]["n"],
                        "p50 ms": r[m]["p50"],
                        "p95 ms": r[m]["p95"],
                    }
                )
        lines += [
            "### Async write path (W1)",
            "",
            "Response-path cost of the long-term write: before, the whole consolidation ran inline "
            "after the answer (`inline_write_ms`); now the response path only enqueues "
            "(`enqueue_ms`, SQLite queue) and the ConsolidationWorker makes the turn visible "
            "later (`enqueue_to_visible_ms`: queue wait + consolidation). "
            "`scripts/bench_writepath.py`, real embedder and LLM, scratch Chroma collection.",
            "",
            _fmt(pd.DataFrame(rows), dec={"p50 ms": 1, "p95 ms": 1}),
            "",
        ]
    cr = Path("runs/cache_replay.json")
    if cr.exists():
        j = json.loads(cr.read_text())
        rows = []
        for nb in (8, 12, 16):
            r = j.get(f"bits_{nb}")
            if not r:
                continue
            nat, rep = r["natural"], r["repeat"]
            rows.append(
                {
                    "LSH bits": nb,
                    "natural: hit rate": nat["hits"] / max(1, nat["eligible"]),
                    "natural: wrong hits": nat["wrong_hits"],
                    # each probe is asked twice; coarse keys can also hit on a first ask
                    "repeat stream: hits / probes": f"{rep['hits']} / {rep['eligible'] // 2}",
                    "repeat: wrong hits": rep["wrong_hits"],
                    "false sharing (distinct queries)": r["false_sharing_rate"],
                }
            )
        lines += [
            "### Decision cache (C1, offline replay on the test split)",
            "",
            "The base gate replayed in the live runs' order with the cache in front of Tier 1 "
            "(Tier-0 decisions are never cached). `natural`: each test probe once, the order the "
            "live runs used; `repeat`: each probe asked twice in a row (a hit on every second ask = hits equal probes); `false sharing`: share of "
            "a learner's distinct queries that fall into a bucket an earlier query already used.",
            "",
            _fmt(
                pd.DataFrame(rows),
                pct=[
                    "natural: hit rate",
                    "false sharing (distinct queries)",
                ],
            ),
        ]
    if not lines:
        return ""
    title = "## 12. Memory system: M2 facts, write path, decision cache"
    return "\n".join([title, "", *lines]) + "\n"


def section_summary(tr: pd.DataFrame | None) -> str:
    """Token use vs quality for every live run, side by side (the efficiency/quality frontier)."""
    if tr is None or "always" not in set(tr.arm):
        return ""
    base_mem = tr[tr.arm == "always"].memory_tokens.mean()
    rows = []
    for arm, g in tr.groupby("arm"):
        pw = _pairwise(str(arm))
        net, ci = None, ""
        if pw is not None:
            net, lo, hi = net_win_ci(pw, "learner")
            ci = f"[{100 * lo:+.1f}%, {100 * hi:+.1f}%]"
        rows.append(
            {
                "run": arm,
                "retrieval_rate": g.retrieved.mean(),
                "memory_tokens": g.memory_tokens.mean(),
                "tokens_saved_vs_always": 1 - g.memory_tokens.mean() / base_mem,
                "net_win_vs_always": 0.0 if arm == "always" else net,
                "net_win_95ci_learner": ci,
            }
        )
    t = pd.DataFrame(rows).sort_values("memory_tokens", ascending=False)
    return (
        "\n".join(
            [
                "## 8. Summary: memory tokens vs answer quality (live runs, test split)",
                "",
                "`net_win_vs_always` = pairwise win rate minus loss rate against `always` (DEEP every "
                "turn); 0 = as good as `always`. A run is better than another only if it uses fewer "
                "memory tokens **and** has a higher net win rate.",
                "",
                _fmt(
                    t,
                    pct=["retrieval_rate", "tokens_saved_vs_always", "net_win_vs_always"],
                    dec={"memory_tokens": 0},
                ),
            ]
        )
        + "\n"
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    OUT.mkdir(parents=True, exist_ok=True)
    parts = [
        "# AMIC results",
        "",
        "_Generated by `python -m amic.eval.report`. Every number comes from this repository's "
        "data and runs; regenerate rather than edit._",
        "",
    ]
    parts.append(section_dataset())
    dec_md, _ = section_decisions()
    parts.append(dec_md)
    parts.append(section_lambda_sweep())
    parts.append(section_calibration())
    eff_md, tr = section_efficiency()
    parts.append(eff_md)
    parts.append(section_breakeven(tr))
    parts.append(section_quality())
    parts.append(section_summary(tr))
    parts.append(section_escalation(tr))
    parts.append(section_tier2())
    parts.append(section_longmemeval())
    parts.append(section_memory_system())
    md = "\n".join(p for p in parts if p)
    (OUT / "main.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
