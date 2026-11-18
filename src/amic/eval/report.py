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
from amic.control.thresholds import tau_star
from amic.control.tier1 import Tier1Model
from amic.eval.replay import (
    decision_metrics,
    fixed_policy,
    load_dataset,
    replay,
    savings_captured,
)

OUT = Path("docs/results")
LAMBDAS = [1.0, 2.0, 4.0, 8.0, 16.0]


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

        m = pd.read_csv(h).merge(df[["probe_id", "label_required"]], on="probe_id")
        if len(m) >= 10:
            k = cohen_kappa_score(m.human_required, m.label_required.astype(int))
            agree = (m.human_required == m.label_required.astype(int)).mean()
            lines += [
                "",
                f"Human vs judge labels (n={len(m)}): agreement {agree:.1%}, Cohen's kappa **{k:.3f}**.",
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


def _decision_rows(df: pd.DataFrame, lam: float) -> pd.DataFrame:
    rows = []
    base = {n: decision_metrics(fixed_policy(df, n), lam) for n in ("never", "always", "oracle")}
    for n in ("never", "always", "oracle"):
        rows.append({"policy": n, **base[n]})
    for name, over, model in VARIANTS:
        if model and not Path(model).exists():
            continue
        s = load_settings(**over)
        s.gate.lam = lam
        t1 = Tier1Model.load(model) if model else None
        m = decision_metrics(replay(df, s, t1), lam)
        m["savings_captured"] = savings_captured(m, base["always"], base["oracle"])
        rows.append({"policy": name, **m})
    return pd.DataFrame(rows)


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
        "`savings_captured` = share of the always->oracle retrieval reduction achieved.",
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
                lt[["policy", "retrieval_rate", "recall", "false_skip_rate", "precision", "cost"]],
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
    r = json.loads(p.read_text())
    rows = []
    for name in ("tier1", "tier1_no_s3", "tier1_lgbm"):
        q = Path(f"runs/{name}.json")
        if q.exists():
            j = json.loads(q.read_text())["test"]
            rows.append(
                {
                    "model": name,
                    "n_test": j["n"],
                    "auc": j["auc"],
                    "brier": j["brier"],
                    "ece": j["ece"],
                }
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
def _traces() -> pd.DataFrame | None:
    db = Path("data/experiments.db")
    if not db.exists():
        return None
    from sqlalchemy import create_engine

    return pd.read_sql_table("traces", create_engine(f"sqlite:///{db}"))


def section_efficiency() -> tuple[str, pd.DataFrame | None]:
    tr = _traces()
    if tr is None or tr.empty:
        return "## 5. Efficiency (live runs)\n\n_no experiment traces yet_\n", None
    tr["lat_gate"] = tr.lat_signals + tr.lat_cache + tr.lat_tier0 + tr.lat_tier1 + tr.lat_tier2
    tr["retrieved"] = tr.route.isin(["SHALLOW", "DEEP"])
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
    return "\n".join(
        [
            "## 5. Efficiency (live runs, test split, all seeds)",
            "",
            "Milliseconds. `gate` = signals + Tier 0 + Tier 1 (excludes the query embedding, which "
            "`always` pays too). Tutor calls are served from a prompt-keyed cache when an identical "
            "prompt+seed was already generated; cached entries keep their first-run timings.",
            "",
            _fmt(
                t,
                pct=["retrieval_rate"],
                dec={c: 1 for c in t.columns if c not in ("arm", "n", "retrieval_rate")},
            ),
        ]
    ) + "\n", tr


def section_breakeven(tr: pd.DataFrame | None) -> str:
    if tr is None or not {"always", "amic"} <= set(tr.arm):
        return ""
    a, b = tr[tr.arm == "amic"], tr[tr.arm == "always"]
    r = a.retrieved.mean()
    c_feat = a.lat_signals.mean()
    c_t0 = a.lat_tier0.mean()
    p_t1 = (a.lat_tier1 > 0).mean()
    c_t1 = a.loc[a.lat_tier1 > 0, "lat_tier1"].mean() if p_t1 > 0 else 0.0
    c_store = b.lat_retrieval.mean()
    # prefill cost of the memory block: always-arm prefill minus never-arm prefill (same probes)
    nv = tr[tr.arm == "never"]
    c_prefill = (b.lat_prefill.mean() - nv.lat_prefill.mean()) if len(nv) else float("nan")
    lhs = c_feat + c_t0 + p_t1 * (c_t1 or 0.0)
    rhs = (1 - r) * (c_store + (c_prefill if not np.isnan(c_prefill) else 0.0))
    rows = pd.DataFrame(
        [
            {"term": "C_feat (signals, excl. embedding)", "ms": c_feat},
            {"term": "C_T0", "ms": c_t0},
            {"term": "P(T1)", "ms": p_t1},
            {"term": "C_T1 (when fired)", "ms": c_t1},
            {"term": "P(T2), P(esc)", "ms": 0.0},
            {"term": "r (AMIC retrieval rate)", "ms": r},
            {"term": "C_ret store (always arm)", "ms": c_store},
            {"term": "C_ret prefill of memory block (always - never)", "ms": c_prefill},
            {"term": "LHS: gate overhead per query", "ms": lhs},
            {"term": "RHS: (1 - r) * (C_store + C_prefill)", "ms": rhs},
        ]
    )
    verdict = "holds" if lhs < rhs else "does NOT hold"
    return (
        "\n".join(
            [
                "## 6. Break-even inequality (measured)",
                "",
                "C_emb cancels (both arms embed the query once; plan R1). Tier 2 and escalation are out of "
                "scope, so P(T2) = P(esc) = 0.",
                "",
                _fmt(rows, dec={"ms": 4}),
                "",
                f"**The inequality {verdict}: {lhs:.3f} ms of gate overhead vs {rhs:.1f} ms saved per query.**",
                "",
                f"End-to-end p50 total: amic {a.lat_total.median():.0f} ms vs always {b.lat_total.median():.0f} ms; "
                f"mean prompt tokens {a.prompt_tokens.mean():.0f} vs {b.prompt_tokens.mean():.0f}.",
            ]
        )
        + "\n"
    )


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
                "identical_to_always": f"{pw.identical.mean():.1%}",
                "fabricated (run)": f"{judged.fab_run.mean():.1%}" if len(judged) else "",
                "fabricated (always)": f"{judged.fab_base.mean():.1%}" if len(judged) else "",
            }
        )
    if not rows:
        return ""
    return (
        "\n".join(
            [
                "## 7. Response quality: pairwise LLM judge vs `always` (test split)",
                "",
                "Mean ± sd across tutor sampling seeds (T = 0.7). Identical answers count as ties without a "
                "judge call. Fabrication rates are over judged (non-identical) pairs.",
                "",
                pd.DataFrame(rows).to_markdown(index=False),
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
    md = "\n".join(p for p in parts if p)
    (OUT / "main.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
