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

        m = pd.read_csv(h).merge(
            df[["probe_id", "label_required", "intended_memory"]], on="probe_id"
        )
        if len(m) >= 10:
            judge = m.label_required.astype(int)
            k = cohen_kappa_score(m.human_required, judge)
            agree = (m.human_required == judge).mean()
            k_hi = cohen_kappa_score(m.human_required, m.intended_memory.astype(int))
            k_ji = cohen_kappa_score(df.intended_memory.astype(int), df.label_required.astype(int))
            lines += [
                "",
                f"Human vs judge labels (n={len(m)}): agreement {agree:.1%}, Cohen's kappa **{k:.3f}**. "
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
    tr["retrieved"] = tr.route.isin(["SHALLOW", "DEEP"])
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
    lhs = c_feat + c_t0 + p_t1 * (c_t1 or 0.0)
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
        {"term": "P(T2), P(esc)", "value": 0.0},
        {"term": "r (retrieval rate)", "value": r},
        {"term": "memory tokens saved / query (vs always)", "value": tok_saved},
        {"term": "store ms saved / query", "value": store_saved},
        {"term": "prefill ms saved / query (measured, prompt-cached)", "value": prefill_saved_meas},
        {"term": "LHS: gate overhead per query (ms)", "value": lhs},
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
    gated = sorted(a for a in set(tr.arm) if a.startswith("amic"))
    if not gated:
        return ""
    rate = _uncached_prefill_ms_per_token()
    lines = [
        "## 6. Break-even inequality (measured)",
        "",
        "C_emb cancels (both arms embed the query once; plan R1). Tier 2 and escalation are out of "
        "scope, so P(T2) = P(esc) = 0. Savings are per query against `always` on the same probes "
        "and seeds. See the prefill caveat in section 5.",
    ]
    for arm in gated:
        summary, rows = _breakeven_one(tr, arm, rate)
        lines += ["", f"### {arm}", "", _fmt(rows, dec={"value": 4}), "", summary]
    return "\n".join(lines) + "\n"


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
        "judge call. Fabrication rates are over judged (non-identical) pairs.",
        "",
        pd.DataFrame(rows).to_markdown(index=False),
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
