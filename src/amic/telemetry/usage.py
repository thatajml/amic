"""Token and cost accounting over DecisionTraces (the live dashboard reads through this).

Traces store token counts only; prices are applied here, at read time, so editing a price never
rewrites history. Per request, with prices in USD per 1M tokens:

    saved_tokens   = baseline_memory_tokens - memory_tokens      (None when not measured)
    spend          = prompt_tokens * in + completion_tokens * out
    baseline_spend = (prompt_tokens + saved_tokens) * in + completion_tokens * out
    saved_usd      = baseline_spend - spend

Assumptions, surfaced in the UI: the always-retrieve answer is taken to have the same completion
length, and memory token counts are the assembler's estimate (~4 chars/token), while prompt and
completion counts are reported by the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sqlalchemy import select

from amic.telemetry.store import SqliteTelemetry, traces

PRICING_PATH = Path("configs/pricing.yaml")
RETRIEVE_ROUTES = ("SHALLOW", "DEEP")
_COLS = [
    "request_id",
    "ts",
    "tenant_id",
    "arm",
    "model",
    "query",
    "route",
    "resolved_by",
    "p",
    "prompt_tokens",
    "completion_tokens",
    "memory_tokens",
    "baseline_memory_tokens",
    "lat_total",
    "lat_retrieval",
    "lat_generation",
]


@dataclass(frozen=True)
class Price:
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    placeholder: bool = False  # True = not a real quote; the UI says so


def load_prices(path: str | Path = PRICING_PATH) -> dict[str, Price]:
    p = Path(path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {name: Price(**(v or {})) for name, v in (raw.get("models") or {}).items()}


def load_traces(db_url: str, since: datetime | None = None) -> pd.DataFrame:
    """Read traces (newest last). Opening through SqliteTelemetry applies pending migrations."""
    engine = SqliteTelemetry(db_url).engine
    q = select(*[traces.c[c] for c in _COLS]).order_by(traces.c.ts)
    if since is not None:
        q = q.where(traces.c.ts >= since.replace(tzinfo=None))
    df = pd.read_sql(q, engine)
    return normalize(df)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["model"] = df["model"].fillna("").replace("", "(unrecorded)")
    for c in ("prompt_tokens", "completion_tokens", "memory_tokens"):
        df[c] = df[c].fillna(0).astype("int64")
    df["baseline_memory_tokens"] = df["baseline_memory_tokens"].astype("Float64")
    return df


def with_costs(df: pd.DataFrame, prices: dict[str, Price]) -> pd.DataFrame:
    df = df.copy()
    p_in = df["model"].map(lambda m: prices.get(m, Price()).input_per_mtok).astype(float) / 1e6
    p_out = df["model"].map(lambda m: prices.get(m, Price()).output_per_mtok).astype(float) / 1e6
    df["total_tokens"] = df["prompt_tokens"] + df["completion_tokens"]
    df["saved_tokens"] = df["baseline_memory_tokens"] - df["memory_tokens"]
    df["spend"] = df["prompt_tokens"] * p_in + df["completion_tokens"] * p_out
    df["baseline_spend"] = (df["prompt_tokens"] + df["saved_tokens"]) * p_in + df[
        "completion_tokens"
    ] * p_out
    df["saved_usd"] = df["baseline_spend"] - df["spend"]
    df["retrieved"] = df["route"].isin(RETRIEVE_ROUTES)
    return df


def summary(df: pd.DataFrame) -> dict[str, Any]:
    """Headline numbers. Savings are over the requests whose baseline was measured."""
    m = df[df["baseline_memory_tokens"].notna()]
    base_mem = float(m["baseline_memory_tokens"].sum())
    saved = float(m["saved_tokens"].sum())
    base_spend = float(m["baseline_spend"].sum())
    saved_usd = float(m["saved_usd"].sum())
    n = len(df)
    return {
        "requests": n,
        "prompt_tokens": int(df["prompt_tokens"].sum()),
        "completion_tokens": int(df["completion_tokens"].sum()),
        "total_tokens": int(df["total_tokens"].sum()),
        "memory_tokens": int(df["memory_tokens"].sum()),
        "measured": len(m),
        "baseline_memory_tokens": base_mem,
        "saved_tokens": saved,
        "saved_pct": saved / base_mem if base_mem else None,
        "skipped": int((df["route"] == "SKIP").sum()),
        "skipped_pct": float((df["route"] == "SKIP").mean()) if n else None,
        "retrievals": int(df["retrieved"].sum()),
        "spend": float(df["spend"].sum()),
        "saved_usd": saved_usd,
        "saved_usd_pct": saved_usd / base_spend if base_spend else None,
    }


def by_model(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("model", sort=True)
    out = pd.DataFrame(
        {
            "requests": g.size(),
            "prompt_tokens": g["prompt_tokens"].sum(),
            "completion_tokens": g["completion_tokens"].sum(),
            "memory_tokens": g["memory_tokens"].sum(),
            "saved_tokens": g["saved_tokens"].sum(min_count=1),
            "skipped_pct": g["route"].apply(lambda r: float((r == "SKIP").mean())),
            "spend": g["spend"].sum(),
            "saved_usd": g["saved_usd"].sum(min_count=1),
            "p50_ms": g["lat_total"].median(),
        }
    )
    return out.reset_index()


def timeline(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Total tokens per time bucket per model; empty buckets are zero (a gap, not a dip)."""
    if df.empty:
        return pd.DataFrame(columns=["ts", "model", "total_tokens"])
    t = df.assign(bucket=df["ts"].dt.floor(freq))
    wide = t.pivot_table(
        index="bucket", columns="model", values="total_tokens", aggfunc="sum", fill_value=0
    )
    full = pd.date_range(wide.index.min(), wide.index.max(), freq=freq, tz=wide.index.tz)
    wide = wide.reindex(full, fill_value=0)
    wide.index.name = "ts"
    return wide.reset_index().melt(id_vars="ts", var_name="model", value_name="total_tokens")


def cumulative_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Running memory tokens injected vs the always-retrieve baseline (measured rows only)."""
    m = df[df["baseline_memory_tokens"].notna()].sort_values("ts")
    return pd.DataFrame(
        {
            "ts": m["ts"],
            "amic": m["memory_tokens"].cumsum(),
            "baseline": m["baseline_memory_tokens"].astype(float).cumsum(),
        }
    )


def bucket_for(span: pd.Timedelta) -> str:
    """A bucket size giving roughly 30-90 points across the visible span."""
    for limit, freq in (
        (pd.Timedelta(minutes=20), "15s"),
        (pd.Timedelta(hours=1), "1min"),
        (pd.Timedelta(hours=6), "5min"),
        (pd.Timedelta(days=1), "15min"),
        (pd.Timedelta(days=7), "2h"),
    ):
        if span <= limit:
            return freq
    return "1D"
