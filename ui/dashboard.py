"""AMIC usage dashboard: live token usage per model, and what the gate saves.

    streamlit run ui/dashboard.py

Reads the DecisionTrace store (settings.metrics_db_url) and refreshes on a timer. Feed it with
the API (`uvicorn amic.service.api:app`, AMIC_TELEMETRY__SHADOW_BASELINE=true) or with
`python scripts/simulate_traffic.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from amic.contracts.config import Settings
from amic.telemetry import usage

st.set_page_config(page_title="AMIC usage", layout="wide")

# Categorical slots in fixed order (validated reference palette); colour follows the model
# name's position among ALL models in the store, so filtering never repaints a survivor.
SERIES = {
    "light": [
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ],
    "dark": [
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#0ca30c",
        "#9085e9",
        "#e66767",
    ],
}
INK = {
    "light": {"secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}
WINDOWS = {
    "Last 15 minutes": timedelta(minutes=15),
    "Last hour": timedelta(hours=1),
    "Last 24 hours": timedelta(days=1),
    "Last 7 days": timedelta(days=7),
    "All time": None,
}


def mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def fmt_int(x: float) -> str:
    return f"{x:,.0f}"


def fmt_usd(x: float) -> str:
    return f"${x:,.4f}" if abs(x) < 1 else f"${x:,.2f}"


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def base_layout(fig: go.Figure, ink: dict[str, str], y_title: str) -> go.Figure:
    fig.update_layout(
        height=320,
        margin={"l": 8, "r": 96, "t": 8, "b": 8},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": None},
        barcornerradius=4,
    )
    fig.update_xaxes(showgrid=False, linecolor=ink["axis"], title=None)
    fig.update_yaxes(
        gridcolor=ink["grid"], gridwidth=1, zeroline=False, title=y_title, rangemode="tozero"
    )
    return fig


def end_label(fig: go.Figure, x: object, y: float, text: str, ink: dict[str, str]) -> None:
    """Direct label at a line's end, in text ink (the line's colour carries identity)."""
    fig.add_annotation(
        x=x,
        y=y,
        text=text,
        xanchor="left",
        xshift=6,
        showarrow=False,
        font={"color": ink["secondary"], "size": 12},
    )


# ------------------------------------------------------------------ sidebar
settings = Settings()
with st.sidebar:
    st.header("AMIC usage")
    db_url = st.text_input("Trace store", settings.metrics_db_url)
    window = st.selectbox("Time range", list(WINDOWS), index=1)
    live = st.toggle("Live", value=True)
    every = st.select_slider("Refresh every (s)", [2, 5, 10, 30], value=5, disabled=not live)

    st.subheader("Prices (USD per 1M tokens)")
    file_prices = usage.load_prices()
    try:
        seen = sorted(usage.load_traces(db_url)["model"].unique())
    except Exception:
        seen: list[str] = []
    models_all = sorted(set(file_prices) | set(seen))
    price_df = pd.DataFrame(
        {
            "model": models_all,
            "input": [file_prices.get(m, usage.Price()).input_per_mtok for m in models_all],
            "output": [file_prices.get(m, usage.Price()).output_per_mtok for m in models_all],
        }
    )
    edited = st.data_editor(
        price_df, hide_index=True, disabled=["model"], width="stretch", key="prices"
    )

    def _price(model: str, p_in: float, p_out: float) -> usage.Price:
        # still a placeholder only if it came from the file as one and was not edited here
        f = file_prices.get(model)
        unchanged = f is not None and (p_in, p_out) == (f.input_per_mtok, f.output_per_mtok)
        return usage.Price(p_in, p_out, placeholder=bool(f and f.placeholder and unchanged))

    prices = {
        r.model: _price(r.model, float(r.input), float(r.output)) for r in edited.itertuples()
    }
    st.caption("Edits apply to this browser session. Defaults: `configs/pricing.yaml`.")

color_of = {m: SERIES[mode()][i % 8] for i, m in enumerate(models_all)}


# ------------------------------------------------------------------ body
@st.fragment(run_every=every if live else None)
def body() -> None:
    m_ = mode()
    ink = INK[m_]
    span = WINDOWS[window]
    since = datetime.now(UTC) - span if span else None
    try:
        raw = usage.load_traces(db_url, since)
    except Exception as e:  # unreadable store: say so instead of a stack trace
        st.error(f"Could not read traces from `{db_url}`: {e}")
        return

    arms = sorted(raw["arm"].unique()) if not raw.empty else []
    sel_arms = st.multiselect("Arms", arms, default=arms, key="arms") if arms else []
    df = usage.with_costs(raw[raw["arm"].isin(sel_arms)], prices) if arms else raw
    if df.empty:
        st.info(
            "No requests in this time range yet. Send traffic with "
            "`python scripts/simulate_traffic.py`, or call `POST /chat` on the API."
        )
        return

    s = usage.summary(df)
    k = st.columns(5)
    k[0].metric("Requests", fmt_int(s["requests"]))
    k[1].metric(
        "Tokens used",
        fmt_int(s["total_tokens"]),
        help=f"{fmt_int(s['prompt_tokens'])} prompt + {fmt_int(s['completion_tokens'])} completion",
    )
    k[2].metric(
        "Memory tokens saved",
        fmt_int(s["saved_tokens"]),
        delta=f"{fmt_pct(s['saved_pct'])} vs always-retrieve"
        if s["saved_pct"] is not None
        else None,
        delta_color="off",
        help="Estimated (~4 chars/token) memory-context tokens the always-retrieve baseline "
        "would have added, minus what AMIC injected. Measured requests only.",
    )
    k[3].metric(
        "Retrievals skipped",
        fmt_int(s["skipped"]),
        delta=f"{fmt_pct(s['skipped_pct'])} of requests",
        delta_color="off",
    )
    k[4].metric(
        "Spend",
        fmt_usd(s["spend"]),
        delta=f"{fmt_usd(s['saved_usd'])} saved ({fmt_pct(s['saved_usd_pct'])})",
        delta_color="off",
    )

    notes = []
    if s["measured"] < s["requests"]:
        notes.append(
            f"Savings are measured on {s['measured']:,} of {s['requests']:,} requests; the rest "
            "were served with `telemetry.shadow_baseline` off."
        )
    if any(prices.get(m, usage.Price(placeholder=True)).placeholder for m in df["model"].unique()):
        notes.append(
            "Dollar figures use placeholder prices, not quotes. Set real rates in "
            "`configs/pricing.yaml` or the sidebar."
        )
    unpriced = sorted(
        m
        for m in df["model"].unique()
        if prices.get(m, usage.Price()).input_per_mtok == 0
        and prices.get(m, usage.Price()).output_per_mtok == 0
    )
    if unpriced:
        notes.append(
            f"No price set for {', '.join(unpriced)}, so their spend and savings count as $0. "
            "Add them in the sidebar or `configs/pricing.yaml`."
        )
    for n in notes:
        st.caption(n)

    left, right = st.columns(2)
    with left:
        st.subheader("Tokens by model")
        tl_span = df["ts"].max() - df["ts"].min()
        freq = usage.bucket_for(span or tl_span)
        tl = usage.timeline(df, freq)
        fig = go.Figure()
        models = sorted(tl["model"].unique())
        for name in models:
            d = tl[tl["model"] == name]
            fig.add_trace(
                go.Scatter(
                    x=d["ts"],
                    y=d["total_tokens"],
                    name=name,
                    mode="lines",
                    line={"color": color_of.get(name, ink["muted"]), "width": 2},
                    hovertemplate="%{y:,.0f} tokens<extra>" + name + "</extra>",
                )
            )
            if len(models) <= 4 and len(d):
                end_label(fig, d["ts"].iloc[-1], float(d["total_tokens"].iloc[-1]), name, ink)
        st.plotly_chart(
            base_layout(fig, ink, f"tokens per {freq}"), width="stretch", theme="streamlit"
        )

    with right:
        st.subheader("Memory tokens: AMIC vs always-retrieve")
        cm = usage.cumulative_memory(df)
        if cm.empty:
            st.caption("No measured baselines in this range. Turn on `telemetry.shadow_baseline`.")
        else:
            fig = go.Figure()
            for col, name, color in (
                ("baseline", "Always retrieve", ink["muted"]),
                ("amic", "AMIC", SERIES[m_][0]),
            ):
                fig.add_trace(
                    go.Scatter(
                        x=cm["ts"],
                        y=cm[col],
                        name=name,
                        mode="lines",
                        line={"color": color, "width": 2, "shape": "hv"},
                        hovertemplate="%{y:,.0f} tokens<extra>" + name + "</extra>",
                    )
                )
                end_label(fig, cm["ts"].iloc[-1], float(cm[col].iloc[-1]), name, ink)
            st.plotly_chart(
                base_layout(fig, ink, "cumulative memory tokens"),
                width="stretch",
                theme="streamlit",
            )

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Routes")
        order = ["SKIP", "SHALLOW", "DEEP", "WRITE"]
        counts = df["route"].value_counts().reindex(order).dropna()
        fig = go.Figure(
            go.Bar(
                y=counts.index,
                x=counts.values,
                orientation="h",
                marker={"color": SERIES[m_][0]},
                text=[f"{v:,.0f} ({v / len(df):.0%})" for v in counts.values],
                textposition="outside",
                cliponaxis=False,
                textfont={"color": ink["secondary"]},
                hovertemplate="%{y}: %{x:,.0f} requests<extra></extra>",
            )
        )
        lay = base_layout(fig, ink, "")
        lay.update_layout(hovermode="closest", height=220, showlegend=False)
        lay.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        lay.update_xaxes(showgrid=True, gridcolor=ink["grid"])
        st.plotly_chart(lay, width="stretch", theme="streamlit")
    with right:
        st.subheader("By model")
        bm = usage.by_model(df)
        st.dataframe(
            bm,
            hide_index=True,
            width="stretch",
            column_config={
                "model": "Model",
                "requests": st.column_config.NumberColumn("Requests", format="%d"),
                "prompt_tokens": st.column_config.NumberColumn("Prompt tok", format="localized"),
                "completion_tokens": st.column_config.NumberColumn(
                    "Completion tok", format="localized"
                ),
                "memory_tokens": st.column_config.NumberColumn("Memory tok", format="localized"),
                "saved_tokens": st.column_config.NumberColumn(
                    "Memory tok saved", format="localized"
                ),
                "skipped_pct": st.column_config.NumberColumn("Skipped", format="percent"),
                "spend": st.column_config.NumberColumn("Spend", format="$%.4f"),
                "saved_usd": st.column_config.NumberColumn("Saved", format="$%.4f"),
                "p50_ms": st.column_config.NumberColumn("p50 latency (ms)", format="%.0f"),
            },
        )

    st.subheader("Latest requests")
    recent = df.sort_values("ts", ascending=False).head(25)
    st.dataframe(
        recent[
            [
                "ts",
                "tenant_id",
                "model",
                "arm",
                "route",
                "resolved_by",
                "p",
                "memory_tokens",
                "baseline_memory_tokens",
                "total_tokens",
                "lat_total",
                "query",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "ts": st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
            "tenant_id": "Learner",
            "model": "Model",
            "arm": "Arm",
            "route": "Route",
            "resolved_by": "Decided by",
            "p": st.column_config.NumberColumn("p(memory)", format="%.2f"),
            "memory_tokens": st.column_config.NumberColumn("Memory tok", format="%d"),
            "baseline_memory_tokens": st.column_config.NumberColumn("Baseline tok", format="%d"),
            "total_tokens": st.column_config.NumberColumn("Tokens", format="%d"),
            "lat_total": st.column_config.NumberColumn("Total ms", format="%.0f"),
            "query": st.column_config.TextColumn("Query", width="large"),
        },
    )
    st.caption(
        f"Updated {datetime.now().strftime('%H:%M:%S')}. Prompt and completion tokens are "
        "reported by the model; memory tokens are estimated. Cost assumes the always-retrieve "
        "answer would be the same length."
    )


body()
