from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from amic.contracts.models import DecisionTrace, ResolvedBy, Route
from amic.telemetry import usage
from amic.telemetry.store import SqliteTelemetry, traces
from tests.conftest import T0


def _trace(i: int, route: Route, model: str, mem: int, base: int | None, **kw) -> DecisionTrace:
    return DecisionTrace(
        request_id=f"r{i}",
        ts=T0 + timedelta(seconds=10 * i),
        tenant_id="t",
        session_id="s",
        arm="amic",
        config_hash="h",
        query="q",
        route=route,
        resolved_by=ResolvedBy.TIER1,
        prompt_tokens=100 + mem,
        completion_tokens=50,
        memory_tokens=mem,
        model=model,
        baseline_memory_tokens=base,
        **kw,
    )


PRICES = {"big": usage.Price(1.0, 2.0), "small": usage.Price(0.5, 0.5)}


def _df(traces: list[DecisionTrace]) -> pd.DataFrame:
    rows = []
    for t in traces:
        d = t.model_dump(exclude={"latency", "features"})
        d["route"] = t.route.value
        d["resolved_by"] = t.resolved_by.value
        d["lat_total"] = d["lat_retrieval"] = d["lat_generation"] = 1.0
        rows.append(d)
    return usage.with_costs(usage.normalize(pd.DataFrame(rows)), PRICES)


def test_costs_and_savings_per_request():
    df = _df([_trace(0, Route.SKIP, "big", 0, 800), _trace(1, Route.DEEP, "big", 800, 800)])
    skip, deep = df.iloc[0], df.iloc[1]
    assert skip.saved_tokens == 800 and deep.saved_tokens == 0
    assert skip.spend == pytest.approx((100 * 1.0 + 50 * 2.0) / 1e6)
    assert skip.saved_usd == pytest.approx(800 * 1.0 / 1e6)  # memory is prompt (input) tokens


def test_summary_counts_only_measured_rows_for_savings():
    df = _df(
        [
            _trace(0, Route.SKIP, "big", 0, 600),
            _trace(1, Route.SHALLOW, "small", 200, 600),
            _trace(2, Route.SKIP, "small", 0, None),  # baseline not measured
        ]
    )
    s = usage.summary(df)
    assert s["requests"] == 3 and s["measured"] == 2
    assert s["saved_tokens"] == 1000 and s["saved_pct"] == pytest.approx(1000 / 1200)
    assert s["skipped"] == 2 and s["retrievals"] == 1
    assert s["saved_usd"] == pytest.approx((600 * 1.0 + 400 * 0.5) / 1e6)


def test_by_model_and_unpriced_model_costs_zero():
    df = _df([_trace(0, Route.SKIP, "big", 0, 500), _trace(1, Route.SKIP, "mystery", 0, 500)])
    bm = usage.by_model(df).set_index("model")
    assert list(bm.index) == ["big", "mystery"]
    assert bm.loc["mystery", "spend"] == 0 and bm.loc["big", "saved_tokens"] == 500


def test_timeline_fills_empty_buckets_with_zero():
    df = _df([_trace(0, Route.SKIP, "big", 0, 0), _trace(9, Route.SKIP, "big", 0, 0)])
    tl = usage.timeline(df, "30s")
    assert len(tl) == 4 and (tl["total_tokens"] == 0).sum() == 2


def test_cumulative_memory_is_monotone():
    df = _df(
        [
            _trace(i, Route.SKIP if i % 2 else Route.DEEP, "big", 0 if i % 2 else 700, 700)
            for i in range(6)
        ]
    )
    cm = usage.cumulative_memory(df)
    assert cm["baseline"].is_monotonic_increasing and cm["amic"].iloc[-1] == 3 * 700


def test_load_prices_reads_repo_file():
    prices = usage.load_prices()
    assert prices and all(p.input_per_mtok >= 0 for p in prices.values())


def test_store_migrates_old_db_and_annotates(tmp_path):
    db = tmp_path / "old.db"
    eng = create_engine(f"sqlite:///{db}")
    with eng.begin() as conn:  # a table from before `model`/`baseline_memory_tokens` existed
        conn.execute(text("CREATE TABLE traces (request_id VARCHAR PRIMARY KEY, ts DATETIME)"))
    tel = SqliteTelemetry(f"sqlite:///{db}")
    cols = {r[1] for r in tel.engine.connect().execute(text("PRAGMA table_info(traces)"))}
    assert {c.name for c in traces.columns} <= cols  # incl. later latency fields (lat_prefill...)
    tel.record(_trace(1, Route.SKIP, "big", 0, None))  # the migrated table accepts a full row

    tel2 = SqliteTelemetry(f"sqlite:///{tmp_path / 'new.db'}")
    tel2.record(_trace(0, Route.SKIP, "big", 0, None))
    tel2.annotate("r0", baseline_memory_tokens=640)
    got = usage.load_traces(f"sqlite:///{tmp_path / 'new.db'}")
    assert got.loc[0, "baseline_memory_tokens"] == 640 and got.loc[0, "model"] == "big"
