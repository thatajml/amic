"""DecisionTrace persistence (P7). SQLite via SQLAlchemy; latency flattened for SQL queries."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    select,
)

from amic.contracts.models import DecisionTrace, LatencySplit

_LAT_FIELDS = list(LatencySplit.model_fields)
_meta = MetaData()
traces = Table(
    "traces",
    _meta,
    Column("request_id", String, primary_key=True),
    Column("ts", DateTime(timezone=True), index=True),
    Column("tenant_id", String, index=True),
    Column("session_id", String),
    Column("arm", String, index=True),
    Column("config_hash", String, index=True),
    Column("query", Text),
    Column("route", String),
    Column("resolved_by", String),
    Column("p", Float),
    Column("tau_low", Float),
    Column("tau_high", Float),
    *[Column(f"lat_{f}", Float) for f in _LAT_FIELDS],
    Column("prompt_tokens", Integer),
    Column("completion_tokens", Integer),
    Column("memory_tokens", Integer),
    Column("n_items", Integer),
    Column("escalated", Boolean),
    Column("cache_hit", Boolean),
    Column("oracle_label", Boolean, nullable=True),
    Column("features", JSON),
)


def _row(t: DecisionTrace) -> dict[str, Any]:
    d = t.model_dump(mode="python", exclude={"latency", "features"})
    d["route"] = t.route.value
    d["resolved_by"] = t.resolved_by.value
    d["features"] = t.features
    for f in _LAT_FIELDS:
        d[f"lat_{f}"] = getattr(t.latency, f)
    return d


def _trace(row: Any) -> DecisionTrace:
    m = dict(row._mapping)
    lat = LatencySplit(**{f: m.pop(f"lat_{f}") or 0.0 for f in _LAT_FIELDS})
    feats = m.pop("features") or {}
    if isinstance(feats, str):
        feats = json.loads(feats)
    if m["ts"].tzinfo is None:
        m["ts"] = m["ts"].replace(tzinfo=UTC)
    return DecisionTrace(latency=lat, features=feats, **m)


class SqliteTelemetry:
    """TelemetryCollector backed by SQLite (or any SQLAlchemy URL)."""

    def __init__(self, url: str = "sqlite:///./data/metrics.db") -> None:
        if url.startswith("sqlite:///") and not url.endswith(":memory:"):
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url)
        _meta.create_all(self.engine)

    def record(self, trace: DecisionTrace) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(traces).values(**_row(trace)))

    def fetch(
        self, limit: int = 100, arm: str | None = None, since: datetime | None = None
    ) -> list[DecisionTrace]:
        q = select(traces).order_by(traces.c.ts.desc()).limit(limit)
        if arm:
            q = q.where(traces.c.arm == arm)
        if since:
            q = q.where(traces.c.ts >= since)
        with self.engine.connect() as conn:
            return [_trace(r) for r in conn.execute(q)]

    def to_parquet(self, path: str | Path) -> int:
        import pandas as pd

        df = pd.read_sql_table("traces", self.engine)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return len(df)


class InMemoryTelemetry:
    def __init__(self) -> None:
        self.traces: list[DecisionTrace] = []

    def record(self, trace: DecisionTrace) -> None:
        self.traces.append(trace)
