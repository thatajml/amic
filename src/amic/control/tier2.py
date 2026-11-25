"""Tier 2: LLM adjudicator for Tier 1's uncertain band (brief 6.2).

Fires only when Tier 1's p falls inside [tau_low, tau_high] (ablation.tiers = "full"). It sees
the query, the last two turns of the current session and the learner's memory SKETCH summary
(record and session counts, days since the last session, most frequent topics). It never reads
memory documents (P3: gate on query x memory-store state, without retrieval).

Fail-open, never SKIP: a timeout, an error, unparseable output or the rate cap returns no route,
and the controller falls back to `gate.abstain_fallback` (SHALLOW). The cap is a sliding window
over ALL gate decisions (brief: Tier 2 on at most 10-15% of traffic). Every verdict is appended
to the DistillationBuffer: training data for the next Tier-1 retrain.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from amic.contracts.models import RequestContext, Route
from amic.generation.gateway.base import LLMGateway

if TYPE_CHECKING:
    from amic.control.distill import DistillationBuffer
    from amic.control.signals.bus import Signals

PROMPT_VERSION = 1
PROMPT = """You are the memory gate of a Python tutoring assistant. Before the tutor answers,
decide whether it must look up this student's long-term memory: notes from their PREVIOUS
tutoring sessions (past questions, mistakes and misconceptions, goals, preferences, projects).

Student's new message: {query}

Current session so far:
{m0}

What the long-term memory holds: {n_records} notes from {n_sessions} past sessions; the last
session was {days} days ago; frequent topics: {topics}.

Options:
- SKIP: the message is self-contained; a correct general answer fully serves this student.
- SHALLOW: a few notes about this student would make the answer better or more personal.
- DEEP: the message explicitly depends on past sessions (e.g. "what did I get wrong last time").
Return JSON only: {{"route": "SKIP" | "SHALLOW" | "DEEP"}}"""

Status = Literal["ok", "timeout", "error", "unparseable", "capped", "replay-miss"]


@dataclass(frozen=True)
class Tier2Result:
    route: Route | None  # None: failed open, the controller uses abstain_fallback
    status: Status
    ms: float = 0.0


def parse_route(raw: str) -> Route:
    j = json.loads(raw)
    if isinstance(j, list) and len(j) == 1 and isinstance(j[0], dict):
        j = j[0]
    v = str(j["route"] if isinstance(j, dict) else j).strip().upper()
    if v not in {"SKIP", "SHALLOW", "DEEP"}:
        raise ValueError(f"not a route: {v!r}")
    return Route(v)


def build_prompt(sig: Signals, ctx: RequestContext) -> str:
    f = sig.features
    m0 = "\n".join(f"{t.role.upper()}: {t.text}" for t in ctx.m0_turns[-2:]) or "(new session)"
    topics = ", ".join(sig.sketch.top_terms[:12]) if sig.sketch and sig.sketch.top_terms else "-"
    return PROMPT.format(
        query=ctx.query,
        m0=m0,
        n_records=sig.sketch.n_records if sig.sketch else 0,
        n_sessions=round(math.expm1(f.get("s2_n_sessions_log", 0.0))),
        days=round(math.expm1(f.get("s2_days_since_log", 0.0))),
        topics=topics,
    )


class RateCap:
    """At most `max_rate` x `window` Tier-2 calls among the last `window` gate decisions."""

    def __init__(self, max_rate: float, window: int = 200) -> None:
        self.budget = max_rate * window
        self._recent: deque[bool] = deque(maxlen=window)

    def allow(self) -> bool:
        return sum(self._recent) + 1 <= self.budget

    def observe(self, fired: bool) -> None:
        self._recent.append(fired)


class _Tier2Base:
    def __init__(self, max_rate: float = 0.12, window: int = 200) -> None:
        self.cap = RateCap(max_rate, window)
        self.fired = 0
        self.capped = 0

    def observe(self, fired: bool) -> None:
        """Called by the controller once per gate decision that did not reach Tier 2."""
        self.cap.observe(fired)

    def adjudicate(self, sig: Signals, ctx: RequestContext | None, p: float) -> Tier2Result:
        if not self.cap.allow():
            self.capped += 1
            self.cap.observe(False)
            return Tier2Result(None, "capped")
        self.fired += 1
        self.cap.observe(True)
        return self._adjudicate(sig, ctx, p)

    def _adjudicate(self, sig: Signals, ctx: RequestContext | None, p: float) -> Tier2Result:
        raise NotImplementedError


class Tier2Adjudicator(_Tier2Base):
    """The live adjudicator: one constrained-JSON call to the Tier-2 model."""

    def __init__(
        self,
        llm: LLMGateway,
        timeout_ms: float = 800.0,
        max_rate: float = 0.12,
        window: int = 200,
        buffer: DistillationBuffer | None = None,
        max_tokens: int = 16,
    ) -> None:
        super().__init__(max_rate, window)
        self.llm = llm
        self.timeout_ms = timeout_ms
        self.buffer = buffer
        self.max_tokens = max_tokens

    def _adjudicate(self, sig: Signals, ctx: RequestContext | None, p: float) -> Tier2Result:
        if ctx is None:
            return Tier2Result(None, "error")
        prompt = build_prompt(sig, ctx)
        t = time.perf_counter()
        try:
            raw = self.llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0.0,
                seed=0,
                json_mode=True,
            ).text
        except Exception as e:  # noqa: BLE001 - Tier 2 must never break the request
            ms = (time.perf_counter() - t) * 1000
            timed_out = "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower()
            return Tier2Result(None, "timeout" if timed_out else "error", ms)
        ms = (time.perf_counter() - t) * 1000
        if ms > self.timeout_ms:  # answered, but too late to be useful: fail open anyway
            return Tier2Result(None, "timeout", ms)
        try:
            route = parse_route(raw)
        except (ValueError, KeyError, TypeError):
            return Tier2Result(None, "unparseable", ms)
        if self.buffer is not None:
            self.buffer.append(ctx, sig.features, p, route, self.llm.model)
        return Tier2Result(route, "ok", ms)


class ReplayTier2(_Tier2Base):
    """Offline replay: verdicts recorded per probe (scripts/replay_tier2.py), keyed by
    ctx.request_id. Same cap and fail-open semantics as the live adjudicator."""

    def __init__(
        self,
        verdicts: dict[str, tuple[str | None, str, float]],
        max_rate: float = 0.12,
        timeout_ms: float = 800.0,
    ) -> None:
        super().__init__(max_rate)
        self.verdicts = verdicts  # probe_id -> (route or None, status, measured ms)
        self.timeout_ms = timeout_ms

    @classmethod
    def from_jsonl(
        cls, path: str, model: str, max_rate: float = 0.12, timeout_ms: float = 800.0
    ) -> ReplayTier2:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(x) for x in f.read().splitlines() if x]
        v = {
            r["probe_id"]: (r["route"], r["status"], float(r["ms"]))
            for r in rows
            if r["model"] == model and r["prompt_version"] == PROMPT_VERSION
        }
        return cls(v, max_rate, timeout_ms)

    def _adjudicate(self, sig: Signals, ctx: RequestContext | None, p: float) -> Tier2Result:
        v = self.verdicts.get(ctx.request_id) if ctx is not None else None
        if v is None:
            return Tier2Result(None, "replay-miss")
        route, status, ms = v
        if status == "ok" and ms > self.timeout_ms:  # the live adjudicator would have given up
            return Tier2Result(None, "timeout", ms)
        return Tier2Result(Route(route) if route else None, status, ms)  # type: ignore[arg-type]
