"""SignalExtractionBus (L4): one shared pass -> fixed-order FeatureVector.

    S1 lexical      regex/lexicon over the query                         (µs)
    S2 session      M0 + per-learner session metadata (Redis hash)       (µs, one HGETALL)
    S3 affinity     dot products vs the learner's M4 centroids + Bloom   (µs, no ANN, no docs)
    S4 parametric   curriculum-concept lookup / general-question form    (µs)

The query is embedded exactly once here (P2); the vector is reused by Tier 1 and retrieval.
Deviation from brief 6.1 (flagged): "M0 already covers the query" uses term overlap with M0
plus follow-up markers instead of embedding every M0 turn (that would cost ~5 ms per turn).
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from amic.contracts.models import RequestContext
from amic.memory.sketch import MemorySketch, terms


# ----------------------------------------------------------------------------- lexicons
def _rx(*alts: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.I)


EXPLICIT_REF = _rx(
    r"last (?:time|week|session|lesson|class)",
    r"what did i (?:say|ask|get wrong|do wrong|mess up|struggle with|learn)",
    r"did i (?:make|get|mess|do)",
    r"remind me",
    r"do you remember",
    r"(?:we|you) (?:discussed|covered|talked about|went over|did|looked at|worked on)",
    r"you (?:told|showed|taught|explained to) me",
    r"my (?:progress|weak (?:spots?|areas?)|mistakes)",
    r"previous(?:ly)? (?:session|lesson|time)",
    r"(?:the )?first time we",
    r"back when",
)
TEMPORAL = _rx(
    r"again",
    r"still",
    r"same",
    r"before",
    r"earlier",
    r"previous(?:ly)?",
    r"a while ago",
    r"the other day",
    r"yesterday",
    r"last (?:week|time)",
    r"back",
)
POSSESSIVE = _rx(r"my", r"mine", r"i'm", r"i've", r"i was", r"i had")
DEFINITE_DEIXIS = _rx(
    r"that (?:bug|error|thing|problem|issue|code|example|exercise|part|one|trick|project)",
    r"the same (?:bug|error|thing|problem|issue|mistake)",
    r"this (?:bug|error|thing|problem|issue)",
    r"the (?:thing|one) (?:with|about|from)",
)
FOLLOWUP = _rx(
    r"what did you mean",
    r"you just",
    r"that last (?:part|bit|line|example|step)",
    r"the example you (?:gave|showed)",
    r"above",
    r"can you show that",
    r"say that again",
    r"in your (?:last|previous) (?:answer|message|reply)",
    r"what you (?:said|wrote) (?:about|there)",
)
REQUEST_PLAN = _rx(
    r"what should i",
    r"next step",
    r"(?:give|set) me (?:an? )?(?:exercise|practice|challenge)",
    r"practice",
    r"study plan",
    r"recommend",
    r"work on next",
    r"tailored",
    r"my level",
)
GENERAL_FORM = re.compile(
    r"^\s*(?:what(?:'s| is| are)|how (?:do|does|can|to)|why (?:does|do|is)|can you (?:explain|show)|"
    r"explain|what's the difference|is it (?:better|true))",
    re.I,
)
CODE = re.compile(r"`|\bdef |\bprint\(|\w+\(.*\)|\btraceback\b|\w+error\b|==|\[\d*:?\d*\]", re.I)
COMMAND = re.compile(r"^\s*/(remember|forget)\b", re.I)

CONCEPT_TERMS = {
    "variable",
    "variables",
    "string",
    "strings",
    "int",
    "integer",
    "float",
    "input",
    "print",
    "if",
    "elif",
    "else",
    "conditional",
    "while",
    "for",
    "loop",
    "loops",
    "range",
    "list",
    "lists",
    "append",
    "extend",
    "index",
    "dictionary",
    "dictionaries",
    "dict",
    "key",
    "keys",
    "function",
    "functions",
    "return",
    "parameter",
    "argument",
    "scope",
    "global",
    "local",
    "exception",
    "exceptions",
    "try",
    "except",
    "error",
    "file",
    "files",
    "open",
    "read",
    "write",
    "comprehension",
    "comprehensions",
    "tuple",
    "tuples",
    "set",
    "sets",
    "class",
    "slicing",
    "recursion",
    "boolean",
    "none",
    "type",
    "types",
    "traceback",
    "debug",
}

# ----------------------------------------------------------------------------- vector
FEATURES: list[str] = [
    # S1
    "s1_explicit_ref",
    "s1_temporal",
    "s1_possessive",
    "s1_deixis",
    "s1_followup",
    "s1_request_plan",
    "s1_general_form",
    "s1_has_code",
    "s1_len_log",
    "s1_question",
    # S2
    "s2_m0_turns",
    "s2_m0_overlap",
    "s2_days_since_log",
    "s2_n_sessions_log",
    "s2_n_memories_log",
    # S3
    "s3_has_memory",
    "s3_max_sim",
    "s3_margin",
    "s3_term_hit_frac",
    "s3_term_hits",
    # S4
    "s4_concept_hits",
    "s4_concept_frac",
    # interactions (help the linear model)
    "x_general_no_personal",
    "x_affinity_personal",
]
S3_FEATURES = [f for f in FEATURES if f.startswith("s3_") or f == "x_affinity_personal"]


@dataclass
class Signals:
    q_vec: np.ndarray
    features: dict[str, float]
    sketch: MemorySketch | None
    is_command: bool
    embed_ms: float
    feat_ms: float

    def vector(self, names: list[str] = FEATURES) -> np.ndarray:
        return np.array([self.features[n] for n in names], dtype=np.float32)


def _count(rx: re.Pattern[str], s: str) -> float:
    return float(min(3, len(rx.findall(s))))


class SignalBus:
    def __init__(
        self,
        embedder: Any,
        sketches: Any,
        sessions: Any,
        use_sketch: bool = True,
        uncertainty: Any = None,
    ):
        self.embedder = embedder
        self.sketches = sketches
        self.sessions = sessions
        self.use_sketch = use_sketch
        self.uncertainty = uncertainty  # optional S5 (UncertaintySignal); adds s5_* features

    def extract(self, ctx: RequestContext, q_vec: np.ndarray | None = None) -> Signals:
        t0 = time.perf_counter()
        if q_vec is None:
            q_vec = self.embedder.encode(ctx.query)  # the ONE embedding of this request (P2)
        t1 = time.perf_counter()
        sketch = self.sketches.get(ctx.tenant_id)
        meta = self.sessions.tenant_meta(ctx.tenant_id)
        feats = compute_features(ctx, q_vec, sketch, meta, self.use_sketch)
        is_command = bool(COMMAND.match(ctx.query))
        if self.uncertainty is not None and not is_command and feats["s3_has_memory"] > 0:
            # only where Tier 1 can run (R1/R4 resolve first); counted in C_feat
            feats.update(self.uncertainty.features(ctx.query, ctx.m0_turns, ctx.now))
        t2 = time.perf_counter()
        return Signals(
            q_vec=q_vec,
            features=feats,
            sketch=sketch,
            is_command=is_command,
            embed_ms=(t1 - t0) * 1000,
            feat_ms=(t2 - t1) * 1000,
        )


def _days_since(ctx: RequestContext, meta: dict[str, Any]) -> float:
    now = (ctx.now or datetime.now(UTC)).timestamp()
    if not meta:
        return 0.0
    if meta.get("last_session_id") == ctx.session_id:  # touched already for this session
        ref = meta.get("prev_session_end", meta.get("last_seen", now))
    else:
        ref = meta.get("last_seen", now)
    return max(0.0, (now - float(ref)) / 86400.0)


def compute_features(
    ctx: RequestContext,
    q_vec: np.ndarray,
    sketch: MemorySketch | None,
    meta: dict[str, Any],
    use_sketch: bool = True,
) -> dict[str, float]:
    q = ctx.query
    words = q.split()
    q_terms = terms(q)
    f: dict[str, float] = {}
    # S1
    f["s1_explicit_ref"] = _count(EXPLICIT_REF, q)
    f["s1_temporal"] = _count(TEMPORAL, q)
    f["s1_possessive"] = _count(POSSESSIVE, q)
    f["s1_deixis"] = _count(DEFINITE_DEIXIS, q)
    f["s1_followup"] = _count(FOLLOWUP, q)
    f["s1_request_plan"] = _count(REQUEST_PLAN, q)
    f["s1_general_form"] = float(bool(GENERAL_FORM.search(q)))
    f["s1_has_code"] = float(bool(CODE.search(q)))
    f["s1_len_log"] = math.log1p(len(words))
    f["s1_question"] = float("?" in q)
    # S2
    m0_text = " ".join(t.text for t in ctx.m0_turns)
    m0_terms = terms(m0_text)
    f["s2_m0_turns"] = float(min(len(ctx.m0_turns), 12))
    f["s2_m0_overlap"] = len(q_terms & m0_terms) / len(q_terms) if q_terms and m0_terms else 0.0
    f["s2_days_since_log"] = math.log1p(_days_since(ctx, meta))
    f["s2_n_sessions_log"] = math.log1p(float(meta.get("n_sessions", 0.0)))
    f["s2_n_memories_log"] = math.log1p(sketch.n_records if sketch else 0)
    # S3 (zeroed when the sketch probe is ablated)
    if sketch is not None and use_sketch:
        max_sim, margin = sketch.affinity(q_vec)
        hits, n_terms = sketch.term_hits(q)
        f["s3_has_memory"] = 1.0
        f["s3_max_sim"] = max_sim
        f["s3_margin"] = margin
        f["s3_term_hit_frac"] = hits / n_terms if n_terms else 0.0
        f["s3_term_hits"] = float(min(hits, 10))
    else:
        f["s3_has_memory"] = float(sketch is not None)
        f["s3_max_sim"] = f["s3_margin"] = f["s3_term_hit_frac"] = f["s3_term_hits"] = 0.0
    # S4
    concept_hits = sum(1 for w in re.findall(r"[a-z]+", q.lower()) if w in CONCEPT_TERMS)
    f["s4_concept_hits"] = float(min(concept_hits, 6))
    f["s4_concept_frac"] = concept_hits / max(1, len(words))
    # interactions
    personal = f["s1_explicit_ref"] + f["s1_possessive"] + f["s1_deixis"] + f["s1_temporal"]
    f["x_general_no_personal"] = float(f["s1_general_form"] > 0 and personal == 0)
    f["x_affinity_personal"] = f["s3_max_sim"] * min(personal, 3.0)
    return f
