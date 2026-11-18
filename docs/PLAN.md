# AMIC: implementation plan (v1, 2026-11-18)

Builder: **@thatajml**, solo across L1–L8. Horizon: **7 days to the final submission** (due 2026-11-25). Anything not done is described in the paper as future work.
All numbers below are design budgets or estimates, not measurements.

## 0. Decisions recorded

| §13 item | Decision |
|---|---|
| 1 LLM hosting | **Local Ollama**: tutor = `llama3.1:8b` (Q4). Logprobs are available, but S5 is deferred (see R7) |
| 1 Tier-2 / judge | Tier 2 = `llama3.2:3b` on Ollama (future work). **Judge = Gemini** (default `gemini-2.5-flash`, configurable) through Gemini's OpenAI-compatible endpoint, used offline only |
| 2 Corpus | **Synthetic education set (primary, ~70%) + LoCoMo (~30%)** |
| 3 Timeline | **1 week to the final submission.** Milestones 6–7 are out of scope and become future work in the paper |
| 4 λ | **4.0** (τ\* = 0.2), a config value, swept in experiments |
| 5 Metrics store | **SQLite** (SQLAlchemy) + Parquet export for analysis |
| 6 Repo shape | **One repo, one Python package, one sub-package per layer** |

## 1. Sprint scope

**In (the minimum the paper needs):** break-even probe · `always` / `never` / `amic` / `oracle`
arms end to end · DecisionTrace telemetry · Tier 0 · signal bus S1–S4 + M4 sketch · calibrated
Tier 1 + cost-sensitive τ · SKIP / SHALLOW / DEEP routes (simplified) · labelled dataset ·
four-arm runner with ablations · break-even table · minimal Decision Inspector · CI running pytest.

**Simplified this sprint:**
- Memory is M1 episodic (raw turns in Chroma) + a heuristic M3 profile. M2 fact extraction comes later.
- Writes are synchronous after the response is returned (a background task), not a Redis-stream worker.
- DEEP = wide-k + MMR. Cross-encoder rerank and the entity hop come later.

**Out (future work in the paper, in this order):** M5 remainder (M2 + async worker + DecisionCache) → M7 verify and
escalate → M6 Tier 2 + distillation → full docker-compose app → Educator Console → LongMemEval.
The design degrades cleanly to Tier 0 + Tier 1 (§10), so cutting these needs no redesign.

**Key efficiency idea:** gate decisions can be **replayed offline**. Once each query's
FeatureVector, oracle label and per-arm cost are logged, every ablation and λ sweep on *decision*
metrics is a cheap re-run of the controller over stored features, with no LLM calls. Only
*response quality* needs generation, and only for the final configurations.

## 2. Repo layout

```
src/amic/
  contracts/        models.py (§3 types) · interfaces.py (5 frozen Protocols) · config.py
  service/          L2  api.py (FastAPI) · session.py (Redis M0) · router.py (arm selection)
  orchestration/    L3  graph.py (LangGraph) · budget.py
  control/          L4  signals/{lexical,session,affinity,parametric}.py · bus.py
                        tier0.py · tier1.py · thresholds.py · route_planner.py · controller.py
                        cache.py (backlog) · tier2.py (backlog)
  memory/           L5  stores/{episodic,profile,sketch}.py · retrieval.py · assembler.py · sketch_builder.py
  generation/       L6  prompt.py · gateway/{base,ollama,openai_compat}.py · verifier.py (backlog)
  writepath/        L7  writer.py (sync for now) · forgetting.py (erasure)
  telemetry/        L8  collector.py · store.py (SQLite) · export.py (Parquet)
  arms/             always.py · never.py · amic.py · oracle.py
  eval/             runner.py · replay.py · metrics.py · breakeven.py · judge.py · datasets/{edu,locomo}.py
ui/                 inspector.py (Streamlit)
scripts/            probe_breakeven.py · gen_edu_dataset.py · label_oracle.py · train_tier1.py
configs/            base.yaml · ablations/{t0_only,t0_t1,no_s3,tau_fixed}.yaml
tests/              unit/ · integration/ · perf/
```

Dependency rule: every layer imports `amic.contracts`. No layer imports another layer's internals,
only its interface. `arms/` and `orchestration/` are where the layers are wired together.

## 3. Contracts (proposed `amic/contracts/models.py`)

```python
from datetime import datetime
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

class Route(str, Enum):
    SKIP = "SKIP"; SHALLOW = "SHALLOW"; DEEP = "DEEP"; WRITE = "WRITE"

class ResolvedBy(str, Enum):
    CACHE = "cache"; TIER0 = "tier0"; TIER1 = "tier1"; TIER2 = "tier2"
    FAILOPEN = "failopen"; FIXED = "fixed"        # FIXED = always/never/oracle arms

class MemoryTier(str, Enum):
    M1 = "M1"; M2 = "M2"; M3 = "M3"

class Turn(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    ts: datetime

class Budget(BaseModel):
    max_prompt_tokens: int = 4096
    max_memory_tokens: int = 1024
    gate_timeout_ms: float = 50.0

class RequestContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    request_id: str
    tenant_id: str                  # = learner/user id (P8)
    session_id: str
    query: str
    m0_turns: list[Turn] = []
    arm: Literal["always", "never", "amic", "oracle"]
    budget: Budget = Budget()
    q_vec: list[float] | None = None    # filled ONCE by the signal bus (P2)
    oracle_label: bool | None = None    # only set in experiments, for the oracle arm

class RetrievalPlan(BaseModel):
    stores: list[MemoryTier] = []
    k: int = 0
    filters: dict[str, str | int | float] = {}
    mmr: bool = False
    mmr_lambda: float = 0.5
    rerank: bool = False
    hops: int = 0
    token_budget: int = 0

class MemoryDecision(BaseModel):
    route: Route
    resolved_by: ResolvedBy
    p: float | None = None          # calibrated P(memory needed); None if decided before Tier 1
    depth_p: float | None = None    # P(DEEP | retrieve)
    tau_low: float | None = None
    tau_high: float | None = None
    plan: RetrievalPlan
    token_budget: int
    rationale: str                  # e.g. "R3: explicit memory reference"

class MemoryItem(BaseModel):
    id: str
    tier: MemoryTier
    text: str
    score: float
    ts: datetime
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    metadata: dict[str, str | int | float | bool] = {}

class LatencySplit(BaseModel):      # milliseconds
    signals: float = 0; cache: float = 0; tier0: float = 0; tier1: float = 0; tier2: float = 0
    retrieval: float = 0; generation: float = 0; verify: float = 0; total: float = 0

class DecisionTrace(BaseModel):
    request_id: str
    ts: datetime
    tenant_id: str
    arm: str
    config_hash: str                # ties a trace to the exact config/ablation
    route: Route
    resolved_by: ResolvedBy
    p: float | None
    tau_low: float | None
    tau_high: float | None
    latency: LatencySplit
    prompt_tokens: int
    completion_tokens: int
    memory_tokens: int              # tokens of injected memory (subset of prompt)
    n_items: int
    escalated: bool = False
    cache_hit: bool = False
    features: dict[str, float]      # FeatureVector snapshot, enables offline replay
    oracle_label: bool | None = None

class AgentResponse(BaseModel):
    answer: str
    decision: MemoryDecision
    items: list[MemoryItem]
    trace_id: str
```

Frozen interfaces (`interfaces.py`, `typing.Protocol`), with signatures exactly as in §8:

```python
class AMICController(Protocol):
    def decide(self, ctx: RequestContext) -> MemoryDecision: ...
class RetrievalEngine(Protocol):
    def retrieve(self, plan: RetrievalPlan, q_vec: np.ndarray, user_id: str) -> list[MemoryItem]: ...
class Sketch(Protocol):
    def affinity(self, q_vec: np.ndarray) -> tuple[float, float]: ...   # (max_sim, margin)
class SketchStore(Protocol):
    def get(self, user_id: str) -> Sketch | None: ...                   # None => no memory (R1)
class TelemetryCollector(Protocol):
    def record(self, trace: DecisionTrace) -> None: ...
class IPipelineArm(Protocol):
    def answer(self, ctx: RequestContext) -> AgentResponse: ...
```

Two flagged refinements (the signatures stay the same):
- `SketchStore.get` may return `None`. That is how R1 (cold start) is detected without touching Chroma.
- `RequestContext.q_vec` is a `list[float]` so it serialises. Internally it is passed as an `np.ndarray`.

## 4. Config schema (`amic/contracts/config.py`, pydantic-settings, YAML + env overrides)

```python
class GateConfig(BaseModel):
    lam: float = 4.0                         # C(false skip) / C(false retrieve)
    tau_mode: Literal["cost", "fixed"] = "cost"   # ablation: fixed => 0.5
    tau_fixed: float = 0.5
    band_low_offset: float = 0.08            # tau_low  = tau* - offset  (tuned on val)
    band_high_offset: float = 0.15           # tau_high = tau* + offset
    tier1_model: Literal["logreg", "lgbm"] = "logreg"
    calibration: Literal["platt", "isotonic"] = "isotonic"
    abstain_fallback: Literal["SHALLOW", "DEEP"] = "SHALLOW"   # band with Tier 2 off
    tier2_max_rate: float = 0.12
    tier2_timeout_ms: float = 800

class AblationFlags(BaseModel):
    tiers: Literal["t0", "t0_t1", "full"] = "t0_t1"
    use_sketch_probe: bool = True            # S3 on/off
    use_cache: bool = False
    use_escalation: bool = False

class RouteConfig(BaseModel):
    shallow_k: int = 3
    deep_k: int = 12
    deep_mmr: bool = True
    always_route: Literal["SHALLOW", "DEEP"] = "DEEP"   # what the `always` baseline retrieves (R2)

class ModelConfig(BaseModel):
    embedding: str = "sentence-transformers/all-MiniLM-L6-v2"
    tutor: str = "llama3.1:8b"
    tier2: str = "llama3.2:3b"
    judge: str = "gemini-2.5-flash"
    judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    # key read from env GEMINI_API_KEY
    temperature: float = 0.0
    seed: int = 0

class Settings(BaseSettings):
    arm: Literal["always", "never", "amic", "oracle"] = "amic"
    gate: GateConfig = GateConfig()
    ablation: AblationFlags = AblationFlags()
    routes: RouteConfig = RouteConfig()
    models: ModelConfig = ModelConfig()
    redis_url: str = "redis://localhost:6379/0"
    chroma_host: str = "localhost"; chroma_port: int = 8000
    chroma_mode: Literal["single", "per_tenant"] = "single"
    metrics_db_url: str = "sqlite:///./data/metrics.db"
    m0_turns: int = 6
```

`config_hash` = a hash of the resolved Settings, stored on every trace.

## 5. Day-by-day plan

Each day ends with the system runnable and the tests green. **DoD** = definition of done.

### Day 1: Milestone 0 (break-even probe) + contracts
- **T1.1** `contracts/` models, interfaces and config, as in §3–4. *DoD:* `tests/unit/test_contracts.py` round-trips every model through JSON; config loads from YAML with env overrides.
- **T1.2** `scripts/probe_breakeven.py`: measure `C_emb` (MiniLM on CPU, 1 query), `C_ret` (Chroma query, k=3 and k=12, at 10², 10³ and 10⁴ stored records), `C_gen` (Ollama 8B with 0 / 300 / 1 000 memory tokens: TTFT, total time, tokens), and rough `C_sig` (regex + 8 centroid dot products). Writes `runs/probe.json` and a short markdown table.
- **T1.3** Compute the inequality for r ∈ {0.3 … 0.9} **in both latency and tokens**. *DoD:* table committed to `docs/results/probe.md`. **Go/no-go:** if the savings are below the gate cost at r = 0.5, stop and rethink before Day 2 (expected: go, because C_sig ≪ C_ret).

### Day 2: Milestones 1 + 2 (skeleton, baselines, telemetry)
- **T2.1** `generation/gateway/ollama.py` (temperature 0, fixed seed, returns token counts and timings); `openai_compat.py` for the judge.
- **T2.2** `memory/stores/episodic.py`: Chroma, single collection + `tenant_id` metadata filter; `retrieval.py` implementing SHALLOW (top-k) and DEEP (wide-k + MMR); `writepath/writer.py` appends the turn after the response.
- **T2.3** `service/session.py` (Redis M0, last N turns), `service/api.py` `POST /chat`, `orchestration/graph.py` (LangGraph: decide → retrieve → generate → record → write).
- **T2.4** `arms/always.py`, `arms/never.py`; `telemetry/` SQLite collector + Parquet export.
- *DoD tests:* `test_tenant_isolation` (tenant A never retrieves B's items); `test_m0_visibility` (turn N content is in the turn N+1 prompt through M0, with no long-term store involved); `test_never_no_chroma` (the never arm makes zero Chroma calls, using a spy); `test_trace_recorded` (one trace per request, all fields populated). Integration tests run against fakeredis + an ephemeral Chroma client.
- **T2.5** `.github/workflows/ci.yml`: ruff + pytest (unit + integration, no Ollama).

### Day 3: Dataset + oracle labels
- **T3.1** `scripts/gen_edu_dataset.py`: synthetic learners (~40), each with 3–5 prior sessions over an **intro Python** curriculum (variables, types, conditionals, loops, functions, lists/dicts, strings, file I/O, exceptions), seeded misconceptions, preferences and goals. Then a probe-query set per learner, mixed by design: general concept questions (should SKIP), questions referring back to earlier sessions (memory-needed), and personalisation-sensitive questions. Seed the histories into memory through the normal write path.
- **T3.2** `eval/datasets/locomo.py` loader; map the adversarial / unanswerable category to "memory-not-required".
- **T3.3** `scripts/label_oracle.py`: run each query through `always` (at both SHALLOW and DEEP) and `never`; the hosted judge scores them pairwise in both orders (to remove position bias); label = memory-required if `never` is materially worse. Depth label = the cheaper route that is not materially worse than DEEP (R5).
- **T3.4** Hand-label a stratified set of **100** queries this sprint (target 200–300 later) and report Cohen's κ against the judge.
- *DoD:* `data/labels.parquet` with ~600–1 000 labelled queries; train/val/test split **by learner** (no leakage); κ reported.
- *Budget note:* ~1 000 queries × 3 arms of 8B generation ≈ several hours of GPU time. Start this overnight at the end of Day 2 with a partial dataset.

### Day 4: Milestone 3 + signal bus + M4 sketch
- **T4.1** `control/tier0.py` rules R1–R4. *DoD:* one unit test per rule plus rule ordering; perf test p50 ≤ 0.1 ms (`pytest-benchmark`, a budget, not a hard CI fail).
- **T4.2** `memory/sketch_builder.py`: mini-batch k-means centroids (k = min(8, n/5)) over the learner's M1 embeddings + a numpy Bloom filter over extracted entities/concepts; rebuilt by the writer. `affinity()` = (max cosine, max − second-best).
- **T4.3** `control/signals/`: S1 lexical (regex/lexicon), S2 session (turn index, M0 coverage = max sim of q_vec to cached M0 turn embeddings, days since last session, memory count), S3 affinity + Bloom hit, S4 parametric (curriculum-concept lookup + "general question" lexical cues). `bus.py` embeds once and fills `ctx.q_vec`. Output: a fixed-order ~24-dim FeatureVector.
- **Report:** the share of labelled traffic Tier 0 resolves alone, and its accuracy (offline replay).
- *DoD tests:* each extractor on fixed inputs; `test_q_vec_computed_once` (embedding model called exactly once per request, using a spy); `test_sketch_no_documents_read`.

### Day 5: Milestone 4 (Tier 1 + thresholds) + the amic and oracle arms
- **T5.1** `scripts/train_tier1.py`: logistic regression (and LightGBM for comparison) on the train split; isotonic calibration on val; depth head (SHALLOW vs DEEP) on the positive class. Save the artefact with its feature schema version.
- **T5.2** `control/thresholds.py`: τ\* = 1/(1+λ); band from config; `tau_mode=fixed` ablation.
- **T5.3** `control/route_planner.py` + `controller.py` (cascade with per-tier timings; band → `abstain_fallback`, since Tier 2 is off this sprint); `arms/amic.py`, `arms/oracle.py`.
- *DoD tests:* `test_threshold_from_lambda` (λ=4 → 0.2); `test_band_abstains`; `test_skip_route_zero_retrieval`; route end-to-end tests for all three routes; reliability curve + Brier score on val written to `runs/`; Tier 0+1 p50 ≲ 5 ms.

### Day 6: Milestone 8 (experiments)
- **T6.1** `eval/replay.py`: run any controller config over stored features → decision metrics (P/R/F1, false-skip rate, cost-weighted error, a λ sweep from 1 to 10).
- **T6.2** `eval/runner.py`: four arms × 3 seeds on the test split → traces; `eval/metrics.py`: retrieval rate, p50/p95 latency split, mean prompt/memory tokens, % of oracle savings captured.
- **T6.3** `eval/breakeven.py`: fill in §9.3 from the measured traces.
- **T6.4** Ablations through replay (decision metrics): T0 only · T0+1 · no S3 · τ = 0.5. Quality (pairwise judge vs `always`) for the main amic config and the no-S3 ablation only.
- *DoD:* `docs/results/` contains the main table, ablation table, break-even table, λ-sweep plot and reliability plot, each produced by **one command** (`python -m amic.eval.report`).

### Day 7: write-up artefacts, demo, buffer
- **T7.1** *Only if Days 1–6 landed:* `ui/inspector.py` (Streamlit): chat with an arm selector, plus a per-turn panel showing route, tier, p against the τ band, feature values, retrieved items, latency split and tokens.
- **T7.2** Fallback, always delivered: `python -m amic.demo` CLI that prints the decision panel per turn. README quick start + a demo script (cold-start learner → SKIP; "what did I get wrong last time" → DEEP; a general concept question → SKIP while memory exists).
- **T7.3** Buffer for whatever slipped (realistically Day 3 or Day 6).

## 6. Risks

| # | Risk | Mitigation |
|---|---|---|
| K1 | **One week, one person, hard final deadline.** Days 3 and 6 are the most likely to slip | Replay design keeps experiments cheap; dataset generation runs overnight; Day 7 is buffer; Streamlit UI is cut first, then LoCoMo, then the LightGBM comparison |
| K2 | **GPU throughput.** 8B generation for ~3 000 labelling runs + ~4 × 3 × test-set runs | Short answers (`num_predict` cap), a smaller labelled set first, reuse `always`/`never` generations from labelling as the arm outputs |
| K3 | **The latency story may be weak.** On a laptop, generation (seconds) dwarfs retrieval (ms), so the end-to-end latency saving will look small | Lead with **memory/prompt tokens and retrieval rate**; report latency honestly, split by component. Prompt-token savings also cut TTFT, which the probe measures |
| K4 | Synthetic data favouring the gate (label leakage through templated phrasing) | Vary phrasing with the LLM; report LoCoMo separately; hand-labelled κ |
| K5 | Gemini judge cost, rate limits / availability (free-tier RPM limits can stall labelling) | Judge runs only offline; cache every judgment keyed by (query, answers, seed) |
| K6 | VRAM: 8B + 3B + embeddings on 8 GB → Ollama swapping models | Tier 2 is out this sprint; embeddings run on CPU; later, set `OLLAMA_MAX_LOADED_MODELS` and measure |

## 7. Inconsistencies / gaps in the brief (my proposed resolutions)

- **R1, break-even double count.** `cost_base` includes `C_emb`, but the "AMIC wins iff" line drops it, while under P2 `C_sig` already contains the embedding. As written, the inequality is too strict. Fix: define `C_sig = C_emb + C_feat`, so the condition becomes `C_feat + P(T1)·C_T1 + P(T2)·C_T2 + P(esc)·(C_ret+C_gen) + C_ver < (1 − r)·C_ret`.
- **R2, `C_ret` isn't one number.** SHALLOW and DEEP cost different amounts, and the brief doesn't say what `always` retrieves. Fix: split into `r_S·C_S + r_D·C_D`; the `always` baseline = DEEP by default (config `always_route`), with SHALLOW-always reported as a second, stronger baseline.
- **R3, missing terms.** Verifier cost (`C_ver`, paid on every SKIP once M7 exists), cache lookup cost, and the fact that `P(T1)` = P(Tier 0 abstains ∧ cache miss). All are captured in `LatencySplit`.
- **R4, cache vs session-dependent rules.** R2 (M0 answers it) depends on the session, but the cache key `(tenant, LSH(q_vec))` ignores it, so a cached SKIP can go stale within a session. Fix: cache only Tier 1/2 outcomes, never Tier 0; include a cheap M0 fingerprint in the key.
- **R5, no source of depth labels.** The Tier-1 depth head needs SHALLOW-vs-DEEP labels, but oracle labelling only compares always vs never. Fix: also run SHALLOW in labelling (T3.3).
- **R6, the WRITE route and `/forget` are under-specified.** Proposal: WRITE = no retrieval + acknowledgement + write intent. `/forget` is an erasure and should be **synchronous** (privacy), an explicit exception to P5.
- **R7, S5 conflicts with P1.** Drafting tokens to measure entropy costs tens of ms, about the same as the retrieval it gates. Keep it as an optional ablation only.
- **R8, dataset vs milestone order.** Tier 1 (M4) needs labels, but the brief never schedules dataset work. It is now Day 3.
- **R9, Chroma tenancy.** Default to a single collection + metadata filter (simpler, no collection-count limits); per-tenant mode stays a config flag.

## 8. Open items

- A `GEMINI_API_KEY` in `.env` before Day 3 labelling.
