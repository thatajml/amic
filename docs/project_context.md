# AMIC — Project Context for Implementation Planning

> All latency, token and traffic numbers here are **design budgets to be validated**, not measured
> results. Never present them as results.

---

## 1. What we are building (one paragraph)

An LLM-based tutoring agent with long-term memory, plus a **controller that decides, before every
query, whether long-term memory retrieval is worth doing — and how much of it**. Most memory-enabled
agents retrieve on every turn, wasting tokens and latency and sometimes injecting irrelevant context.
AMIC gates retrieval with a cheap three-tier cascade and routes each query to **SKIP / SHALLOW /
DEEP** retrieval. It is evaluated against an always-retrieve baseline (and a never-retrieve floor
and an oracle ceiling) on retrieval rate, latency, tokens, cost and answer quality. **Application
domain: intelligent tutoring (education).**

**Research claim to make falsifiable:** AMIC skips memory on a large share of queries, cutting
retrieval rate / latency / prompt tokens, while holding answer quality at always-retrieve levels —
and its gate overhead is smaller than the retrieval it saves (break-even inequality, §9.3).

---

## 2. Project facts

| Item | Value |
|---|---|
| Type | Final-year capstone (B.Tech CSE – AI & Data Science), Group 8 / SP8, MIT World Peace University, Pune |
| Guide | Prof. Shoma Mitkari |
| Stage | Level-2 synopsis approved; architecture revised to **AMIC v2** (this doc reflects v2) |
| Publications | Paper 1: literature review (IEEE conference, in preparation). Paper 2: implementation paper — *"AMIC: Cascaded Cost-Sensitive Memory Gating for LLM-Based Tutoring Agents"* (target IEEE Access / IEEE conference). **The code must produce the tables and ablations that paper needs.** |
| Language | Python 3.11+ |

### Team and ownership

| Member | Owns | Layers |
|---|---|---|
| **Ala Ajmal Malik** | AMIC decision module | L4 — signal bus, Tier 0/1/2, ThresholdPolicy, RoutePlanner, DecisionCache, DistillationBuffer |
| **Aadil Salim** | Long-term memory & retrieval | L5 + L7 — M1–M4 stores, RetrievalEngine, ContextAssembler, ConsolidationWorker, SketchBuilder, ForgettingManager |
| **Fahim Abullais** | LLM agent & backend, end-to-end integration | L2, L3, L6 — FastAPI service, SessionManager, LangGraph orchestrator, PromptBuilder, LLMGateway, ResponseVerifier, EscalationController, the baseline arms |
| **Haamid Ashraf** | Evaluation, UI, dataset | L1 + L8 — Tutor UI, Decision Inspector, ExperimentRunner, Evaluator, MetricsRepository, dataset curation |

**Shared boundary:** the M4 memory sketch is *built* by Aadil and *consumed* by Ala. Its
`affinity()` signature must be frozen early.

---

## 3. Tech stack

| Concern | Choice |
|---|---|
| API / service | FastAPI |
| Agent orchestration | LangGraph (state machine), LangChain for LLM/prompt/retrieval glue |
| LLM | Llama 3 or similar open model, **or** a hosted API — *open decision, see §13*. Accessed only through an `LLMGateway` adapter |
| Embeddings | Sentence Transformers |
| Vector store | ChromaDB (HNSW; per-tenant collections, fallback to single collection + tenant metadata filter) |
| Session state + write queue | Redis (sessions, WriteIntent stream) |
| Tier-1 gate model | scikit-learn logistic regression or a GBDT (LightGBM/XGBoost), with Platt/isotonic calibration |
| ML runtime | PyTorch / Hugging Face Transformers (if local models) |
| UI | Streamlit (or equivalent) |
| Packaging / deploy | docker-compose; GitHub Actions CI (tests + gate-metric regression check) |
| Tests | pytest |

Hardware assumption: 16 GB RAM dev laptops; a CUDA GPU only if the LLM is hosted locally.

---

## 4. Design principles (non-negotiable)

| # | Principle | Consequence |
|---|---|---|
| P1 | The gate must be **structurally cheaper** than what it gates | Three-tier cascade; expensive tiers fire only on abstention |
| P2 | **Compute the query embedding once** | `q_vec` produced in the signal bus and reused by Tier 1, the sketch probe and retrieval |
| P3 | Gate on **(query × memory-store state)**, not the query alone | S3 sketch probe against the user's M4 sketch — core novelty claim |
| P4 | **Thresholds follow from costs**, not argmax | `ThresholdPolicy` derives τ from cost ratio λ |
| P5 | **Never pay for writes on the response path** | All consolidation async behind a queue |
| P6 | **A wrong skip must be recoverable** | Bounded verify-and-escalate, max one retry |
| P7 | **Every decision is an observable data point** | `DecisionTrace` is a first-class artefact |
| P8 | **Stateless request path, per-tenant state** | Memory namespaced per learner; horizontally scalable |

---

## 5. Layered architecture

| Layer | Responsibility | Components |
|---|---|---|
| L1 Experience | Student chat, decision transparency, educator/researcher views | Tutor UI, Decision Inspector, Educator Console, Evaluation Dashboard |
| L2 Service | Auth, tenancy, session state, experiment-arm selection | API Gateway, SessionManager (M0 buffer), ExperimentRouter |
| L3 Orchestration | LangGraph state machine, per-request budgets | AgentOrchestrator, Budget & Policy Manager |
| **L4 AMIC Control Plane** | **The contribution** — decide whether and how much memory to invoke | SignalExtractionBus, Tier 0/1/2, RoutePlanner, DecisionCache, ThresholdCalibrator |
| L5 Memory Fabric | Store and retrieve long-term memory | RetrievalEngine, ContextAssembler, M1–M4 stores |
| L6 Generation | Prompting, inference, verification, escalation | PromptBuilder, LLMGateway, ResponseVerifier, EscalationController |
| L7 Async Write Path | Turn interactions into durable memory, off the hot path | WriteIntentQueue, ConsolidationWorker, ForgettingManager |
| L8 Observability & Eval | Make claims measurable | DecisionTraceLogger, MetricsStore, ExperimentRunner, Evaluator |

### Request flow (hot path)

```
user query
  → L2: resolve tenant, load session (M0), pick experiment arm
  → L3: orchestrator starts LangGraph run with budget
  → L4: SignalBus (one pass, computes q_vec once) → FeatureVector
        → DecisionCache hit? → reuse decision
        → Tier 0 rules → resolved? else
        → Tier 1 calibrated model → resolved? else
        → Tier 2 LLM adjudicator (≤10–15% of traffic; timeout ⇒ SHALLOW, never SKIP)
        → RoutePlanner → MemoryDecision {route, RetrievalPlan, token budget}
  → L5: SKIP: M0 only | SHALLOW: M3 + top-2/3 M2 | DEEP: wide-k M1+M2+M3, filters, MMR, rerank, 1 hop
  → L6: PromptBuilder → LLMGateway → ResponseVerifier
        → if route was SKIP and verify fails → escalate once to SHALLOW/DEEP and regenerate
  → L7: enqueue WriteIntent (non-blocking) and return
  → L8: record DecisionTrace
```

---

## 6. AMIC control plane (L4) — detail

### 6.1 Signal extraction (single shared pass → ~24-dim `FeatureVector`)

| Signal | Content | Cost budget |
|---|---|---|
| S1 Lexical | Deictic/anaphoric markers ("again", "last time", "you said", "my"), unresolved pronouns, question type, length | µs — regex + lexicon |
| S2 Session | Turn index, whether M0 already covers the query, days since last session, # stored memories | µs — in-memory |
| S3 Memory affinity | `max_sim` and `margin` from dot products vs the user's M4 centroid sketch + Bloom-filter entity hit | µs–ms — **no ANN search, no documents read** |
| S4 Parametric proxy | Is this general knowledge the base model already has? (curriculum-concept lookup / small classifier) | ms |
| S5 Uncertainty *(optional)* | Token entropy of a short draft — needs logprobs ⇒ local models only | tens of ms — drop for hosted APIs |

### 6.2 The cascade

**Tier 0 — Deterministic Policy Engine (~0.1 ms).** Ordered rules:
- R1 no memory exists for this learner → **SKIP** (cold-start guarantee)
- R2 M0 working buffer already answers it → **SKIP**
- R3 explicit memory reference ("what did I say about…", "my progress") → **DEEP**
- R4 explicit command (`/remember`, `/forget`) → **WRITE**
- otherwise → **ABSTAIN**

**Tier 1 — Calibrated Gate Model (~5 ms).** LR/GBDT over the FeatureVector + Platt/isotonic
calibration so `p` is an honest probability. Second head predicts depth (SHALLOW vs DEEP). Resolves
when `p < τ_low` or `p > τ_high`; otherwise ABSTAIN.

**Tier 2 — LLM Adjudicator.** Small model, constrained JSON output, only inside the uncertain band.
Hard cap ≤ 10–15 % of traffic. **Timeout fails open to SHALLOW, never SKIP.** Every verdict is
written to a `DistillationBuffer` → training data for the next Tier-1 retrain.

### 6.3 Cost-sensitive thresholding

`λ = C(false skip) / C(false retrieve)` → retrieve-if threshold `τ* = 1 / (1 + λ)`.
With λ = 4, τ* = 0.2 (not 0.5). The abstention band `[τ_low, τ_high]` is set around τ* and tuned on
the validation set to hit the target Tier-2 firing rate. λ must be a **config value**, and the code
must support sweeping it.

### 6.4 Decision cache

Key `(tenant, LSH bucket of q_vec)`, TTL ~15 min, **invalidated whenever the consolidation worker
writes for that tenant**. Must be disable-able (ablation).

---

## 7. Memory fabric (L5) and write path (L7)

| Tier | Contents | Used by |
|---|---|---|
| M0 Working | Last N turns, in context | Always present; never retrieved |
| M1 Episodic | Raw turn records, embedded, timestamped | DEEP |
| M2 Semantic | Distilled atomic facts with confidence + validity intervals | SHALLOW, DEEP |
| M3 Learner Profile | Structured record: per-concept mastery, preferences, goals (~100–200 tokens) | SHALLOW, DEEP |
| M4 Memory Sketch | Cluster centroids, topic tags, Bloom filter over entities | **Gate only — never injected into a prompt** |

**Routes:** SKIP = M0 only, zero vector-store contact · SHALLOW = M3 block + top-2/3 from M2, no
rerank/hops · DEEP = wide-k over M1+M2+M3, metadata filters, MMR, cross-encoder rerank, one hop of
entity expansion.

**Async write path:** response path only enqueues a `WriteIntent`. Worker runs: salience filter →
LLM fact extraction → conflict resolution (insert / update / invalidate, validity intervals kept) →
upsert M1, M2 → EWMA update of M3 mastery → rebuild M4 sketch → invalidate tenant's decision cache.
`ForgettingManager`: decay, TTL, compaction, per-learner erasure.

**Consistency contract:** turn-N content is visible via M0 immediately and via M1/M2 within the
consolidation SLA. There is **no read-your-writes guarantee on long-term stores** — tests must assert
the M0 path, not a synchronous one.

---

## 8. Frozen interfaces

```python
AMICController.decide(ctx: RequestContext) -> MemoryDecision
RetrievalEngine.retrieve(plan: RetrievalPlan, q_vec, user_id) -> list[MemoryItem]
SketchStore.get(user_id).affinity(q_vec) -> tuple[float, float]   # (max_sim, margin)
TelemetryCollector.record(trace: DecisionTrace) -> None
IPipelineArm.answer(ctx: RequestContext) -> AgentResponse
```

Minimum fields:

- **RequestContext** — tenant/user id, session id, query text, M0 turns, arm name, budget, `q_vec` (filled once by signal bus)
- **MemoryDecision** — route (`SKIP|SHALLOW|DEEP|WRITE`), tier that resolved it (0/1/2/cache), `p`, τ_low/τ_high used, `RetrievalPlan`, token budget, rationale
- **RetrievalPlan** — stores to hit, k, filters, MMR on/off, rerank on/off, hops, token budget
- **MemoryItem** — id, tier (M1/M2/M3), text, score, timestamp, validity interval, metadata
- **DecisionTrace** — request id, arm, tier used, p, τ, route, latency split (signals / tiers / retrieval / generation / verify), prompt + completion tokens, escalation flag, cache hit, feature vector snapshot
- **AgentResponse** — answer text, decision, retrieved items, trace id

`IPipelineArm` has four implementations: `always`, `never`, `amic`, `oracle`.

---

## 9. Evaluation

### 9.1 Four arms

| Arm | Behaviour | Purpose |
|---|---|---|
| `always` | Retrieve every query | Conventional baseline |
| `never` | Never retrieve | Cost floor / impersonality ceiling |
| `amic` | Proposed system | Contribution |
| `oracle` | Retrieve iff ground-truth label says so | Upper bound → "% of achievable savings captured" |

### 9.2 Metrics

- **Decision quality:** precision/recall/F1 on *memory-needed*, false-skip rate separately, cost-weighted error at λ (and across a λ sweep), Tier-1 calibration (reliability curve, Brier score)
- **Efficiency:** retrieval rate, p50/p95 latency split (gate / retrieval / generation), mean prompt tokens, cost per 1 000 queries
- **Response quality:** LLM-as-judge pairwise vs `always`, ≥ 3 fixed seeds, variance reported; groundedness; personalization
- **Headroom:** share of `oracle` savings that `amic` captures

### 9.3 Break-even inequality

```
cost_AMIC = C_sig + P(T1)·C_T1 + P(T2)·C_T2 + r·C_ret + C_gen + P(esc)·(C_ret + C_gen)
cost_base = C_emb + C_ret + C_gen

AMIC wins iff:  C_sig + P(T1)·C_T1 + P(T2)·C_T2 + P(esc)·(C_ret + C_gen)  <  (1 − r)·C_ret
```

Telemetry must capture every term so this can be filled in from measured data automatically.

### 9.4 Ablations (config flags)

- Tier 0 only · Tier 0+1 · full cascade
- S3 sketch probe removed (query-only gating)
- τ fixed at 0.5 vs cost-sensitive τ from λ
- Escalation disabled
- DecisionCache disabled

### 9.5 Dataset — counterfactual oracle labelling

Run each query through `always` and `never`; an LLM judge scores both; materially worse without
memory ⇒ **memory-required**, else **memory-not-required**. Validate against ~200–300 human-labelled
queries (report agreement). Supplement with LoCoMo (incl. its adversarial/unanswerable category as a
memory-not-required analogue), possibly LongMemEval, and a synthetic education-domain tutoring set.

---

## 10. Build order

0. **Break-even probe:** stub pipeline; measure `C_emb`, `C_ret`, `C_gen`, rough `C_sig`; check the inequality can plausibly hold for realistic `r`.
1. **Skeleton + `always` / `never` arms.**
2. **Telemetry:** `DecisionTrace` + metrics store.
3. **Tier 0 alone.**
4. **Signal bus + M4 sketch + Tier 1** + ThresholdPolicy from λ.
5. **RoutePlanner, layered M0–M3 fabric, async write path, DecisionCache.**
6. **Tier 2 adjudicator + distillation loop.**
7. **ResponseVerifier + bounded escalation.**
8. **Experiments:** four-arm ExperimentRunner, ablations, λ/τ sweep, results tables.

Parallel track from milestone 2: Tutor UI + Decision Inspector + evaluation dashboard.
If time runs short: cut milestones 6 and 7.

---

## 11. Testing strategy

- **Unit:** each signal extractor; each tier in isolation; DecisionCache keying + invalidation; conflict resolver validity intervals; forgetting/decay
- **Route/path:** all three routes end to end; cache-hit path; Tier-2 timeout ⇒ SHALLOW; escalation never exceeds one retry
- **Consistency/isolation:** stale cache invalidated on write; M0 visibility of turn N; strict per-tenant separation; verified erasure across M1–M4
- **Performance budgets:** Tier 0 ≈ 0.1 ms, Tier 0+1 path ≲ 5 ms, Tier 2 ≤ 10–15 % of traffic; memory growth 10² → 10⁴ records
- **CI:** GitHub Actions runs tests + a regression check on gate metrics on each push

---

## 12. Deployment

docker-compose services: FastAPI ingress · consolidation worker · Redis · ChromaDB · metrics store ·
LLM gateway · Streamlit UI. Scheduled jobs: nightly Tier-1 distillation retrain, threshold
recalibration, memory decay/compaction. Staged rollout: local → single cloud GPU or hosted endpoint
→ shadow mode against `always`.

---

## 13. Open decisions

1. **LLM hosting** — local Llama 3 via Ollama/vLLM vs hosted API; small model for Tier 2 and judge.
2. **Evaluation corpus** — LoCoMo/LongMemEval, education-domain set, or both.
3. **Timeline** — decides whether milestones 6–7 are in scope.
4. **λ value** — default λ = 4 ⇒ τ* = 0.2.
5. **Metrics store** — SQLite/Postgres vs Parquet vs other.
6. **Repo shape** — monorepo with one package per layer vs separate services.

---

## Known risks

| Risk | Mitigation |
|---|---|
| Gate overhead exceeds savings | Milestone 0 break-even probe |
| No labelled memory-need data | Counterfactual oracle labelling |
| Tier 2 fires too often | Hard firing cap; widen τ band; fail-open on timeout |
| Per-tenant Chroma collection limits | Single collection + tenant metadata filter |
| LLM-judge variance | Pairwise, fixed seeds, report variance |
| Scope creep across four workstreams | Freeze §8 interfaces; `always` baseline first |
