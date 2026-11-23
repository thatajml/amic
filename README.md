# AMIC: Adaptive Memory Invocation Controller

Cascaded, cost-sensitive gating of long-term memory retrieval for an LLM tutoring agent. Before
each query, AMIC decides whether to **SKIP** memory, do a **SHALLOW** retrieval (top 3 items) or a
**DEEP** retrieval (wide search + MMR, 12 items). It is evaluated against `always` / `never` /
`oracle` baselines on a synthetic intro-Python tutoring set, with LoCoMo as an external check.

Final-year capstone, Group 8 / SP8, MIT-WPU Pune. Guide: Prof. Shoma Mitkari.

- Full brief: [docs/project_context.md](docs/project_context.md)
- Implementation plan and decisions: [docs/PLAN.md](docs/PLAN.md)
- **Results (generated, every number from this repo's runs): [docs/results/main.md](docs/results/main.md)**
- Local build and dependencies: [docs/LOCAL_BUILD.md](docs/LOCAL_BUILD.md)

## How it works

```
query ──► signal bus (embed once; S1 lexical, S2 session, S3 memory sketch, S4 parametric)
      ──► Tier 0 rules  (no memory → SKIP; explicit "last time…" → DEEP; …)
      ──► Tier 1 calibrated classifier: p = P(memory needed)
            p < τ_low → SKIP · p > τ_high → retrieve (depth head picks SHALLOW/DEEP)
            in the band → SHALLOW (fail open, never SKIP)
      ──► retrieve (Chroma) ──► generate (any LLM) ──► trace (SQLite)
```

τ\* = 1/(1+λ), where λ = cost of a wrong skip ÷ cost of a needless retrieval (default λ = 4,
τ\* = 0.2). The same LangGraph pipeline serves every arm; arms differ only in the decision step.

## Status (2026-11-23)

Every component of the brief except the M3 learner profile, DEEP rerank/entity hop and the
Educator Console is implemented: Tier 0/1/2 with cost-sensitive thresholds, the M4 sketch,
verify-and-escalate, the decision cache, M1 episodic + M2 fact memory with validity intervals, an
async write path (SQLite or Redis Streams queue + ConsolidationWorker), and LongMemEval. The first
evaluation (805 labelled probes, 98 hand labels, seven live runs × 3 seeds, pairwise judging)
was followed by a **pre-registered extension** for the later components
([docs/PLAN.md](docs/PLAN.md) section 9). All intervals are 95% cluster-bootstrap CIs.

**Education set** (test split, 161 probes × 3 seeds; net win = pairwise wins − losses vs `always`):

| Run | Memory tokens / query | Net win vs `always` (learner CI) |
|---|---|---|
| `always` (DEEP every turn) | 800 | 0 |
| `always_shallow` (top-3 raw turns every turn, no gate) | 314 | −4.3 [−11.7, +3.1] |
| `amic` (T0+T1, λ = 4) | 369 | −6.8 [−14.6, +1.2] |
| `amic_esc` (+ verify-and-escalate) | 370 | −6.6 [−14.6, +1.5] |
| `amic_full` (+ Tier 2) | 370 | −6.8 [−14.4, +0.8] |
| `amic_lam1` (λ = 1) / `amic_lam1_esc` | 269 / 290 | −19.5 / −15.1 |
| `always_m2` (top-3 extracted facts every turn) | 69 | −40.4 [−46.6, −33.8] |
| `amic_v2` (full system: T0–T2, cache, escalation, M2 routes) | 200 | −30.0 [−36.4, −22.9] |
| `never` | 0 | −62.9 |

**LongMemEval_S** (cleaned; gold-answer accuracy on a stratified 150; gate decisions on all 500):
`always` 45.8%, `always_shallow` 45.8% with 4× fewer memory tokens, `amic` 48.3%, `never` 5.0%.
The education-trained gate retrieves on 100% of LongMemEval questions.

- **SHALLOW retrieval is as good as DEEP**, on the education set and on LongMemEval's objective
  accuracy, at 60–75% fewer memory tokens.
- **Skipping memory hurts**: SKIPs lose ~70% of pairwise comparisons, even the oracle's.
- **The gate does not beat always-SHALLOW**; the S3 sketch probe and Tier 2 change nothing
  measurable (`amic_full − amic` = +0.0).
- **Verify-and-escalate recovers part of the loss only where skipping is frequent** (+4.3 pts at
  λ = 1, +18.8 for `never`), and gating first beats verifying afterwards by 37.5 pts.
- **Extracted facts are far cheaper but much worse** than raw turns (−36 pts vs `always_shallow`).
- **The async write path** takes the write off the response path: 2 ms to enqueue vs 17 ms
  inline (4.5 s with fact extraction).
- **Labels and the judge bound everything**: human vs judge κ = 0.26; the judge agrees with
  itself on ~73% of identical pairs judged twice.

## Quick start

Requirements: Python 3.11–3.13, [Ollama](https://ollama.com) with `llama3.1:8b`; Docker only if
you use Redis for sessions.

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows (source .venv/bin/activate elsewhere)
pip install -e ".[gate,ui,eval,dev,llm-local,llm-hosted]"
ollama pull llama3.1:8b
cp .env.example .env            # add GEMINI_API_KEY (judge/dataset only); set AMIC_SESSION_BACKEND=sqlite to skip Docker
pytest                          # unit + integration tests, no Ollama or API key needed
```

Try the gate on the seeded learners (needs the dataset below):

```bash
python -m amic.demo --script               # cold start → SKIP, general question, "what did I get wrong…" → DEEP
python -m amic.demo --learner edu-001      # interactive, with a decision panel per turn
```

## Reproducing the results

Each step is resumable; LLM-judge steps sleep through Gemini daily-quota resets with
`--wait-for-quota` and fall back to `GEMINI_API_KEY_2`, `_3`, … if set.

```bash
# 1. data
python scripts/gen_edu_dataset.py --learners 40 && python scripts/gen_edu_dataset.py --seed-memory
python scripts/prepare_locomo.py && python scripts/seed_sessions.py
python scripts/extract_features.py

# 2. counterfactual oracle labels (SKIP / SHALLOW / DEEP answers, judged twice each)
python scripts/label_oracle.py generate
python scripts/label_oracle.py judge --wait-for-quota
python scripts/label_oracle.py label
python scripts/hand_label.py --score                       # human vs judge kappa

# 3. Tier 1 (+ ablation variants)
python scripts/train_tier1.py
python scripts/train_tier1.py --no-s3 --name tier1_no_s3
python scripts/train_tier1.py --kind lgbm --name tier1_lgbm

# 4. live runs on the test split (3 seeds)
python -m amic.eval.runner --arms always never oracle amic
python -m amic.eval.runner --arms amic --name amic_lam1 --config configs/ablations/lam1.yaml
python -m amic.eval.runner --arms amic --name amic_no_s3 --config configs/ablations/no_s3.yaml --tier1 models/tier1_no_s3.joblib
python -m amic.eval.runner --arms always --name always_shallow --config configs/ablations/always_shallow.yaml

# 5. pairwise quality vs always, then every table and figure
python scripts/judge_pairwise.py --runs amic never oracle amic_lam1 amic_no_s3 always_shallow --wait-for-quota
python -m amic.eval.report                                 # → docs/results/main.md
```

Extension (pre-registered, docs/PLAN.md section 9); tutor answers come from the generation
cache and identical answer pairs reuse earlier verdicts (`--reuse-from`):

```bash
A=configs/ablations
# verify-and-escalate (E1-E4) and Tier 2 (T1, T2)
python -m amic.eval.runner --arms amic  --name amic_esc      --config $A/escalate.yaml
python -m amic.eval.runner --arms amic  --name amic_lam1_esc --config $A/lam1_escalate.yaml
python -m amic.eval.runner --arms never --name never_esc     --config $A/escalate.yaml
python scripts/replay_verify.py --runs never amic amic_lam1
python scripts/tune_band.py --rate 0.12 && python scripts/replay_tier2.py
python -m amic.eval.runner --arms amic  --name amic_full     --config $A/full.yaml
# M2 facts, the full system, cache and write path (F1, V1, C1, W1)
python scripts/extract_facts.py
python -m amic.eval.runner --arms always --name always_m2 --config $A/m2.yaml $A/always_shallow.yaml
python -m amic.eval.runner --arms amic --name amic_v2 --config $A/full.yaml $A/escalate.yaml $A/m2.yaml $A/cache.yaml $A/v2.yaml
python scripts/replay_cache.py && python scripts/bench_writepath.py && python scripts/bench_writepath.py --extract
python scripts/judge_pairwise.py --runs amic_esc --reuse-from amic always_shallow   # etc. per run
# LongMemEval (L1, L2): download per amic/eval/datasets/longmemeval.py, then
python scripts/prepare_longmemeval.py --device cuda
python -m amic.eval.runner --arms always --name lme_always --dataset longmemeval --seeds 1 --temperature 0 --config configs/longmemeval.yaml
python scripts/judge_longmemeval.py --runs lme_always lme_always_shallow lme_never lme_amic lme_amic_esc
python -m amic.eval.report
```

The async worker for a live service: `python -m amic.writepath.worker` (set
`AMIC_WRITEPATH__QUEUE=sqlite` or `redis`, and `AMIC_WRITEPATH__EXTRACT_FACTS=true` for M2).

Decision-quality ablations and the λ sweep are replayed offline from stored features (no LLM
calls); only response quality needs generation.

## Repository layout

```
src/amic/
  contracts/      Pydantic models, Protocols, Settings (YAML + AMIC_* env)
  control/        signal bus, Tier 0, Tier 1, thresholds, route planner, controller
  memory/         embedder, Chroma episodic store, retrieval (SHALLOW / DEEP+MMR), M4 sketch
  generation/     prompt builder; Ollama and OpenAI-compatible gateways; experiment cache
  orchestration/  LangGraph pipeline shared by all arms
  service/        FastAPI app, session store (Redis or SQLite)
  arms/           always / never / oracle / amic
  eval/           datasets, runner, offline replay, quota-aware judging, report
scripts/          dataset, labelling, training and judging entry points
configs/          base.yaml, ablations/*.yaml
docs/             brief, plan, generated results
```

## Branches

- `main`: the system, experiments and results.
- Paper artefacts (data, answers, judgments, models): GitHub releases `artifacts-2026-11-23*`.
- `ui-chat`: a Streamlit chat UI to run any LLM (local Ollama or any OpenAI-compatible API) with
  AMIC and inspect each decision. Not merged yet.
- `dashboard-and-startup-command`: usage/cost dashboard and a one-command startup. Not merged yet.

## Ownership

The initial full build across all layers (L1–L8) is done by **@thatajml** (Ala Ajmal Malik). The
per-member ownership table in the brief describes the original split. The repo still follows those
layer boundaries so the parts can be handed back later.
