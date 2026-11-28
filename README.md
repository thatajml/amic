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

Every component of the brief is implemented on `main`: Tier 0/1/2 with cost-sensitive thresholds,
the S1-S5 signals and the M4 sketch, the decision cache, verify-and-escalate, M1 episodic + M2
fact memory with validity intervals + the M3 learner profile (EWMA mastery), the brief's SHALLOW
(M3 + top facts) and DEEP (M1 + M2 + M3, MMR, cross-encoder rerank, one entity hop) routes, the
WRITE route (`/remember`, `/forget`), an async write path (SQLite or Redis Streams queue +
ConsolidationWorker), the ForgettingManager (decay, TTL, compaction, erasure), scheduled jobs
(distillation retrain, threshold recalibration, forgetting), shadow mode against `always`, the
Educator Console, performance-budget tests and a CI gate-metric regression check. This branch
(`full-stack-docker`) adds the usage dashboard, one-command startup and the docker-compose stack
with every service. The chat UI / Decision Inspector lives on the unmerged `ui-chat` branch.

The first evaluation (805 labelled probes, 98 hand labels, seven live runs × 3 seeds, pairwise
judging) was followed by two **pre-registered extensions** for the later components
([docs/PLAN.md](docs/PLAN.md) sections 9 and 10). All intervals are 95% cluster-bootstrap CIs.

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
| `always_m3` (brief SHALLOW: M3 profile + top-3 facts every turn) | 212 | −39.3 [−46.4, −32.1] |
| `always_deep2` (brief DEEP: M1+M2+M3, rerank, 1 hop, every turn) | 794 | −1.2 [−9.0, +6.6] |
| `amic_v3` (brief-complete: T0–T2, cache, escalation, brief routes) | 293 | −30.4 [−39.8, −20.2] |
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
- **Extracted facts are far cheaper but much worse** than raw turns (−36 pts vs `always_shallow`);
  adding the M3 profile does not fix it (brief SHALLOW −35 pts vs `always_shallow`).
- **Rerank + entity hop add nothing to DEEP** (−1.2 pts, n.s.) and cost ~170 ms of retrieval.
- **Distilling Tier 2 into Tier 1** raises test AUC by +0.04 (learner CI includes 0) and
  changes no gate decision.
- **The S5 uncertainty signal** (draft-token entropy) adds +0.035 AUC and −0.019 expected cost,
  neither significant, for ~360 ms per request (the whole T0+T1 gate takes ~1 ms).
- **Performance budgets hold**: Tier 0 < 0.001 ms, Tier 0+1 ~0.95 ms (budget 5 ms); gate signal
  cost stays flat (0.07 ms) from 10^2 to 10^4 memory records.
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

### Python environment: venv or conda

`setup-amic.cmd` creates or updates the environment with every extra installed. On an NVIDIA
machine it installs CUDA torch first. Re-run it to update.

```bat
setup-amic.cmd conda        :: conda env "amic" from environment.yml (Miniconda, conda-forge only)
setup-amic.cmd venv         :: .venv in the repo
setup-amic.cmd conda -Cpu   :: skip CUDA torch
```

It records the choice as `AMIC_ENV` in `.env`, and `start-amic.cmd` and `chat-amic.cmd` use it:
`venv`, `conda` (env `AMIC_CONDA_ENV`, default `amic`), or `auto` (`.venv` if present, else
conda). `AMIC_PYTHON=<path>` overrides all of these. To work in a shell, run `conda activate amic`.
`conda env create -f environment.yml` also works, but on Windows it installs CPU-only torch.

## Configuration

Defaults live in [src/amic/contracts/config.py](src/amic/contracts/config.py). Change them in
`.env` (start with `copy .env.example .env`, which lists every setting) or with environment
variables:

- The app reads only `AMIC_*` names. Nested settings use `__`: `AMIC_MODELS__TUTOR=llama3.2:3b`
  sets `models.tutor`, and `AMIC_GATE__LAM=4.0` sets `gate.lam`. A name without the prefix, such
  as `LLM_MODEL`, is ignored.
- `GEMINI_API_KEY` is the exception: the judge reads it by that name. Only the offline
  evaluation needs it.
- `configs/base.yaml` and `configs/ablations/*.yaml` are experiment configs, passed with
  `--config` to `python -m amic.eval.runner` and `python -m amic.demo`. A YAML file outranks
  `.env`. The API and dashboard don't read these files.
- `configs/pricing.yaml` holds the per-model prices the dashboard uses.
- Under Docker, `docker-compose.yml` sets the Ollama URL, the Redis and Chroma hosts, the
  shadow baseline, and the shared-store topology (Redis sessions and write queue, Chroma server)
  for the containers. Those values override `.env`.

## Run with Docker (one command)

**Requirements**

- **Docker Desktop**, running.
- **Ollama on the host** (the Windows app), running, with the tutor model pulled:
  `ollama pull llama3.1:8b`. The LLM is not containerised. The containers call this Ollama at
  `host.docker.internal:11434`, so it keeps using your GPU and the models you already downloaded.
  Ollama's default settings work. You don't need to change `OLLAMA_HOST`.
- A `.env` file (`copy .env.example .env`, see [Configuration](#configuration)). The containers
  read the same settings as a local run.

```bat
docker compose up -d                 :: build once, then start everything below
docker compose run --rm chat         :: terminal chat (same options as chat-amic.cmd, e.g. --frozen)
docker compose logs -f worker jobs   :: what the background services are doing
docker compose down                  :: stop everything
docker compose up -d --build         :: rebuild after changing code
```

| Service | What it is |
|---|---|
| `api` | `POST /chat`, `GET`/`DELETE /learners/{id}`, `/traces` on http://localhost:8080 (docs at `/docs`) |
| `worker` | Consolidation worker: stores each answered turn in M1, updates the M3 profile, applies `/remember` and `/forget`. Set `AMIC_WRITEPATH__EXTRACT_FACTS=true` in `.env` to also extract M2 facts with the host's Ollama |
| `educator` | Educator Console on http://localhost:8502: a learner's memory and gate decisions, plus close-fact, add-note, correct-mastery and erase |
| `jobs` | Nightly jobs, once at start and then every 24 h: Tier-1 retrain with a promotion gate, band recalibration, forgetting. Results go to `data/jobs.jsonl` |
| `dashboard` | Usage and savings dashboard on http://localhost:8501 |
| `redis`, `chroma` | Shared stores: sessions, memory version, M3 profiles and the write queue in Redis; M1 and M2 in the Chroma server |

The API, worker, console and jobs are separate processes. So under Docker, memory lives in the
Redis and Chroma containers (named volumes `redis-data` and `chroma-data`), not in `data/chroma`
or `data/sessions.sqlite`. The API answers without waiting for the worker. If the worker is
stopped, new turns wait in the Redis stream until it is back. The terminal chat is the
exception: like `chat-amic.cmd`, it runs in one process on `data/chroma` (the seeded `edu-*`
learners) and writes its own turns.

`data/`, `models/` and `configs/` are shared with the host, so traces, the trained Tier-1 model
and the dashboard are the same as in a local run. The image uses CPU-only torch, because it only
embeds queries, and has the embedding model built in. It is about 3.4 GB. The services restart
with Docker Desktop. Don't run this and `start-amic.cmd` at the same time, because both use
ports 8080 and 8501. The Educator Console and the API have no login, so their ports are bound to
localhost only.

Run `docker compose` from a normal Windows terminal (PowerShell or cmd). From Git Bash, a
lowercase `/c/...` path can make the build stop with `additional privileges requested`.

## Run without Docker

```bat
start-amic.cmd              :: Ollama (+ tutor model preloaded), API on :8080, dashboard on :8501
start-amic.cmd -Infra       :: also Docker Redis/Chroma (automatic when the config needs them)
start-amic.cmd -Stop        :: stop the API and dashboard
```

To chat with the tutor in the terminal, run `chat-amic.cmd`. Each answer is followed by AMIC's
decision panel (route, reason, memory used, latency, tokens). It chats as seeded learner `edu-001`
and saves new turns to memory. Use `--learner <name>` for a new learner, `--frozen` to save
nothing, `--arm always` to compare with the always-retrieve baseline, and `--script` for 3 canned turns.

Each step is skipped if it is already up. Logs and PIDs are in `data/run/`. The API runs with
the shadow baseline on, so the dashboard shows savings.

## Usage dashboard

A live view of token usage per model and of what the gate saves compared with always retrieving:

```bash
pip install -e ".[ui]"
streamlit run ui/dashboard.py                         # reads data/metrics.db, refreshes every 5 s
python scripts/simulate_traffic.py --rate 0.5         # replays edu questions (frozen edu_v1 memory)
python scripts/simulate_traffic.py --fake-llm --rate 5 --models fake-a fake-b   # no Ollama
```

For real API traffic, start the API with `AMIC_TELEMETRY__SHADOW_BASELINE=true`. After each
response it measures how many memory tokens the always-retrieve baseline would have used. To
compare several models, list the extra ones in `models.available` and pass `"model"` in
`POST /chat`. Prices are in `configs/pricing.yaml` and are placeholders until you set real ones.
Try the gate on the seeded learners (needs the dataset below):

```bash
python -m amic.demo --script               # cold start → SKIP, general question, "what did I get wrong…" → DEEP
python -m amic.demo --learner edu-001      # interactive, with a decision panel per turn
```

### Chat UI: any LLM with AMIC

```bash
streamlit run ui/chat.py
```

Pick the tutor model (a local Ollama model, or any OpenAI-compatible API: Gemini, OpenAI,
OpenRouter, Groq, LM Studio/vLLM or a custom URL), a learner (a seeded one, or a new learner with no
memory), the arm (`amic` / `always` / `never`) and λ. Every answer shows the route, which tier
decided, p against the threshold band, the rationale, the strongest signals, the retrieved memory,
the latency split and tokens; the sidebar keeps session totals. API keys are read from `.env` or
pasted for the session only, never stored.

**Accounts.** Create an account with a name and a PIN (4+ characters) and sign in again later with
the same pair. Every turn is saved to your own long-term memory and your M3 learner profile (your
name, goals, concepts you struggle with) is created at sign-up, so the tutor remembers you across
sessions. Accounts live in `data/chat_users.sqlite` (PIN stored only as a salted scrypt hash; there
is no recovery, a forgotten PIN means a new account). Account memory goes to its own Chroma
collection `amic_users`, never to the seeded `edu_v1`. **Guest** keeps nothing. **Seeded learner**
(`edu-*`) is read-only unless you turn writes on, so chatting cannot change the memory the
experiments use. While the model answers, the chat shows a shimmering "thinking" indicator.

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

Completion (pre-registered, docs/PLAN.md section 10):

```bash
python scripts/build_profiles.py                                   # M3 from history + facts
python -m amic.eval.runner --arms always --name always_m3 --config $A/always_shallow.yaml $A/m3.yaml
python -m amic.eval.runner --arms always --name always_deep2 --config $A/deep_v2.yaml
python -m amic.eval.runner --arms amic --name amic_v3 --config $A/full.yaml $A/escalate.yaml $A/cache.yaml $A/m3.yaml $A/deep_v2.yaml $A/v3.yaml
python scripts/judge_pairwise.py --runs amic_v3 --reuse-from always_m3 always_deep2 amic_v2 amic never
python scripts/extract_s5.py && python scripts/train_tier1.py --s5 --name tier1_s5             # S1
python scripts/train_tier1.py --distill data/distill/replay_train.jsonl --distill-weight 0.5 --name tier1_distill05
python scripts/compare_tier1.py tier1_distill05 tier1_distill10 tier1_distill_only tier1_s5  # S1, DS1
python scripts/bench_gate.py                                      # PB1
```

Running the system:

```bash
uvicorn amic.service.api:app                  # API: /chat, /learners/{id} (GET, DELETE), /traces
python -m amic.writepath.worker               # consolidation (AMIC_WRITEPATH__QUEUE=sqlite|redis;
                                              #  AMIC_WRITEPATH__EXTRACT_FACTS=true for M2)
python -m amic.jobs nightly --every 24        # retrain (promotion gate) + recalibrate + forget
streamlit run ui/educator.py                  # Educator Console
AMIC_SHADOW__ENABLED=true uvicorn ...         # shadow mode vs always; scripts/shadow_report.py
python scripts/smoke_service.py               # API + worker processes on real Redis + Chroma
```

`/remember <text>` and `/forget <text>` in a chat are explicit memory commands (route WRITE).

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
- Chat UI (`ui/chat.py`, `src/amic/chatui.py`): a Streamlit chat to run any LLM (local Ollama or any
  OpenAI-compatible API) with AMIC and inspect each decision. Merged into `main`.
- Usage/cost dashboard, one-command startup and the full docker-compose stack (worker, Educator
  Console, jobs): merged into `main`.

## Ownership

The initial full build across all layers (L1–L8) is done by **@thatajml** (Ala Ajmal Malik). The
per-member ownership table in the brief describes the original split. The repo still follows those
layer boundaries so the parts can be handed back later.
