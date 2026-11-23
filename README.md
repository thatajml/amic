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

Milestones 0–4 and 8 of the brief are implemented; Tier 2, verification/escalation and fact
extraction are future work (see [docs/PLAN.md](docs/PLAN.md) §1). The full evaluation has been run:
805 labelled probes (two LLM-judge judgments each, 98 hand labels), seven live runs × 3 seeds on
the held-out test split, and pairwise quality judging of every run against `always`.

**Headline results** (test split, 161 probes × 3 seeds; details and caveats in
[docs/results/main.md](docs/results/main.md)):

| Run | Memory tokens / query | Saved vs `always` | Net win vs `always` |
|---|---|---|---|
| `always` (DEEP every turn) | 800 | – | 0 |
| `amic` (λ = 4) | 369 | 54% | −6.8 pts |
| `always_shallow` (SHALLOW every turn, no gate) | 314 | 61% | −4.3 pts |
| `amic_lam1` (λ = 1) | 269 | 66% | −19.5 pts |
| `oracle` (labels) | 239 | 70% | −18.4 pts |
| `never` | 0 | 100% | −62.9 pts |

- **SHALLOW retrieval is as good as DEEP** for this dataset and saves ~60% of memory tokens.
- **Skipping memory hurts**: a SKIP loses ~70% of pairwise comparisons, even when the oracle
  label said memory was not needed.
- **The gate does not beat the simple always-SHALLOW baseline** here: `always_shallow` uses fewer
  tokens with better quality than `amic`. The S3 sketch ablation changes nothing measurable.
- **The gate is cheap** (~2 ms per query vs 10–150 ms of retrieval + prefill saved), so the
  break-even inequality holds; end-to-end latency barely changes because decoding dominates.
- **Labels are the main limitation**: human vs judge Cohen's κ = 0.26, and the judge's necessity
  label disagrees with its own pairwise preferences on SKIP.

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

### Chat UI: any LLM with AMIC

```bash
streamlit run ui/chat.py
```

Pick the tutor model (a local Ollama model, or any OpenAI-compatible API: Gemini, OpenAI,
OpenRouter, Groq, LM Studio/vLLM or a custom URL), a learner (a seeded one, or a new learner with no
memory), the arm (`amic` / `always` / `never`) and λ. Every answer shows the route, which tier
decided, p against the threshold band, the rationale, the strongest signals, the retrieved memory,
the latency split and tokens; the sidebar keeps session totals. API keys are read from `.env` or
pasted for the session only, never stored. Long-term memory is read-only unless you turn writes on,
so chatting cannot change the seeded memory the experiments use.

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
- `ui-chat` (this branch): adds the chat UI above (`ui/chat.py`, `src/amic/chatui.py`).
- `dashboard-and-startup-command`: usage/cost dashboard and a one-command startup. Not merged yet.

## Ownership

The initial full build across all layers (L1–L8) is done by **@thatajml** (Ala Ajmal Malik). The
per-member ownership table in the brief describes the original split. The repo still follows those
layer boundaries so the parts can be handed back later.
