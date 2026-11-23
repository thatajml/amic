# AMIC: Adaptive Memory Invocation Controller

Cascaded, cost-sensitive gating of long-term memory retrieval for an LLM tutoring agent. Before
each query, AMIC decides whether to **SKIP**, do a **SHALLOW** retrieval or a **DEEP** retrieval.
It is evaluated against `always` / `never` / `oracle` baselines.

Final-year capstone, Group 8 / SP8, MIT-WPU Pune. Guide: Prof. Shoma Mitkari.

- Full brief: [docs/project_context.md](docs/project_context.md)
- Local build and dependencies: [docs/LOCAL_BUILD.md](docs/LOCAL_BUILD.md)

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -e ".[gate,ui,eval,dev,llm-local]"
docker compose -f docker-compose.dev.yml up -d
cp .env.example .env
pytest
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
- Under Docker, `docker-compose.yml` sets the Ollama URL, the Redis and Chroma hosts and the
  shadow baseline for the containers. Those values override `.env`.

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
docker compose up -d                 :: build once, then API :8080, dashboard :8501, Redis, Chroma
docker compose run --rm chat         :: terminal chat (same options as chat-amic.cmd, e.g. --frozen)
docker compose down                  :: stop everything
docker compose up -d --build         :: rebuild after changing code
```

`data/`, `models/` and `configs/` are shared with the host, so traces, memory, the trained Tier-1
model and the dashboard are the same as in a local run. The image uses CPU-only torch, because
it only embeds queries, and has the embedding model built in. It is about 3.4 GB. The services
restart with Docker Desktop. Don't run this and `start-amic.cmd` at the same time, because both
use ports 8080 and 8501.

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

## Ownership

The initial full build across all layers (L1–L8) is done by **@thatajml** (Ala Ajmal Malik). The
per-member ownership table in the brief describes the original split. The repo still follows those
layer boundaries so the parts can be handed back later.

## Status

Planning phase. The repo layout, Pydantic contracts and milestone plan are still pending
(see §13–14 of the brief).
