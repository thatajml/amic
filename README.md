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

## Run the whole stack

```bat
start-amic.cmd              :: Ollama (+ tutor model preloaded), API on :8080, dashboard on :8501
start-amic.cmd -Infra       :: also Docker Redis/Chroma (automatic when the config needs them)
start-amic.cmd -Stop        :: stop the API and dashboard
```

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
