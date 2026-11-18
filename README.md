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

## Ownership

The initial full build across all layers (L1–L8) is done by **@thatajml** (Ala Ajmal Malik). The
per-member ownership table in the brief describes the original split. The repo still follows those
layer boundaries so the parts can be handed back later.

## Status

Planning phase. The repo layout, Pydantic contracts and milestone plan are still pending
(see §13–14 of the brief).
