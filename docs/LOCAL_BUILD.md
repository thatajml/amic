# Local build: dependencies and setup

Target machine: Windows 11 / macOS / Linux dev laptop, 16 GB RAM. A CUDA GPU is needed only if
the LLM runs locally (open decision §13.1).

## 1. System dependencies

| Tool | Version | Required? | Purpose |
|---|---|---|---|
| Python | 3.11 – 3.13 | **Yes** | Runtime (`requires-python = ">=3.11,<3.14"`) |
| Git | 2.40+ | **Yes** | Source control |
| Docker Desktop (with Compose v2) | 24+ | **Yes** | Runs Redis locally (no native Windows build); optional Chroma server |
| GitHub CLI (`gh`) | 2.x | Recommended | Repo, PR and CI operations |
| uv | 0.5+ | Recommended | Fast installs and a reproducible lock file (`uv.lock`); plain `pip` also works |
| NVIDIA driver + CUDA 12.x runtime | recent | Only for a local LLM / GPU embeddings | RTX-class GPU; PyTorch wheels bundle the CUDA libraries |
| Ollama | 0.40+ | **Yes** (decided: local) | Serves Llama 3 (8B Q4 fits in 8 GB VRAM) plus the small Tier-2 / judge model |
| MSVC Build Tools (C++) | 2022 | Fallback only | Needed only if a wheel such as `chroma-hnswlib` has to compile from source on Windows |

## 2. Infrastructure services (docker-compose)

| Service | Image | Port |
|---|---|---|
| Redis | `redis:7-alpine` | 6379 |
| ChromaDB (optional; default is embedded in-process) | `chromadb/chroma:1.5.9` | 8000 |

The metrics store defaults to SQLite (one file, no service) until §13.5 is decided.

## 3. Python packages (from `pyproject.toml`)

**Core** (installed with `pip install -e .`):
fastapi, uvicorn[standard], pydantic v2, pydantic-settings, httpx, pyyaml, langgraph, langchain-core,
numpy, sentence-transformers, chromadb, redis, sqlalchemy, pandas, pyarrow, structlog, tenacity, orjson.

**Optional extras**, grouped by layer:

| Extra | Packages | Used by (layer) |
|---|---|---|
| `gate` | scikit-learn, lightgbm, joblib | L4 Tier-1 gate |
| `llm-local` | ollama, langchain-ollama | L6 LLMGateway (local) |
| `llm-hosted` | openai, langchain-openai | L6 LLMGateway (hosted / OpenAI-compatible) |
| `ml` | torch, transformers, accelerate | L5 rerank, local HF models |
| `ui` | streamlit, plotly | L1 UI |
| `eval` | scipy, matplotlib, datasets, tqdm | L8 evaluation |
| `dev` | pytest, pytest-asyncio, pytest-cov, pytest-benchmark, pytest-timeout, hypothesis, fakeredis, ruff, mypy, pre-commit | All |

> MMR, the LSH decision-cache key and the M4 Bloom filter are small enough to write in numpy, so
> they add no dependency.

## 4. Setup steps

```bash
git clone <repo-url> amic && cd amic

# virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate

# GPU only: install CUDA PyTorch FIRST. On Windows, PyPI's default torch wheel is CPU-only.
pip install torch --index-url https://download.pytorch.org/whl/cu124

# project + extras
pip install -e ".[gate,ui,eval,dev,llm-local]"     # or llm-hosted, or [all]

# infrastructure
docker compose -f docker-compose.dev.yml up -d

# config
cp .env.example .env

# sanity check
pytest
```

With uv: `uv venv && uv pip install -e ".[all]"`. Then commit `uv.lock` (`uv lock`) so every machine resolves the
same versions.

## 5. Models to pull (once §13.1 is decided)

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-d, CPU-friendly). Downloaded automatically on first use.
- Cross-encoder reranker (DEEP route): `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Local LLM: `ollama pull llama3.1:8b`, plus a small model for Tier 2, e.g. `ollama pull llama3.2:3b`.
