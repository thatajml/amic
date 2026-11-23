# AMIC app image: API, dashboard and terminal chat. Used by docker-compose.yml.
# The LLM is not inside: containers call the Ollama running on the host.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

# libgomp1: OpenMP for LightGBM (Tier-1 variant). curl: health checks.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch is enough: torch only embeds queries (MiniLM, models.embedding_device=cpu);
# generation runs in Ollama. Installed first so pip never pulls the multi-GB CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies before code, so editing src/ does not reinstall them.
COPY pyproject.toml README.md ./
RUN mkdir -p src/amic && touch src/amic/__init__.py \
 && pip install ".[gate,llm-local,llm-hosted,ui]" \
 && pip uninstall -y amic

# Embedding model baked in, so containers start without a download.
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"

COPY src ./src
COPY ui ./ui
COPY scripts ./scripts
COPY configs ./configs
RUN pip install --no-deps .

EXPOSE 8080 8501
CMD ["uvicorn", "amic.service.api:app", "--host", "0.0.0.0", "--port", "8080"]
