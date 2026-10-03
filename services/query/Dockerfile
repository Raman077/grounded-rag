# Build from the repository root:  docker build -f services/query/Dockerfile -t grounded-rag-query .
FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY libs ./libs
COPY services ./services
RUN uv sync --frozen --no-dev --package query-service --package ingest-service

COPY prompts ./prompts
ENV PATH="/app/.venv/bin:$PATH" RAG_PROMPT_DIR=/app/prompts
# Download the default embedding model at build time so containers start offline.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "query_service.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
