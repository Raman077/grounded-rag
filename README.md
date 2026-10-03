# grounded-rag

A production-grade Retrieval-Augmented Generation (RAG) platform built for answers that are **grounded in sources, cited, fast and fresh**.

The design focuses on what prototypes usually skip:

- **Latency**: streaming SSE responses, hybrid retrieval in a single round trip, semantic and prompt caching.
- **Hallucination control**: a sufficient-context gate, citation validation, cross-encoder reranking and groundedness checks.
- **Observability**: OpenTelemetry spans for every pipeline stage, Langfuse traces, and evaluation gates in CI.
- **Data freshness**: hash-diffed incremental ingestion now; CDC- and webhook-driven ingestion and blue/green index versions next.

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the full design: architecture diagram, ingestion, retrieval, generation and guardrails, the API contract, operations, tech stack and roadmap.

## What works today (Phase 1)

```
files -> parse (Markdown / HTML / PDF / text) -> structure-aware chunks -> embed -> Qdrant (tenant + ACL payload)
query -> JWT (tenant, groups) -> embed -> dense search, ACL-filtered in Qdrant -> relevance gate
      -> pack context -> LLM (Claude or OpenCode) -> citation check -> JSON or SSE stream
```

- `POST /v1/query` implements the contract in [docs/ARCHITECTURE.md §5](docs/ARCHITECTURE.md#5-main-query-api-contract-rest--sse): JSON or Server-Sent Events, RFC 7807 errors, 401/413/422/503 handling.
- Tenant and ACL groups come only from the verified JWT and are enforced inside the Qdrant query.
- "I don't know" in two layers: a similarity threshold skips the LLM when nothing relevant is found, and the model's `INSUFFICIENT_CONTEXT` reply becomes a clean fallback with related sources.
- Citations are validated: markers pointing at sources that were not supplied are removed.
- Re-ingesting unchanged files is a no-op (content hash); changed files replace their chunks without a gap.
- Every stage is an OpenTelemetry span; set `OTEL_EXPORTER_OTLP_ENDPOINT` to export to Jaeger or Langfuse.

### Phase 1 baseline

Measured on the 25-question golden set (`eval/golden/golden_v0.jsonl`) against the sample corpus, with
`BAAI/bge-small-en-v1.5` embeddings and `openrouter/qwen/qwen3.8-27b:free` via OpenCode as both the
generator and the judge. Full report: `eval/baselines/phase1.json`.

| Metric | Phase 1 | What it measures |
|---|---|---|
| Source recall | **1.00** | Expected source document cited (21 answerable questions) |
| Answer rate | **1.00** | Answerable questions answered rather than declined |
| Correct-refusal rate | **0.75** | No-answer and ACL-restricted questions declined (3 of 4) |
| Correctness (judge) | **0.97** | Answer matches the reference |
| Faithfulness (judge) | **0.83** | Claims supported by the cited snippets |
| Latency p50 / p95 | **4.3 s / 24.0 s** | End-to-end, free-tier model through OpenCode |

Notes:
- Faithfulness is a lower bound. The judge sees only the 300-character citation snippet, so correct answers
  whose supporting sentence sits later in the chunk were marked down (g002, g003, g009, g021). Phase 3 adds
  the in-service groundedness check against the full chunk text.
- The one missed refusal (g022, "Does Nimbus offer a GPU compute plan?") was answered "none of the listed
  plans is a GPU plan", citing the pricing table. That is grounded, but the metric counts it as a miss.
- The off-topic and ACL-restricted questions are stopped by the relevance gate in about 0.1 s without
  calling the LLM. Latency is dominated by the free model, not by retrieval.

## Quick start (no Docker needed)

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-packages
cp .env.example .env            # defaults: embedded Qdrant in ./data, OpenCode as the LLM

# 1. Index the sample corpus (first run downloads the ~130 MB embedding model)
uv run rag-ingest ingest eval/fixtures/corpus --tenant nimbus --acl public
uv run rag-ingest ingest eval/fixtures/corpus-internal --tenant nimbus --acl engineering

# 2. Start an LLM backend - either OpenCode (free models, no API key) ...
cd deploy/opencode && opencode serve --port 4096 &   # uses the tool-less "rag" agent defined here
cd ../..
#    ... or Claude: set RAG_LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY in .env

# 3. Run the API
uv run uvicorn query_service.main:create_app --factory --port 8000

# 4. Ask a question
TOKEN=$(uv run rag-devtoken --tenant nimbus --group public)
curl -N localhost:8000/v1/query \
  -H "Authorization: Bearer $TOKEN" -H "Accept: text/event-stream" -H "Content-Type: application/json" \
  -d '{"query": "What service credit do I get if uptime falls below 95%?"}'
```

Embedded Qdrant (`RAG_QDRANT_LOCATION=./data/qdrant`) locks its folder, so stop the API before
re-running ingestion. Use a Qdrant server (`docker compose -f deploy/docker-compose.yml up qdrant`,
then `RAG_QDRANT_LOCATION=http://localhost:6333`) to do both at once.

### With Docker

```bash
docker compose -f deploy/docker-compose.yml up --build                    # Qdrant, Jaeger, query API
docker compose -f deploy/docker-compose.yml run --rm ingest ingest /data --tenant nimbus --acl public
```

Traces appear in Jaeger at http://localhost:16686.

## LLM providers

| `RAG_LLM_PROVIDER` | Use | Notes |
|---|---|---|
| `anthropic` (default) | Production | Streams token by token from the Messages API; model `RAG_GENERATION_MODEL` (default `claude-sonnet-5`), low effort, thinking off for fast first tokens. |
| `opencode` | Local development | Calls a running `opencode serve` over HTTP, so any model OpenCode can reach works (OpenRouter free models, OpenCode Zen, ...). Run it from `deploy/opencode` so the tool-less `rag` agent is used. The OpenCode API returns the finished message, so the answer arrives in one SSE `token` event. |

## Evaluation

```bash
# with the API running
uv run python eval/harness/run_eval.py --judge opencode --out eval/reports/run.json \
    --baseline eval/baselines/phase1.json
```

Reports source recall, answer rate, correct-refusal rate (no-answer and ACL-restricted questions),
judge-scored correctness and faithfulness, and p50/p95 latency. With `--baseline` it exits non-zero
on a regression, which is what the CI gate in Phase 2 will use.

## Development

```bash
uv run pytest                     # offline: in-memory Qdrant, hashing embedder, scripted LLM
uv run pytest -m opencode         # live check against a running `opencode serve`
uv run ruff check . && uv run ruff format --check .
uv run mypy services/query/src services/ingest/src libs/core/src eval/harness
```

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| 1. Baseline | End-to-end dense RAG, SSE, JWT + ACL, tracing, golden set v0 | Done |
| 2. Retrieval quality | Hybrid search, reranking, parent-child chunking, query rewriting, CI eval gate | Next |
| 3. Production hardening | Guardrails, fallbacks, Temporal ingestion with CDC, SLOs, load tests | Planned |
| 4. Optimization | Semantic cache, model tiering, multi-query/HyDE, online eval loop | Planned |

## Repository layout

```
services/query/    FastAPI query service: api/, pipeline/, providers/ (Anthropic, OpenCode)
services/ingest/   Ingestion CLI: parsers/, chunking/, pipeline
libs/core/         Shared config, schemas (API contract), embeddings, vector store, telemetry
prompts/           Versioned prompt templates
eval/              Sample corpus, golden set, evaluation harness, baselines
deploy/            docker-compose, OpenCode agent config
```

## License

MIT
