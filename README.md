# grounded-rag

A production-grade Retrieval-Augmented Generation (RAG) platform built for answers that are **grounded in sources, cited, fast and fresh**.

The design focuses on what prototypes usually skip:

- **Latency**: streaming SSE responses, hybrid retrieval in a single round trip, semantic and prompt caching.
- **Hallucination control**: cross-encoder reranking, a sufficient-context gate, citation validation and groundedness checks.
- **Observability**: OpenTelemetry spans for every pipeline stage, Langfuse traces, and RAGAS evaluation gates in CI.
- **Data freshness**: CDC- and webhook-driven incremental ingestion with hash diffing and blue/green index versions.

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the full design: architecture diagram, ingestion, retrieval, generation and guardrails, the API contract, operations, tech stack and roadmap.

## Architecture at a glance

```
Sources -> Ingestion (Temporal: parse -> chunk -> enrich -> embed)
        -> Storage (Qdrant dense+sparse, Postgres, Redis, object store)
        -> Query service (FastAPI: guard -> understand -> hybrid retrieve -> rerank -> pack -> generate -> guard)
        -> SSE stream with citations
```

## Tech stack

| Area | Choice |
|---|---|
| Language and API | Python 3.12, FastAPI, Pydantic v2 |
| Ingestion orchestration | Temporal |
| Vector store | Qdrant (dense and sparse vectors, RRF fusion) |
| Data and cache | Postgres 16, Redis Stack |
| LLM | Claude, via the Anthropic SDK directly |
| Reranker | Cohere Rerank or bge-reranker-v2-m3 |
| Eval and observability | RAGAS, OpenTelemetry, Langfuse, Prometheus/Grafana |

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| 1. Baseline | End-to-end dense RAG, SSE, tracing, golden set v0 | Planned |
| 2. Retrieval quality | Hybrid search, reranking, parent-child chunking, query rewriting, CI eval gate | Planned |
| 3. Production hardening | Guardrails, fallbacks, Temporal ingestion with CDC, SLOs, load tests | Planned |
| 4. Optimization | Semantic cache, model tiering, multi-query/HyDE, online eval loop | Planned |

## Repository layout

```
services/query/    FastAPI query service
services/ingest/   Temporal ingestion workflows and activities
libs/core/         Shared schemas, config and telemetry
prompts/           Versioned prompt templates
eval/              Golden datasets and evaluation harness
deploy/            docker-compose, Helm and Terraform
```

## License

MIT
