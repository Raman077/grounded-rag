"""App factory. Run with:

uvicorn query_service.main:create_app --factory --port 8000
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from rag_core.config import Settings, get_settings
from rag_core.embeddings import Embedder, build_embedder
from rag_core.telemetry import setup_telemetry
from rag_core.vectorstore import VectorStore

from query_service.api.errors import install_error_handlers
from query_service.api.routes import router
from query_service.pipeline.orchestrator import QueryPipeline
from query_service.pipeline.prompt import load_prompt
from query_service.providers import LLMProvider, build_provider

log = logging.getLogger(__name__)

_DEFAULT_SECRET = Settings.model_fields["jwt_secret"].default


def create_app(
    settings: Settings | None = None,
    *,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    llm: LLMProvider | None = None,
) -> FastAPI:
    """Build the app. Components can be injected (tests); otherwise they come from settings."""
    settings = settings or get_settings()
    if settings.environment == "prod" and settings.jwt_secret == _DEFAULT_SECRET:
        raise RuntimeError("RAG_JWT_SECRET must be set in production")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        vector_store = store or VectorStore.from_settings(settings)
        await vector_store.ensure_collection()
        provider = llm or build_provider(settings)
        app.state.pipeline = QueryPipeline(
            settings=settings,
            store=vector_store,
            embedder=embedder or build_embedder(settings),
            llm=provider,
            prompt=load_prompt(settings.prompt_dir, settings.prompt_name),
        )
        log.info("query service ready: llm=%s index=%s", provider.model, settings.collection_alias)
        try:
            yield
        finally:
            if llm is None:
                await provider.aclose()
            if store is None:
                await vector_store.client.close()

    setup_telemetry("rag-query")
    app = FastAPI(title="grounded-rag query API", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    install_error_handlers(app)
    app.include_router(router)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz")
    return app
