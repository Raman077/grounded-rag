"""Retrieval strategies behind one protocol, so they can be swapped and compared.

Three, in increasing cost and quality:

``dense``          one vector search. Fast, and the Phase 1 behaviour.
``hybrid``         dense + BM25, fused by RRF inside Qdrant. Recovers exact-token
                   matches -- error codes, versions, identifiers -- that similarity
                   search alone misses.
``hybrid+rerank``  hybrid proposes ``rerank_candidates``, a cross-encoder reads each
                   (query, chunk) pair together and keeps the best ``limit``.

They share one interface so the eval harness can run the same golden set through each
and the difference is measured rather than assumed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from rag_core.config import Settings
from rag_core.embeddings import Embedder
from rag_core.rerank import Reranker, build_reranker
from rag_core.schemas import QueryFilters
from rag_core.sparse import SparseEmbedder, build_sparse_embedder
from rag_core.telemetry import get_tracer, timed_span
from rag_core.vectorstore import ScoredChunk, VectorStore

tracer = get_tracer(__name__)

RetrieverName = Literal["dense", "hybrid", "hybrid+rerank"]


class Retriever(Protocol):
    name: str

    async def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        acl_groups: Sequence[str],
        limit: int,
        filters: QueryFilters | None = None,
    ) -> list[ScoredChunk]: ...


class DenseRetriever:
    name = "dense"

    def __init__(self, *, store: VectorStore, embedder: Embedder, candidates: int) -> None:
        self.store = store
        self.embedder = embedder
        self.candidates = candidates

    async def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        acl_groups: Sequence[str],
        limit: int,
        filters: QueryFilters | None = None,
    ) -> list[ScoredChunk]:
        with timed_span(tracer, "retrieve.dense", limit=limit) as span:
            vector = await self.embedder.embed_query(query)
            hits = await self.store.search(
                vector,
                tenant_id=tenant_id,
                acl_groups=acl_groups,
                limit=max(self.candidates, limit),
                filters=filters,
            )
            span.set_attribute("hits", len(hits))
        return hits[:limit]


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        *,
        store: VectorStore,
        embedder: Embedder,
        sparse: SparseEmbedder,
        candidates: int,
        prefetch_limit: int,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.sparse = sparse
        self.candidates = candidates
        self.prefetch_limit = prefetch_limit

    async def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        acl_groups: Sequence[str],
        limit: int,
        filters: QueryFilters | None = None,
    ) -> list[ScoredChunk]:
        with timed_span(tracer, "retrieve.hybrid", limit=limit, prefetch=self.prefetch_limit) as span:
            dense_vector = await self.embedder.embed_query(query)
            sparse_vector = await self.sparse.embed_query(query)
            span.set_attribute("sparse_terms", len(sparse_vector.indices))
            hits = await self.store.search_hybrid(
                dense_vector,
                sparse_vector,
                tenant_id=tenant_id,
                acl_groups=acl_groups,
                limit=max(self.candidates, limit),
                prefetch_limit=self.prefetch_limit,
                filters=filters,
            )
            span.set_attribute("hits", len(hits))
        return hits[:limit]


class RerankingRetriever:
    """Wraps another retriever: over-fetch, then let a cross-encoder choose.

    The inner retriever is asked for ``candidates`` rather than ``limit`` -- reranking
    can only reorder what retrieval already found, so recall at the candidate depth is
    the ceiling on everything downstream.
    """

    def __init__(self, *, inner: Retriever, reranker: Reranker, candidates: int) -> None:
        self.inner = inner
        self.reranker = reranker
        self.candidates = candidates
        self.name = f"{inner.name}+rerank"

    async def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        acl_groups: Sequence[str],
        limit: int,
        filters: QueryFilters | None = None,
    ) -> list[ScoredChunk]:
        pool = await self.inner.retrieve(
            query,
            tenant_id=tenant_id,
            acl_groups=acl_groups,
            limit=max(self.candidates, limit),
            filters=filters,
        )
        return await self.reranker.rerank(query, pool, limit=limit)


async def build_retriever(
    name: RetrieverName | str,
    *,
    settings: Settings,
    store: VectorStore,
    embedder: Embedder,
) -> Retriever:
    """Construct a retriever by name. Models load lazily on first use, so building a
    reranker is cheap until something is actually reranked."""
    if name == "dense":
        return DenseRetriever(store=store, embedder=embedder, candidates=settings.retrieval_candidates)

    hybrid = HybridRetriever(
        store=store,
        embedder=embedder,
        sparse=build_sparse_embedder(settings),
        candidates=settings.retrieval_candidates,
        prefetch_limit=settings.hybrid_prefetch_limit,
    )
    if name == "hybrid":
        return hybrid
    if name == "hybrid+rerank":
        return RerankingRetriever(
            inner=hybrid,
            reranker=build_reranker(settings.model_copy(update={"rerank_enabled": True})),
            candidates=settings.rerank_candidates,
        )
    raise ValueError(f"unknown retriever: {name}")


def build_default_retriever(*, settings: Settings, store: VectorStore, embedder: Embedder) -> Retriever:
    """The retriever the query service uses, chosen by configuration."""
    sparse = build_sparse_embedder(settings)
    base: Retriever
    if settings.retrieval_mode == "dense":
        base = DenseRetriever(store=store, embedder=embedder, candidates=settings.retrieval_candidates)
    else:
        base = HybridRetriever(
            store=store,
            embedder=embedder,
            sparse=sparse,
            candidates=settings.retrieval_candidates,
            prefetch_limit=settings.hybrid_prefetch_limit,
        )
    if settings.rerank_enabled:
        return RerankingRetriever(inner=base, reranker=build_reranker(settings), candidates=settings.rerank_candidates)
    return base
