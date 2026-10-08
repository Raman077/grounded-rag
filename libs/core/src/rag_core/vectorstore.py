"""Qdrant access shared by ingestion and query.

Queries always go through the alias (``corpus_live``) so an index rebuild can be promoted
atomically. Every search is filtered on ``tenant_id`` and ``acl_groups`` inside Qdrant;
results are never filtered after retrieval.

Each point carries two named vectors: ``dense`` (semantic) and ``sparse`` (lexical,
BM25). The sparse config sets ``Modifier.IDF`` so Qdrant computes the inverse document
frequency from the live collection -- the fastembed side only emits term frequencies,
and IDF computed over a stale snapshot would drift as documents change.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from rag_core.config import Settings
from rag_core.schemas import ChunkPayload, QueryFilters
from rag_core.sparse import SparseVector

_KEYWORD_FIELDS = ("acl_groups", "doc_id", "doc_type", "source")

DENSE = "dense"
SPARSE = "sparse"


@dataclass(frozen=True)
class ScoredChunk:
    id: str
    score: float
    payload: ChunkPayload
    #: A 0-1 relevance figure that a gate can threshold on. Cosine similarity carries
    #: straight over; a fused (RRF) score has no such scale, so it is ``None`` until a
    #: reranker supplies one. See ``rag_core.gate``.
    normalized_score: float | None = None
    #: The score from the retrieval stage, kept when a reranker overwrites ``score``.
    retrieval_score: float | None = None


@dataclass(frozen=True)
class ChunkPoint:
    id: str
    vector: list[float]
    payload: ChunkPayload
    sparse: SparseVector | None = None


def build_client(settings: Settings) -> AsyncQdrantClient:
    location = settings.qdrant_location
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    if location.startswith(("http://", "https://")):
        return AsyncQdrantClient(url=location, api_key=api_key)
    if location == ":memory:":
        return AsyncQdrantClient(location=":memory:")
    return AsyncQdrantClient(path=location)


class VectorStore:
    def __init__(self, client: AsyncQdrantClient, *, collection: str, alias: str, dim: int) -> None:
        self.client = client
        self.collection = collection
        self.alias = alias
        self.dim = dim

    @classmethod
    def from_settings(cls, settings: Settings) -> VectorStore:
        return cls(
            build_client(settings),
            collection=settings.collection_name,
            alias=settings.collection_alias,
            dim=settings.embedding_dim,
        )

    async def ensure_collection(self) -> None:
        """Create the collection, its payload indexes and the alias if they are missing."""
        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(
                self.collection,
                vectors_config={DENSE: models.VectorParams(size=self.dim, distance=models.Distance.COSINE)},
                sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200),
            )
            with warnings.catch_warnings():
                # Embedded (local) Qdrant ignores payload indexes and warns about it.
                warnings.filterwarnings("ignore", message="Payload indexes have no effect")
                await self._create_payload_indexes()

        aliases = await self.client.get_aliases()
        if not any(a.alias_name == self.alias for a in aliases.aliases):
            await self.client.update_collection_aliases(
                change_aliases_operations=[
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(collection_name=self.collection, alias_name=self.alias)
                    )
                ]
            )

    async def _create_payload_indexes(self) -> None:
        await self.client.create_payload_index(
            self.collection,
            "tenant_id",
            field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
        )
        for field in _KEYWORD_FIELDS:
            await self.client.create_payload_index(
                self.collection, field, field_schema=models.PayloadSchemaType.KEYWORD
            )
        await self.client.create_payload_index(
            self.collection, "updated_at", field_schema=models.PayloadSchemaType.DATETIME
        )

    async def ping(self) -> bool:
        try:
            await self.client.get_collections()
        except Exception:
            return False
        return True

    # ------------------------------------------------------------- writes

    async def get_doc_version(self, tenant_id: str, doc_id: str) -> str | None:
        points, _ = await self.client.scroll(
            self.alias,
            scroll_filter=models.Filter(must=[_match("tenant_id", tenant_id), _match("doc_id", doc_id)]),
            limit=1,
            with_payload=["doc_version"],
            with_vectors=False,
        )
        if not points or points[0].payload is None:
            return None
        version = points[0].payload.get("doc_version")
        return str(version) if version is not None else None

    async def replace_document(self, tenant_id: str, doc_id: str, points: Sequence[ChunkPoint]) -> None:
        """Upsert the new chunks first, then delete the document's stale ones, so readers
        never see the document disappear mid-update."""
        if points:
            await self.client.upsert(
                self.alias,
                points=[
                    models.PointStruct(id=p.id, vector=_vectors(p), payload=p.payload.model_dump(mode="json"))
                    for p in points
                ],
                wait=True,
            )
        stale = models.Filter(
            must=[_match("tenant_id", tenant_id), _match("doc_id", doc_id)],
            must_not=[models.HasIdCondition(has_id=[p.id for p in points])] if points else None,
        )
        await self.client.delete(self.alias, points_selector=models.FilterSelector(filter=stale), wait=True)

    async def delete_document(self, tenant_id: str, doc_id: str) -> None:
        await self.replace_document(tenant_id, doc_id, [])

    # -------------------------------------------------------------- reads

    def _access_filter(self, tenant_id: str, acl_groups: Sequence[str], filters: QueryFilters | None) -> models.Filter:
        must: list[models.Condition] = [
            _match("tenant_id", tenant_id),
            models.FieldCondition(key="acl_groups", match=models.MatchAny(any=list(acl_groups))),
        ]
        if filters is not None:
            must.extend(_filter_conditions(filters))
        return models.Filter(must=must)

    async def search(
        self,
        vector: list[float],
        *,
        tenant_id: str,
        acl_groups: Sequence[str],
        limit: int,
        filters: QueryFilters | None = None,
    ) -> list[ScoredChunk]:
        """Dense-only search. Cosine similarity carries a usable 0-1 scale, so the
        result's ``normalized_score`` is the score itself."""
        if not acl_groups:
            return []
        result = await self.client.query_points(
            self.alias,
            query=vector,
            using=DENSE,
            query_filter=self._access_filter(tenant_id, acl_groups, filters),
            limit=limit,
            with_payload=True,
        )
        return [_chunk(p, normalized=p.score) for p in result.points]

    async def search_hybrid(
        self,
        vector: list[float],
        sparse: SparseVector,
        *,
        tenant_id: str,
        acl_groups: Sequence[str],
        limit: int,
        prefetch_limit: int,
        filters: QueryFilters | None = None,
    ) -> list[ScoredChunk]:
        """Dense and sparse candidates fused by Reciprocal Rank Fusion, server-side.

        RRF scores rank positions, not similarity, so the result carries no
        ``normalized_score``: there is no threshold that means the same thing across
        queries. Gating a hybrid result needs a reranker (or a rank-based gate).
        """
        if not acl_groups:
            return []
        query_filter = self._access_filter(tenant_id, acl_groups, filters)
        prefetch = [models.Prefetch(query=vector, using=DENSE, limit=prefetch_limit, filter=query_filter)]
        if sparse:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    using=SPARSE,
                    limit=prefetch_limit,
                    filter=query_filter,
                )
            )
        result = await self.client.query_points(
            self.alias,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return [_chunk(p, normalized=None) for p in result.points]


def _vectors(point: ChunkPoint) -> dict[str, Any]:
    vectors: dict[str, Any] = {DENSE: point.vector}
    if point.sparse:
        vectors[SPARSE] = models.SparseVector(indices=point.sparse.indices, values=point.sparse.values)
    return vectors


def _chunk(point: Any, *, normalized: float | None) -> ScoredChunk:
    return ScoredChunk(
        id=str(point.id),
        score=point.score,
        payload=ChunkPayload.model_validate(point.payload),
        normalized_score=normalized,
        retrieval_score=point.score,
    )


def _match(key: str, value: Any) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _filter_conditions(filters: QueryFilters) -> list[models.Condition]:
    conditions: list[models.Condition] = []
    if filters.doc_type:
        conditions.append(models.FieldCondition(key="doc_type", match=models.MatchAny(any=filters.doc_type)))
    if filters.source:
        conditions.append(models.FieldCondition(key="source", match=models.MatchAny(any=filters.source)))
    if filters.updated_after:
        conditions.append(
            models.FieldCondition(key="updated_at", range=models.DatetimeRange(gte=filters.updated_after))
        )
    for key, value in (filters.metadata or {}).items():
        conditions.append(_match(f"metadata.{key}", value))
    return conditions
