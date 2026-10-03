import uuid
from datetime import UTC, datetime

from rag_core.embeddings import HashingEmbedder
from rag_core.schemas import ChunkPayload, QueryFilters
from rag_core.vectorstore import ChunkPoint, VectorStore


def _point(
    embedder_vec: list[float],
    *,
    tenant: str,
    doc: str,
    text: str,
    acl: list[str],
    index: int = 0,
    doc_type: str = "doc",
    updated: datetime | None = None,
) -> ChunkPoint:
    return ChunkPoint(
        id=str(uuid.uuid4()),
        vector=embedder_vec,
        payload=ChunkPayload(
            tenant_id=tenant,
            acl_groups=acl,
            doc_id=doc,
            doc_version="v1",
            source="test",
            doc_type=doc_type,
            title=doc,
            chunk_index=index,
            text=text,
            updated_at=updated,
        ),
    )


async def test_ensure_collection_is_idempotent(store: VectorStore) -> None:
    await store.ensure_collection()
    aliases = await store.client.get_aliases()
    assert [a.alias_name for a in aliases.aliases] == [store.alias]


async def test_search_enforces_tenant_and_acl(store: VectorStore, embedder: HashingEmbedder) -> None:
    text = "the pro plan costs 49 dollars"
    vec = await embedder.embed_query(text)
    await store.replace_document(
        "t1", "public-doc", [_point(vec, tenant="t1", doc="public-doc", text=text, acl=["public"])]
    )
    await store.replace_document("t1", "eng-doc", [_point(vec, tenant="t1", doc="eng-doc", text=text, acl=["eng"])])
    await store.replace_document("t2", "other", [_point(vec, tenant="t2", doc="other", text=text, acl=["public"])])

    public = await store.search(vec, tenant_id="t1", acl_groups=["public"], limit=10)
    assert {h.payload.doc_id for h in public} == {"public-doc"}

    both = await store.search(vec, tenant_id="t1", acl_groups=["public", "eng"], limit=10)
    assert {h.payload.doc_id for h in both} == {"public-doc", "eng-doc"}

    assert await store.search(vec, tenant_id="t1", acl_groups=[], limit=10) == []


async def test_replace_document_removes_stale_chunks(store: VectorStore, embedder: HashingEmbedder) -> None:
    vec = await embedder.embed_query("x")
    old = [_point(vec, tenant="t", doc="d", text=f"old {i}", acl=["public"], index=i) for i in range(3)]
    await store.replace_document("t", "d", old)
    new = [_point(vec, tenant="t", doc="d", text="new", acl=["public"])]
    await store.replace_document("t", "d", new)

    hits = await store.search(vec, tenant_id="t", acl_groups=["public"], limit=10)
    assert [h.payload.text for h in hits] == ["new"]

    await store.delete_document("t", "d")
    assert await store.search(vec, tenant_id="t", acl_groups=["public"], limit=10) == []


async def test_search_applies_request_filters(store: VectorStore, embedder: HashingEmbedder) -> None:
    vec = await embedder.embed_query("x")
    await store.replace_document(
        "t",
        "faq",
        [
            _point(
                vec,
                tenant="t",
                doc="faq",
                text="a",
                acl=["public"],
                doc_type="faq",
                updated=datetime(2026, 9, 1, tzinfo=UTC),
            )
        ],
    )
    await store.replace_document(
        "t",
        "policy",
        [
            _point(
                vec,
                tenant="t",
                doc="policy",
                text="b",
                acl=["public"],
                doc_type="policy",
                updated=datetime(2025, 1, 1, tzinfo=UTC),
            )
        ],
    )

    by_type = await store.search(
        vec, tenant_id="t", acl_groups=["public"], limit=10, filters=QueryFilters(doc_type=["faq"])
    )
    assert [h.payload.doc_id for h in by_type] == ["faq"]

    recent = await store.search(
        vec,
        tenant_id="t",
        acl_groups=["public"],
        limit=10,
        filters=QueryFilters(updated_after=datetime(2026, 1, 1, tzinfo=UTC)),
    )
    assert [h.payload.doc_id for h in recent] == ["faq"]


async def test_get_doc_version(store: VectorStore, embedder: HashingEmbedder) -> None:
    vec = await embedder.embed_query("x")
    assert await store.get_doc_version("t", "d") is None
    await store.replace_document("t", "d", [_point(vec, tenant="t", doc="d", text="a", acl=["public"])])
    assert await store.get_doc_version("t", "d") == "v1"
