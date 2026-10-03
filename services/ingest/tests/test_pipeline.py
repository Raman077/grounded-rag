import shutil
from pathlib import Path

from ingest_service.pipeline import IngestOptions, ingest_file, make_doc_id
from rag_core.embeddings import HashingEmbedder
from rag_core.vectorstore import VectorStore

CORPUS = Path(__file__).parents[3] / "eval" / "fixtures" / "corpus"
OPTIONS = IngestOptions(tenant_id="acme", acl_groups=["public"], max_tokens=200, overlap_tokens=30)


async def _search(store: VectorStore, embedder: HashingEmbedder, text: str) -> list[str]:
    hits = await store.search(await embedder.embed_query(text), tenant_id="acme", acl_groups=["public"], limit=50)
    return [h.payload.text for h in hits]


async def test_ingest_indexes_then_skips_unchanged_file(
    tmp_path: Path, store: VectorStore, embedder: HashingEmbedder
) -> None:
    shutil.copy(CORPUS / "pricing.md", tmp_path / "pricing.md")
    first = await ingest_file(tmp_path / "pricing.md", root=tmp_path, options=OPTIONS, store=store, embedder=embedder)
    assert first.status == "indexed" and first.chunks > 1
    assert first.doc_id == make_doc_id("acme", "local", "pricing.md")

    second = await ingest_file(tmp_path / "pricing.md", root=tmp_path, options=OPTIONS, store=store, embedder=embedder)
    assert second.status == "unchanged"


async def test_changed_file_replaces_old_chunks(tmp_path: Path, store: VectorStore, embedder: HashingEmbedder) -> None:
    path = tmp_path / "faq.md"
    path.write_text("# FAQ\n\nThe old answer is forty two.", encoding="utf-8")
    await ingest_file(path, root=tmp_path, options=OPTIONS, store=store, embedder=embedder)
    path.write_text("# FAQ\n\nThe new answer is seventeen.", encoding="utf-8")
    result = await ingest_file(path, root=tmp_path, options=OPTIONS, store=store, embedder=embedder)

    assert result.status == "indexed"
    texts = await _search(store, embedder, "answer")
    assert texts == ["The new answer is seventeen."]


async def test_payload_carries_citation_metadata(tmp_path: Path, store: VectorStore, embedder: HashingEmbedder) -> None:
    shutil.copy(CORPUS / "sla.md", tmp_path / "sla.md")
    await ingest_file(tmp_path / "sla.md", root=tmp_path, options=OPTIONS, store=store, embedder=embedder)
    hits = await store.search(
        await embedder.embed_query("service credits uptime"), tenant_id="acme", acl_groups=["public"], limit=3
    )
    payload = hits[0].payload
    assert payload.title == "Service Level Agreement"
    assert payload.url == "sla.md"
    assert payload.section_path[0] == "Service Level Agreement"
    assert payload.updated_at is not None


async def test_unparseable_file_is_skipped(tmp_path: Path, store: VectorStore, embedder: HashingEmbedder) -> None:
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")
    result = await ingest_file(tmp_path / "empty.md", root=tmp_path, options=OPTIONS, store=store, embedder=embedder)
    assert result.status == "skipped"
