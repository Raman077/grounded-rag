"""Ingest one file: hash -> skip if unchanged -> parse -> chunk -> embed -> replace in index."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from rag_core.embeddings import Embedder
from rag_core.schemas import ChunkPayload
from rag_core.telemetry import get_tracer, timed_span
from rag_core.vectorstore import ChunkPoint, VectorStore

from ingest_service.chunking import Chunk, chunk_document
from ingest_service.parsers import ParseError, parse_file

log = logging.getLogger(__name__)
tracer = get_tracer(__name__)

_CHUNK_NAMESPACE = uuid.UUID("6f1c7d0e-3b7a-4f0e-9d55-6a1f0b2f9c11")


@dataclass(frozen=True)
class IngestOptions:
    tenant_id: str
    acl_groups: Sequence[str]
    source: str = "local"
    doc_type: str = "doc"
    max_tokens: int = 350
    overlap_tokens: int = 40
    force: bool = False


@dataclass(frozen=True)
class IngestResult:
    path: str
    doc_id: str
    status: Literal["indexed", "unchanged", "skipped", "failed"]
    chunks: int = 0
    reason: str | None = None


def make_doc_id(tenant_id: str, source: str, rel_path: str) -> str:
    return hashlib.sha256(f"{tenant_id}\x00{source}\x00{rel_path}".encode()).hexdigest()[:32]


def embedding_text(title: str, chunk: Chunk) -> str:
    """Prefix each chunk with its title and heading path so it embeds with its context."""
    header = " > ".join([title, *chunk.section_path])
    return f"{header}\n\n{chunk.text}"


async def ingest_file(
    path: Path, *, root: Path, options: IngestOptions, store: VectorStore, embedder: Embedder
) -> IngestResult:
    rel_path = path.relative_to(root).as_posix()
    doc_id = make_doc_id(options.tenant_id, options.source, rel_path)

    with timed_span(tracer, "ingest.document", doc_id=doc_id, path=rel_path) as span:
        raw = await asyncio.to_thread(path.read_bytes)
        version = hashlib.sha256(raw).hexdigest()
        if not options.force and await store.get_doc_version(options.tenant_id, doc_id) == version:
            span.set_attribute("status", "unchanged")
            return IngestResult(rel_path, doc_id, "unchanged")

        try:
            doc = await asyncio.to_thread(parse_file, path)
        except ParseError as exc:
            log.warning("skipping %s: %s", rel_path, exc)
            span.set_attribute("status", "skipped")
            return IngestResult(rel_path, doc_id, "skipped", reason=str(exc))

        chunks = chunk_document(doc, max_tokens=options.max_tokens, overlap_tokens=options.overlap_tokens)
        vectors = await embedder.embed_documents([embedding_text(doc.title, c) for c in chunks])
        stat = await asyncio.to_thread(path.stat)
        updated_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)

        points = [
            ChunkPoint(
                id=str(uuid.uuid5(_CHUNK_NAMESPACE, f"{doc_id}:{chunk.index}:{chunk.text}")),
                vector=vector,
                payload=ChunkPayload(
                    tenant_id=options.tenant_id,
                    acl_groups=list(options.acl_groups),
                    doc_id=doc_id,
                    doc_version=version,
                    source=options.source,
                    doc_type=options.doc_type,
                    title=doc.title,
                    url=rel_path,
                    section_path=chunk.section_path,
                    page=chunk.page,
                    chunk_index=chunk.index,
                    text=chunk.text,
                    updated_at=updated_at,
                ),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await store.replace_document(options.tenant_id, doc_id, points)
        span.set_attribute("status", "indexed")
        span.set_attribute("chunks", len(points))
        return IngestResult(rel_path, doc_id, "indexed", chunks=len(points))
