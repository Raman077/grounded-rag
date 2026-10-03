"""Shared test fixtures: in-memory Qdrant, hashing embedder and a scripted LLM, so the whole
pipeline runs offline and deterministically. Real-model checks live behind markers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from ingest_service.pipeline import IngestOptions, ingest_file
from qdrant_client import AsyncQdrantClient
from query_service.api.auth import issue_token
from query_service.main import create_app
from query_service.providers import ChatMessage, Completion, LLMError, LLMEvent, TextDelta
from rag_core.config import Settings
from rag_core.embeddings import HashingEmbedder
from rag_core.vectorstore import VectorStore

REPO_ROOT = Path(__file__).parent
CORPUS = REPO_ROOT / "eval" / "fixtures" / "corpus"
INTERNAL_CORPUS = REPO_ROOT / "eval" / "fixtures" / "corpus-internal"
TENANT = "nimbus"


class ScriptedLLM:
    """Stand-in for a real model: replies with a fixed string (or a function of the prompt)
    in a few deltas, and records every call."""

    model = "scripted-llm"

    def __init__(
        self,
        reply: str | Callable[[str], str] = "The Pro plan costs $49 per month [S1].",
        *,
        stop: str = "end",
        error: LLMError | None = None,
    ) -> None:
        self.reply = reply
        self.stop = stop
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def stream(self, *, system: str, messages: list[ChatMessage], max_tokens: int) -> AsyncIterator[LLMEvent]:
        self.calls.append({"system": system, "messages": messages, "max_tokens": max_tokens})
        if self.error is not None:
            raise self.error
        text = self.reply(messages[-1]["content"]) if callable(self.reply) else self.reply
        step = max(1, len(text) // 3)
        for i in range(0, len(text), step):
            yield TextDelta(text[i : i + step])
        yield Completion(model=self.model, stop=self.stop, input_tokens=100, output_tokens=len(text) // 4)  # type: ignore[arg-type]

    async def aclose(self) -> None:
        return None


@pytest.fixture
def settings() -> Settings:
    return Settings(
        environment="test",
        qdrant_location=":memory:",
        embedding_provider="hashing",
        embedding_dim=384,
        prompt_dir=REPO_ROOT / "prompts",
        min_relevance_score=0.2,
        chunk_max_tokens=200,
        chunk_overlap_tokens=30,
    )


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(384)


@pytest.fixture
def store(settings: Settings) -> Iterator[VectorStore]:
    vs = VectorStore(
        AsyncQdrantClient(location=":memory:"),
        collection=settings.collection_name,
        alias=settings.collection_alias,
        dim=settings.embedding_dim,
    )
    asyncio.run(vs.ensure_collection())
    yield vs
    asyncio.run(vs.client.close())


async def ingest_folder(folder: Path, store: VectorStore, embedder: HashingEmbedder, acl: list[str]) -> None:
    options = IngestOptions(tenant_id=TENANT, acl_groups=acl, max_tokens=200, overlap_tokens=30)
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        await ingest_file(path, root=folder, options=options, store=store, embedder=embedder)


@pytest.fixture
def indexed_store(store: VectorStore, embedder: HashingEmbedder) -> VectorStore:
    asyncio.run(ingest_folder(CORPUS, store, embedder, ["public"]))
    asyncio.run(ingest_folder(INTERNAL_CORPUS, store, embedder, ["engineering"]))
    return store


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def client(
    settings: Settings, indexed_store: VectorStore, embedder: HashingEmbedder, llm: ScriptedLLM
) -> Iterator[TestClient]:
    app = create_app(settings, store=indexed_store, embedder=embedder, llm=llm)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def make_token(settings: Settings) -> Callable[..., str]:
    def _make(groups: list[str] | None = None, tenant: str = TENANT) -> str:
        return issue_token(settings, subject="tester", tenant_id=tenant, groups=groups or ["public"])

    return _make


@pytest.fixture
def auth(make_token: Callable[..., str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {make_token()}"}
