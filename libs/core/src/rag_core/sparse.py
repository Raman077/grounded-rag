"""Sparse (lexical) embeddings, the other half of hybrid retrieval.

Dense vectors match meaning but blur exact tokens: an error code, a version string
or a product name that never appeared in training is not reliably retrievable by
similarity alone. A sparse vector keeps the term signal, so ``E4019`` matches
``E4019`` and nothing else.

* ``bm25``    - ``Qdrant/bm25`` via fastembed. Term frequencies only; the IDF half of
                the formula is computed by Qdrant itself, which is why the sparse
                vector config carries ``Modifier.IDF`` (see ``vectorstore``).
* ``hashing`` - deterministic token hashing, no model download. For tests and offline
                smoke runs, mirroring ``HashingEmbedder``.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rag_core.config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SPARSE_DIM = 2**20


@dataclass(frozen=True)
class SparseVector:
    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.indices)


class SparseEmbedder(Protocol):
    model_name: str

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    async def embed_query(self, text: str) -> SparseVector: ...


class HashingSparseEmbedder:
    """Token-hashing bag of words. Lexical and deterministic, but with no corpus
    statistics -- good enough to exercise the hybrid path without a model download."""

    def __init__(self) -> None:
        self.model_name = "hashing-sparse"

    def _embed(self, text: str) -> SparseVector:
        counts: dict[int, float] = {}
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % _SPARSE_DIM
            counts[index] = counts.get(index, 0.0) + 1.0
        ordered = sorted(counts.items())
        return SparseVector([i for i, _ in ordered], [v for _, v in ordered])

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> SparseVector:
        return self._embed(text)


class Bm25SparseEmbedder:
    def __init__(self, model_name: str) -> None:
        from fastembed import SparseTextEmbedding

        self.model_name = model_name
        self._model = SparseTextEmbedding(model_name=model_name)

    @staticmethod
    def _convert(raw: object) -> SparseVector:
        indices = raw.indices.tolist()  # type: ignore[attr-defined]
        values = raw.values.tolist()  # type: ignore[attr-defined]
        return SparseVector(indices, values)

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        def run() -> list[SparseVector]:
            return [self._convert(v) for v in self._model.embed(list(texts))]

        return await asyncio.to_thread(run)

    async def embed_query(self, text: str) -> SparseVector:
        def run() -> SparseVector:
            return self._convert(next(iter(self._model.query_embed(text))))

        return await asyncio.to_thread(run)


def build_sparse_embedder(settings: Settings) -> SparseEmbedder:
    if settings.sparse_provider == "hashing":
        return HashingSparseEmbedder()
    return Bm25SparseEmbedder(settings.sparse_model)
