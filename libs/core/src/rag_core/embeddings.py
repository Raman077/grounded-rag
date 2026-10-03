"""Dense embedding providers behind one small protocol.

* ``fastembed`` - local ONNX models (default for development, no API key).
* ``voyage``    - Voyage AI REST API (production option).
* ``hashing``   - deterministic feature hashing; no model download. Used by tests and
                  for smoke-testing the pipeline offline. Lexical only, not for real use.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any, Protocol

import httpx

from rag_core.config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    model_name: str
    dim: int

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class HashingEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.model_name = f"hashing-{dim}"

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _TOKEN_RE.findall(text.lower())
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign
        return _normalize(vec)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FastEmbedEmbedder:
    def __init__(self, model_name: str, dim: int) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.dim = dim
        self._model = TextEmbedding(model_name=model_name)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        def run() -> list[list[float]]:
            return [v.tolist() for v in self._model.embed(list(texts))]

        return await asyncio.to_thread(run)

    async def embed_query(self, text: str) -> list[float]:
        def run() -> list[float]:
            vector: list[float] = next(iter(self._model.query_embed(text))).tolist()
            return vector

        return await asyncio.to_thread(run)


class VoyageEmbedder:
    _URL = "https://api.voyageai.com/v1/embeddings"
    _BATCH = 128

    def __init__(self, model_name: str, dim: int, api_key: str) -> None:
        self.model_name = model_name
        self.dim = dim
        self._client = httpx.AsyncClient(headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0)

    async def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._BATCH):
            batch = list(texts[start : start + self._BATCH])
            resp = await self._client.post(
                self._URL,
                json={"input": batch, "model": self.model_name, "input_type": input_type},
            )
            resp.raise_for_status()
            data: list[dict[str, Any]] = sorted(resp.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        return out

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, "document")

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], "query"))[0]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hashing":
        return HashingEmbedder(settings.embedding_dim)
    if settings.embedding_provider == "voyage":
        if settings.voyage_api_key is None:
            raise ValueError("RAG_VOYAGE_API_KEY is required for the voyage embedding provider")
        return VoyageEmbedder(
            settings.embedding_model,
            settings.embedding_dim,
            settings.voyage_api_key.get_secret_value(),
        )
    return FastEmbedEmbedder(settings.embedding_model, settings.embedding_dim)
