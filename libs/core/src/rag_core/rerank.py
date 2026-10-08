"""Cross-encoder reranking.

Retrieval scores a query and a chunk independently -- two vectors compared after the
fact. A cross-encoder reads the pair together and scores the relationship, which is
far more accurate and far more expensive. The standard trade is to let the cheap
retriever propose many candidates and let the expensive model order the few that
survive: retrieve 50, rerank, keep 8.

Scores are raw logits, not similarities: they are comparable within one query's
candidate list and meaningless across queries. Anything downstream that needs a
0-1 relevance gate reads ``normalized_score`` (a sigmoid), never ``score``.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Protocol

from rag_core.config import Settings
from rag_core.telemetry import get_tracer, timed_span
from rag_core.vectorstore import ScoredChunk

tracer = get_tracer(__name__)


class Reranker(Protocol):
    model_name: str

    async def rerank(self, query: str, chunks: Sequence[ScoredChunk], *, limit: int) -> list[ScoredChunk]: ...


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp = math.exp(x)  # avoids overflow for large negative logits
    return exp / (1.0 + exp)


class NullReranker:
    """Keeps retrieval order. Used when reranking is disabled, so callers need no branch."""

    model_name = "none"

    async def rerank(self, query: str, chunks: Sequence[ScoredChunk], *, limit: int) -> list[ScoredChunk]:
        return list(chunks[:limit])


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    async def rerank(self, query: str, chunks: Sequence[ScoredChunk], *, limit: int) -> list[ScoredChunk]:
        if not chunks:
            return []

        documents = [c.payload.text for c in chunks]

        def run() -> list[float]:
            return [float(s) for s in self._model.rerank(query, documents)]

        with timed_span(tracer, "rerank.cross_encoder", model=self.model_name, candidates=len(chunks)) as span:
            scores = await asyncio.to_thread(run)
            ranked = sorted(
                (
                    ScoredChunk(
                        id=chunk.id,
                        score=score,
                        payload=chunk.payload,
                        normalized_score=_sigmoid(score),
                        retrieval_score=chunk.score,
                    )
                    for chunk, score in zip(chunks, scores, strict=True)
                ),
                key=lambda c: c.score,
                reverse=True,
            )[:limit]
            span.set_attribute("kept", len(ranked))
            if ranked:
                span.set_attribute("top_score", ranked[0].normalized_score)
        return ranked


def build_reranker(settings: Settings) -> Reranker:
    if not settings.rerank_enabled or settings.rerank_provider == "none":
        return NullReranker()
    return CrossEncoderReranker(settings.rerank_model)
