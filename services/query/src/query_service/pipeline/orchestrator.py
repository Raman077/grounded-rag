"""The query pipeline: embed -> dense retrieve -> sufficiency gate -> pack -> generate -> cite.

``QueryPipeline.run`` yields events that map one-to-one onto the SSE stream; the blocking
JSON endpoint consumes the same events and returns the ``Final`` payload.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from rag_core.config import Settings
from rag_core.embeddings import Embedder
from rag_core.schemas import (
    AnswerType,
    Citation,
    QueryRequest,
    QueryResponse,
    Timings,
    Usage,
)
from rag_core.telemetry import Stopwatch, current_trace_id, get_tracer, timed_span
from rag_core.vectorstore import ScoredChunk, VectorStore

from query_service.api.auth import Principal
from query_service.pipeline.prompt import (
    INSUFFICIENT_CONTEXT,
    PromptTemplate,
    build_messages,
    pack_context,
)
from query_service.providers import Completion, LLMError, LLMProvider, TextDelta

tracer = get_tracer(__name__)

_CITATION_RE = re.compile(r"\[(S\d+)\]")
_SNIPPET_CHARS = 300
_RELATED_LIMIT = 3

FALLBACK_ANSWER = (
    "I couldn't find enough information in the available documents to answer that. The sources below may be related."
)
REFUSAL_ANSWER = "I can't help with that request."


@dataclass(frozen=True)
class Meta:
    request_id: uuid.UUID
    trace_id: str


@dataclass(frozen=True)
class Retrieval:
    citations: list[Citation]


@dataclass(frozen=True)
class Token:
    delta: str


@dataclass(frozen=True)
class Final:
    response: QueryResponse


@dataclass(frozen=True)
class Error:
    code: Literal["llm_unavailable", "llm_error"]
    message: str
    retryable: bool


PipelineEvent = Meta | Retrieval | Token | Final | Error


class QueryPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        store: VectorStore,
        embedder: Embedder,
        llm: LLMProvider,
        prompt: PromptTemplate,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.prompt = prompt

    async def run(self, req: QueryRequest, principal: Principal) -> AsyncIterator[PipelineEvent]:
        total = Stopwatch()
        request_id = uuid.uuid4()
        timings = Timings()

        with tracer.start_as_current_span("rag.query") as root:
            root.set_attribute("rag.request_id", str(request_id))
            root.set_attribute("rag.tenant_id", principal.tenant_id)
            root.set_attribute("rag.mode", req.options.mode)
            yield Meta(request_id=request_id, trace_id=current_trace_id())

            # 1. Retrieve -----------------------------------------------------
            watch = Stopwatch()
            with timed_span(tracer, "retrieve.dense", top_k=req.options.top_k) as span:
                vector = await self.embedder.embed_query(req.query)
                hits = await self.store.search(
                    vector,
                    tenant_id=principal.tenant_id,
                    acl_groups=principal.groups,
                    limit=max(self.settings.retrieval_candidates, req.options.top_k),
                    filters=req.filters,
                )
                span.set_attribute("hits", len(hits))
                span.set_attribute("top_score", hits[0].score if hits else 0.0)
            timings.retrieval = watch.ms()

            # 2. Sufficiency gate --------------------------------------------
            threshold = self.settings.min_relevance_score
            if req.options.mode == "strict":
                threshold += self.settings.strict_score_margin
            relevant = [h for h in hits if h.score >= threshold][: req.options.top_k]
            root.set_attribute("rag.relevant_chunks", len(relevant))

            if not relevant:
                related = [_citation(f"S{i + 1}", h) for i, h in enumerate(hits[:_RELATED_LIMIT])]
                yield Retrieval(citations=[])
                timings.total = total.ms()
                yield Final(
                    self._response(
                        request_id,
                        answer_type="insufficient_context",
                        answer=FALLBACK_ANSWER,
                        related=related,
                        timings=timings,
                        model=self.llm.model,
                    )
                )
                return

            sources = pack_context(relevant, max_tokens=self.settings.max_context_tokens)
            citations = [_citation(s.source_id, s.chunk) for s in sources]
            yield Retrieval(citations=citations if req.options.include_sources else [])

            # 3. Generate -----------------------------------------------------
            system, messages = build_messages(self.prompt, question=req.query, history=req.history, sources=sources)
            text_parts: list[str] = []
            completion: Completion | None = None
            gate = _SentinelGate(INSUFFICIENT_CONTEXT)
            watch = Stopwatch()
            with timed_span(tracer, "llm.generate", model=self.llm.model) as span:
                try:
                    async for event in self.llm.stream(
                        system=system, messages=messages, max_tokens=req.options.max_output_tokens
                    ):
                        if isinstance(event, TextDelta):
                            if timings.ttft is None:
                                timings.ttft = watch.ms()
                            text_parts.append(event.text)
                            if released := gate.feed(event.text):
                                yield Token(released)
                        else:
                            completion = event
                except LLMError as exc:
                    span.record_exception(exc)
                    yield Error(
                        code="llm_unavailable" if exc.retryable else "llm_error",
                        message=str(exc),
                        retryable=exc.retryable,
                    )
                    return
                if not gate.matched and (rest := gate.flush()):
                    yield Token(rest)
                if completion is not None:
                    span.set_attribute("input_tokens", completion.input_tokens)
                    span.set_attribute("output_tokens", completion.output_tokens)
                    span.set_attribute("stop", completion.stop)

            # 4. Assemble -----------------------------------------------------
            answer = "".join(text_parts).strip()
            usage = Usage()
            stop = "end"
            model = self.llm.model
            if completion is not None:
                usage = Usage(
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    cached_input_tokens=completion.cached_input_tokens,
                )
                stop = completion.stop
                model = completion.model

            timings.total = total.ms()
            if gate.matched or not answer:
                response = self._response(
                    request_id,
                    answer_type="insufficient_context",
                    answer=FALLBACK_ANSWER,
                    related=citations[:_RELATED_LIMIT],
                    timings=timings,
                    usage=usage,
                    model=model,
                )
            elif stop == "refusal":
                response = self._response(
                    request_id, answer_type="refused", answer=REFUSAL_ANSWER, timings=timings, usage=usage, model=model
                )
            else:
                cited, answer = _resolve_citations(answer, citations)
                response = self._response(
                    request_id,
                    answer_type="partial" if stop == "max_tokens" else "answer",
                    answer=answer,
                    citations=cited,
                    timings=timings,
                    usage=usage,
                    model=model,
                )
            root.set_attribute("rag.answer_type", response.answer_type)
            yield Final(response)

    def _response(
        self,
        request_id: uuid.UUID,
        *,
        answer_type: AnswerType,
        answer: str,
        timings: Timings,
        model: str,
        citations: list[Citation] | None = None,
        related: list[Citation] | None = None,
        usage: Usage | None = None,
    ) -> QueryResponse:
        return QueryResponse(
            request_id=request_id,
            trace_id=current_trace_id(),
            answer_type=answer_type,
            answer=answer,
            citations=citations or [],
            related_sources=related or [],
            usage=usage or Usage(),
            timings_ms=timings,
            model=model,
            prompt_version=self.prompt.key,
            index_version=self.settings.collection_name,
        )


class _SentinelGate:
    """Holds back the start of the stream until it is clear the model is not answering with
    the INSUFFICIENT_CONTEXT sentinel, so the sentinel never reaches the client."""

    def __init__(self, sentinel: str) -> None:
        self._sentinel = sentinel
        self._buffer = ""
        self._open = False
        self.matched = False

    def feed(self, text: str) -> str:
        if self._open:
            return text
        if self.matched:
            return ""
        self._buffer += text
        head = self._buffer.lstrip()
        if head.startswith(self._sentinel):
            self.matched = True
            return ""
        if self._sentinel.startswith(head):
            return ""  # still ambiguous; keep buffering
        self._open = True
        released, self._buffer = self._buffer, ""
        return released

    def flush(self) -> str:
        released, self._buffer = self._buffer, ""
        return released


def _citation(source_id: str, hit: ScoredChunk) -> Citation:
    p = hit.payload
    snippet = p.text if len(p.text) <= _SNIPPET_CHARS else p.text[:_SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
    return Citation(
        id=source_id,
        doc_id=p.doc_id,
        chunk_id=hit.id,
        title=p.title,
        url=p.url,
        page=p.page,
        section_path=" > ".join(p.section_path) or None,
        snippet=snippet,
        score=round(hit.score, 4),
        updated_at=p.updated_at,
    )


def _resolve_citations(answer: str, citations: list[Citation]) -> tuple[list[Citation], str]:
    """Return the sources the answer actually cites, in citation order, and drop any
    [Sx] marker that does not refer to a supplied source."""
    by_id = {c.id: c for c in citations}
    order: list[str] = []
    removed = False

    def keep(match: re.Match[str]) -> str:
        nonlocal removed
        source_id = match.group(1)
        if source_id not in by_id:
            removed = True
            return ""
        if source_id not in order:
            order.append(source_id)
        return match.group(0)

    cleaned = _CITATION_RE.sub(keep, answer)
    if removed:  # tidy the space a dropped marker leaves before punctuation
        cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    return [by_id[i] for i in order], cleaned
