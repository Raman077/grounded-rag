"""Wire contract for the query API and the payload stored with every indexed chunk.

See docs/ARCHITECTURE.md section 5 for the authoritative description.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- request


class HistoryTurn(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class QueryFilters(StrictModel):
    doc_type: list[str] | None = None
    source: list[str] | None = None
    updated_after: datetime | None = None
    metadata: dict[str, str] | None = None


class QueryOptions(StrictModel):
    top_k: int = Field(default=8, ge=1, le=20)
    mode: Literal["balanced", "strict", "fast"] = "balanced"
    include_sources: bool = True
    use_cache: bool = True
    language: str = "auto"
    max_output_tokens: int = Field(default=1024, ge=64, le=8192)


class QueryRequest(StrictModel):
    # The length limit is enforced by the route so it can answer 413 instead of 422.
    query: str = Field(min_length=1)
    conversation_id: UUID | None = None
    history: list[HistoryTurn] = Field(default_factory=list, max_length=20)
    filters: QueryFilters = Field(default_factory=QueryFilters)
    options: QueryOptions = Field(default_factory=QueryOptions)


# --------------------------------------------------------------- response

AnswerType = Literal["answer", "partial", "insufficient_context", "refused", "clarification"]


class Citation(BaseModel):
    id: str
    doc_id: str
    chunk_id: str
    title: str
    url: str | None = None
    page: int | None = None
    section_path: str | None = None
    snippet: str
    score: float
    updated_at: datetime | None = None


class Grounding(BaseModel):
    score: float | None = None
    unsupported_sentences: int = 0
    checked: bool = False


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


class Timings(BaseModel):
    guard_in: float | None = None
    query_understanding: float | None = None
    retrieval: float | None = None
    rerank: float | None = None
    ttft: float | None = None
    total: float | None = None


class CacheInfo(BaseModel):
    hit: bool = False
    type: Literal["exact", "semantic"] | None = None


class QueryResponse(BaseModel):
    request_id: UUID
    trace_id: str
    answer_type: AnswerType
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    related_sources: list[Citation] = Field(default_factory=list)
    grounding: Grounding = Field(default_factory=Grounding)
    usage: Usage = Field(default_factory=Usage)
    timings_ms: Timings = Field(default_factory=Timings)
    cache: CacheInfo = Field(default_factory=CacheInfo)
    model: str
    prompt_version: str
    index_version: str
    degraded: bool = False


# ----------------------------------------------------------- index payload


class ChunkPayload(BaseModel):
    """Stored alongside each vector. ``tenant_id`` and ``acl_groups`` are always filtered on."""

    tenant_id: str
    acl_groups: list[str]
    doc_id: str
    doc_version: str
    source: str
    doc_type: str
    title: str
    url: str | None = None
    language: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page: int | None = None
    chunk_index: int
    text: str
    updated_at: datetime | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
