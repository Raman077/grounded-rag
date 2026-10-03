from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from rag_core.schemas import QueryRequest, QueryResponse

from query_service.api.auth import Principal, get_principal
from query_service.api.errors import ProblemError, problem_response
from query_service.pipeline.orchestrator import (
    Error,
    Final,
    Meta,
    PipelineEvent,
    QueryPipeline,
    Retrieval,
    Token,
)

router = APIRouter()

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", include_in_schema=False)
async def readyz(request: Request) -> JSONResponse:
    pipeline: QueryPipeline = request.app.state.pipeline
    ok = await pipeline.store.ping()
    return JSONResponse({"status": "ok" if ok else "unavailable"}, status_code=200 if ok else 503)


@router.post(
    "/v1/query",
    response_model=QueryResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
    summary="Answer a question from the indexed documents, with citations",
)
async def query(
    request: Request,
    body: QueryRequest,
    principal: Annotated[Principal, Depends(get_principal)],
) -> Any:
    pipeline: QueryPipeline = request.app.state.pipeline
    limit = pipeline.settings.max_query_chars
    if len(body.query) > limit:
        raise ProblemError(413, "Query too long", f"Query must be at most {limit} characters.")

    events = pipeline.run(body, principal)
    if "text/event-stream" in request.headers.get("accept", ""):
        return StreamingResponse(_sse(events), media_type="text/event-stream", headers=_SSE_HEADERS)

    async for event in events:
        if isinstance(event, Final):
            return event.response
        if isinstance(event, Error):
            status = 503 if event.retryable else 502
            headers = {"Retry-After": "5"} if event.retryable else None
            return problem_response(
                request,
                status,
                "Generation failed",
                event.message,
                headers=headers,
                extra={"code": event.code, "retryable": event.retryable},
            )
    raise RuntimeError("query pipeline ended without a final event")


async def _sse(events: AsyncIterator[PipelineEvent]) -> AsyncIterator[str]:
    async for event in events:
        if isinstance(event, Meta):
            yield _frame(
                "meta", {"request_id": str(event.request_id), "trace_id": event.trace_id, "cache": {"hit": False}}
            )
        elif isinstance(event, Retrieval):
            yield _frame("retrieval", {"citations": [c.model_dump(mode="json") for c in event.citations]})
        elif isinstance(event, Token):
            yield _frame("token", {"delta": event.delta})
        elif isinstance(event, Final):
            yield _frame("final", event.response.model_dump(mode="json"))
        elif isinstance(event, Error):
            yield _frame("error", {"code": event.code, "message": event.message, "retryable": event.retryable})


def _frame(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
