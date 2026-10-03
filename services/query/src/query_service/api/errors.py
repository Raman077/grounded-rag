"""RFC 7807 problem+json responses for every error the API returns."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

PROBLEM_JSON = "application/problem+json"


class ProblemError(Exception):
    def __init__(
        self,
        status: int,
        title: str,
        detail: str | None = None,
        *,
        headers: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(detail or title)
        self.status = status
        self.title = title
        self.detail = detail
        self.headers = headers
        self.extra = extra or {}


def problem_response(
    request: Request,
    status: int,
    title: str,
    detail: str | None = None,
    *,
    headers: Mapping[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {"type": "about:blank", "title": title, "status": status, "instance": request.url.path}
    if detail:
        body["detail"] = detail
    body.update(extra or {})
    return JSONResponse(body, status_code=status, headers=headers, media_type=PROBLEM_JSON)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProblemError)
    async def _problem(request: Request, exc: ProblemError) -> JSONResponse:
        return problem_response(request, exc.status, exc.title, exc.detail, headers=exc.headers, extra=exc.extra)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": list(e.get("loc", [])), "msg": e.get("msg", ""), "type": e.get("type", "")} for e in exc.errors()
        ]
        return problem_response(
            request,
            422,
            "Request validation failed",
            "The request body does not match the schema.",
            extra={"errors": errors},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return problem_response(request, exc.status_code, str(exc.detail), headers=exc.headers)
