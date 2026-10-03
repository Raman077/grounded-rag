"""Bearer JWT authentication. Tenant and ACL groups come only from verified token claims,
never from the request body.

Expected claims: ``sub``, ``tenant_id``, ``groups`` (list of strings), ``exp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Request
from rag_core.config import Settings

from query_service.api.errors import ProblemError

_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant_id: str
    groups: tuple[str, ...]


def decode_token(token: str, settings: Settings) -> Principal:
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            options={"require": ["sub", "exp"], "verify_aud": settings.jwt_audience is not None},
        )
    except jwt.ExpiredSignatureError as exc:
        raise ProblemError(401, "Unauthorized", "Token has expired.", headers=_UNAUTHORIZED_HEADERS) from exc
    except jwt.InvalidTokenError as exc:
        raise ProblemError(401, "Unauthorized", "Invalid token.", headers=_UNAUTHORIZED_HEADERS) from exc

    tenant_id = claims.get("tenant_id")
    groups = claims.get("groups", [])
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ProblemError(403, "Forbidden", "Token has no tenant_id claim.")
    if not isinstance(groups, list) or not all(isinstance(g, str) for g in groups):
        raise ProblemError(403, "Forbidden", "Token groups claim must be a list of strings.")
    return Principal(subject=str(claims["sub"]), tenant_id=tenant_id, groups=tuple(groups))


async def get_principal(request: Request) -> Principal:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise ProblemError(401, "Unauthorized", "Missing bearer token.", headers=_UNAUTHORIZED_HEADERS)
    settings: Settings = request.app.state.settings
    return decode_token(token.strip(), settings)


def issue_token(
    settings: Settings, *, subject: str, tenant_id: str, groups: list[str], ttl: timedelta = timedelta(hours=12)
) -> str:
    """Mint a token signed with the configured secret. For development and tests only."""
    now = datetime.now(UTC)
    claims = {"sub": subject, "tenant_id": tenant_id, "groups": groups, "iat": now, "exp": now + ttl}
    if settings.jwt_audience:
        claims["aud"] = settings.jwt_audience
    return jwt.encode(claims, settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm)
