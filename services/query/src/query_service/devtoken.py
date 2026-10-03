"""Print a development JWT signed with RAG_JWT_SECRET.

rag-devtoken --tenant acme --group public --group engineering
"""

from __future__ import annotations

import argparse
from datetime import timedelta

from rag_core.config import get_settings

from query_service.api.auth import issue_token


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rag-devtoken", description=__doc__)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--group", action="append", default=[], help="repeatable")
    parser.add_argument("--subject", default="dev-user")
    parser.add_argument("--hours", type=float, default=12)
    args = parser.parse_args(argv)

    settings = get_settings()
    if settings.environment == "prod":
        parser.error("refusing to mint tokens with RAG_ENVIRONMENT=prod")
    print(
        issue_token(
            settings,
            subject=args.subject,
            tenant_id=args.tenant,
            groups=args.group,
            ttl=timedelta(hours=args.hours),
        )
    )


if __name__ == "__main__":
    main()
