"""Batch ingestion CLI.

rag-ingest ingest ./docs --tenant acme --acl public --acl engineering
rag-ingest delete --tenant acme --source local --path guides/setup.md
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections import Counter
from pathlib import Path

from rag_core.config import get_settings
from rag_core.embeddings import build_embedder
from rag_core.telemetry import setup_telemetry
from rag_core.vectorstore import VectorStore

from ingest_service.parsers import SUPPORTED_SUFFIXES
from ingest_service.pipeline import IngestOptions, IngestResult, ingest_file, make_doc_id


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-ingest", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="index a file or every supported file under a folder")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--tenant", required=True)
    ingest.add_argument("--acl", action="append", required=True, help="group allowed to read; repeatable")
    ingest.add_argument("--source", default="local")
    ingest.add_argument("--doc-type", default="doc")
    ingest.add_argument("--force", action="store_true", help="re-index even if the file is unchanged")

    delete = sub.add_parser("delete", help="remove one document from the index")
    delete.add_argument("--tenant", required=True)
    delete.add_argument("--source", default="local")
    delete.add_argument("--path", required=True, help="path relative to the folder it was ingested from")
    return parser


def _discover(path: Path) -> tuple[Path, list[Path]]:
    if path.is_file():
        return path.parent, [path]
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    return path, files


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    setup_telemetry("rag-ingest")
    store = VectorStore.from_settings(settings)
    await store.ensure_collection()

    if args.command == "delete":
        doc_id = make_doc_id(args.tenant, args.source, Path(args.path).as_posix())
        await store.delete_document(args.tenant, doc_id)
        print(f"deleted {args.path} ({doc_id})")
        return 0

    root, files = _discover(args.path.resolve())
    if not files:
        print(f"no supported files under {args.path} ({', '.join(sorted(SUPPORTED_SUFFIXES))})")
        return 1

    embedder = build_embedder(settings)
    options = IngestOptions(
        tenant_id=args.tenant,
        acl_groups=args.acl,
        source=args.source,
        doc_type=args.doc_type,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        force=args.force,
    )
    results: list[IngestResult] = []
    for file in files:
        try:
            result = await ingest_file(file, root=root, options=options, store=store, embedder=embedder)
        except Exception as exc:  # keep going; report at the end
            logging.exception("failed to ingest %s", file)
            result = IngestResult(file.relative_to(root).as_posix(), "", "failed", reason=str(exc))
        results.append(result)
        detail = f" ({result.chunks} chunks)" if result.chunks else f" - {result.reason}" if result.reason else ""
        print(f"{result.status:<9} {result.path}{detail}")

    counts = Counter(r.status for r in results)
    print("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    await store.client.close()
    return 1 if counts.get("failed") else 0


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(asyncio.run(_run(_parser().parse_args(argv))))


if __name__ == "__main__":
    main()
