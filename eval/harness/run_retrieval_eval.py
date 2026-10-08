"""Measure retrieval quality directly, with no LLM in the loop.

    python eval/harness/run_retrieval_eval.py [--golden eval/golden/golden_v0.jsonl]
        [--retriever dense|hybrid|hybrid+rerank] [--out report.json] [--baseline prior.json]

Phase 2 changes retrieval, so it is measured against retrieval -- not through the
generator and an LLM judge. That makes the numbers deterministic, free and fast
enough to run on every commit, and it removes the judge as a confound.

Metrics (per query, averaged; k = --k)
  recall@k    share of the query's expected source documents that appear in the top k
  hit@k       1.0 if any expected source appears in the top k
  mrr         1 / rank of the first expected source (0 if absent)
  ndcg@k      graded by whether a chunk belongs to an expected source
  noise@k     share of retrieved chunks from documents that are NOT expected --
              the precision side, which recall alone hides

Negative items (``no_answer``, ``acl``) are scored separately: they have no expected
source, so the thing to measure is that nothing scores above the relevance gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ingest_service.pipeline import IngestOptions, ingest_file
from rag_core.config import Settings, get_settings
from rag_core.embeddings import build_embedder
from rag_core.retrieval import Retriever, build_retriever
from rag_core.sparse import build_sparse_embedder
from rag_core.vectorstore import VectorStore

NEGATIVE_TYPES = {"no_answer", "acl"}
QUALITY_METRICS = ("recall_at_k", "hit_at_k", "mrr", "ndcg_at_k", "correct_rejection_rate")


@dataclass(frozen=True)
class Corpus:
    path: Path
    groups: tuple[str, ...]


async def index_corpus(
    store: VectorStore,
    embedder: Any,
    sparse_embedder: Any,
    settings: Settings,
    corpora: list[Corpus],
    tenant: str,
) -> int:
    await store.ensure_collection()
    total = 0
    for corpus in corpora:
        options = IngestOptions(
            tenant_id=tenant,
            acl_groups=corpus.groups,
            source="eval",
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            force=True,
        )
        for path in sorted(corpus.path.rglob("*")):
            if not path.is_file():
                continue
            result = await ingest_file(
                path,
                root=corpus.path,
                options=options,
                store=store,
                embedder=embedder,
                sparse_embedder=sparse_embedder,
            )
            total += result.chunks
    return total


def ndcg(relevances: list[int], k: int) -> float:
    """Binary-relevance nDCG@k. Ideal ranking puts every relevant chunk first."""
    gains = [r / math.log2(i + 2) for i, r in enumerate(relevances[:k])]
    ideal = sorted(relevances, reverse=True)[:k]
    best = [r / math.log2(i + 2) for i, r in enumerate(ideal)]
    denominator = sum(best)
    return round(sum(gains) / denominator, 4) if denominator else 0.0


def score_positive(expected: set[str], retrieved_urls: list[str], k: int) -> dict[str, float]:
    found = {url for url in retrieved_urls[:k] if url in expected}
    relevances = [1 if url in expected else 0 for url in retrieved_urls]
    rank = next((i + 1 for i, url in enumerate(retrieved_urls) if url in expected), 0)
    noise = [url for url in retrieved_urls[:k] if url not in expected]
    return {
        "recall_at_k": len(found) / len(expected) if expected else 1.0,
        "hit_at_k": 1.0 if found else 0.0,
        "mrr": 1.0 / rank if rank else 0.0,
        "ndcg_at_k": ndcg(relevances, k),
        "noise_at_k": len(noise) / k if k else 0.0,
    }


def summarize(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    positive = [r for r in rows if r["type"] not in NEGATIVE_TYPES]
    negative = [r for r in rows if r["type"] in NEGATIVE_TYPES]
    metrics: dict[str, Any] = {}

    for key in ("recall_at_k", "hit_at_k", "mrr", "ndcg_at_k", "noise_at_k"):
        values = [r["scores"][key] for r in positive if "scores" in r]
        if values:
            metrics[key] = round(statistics.mean(values), 4)

    if negative:
        # A negative item is handled correctly when nothing clears the relevance gate.
        metrics["correct_rejection_rate"] = round(sum(1.0 for r in negative if not r["above_gate"]) / len(negative), 4)

    by_type: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positive:
        grouped[row["type"]].append(row)
    for name, items in sorted(grouped.items()):
        by_type[name] = {
            "n": len(items),
            "recall_at_k": round(statistics.mean(i["scores"]["recall_at_k"] for i in items), 4),
            "mrr": round(statistics.mean(i["scores"]["mrr"] for i in items), 4),
        }

    latencies = [r["latency_ms"] for r in rows]
    metrics["latency_p50_ms"] = round(statistics.median(latencies), 1) if latencies else 0.0
    metrics["latency_mean_ms"] = round(statistics.mean(latencies), 1) if latencies else 0.0
    metrics["k"] = k
    return {"overall": metrics, "by_type": by_type}


def compare(current: dict[str, Any], baseline: dict[str, Any], max_drop: float) -> list[str]:
    failures = []
    for key in QUALITY_METRICS:
        now, before = current["overall"].get(key), baseline["overall"].get(key)
        if now is not None and before is not None and now < before - max_drop:
            failures.append(f"{key}: {now} < baseline {before} - {max_drop}")
    return failures


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.k > settings.retrieval_candidates:
        settings = settings.model_copy(update={"retrieval_candidates": args.k})

    items = [json.loads(line) for line in args.golden.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        items = items[: args.limit]

    embedder = build_embedder(settings)
    sparse_embedder = build_sparse_embedder(settings)
    store = VectorStore.from_settings(settings)
    corpora = [
        Corpus(args.corpus, ("public",)),
        *([Corpus(args.corpus_internal, ("internal",))] if args.corpus_internal.is_dir() else []),
    ]
    chunks = await index_corpus(store, embedder, sparse_embedder, settings, corpora, args.tenant)
    retriever: Retriever = await build_retriever(args.retriever, settings=settings, store=store, embedder=embedder)
    print(f"indexed {chunks} chunks  retriever={args.retriever}  k={args.k}\n")

    rows: list[dict[str, Any]] = []
    for item in items:
        started = datetime.now(UTC)
        hits = await retriever.retrieve(item["question"], tenant_id=args.tenant, acl_groups=args.group, limit=args.k)
        latency = (datetime.now(UTC) - started).total_seconds() * 1000
        urls = [h.payload.url or "?" for h in hits]
        row: dict[str, Any] = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "retrieved": urls[: args.k],
            "top_score": round(hits[0].score, 4) if hits else 0.0,
            "above_gate": bool(hits and hits[0].score >= settings.min_relevance_score),
            "latency_ms": round(latency, 1),
        }
        if item["type"] not in NEGATIVE_TYPES:
            row["expected"] = item["expected_sources"]
            row["scores"] = score_positive(set(item["expected_sources"]), urls, args.k)
            flag = "ok " if row["scores"]["hit_at_k"] else "MISS"
        else:
            flag = "ok " if not row["above_gate"] else "LEAK"
        rows.append(row)
        print(f"{item['id']} {item['type']:<9} {flag}  top={row['top_score']:<7} {','.join(urls[:3])}")

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "retriever": args.retriever,
        "golden": str(args.golden),
        "embedding_model": settings.embedding_model,
        "items": len(rows),
        "chunks_indexed": chunks,
        **summarize(rows, args.k),
        "rows": rows,
    }
    print("\n" + json.dumps({"overall": report["overall"], "by_type": report["by_type"]}, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nreport written to {args.out}")

    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        failures = compare(report, baseline, args.max_drop)
        if failures:
            print("\nREGRESSION:\n  " + "\n  ".join(failures))
            return 1
        print("\nno regression against baseline")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", type=Path, default=Path("eval/golden/golden_v0.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("eval/fixtures/corpus"))
    parser.add_argument("--corpus-internal", type=Path, default=Path("eval/fixtures/corpus-internal"))
    parser.add_argument("--retriever", default="dense", choices=["dense", "hybrid", "hybrid+rerank"])
    parser.add_argument("--tenant", default="nimbus")
    parser.add_argument("--group", action="append", default=None)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-drop", type=float, default=0.02)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    args.group = args.group or ["public"]
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
