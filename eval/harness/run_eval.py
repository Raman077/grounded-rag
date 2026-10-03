"""Run the golden set against a running query API and report quality and latency metrics.

    python eval/harness/run_eval.py --base-url http://localhost:8000 \
        --golden eval/golden/golden_v0.jsonl --tenant nimbus --group public \
        [--judge opencode|anthropic] [--baseline eval/baselines/phase1.json] [--out report.json]

Metrics
  source_recall          answerable items: share of expected source files that were cited
  answer_rate            answerable items: share answered (not refused / insufficient)
  correct_refusal_rate   no-answer and ACL items: share correctly declined
  correctness, faithfulness   (with --judge) 0-1 scores from an LLM judge
  latency_p50_ms, latency_p95_ms

With --baseline, exits 1 if any metric drops by more than --max-drop (absolute) or a latency
grows by more than --max-latency-increase (relative).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from query_service.api.auth import issue_token
from query_service.providers import LLMProvider, TextDelta, build_provider
from rag_core.config import get_settings

DECLINED = {"insufficient_context", "refused"}
NEGATIVE_TYPES = {"no_answer", "acl"}
QUALITY_METRICS = ("source_recall", "answer_rate", "correct_refusal_rate", "correctness", "faithfulness")
LATENCY_METRICS = ("latency_p50_ms", "latency_p95_ms")

JUDGE_SYSTEM = """You grade answers from a question-answering system. Reply with JSON only, no prose:
{"correctness": <0..1>, "faithfulness": <0..1>, "reason": "<one sentence>"}
correctness: how well the answer matches the reference answer (1 = same facts, 0 = wrong or missing).
faithfulness: how well every claim in the answer is supported by the cited source snippets
(1 = fully supported, 0 = unsupported). Snippets may be truncated; judge what is shown."""


async def ask(client: httpx.AsyncClient, token: str, question: str) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    resp = await client.post(
        "/v1/query",
        json={"query": question},
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    elapsed = (time.perf_counter() - start) * 1000
    if resp.status_code != 200:
        return {"answer_type": "error", "error": resp.text[:300], "citations": []}, elapsed
    data: dict[str, Any] = resp.json()
    return data, elapsed


async def judge(provider: LLMProvider, item: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    snippets = "\n".join(f"[{c['id']}] {c['snippet']}" for c in result.get("citations", []))
    prompt = (
        f"Question: {item['question']}\n\nReference answer: {item['expected_answer']}\n\n"
        f"System answer: {result.get('answer', '')}\n\nCited source snippets:\n{snippets or '(none)'}"
    )
    text = ""
    async for event in provider.stream(
        system=JUDGE_SYSTEM, messages=[{"role": "user", "content": prompt}], max_tokens=300
    ):
        if isinstance(event, TextDelta):
            text += event.text
    start, end = text.find("{"), text.rfind("}")
    try:
        parsed: dict[str, Any] = json.loads(text[start : end + 1])
        return {
            "correctness": float(parsed["correctness"]),
            "faithfulness": float(parsed["faithfulness"]),
            "reason": str(parsed.get("reason", "")),
        }
    except (ValueError, KeyError, TypeError):
        return {"judge_error": text[:200]}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return round(ordered[index], 1)


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    positive = [r for r in rows if r["type"] not in NEGATIVE_TYPES]
    negative = [r for r in rows if r["type"] in NEGATIVE_TYPES]
    metrics: dict[str, float] = {}

    recalls = []
    for r in positive:
        expected = set(r["expected_sources"])
        cited = {c.get("url") for c in r["result"].get("citations", [])}
        recalls.append(len(expected & cited) / len(expected) if expected else 1.0)
    if positive:
        metrics["source_recall"] = round(statistics.mean(recalls), 3)
        metrics["answer_rate"] = round(
            sum(r["result"]["answer_type"] not in DECLINED | {"error"} for r in positive) / len(positive), 3
        )
    if negative:
        metrics["correct_refusal_rate"] = round(
            sum(r["result"]["answer_type"] in DECLINED for r in negative) / len(negative), 3
        )
    for key in ("correctness", "faithfulness"):
        scores = [r["judge"][key] for r in positive if key in r.get("judge", {})]
        if scores:
            metrics[key] = round(statistics.mean(scores), 3)
    latencies = [r["latency_ms"] for r in rows]
    metrics["latency_p50_ms"] = percentile(latencies, 50)
    metrics["latency_p95_ms"] = percentile(latencies, 95)
    metrics["errors"] = sum(r["result"]["answer_type"] == "error" for r in rows)
    return metrics


def compare(
    current: dict[str, float], baseline: dict[str, float], max_drop: float, max_latency_increase: float
) -> list[str]:
    failures = []
    for key in QUALITY_METRICS:
        if key in current and key in baseline and current[key] < baseline[key] - max_drop:
            failures.append(f"{key}: {current[key]} < baseline {baseline[key]} - {max_drop}")
    for key in LATENCY_METRICS:
        if key in current and key in baseline and current[key] > baseline[key] * (1 + max_latency_increase):
            failures.append(f"{key}: {current[key]} > baseline {baseline[key]} * {1 + max_latency_increase}")
    return failures


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    token = issue_token(settings, subject="eval", tenant_id=args.tenant, groups=args.group)
    items = [json.loads(line) for line in args.golden.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        items = items[: args.limit]

    judge_provider: LLMProvider | None = None
    if args.judge:
        judge_provider = build_provider(settings.model_copy(update={"llm_provider": args.judge}))

    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout) as client:
        for item in items:
            result, latency = await ask(client, token, item["question"])
            row = {**item, "result": result, "latency_ms": round(latency, 1)}
            if (
                judge_provider
                and item["type"] not in NEGATIVE_TYPES
                and result["answer_type"] not in DECLINED | {"error"}
            ):
                row["judge"] = await judge(judge_provider, item, result)
            rows.append(row)
            cited = ",".join(sorted({c.get("url", "?") for c in result.get("citations", [])})) or "-"
            print(f"{item['id']} {item['type']:<9} {result['answer_type']:<20} {latency:>8.0f} ms  cited={cited}")
    if judge_provider:
        await judge_provider.aclose()

    metrics = summarize(rows)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "golden": str(args.golden),
        "items": len(rows),
        "metrics": metrics,
        "rows": rows,
    }
    print("\n" + json.dumps(metrics, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"report written to {args.out}")

    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["metrics"]
        failures = compare(metrics, baseline, args.max_drop, args.max_latency_increase)
        if failures:
            print("\nREGRESSION:\n  " + "\n  ".join(failures))
            return 1
        print("\nno regression against baseline")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--golden", type=Path, default=Path("eval/golden/golden_v0.jsonl"))
    parser.add_argument("--tenant", default="nimbus")
    parser.add_argument("--group", action="append", default=None)
    parser.add_argument("--judge", choices=["anthropic", "opencode"])
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-drop", type=float, default=0.05)
    parser.add_argument("--max-latency-increase", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    args.group = args.group or ["public"]
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
