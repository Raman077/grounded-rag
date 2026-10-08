# Learning AI engineering with this repo

This project is the textbook. Every concept worth knowing in RAG and agent
engineering becomes a change to this codebase, and the eval harness says whether
the change was worth making.

That last part is the whole point. Most people learn this field by reading about
techniques and adopting the ones that sound sophisticated. That produces engineers
who can name twelve retrieval strategies and cannot tell you which one their system
needs. The alternative is slower and much more durable: change one thing, measure it
against a benchmark you trust, and keep a record of what actually moved.

## The method

Every lesson runs the same loop.

1. **Question.** One sentence, answerable with a number. "Does BM25 improve recall on
   exact-identifier queries?" — not "should we add hybrid search?"
2. **Predict.** Write down the number you expect, *before* running anything. Commit to
   it. This is the step everyone skips and the step that teaches most: the gap between
   what you expected and what happened is the shape of what you did not understand.
3. **Change.** The smallest edit that answers the question. One variable.
4. **Measure.** `uv run python eval/harness/run_retrieval_eval.py --retriever ... --out ...`
5. **Conclude.** Write it down — in the commit message, in a baseline file, or in
   `docs/experiments/`. A negative result is a result. Phase 2a's whole finding was
   that the benchmark could not detect the change; that was worth more than a win.

Rule: **no technique enters the live query path without a number justifying it.**
`rag_core.retrieval` exists so strategies can sit side by side, unproven, until the
eval earns them a promotion.

## Why the eval came first

Phase 1 reported `source_recall = 1.00` and that looked like success. It was not. The
corpus is six documents and `k` is eight, so retrieving eight chunks from six
documents finds the right one every time, whatever the retriever does. The metric was
pinned at its ceiling before any work began.

Worse, the generation metrics used `qwen3.8-27b` as both the generator and the judge.
A model grading its own output is not evidence.

Two lessons, both general:

- **A metric at its ceiling measures nothing.** Before trusting a benchmark, confirm
  it can fail. Run the dumbest possible baseline — random retrieval, no retrieval — and
  check the score drops. If it does not, the benchmark is broken.
- **Know what your judge is.** An LLM judge is a model with preferences, including a
  preference for its own output. Measure its agreement with human labels before
  quoting its scores.

## The ladder

Each rung is a lesson: a question, a change in this repo, and a metric that answers it.
Rungs build on each other, so work up in order.

| # | Concept | The change here | What the number tells you |
|---|---|---|---|
| 1 | **Benchmark design** | corpus v1 + golden set v1 | Whether you can measure at all |
| 2 | **Chunking** | sweep size, overlap, structure-aware vs fixed | Usually the biggest single lever, and the most ignored |
| 3 | **Embeddings** | swap bge-small → bge-base → voyage | What model choice is actually worth, against its cost |
| 4 | **Hybrid & fusion** | already built, still unproven | Where lexical beats semantic, and by how much |
| 5 | **Reranking** | already built, still unproven | The quality/latency exchange rate |
| 6 | **Context engineering** | parent expansion, ordering, packing | Retrieval found it; did the generator use it? |
| 7 | **Contextual Retrieval** | LLM-written chunk headers at ingest | Anthropic's technique, measured on your own corpus |
| 8 | **Generation & grounding** | prompt variants, citation enforcement | Faithfulness vs. helpfulness, which trade against each other |
| 9 | **LLM-as-judge** | hand-label, measure agreement | How much to trust every number above |
| 10 | **Agents (LangGraph)** | `/v1/agent`, loops and state | When a loop beats a pipeline — and when it just costs more |
| 11 | **Caching & cost** | semantic cache, prompt caching | Latency and money, the metrics that decide deployments |
| 12 | **Production** | index versioning, SLOs, drift | What separates a demo from a service |

Rungs 4 and 5 are built but unproven. That is deliberate and it is the honest state of
the project: the code exists, the evidence does not yet.

## Lesson 1 — build a benchmark that can fail

**Question.** Can a corpus be built where dense-only retrieval measurably loses?

**Predict before you start.** Write your guesses down:

- Dense recall@8 on exact-identifier queries (error codes, version strings): ____
- BM25 recall@8 on the same: ____
- Dense recall@8 on pure-paraphrase queries: ____
- BM25 recall@8 on the same: ____

**The change.** `eval/fixtures/corpus-v1/`, 50+ documents, with four query families in
`eval/golden/golden_v1.jsonl`:

- **exact identifier** — `E4019`, `v2.14.3`, `SKU-88421`. Terms that carry no semantic
  neighbourhood. Dense embeddings should struggle; BM25 should not.
- **pure paraphrase** — the question shares no content words with the passage that
  answers it. The mirror image: dense should win, BM25 should fail.
- **near-duplicate distractors** — five documents describing similar policies, one
  correct. Retrieval finds all five; only a reranker can order them.
- **multi-hop** — the answer needs two documents joined. Nothing in Phase 2 handles
  this; it exists to be failed now and fixed at rung 10.

Enough documents that `k=8` is a small slice of the corpus, not most of it.

**Measure.** Run all three retrievers against `golden_v1`, broken down by family. The
`by_type` block in the report already does this.

**Expected conclusion.** Hybrid should win decisively on exact identifiers and roughly
tie elsewhere. Reranking should win on near-duplicates. Both should cost latency. If
that is *not* what the numbers say, the corpus is still wrong — or one of the
assumptions above is, which is the more interesting outcome.

## How to use me

Ask for the concept before the code. "Why does RRF use `1/(k+rank)` instead of just
averaging the scores?" is a better question than "add RRF", and you only get to ask it
once per concept.

Useful shapes of question:

- *Explain the mechanism* — "why does IDF have to be computed server-side here?"
- *Predict with me* — "what do you expect reranking to do to p95, and why?"
- *Challenge the result* — "recall went up but nDCG went down; what does that mean?"
- *Read the code back* — "walk me through what happens between my HTTP request and
  the first streamed token."

Ask me to be wrong out loud. If I say a technique will help and the number says
otherwise, that disagreement is the lesson — and it has already happened once in this
repo.

## Running things

```bash
uv sync --all-extras --all-packages          # once
uv run pytest -q                              # 46 tests, ~8s
uv run ruff check . && uv run ruff format --check . && uv run mypy libs services

# the retrieval eval — no API key, no judge, about a second
$env:RAG_QDRANT_LOCATION = ":memory:"        # PowerShell; export on bash
uv run python eval/harness/run_retrieval_eval.py --retriever dense --k 8
uv run python eval/harness/run_retrieval_eval.py --retriever hybrid+rerank --k 8 \
    --baseline eval/baselines/retrieval-dense-v0.json
```

`--baseline` exits non-zero on a regression, which is how this becomes a CI gate rather
than a thing you remember to run.
