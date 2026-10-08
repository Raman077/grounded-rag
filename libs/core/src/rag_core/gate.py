"""The relevance gate: deciding when retrieval found nothing worth answering from.

The gate is what makes "I don't know" possible. Without it, the generator is handed
the best of a bad set and writes something plausible from it.

Phase 1 gated on a hand-picked cosine threshold (``0.55``) with a comment conceding it
had to be re-tuned whenever the embedding model changed. That is a real weakness:

* cosine thresholds do not transfer between embedding models, and barely transfer
  between corpora;
* a fused (RRF) score has no absolute scale at all -- it ranks, it does not measure;
* reranker outputs are logits, on yet another scale.

So the threshold is no longer a constant anyone types. ``calibrate`` fits it from
labelled data by maximising F1, and each retriever declares which scale its scores are
on. A score with no meaningful scale gets a rank-based gate and says so, rather than
being compared against a number that means nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from rag_core.vectorstore import ScoredChunk

Strategy = Literal["score", "rank"]


@dataclass(frozen=True)
class GateDecision:
    relevant: list[ScoredChunk]
    strategy: Strategy
    threshold: float | None
    top_score: float | None

    @property
    def passed(self) -> bool:
        return bool(self.relevant)


@dataclass(frozen=True)
class RelevanceGate:
    """Keeps the chunks worth answering from.

    ``threshold`` applies to ``ScoredChunk.normalized_score``, which every stage puts on
    a 0-1 scale. When the chunks carry no normalized score -- a fused result that was
    never reranked -- there is nothing to threshold, so the gate falls back to rank and
    reports that, letting the caller record a degraded decision instead of a silent one.
    """

    threshold: float
    margin: float = 0.0

    def apply(self, chunks: Sequence[ScoredChunk], *, limit: int, strict: bool = False) -> GateDecision:
        if not chunks:
            return GateDecision([], "score", self.threshold, None)

        if any(c.normalized_score is None for c in chunks):
            return GateDecision(list(chunks[:limit]), "rank", None, chunks[0].score)

        cutoff = self.threshold + (self.margin if strict else 0.0)
        relevant = [c for c in chunks if (c.normalized_score or 0.0) >= cutoff][:limit]
        return GateDecision(relevant, "score", cutoff, chunks[0].normalized_score)


@dataclass(frozen=True)
class Calibration:
    threshold: float
    f1: float
    precision: float
    recall: float
    positives: int
    negatives: int


def calibrate(positives: Sequence[float], negatives: Sequence[float], *, steps: int = 200) -> Calibration:
    """Pick the threshold with the best F1 separating answerable from unanswerable queries.

    ``positives`` are top scores for queries the corpus can answer, ``negatives`` for
    queries it cannot. F1 rather than accuracy because the two classes are rarely
    balanced, and both error directions matter: too low a threshold invents answers,
    too high refuses ones it could have given.
    """
    if not positives or not negatives:
        raise ValueError("calibration needs both positive and negative examples")

    lo, hi = min([*positives, *negatives]), max([*positives, *negatives])
    best = Calibration(
        threshold=lo, f1=0.0, precision=0.0, recall=0.0, positives=len(positives), negatives=len(negatives)
    )
    for i in range(steps + 1):
        threshold = lo + (hi - lo) * i / steps
        tp = sum(1 for s in positives if s >= threshold)
        fp = sum(1 for s in negatives if s >= threshold)
        fn = len(positives) - tp
        if tp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best.f1:
            best = Calibration(
                threshold=round(threshold, 4),
                f1=round(f1, 4),
                precision=round(precision, 4),
                recall=round(recall, 4),
                positives=len(positives),
                negatives=len(negatives),
            )
    return best
