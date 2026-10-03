"""Structure-aware chunking (Phase 1 baseline).

Sections come from the document's headings, so a chunk never spans two sections. Inside a
section the text is split into blocks (paragraphs, tables, fenced code) which are packed
greedily up to ``max_tokens``. Tables and code blocks are never cut in the middle unless a
single block is itself too large; oversized tables are split by rows with the header
repeated. Consecutive prose chunks share up to ``overlap_tokens`` of trailing sentences.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from ingest_service.parsers.base import ParsedDocument

BlockKind = Literal["prose", "table", "code"]

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def estimate_tokens(text: str) -> int:
    """Rough count (about 4 characters per token for English). Good enough for packing."""
    return max(1, math.ceil(len(text) / 4))


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    section_path: list[str]
    page: int | None


@dataclass(frozen=True)
class _Block:
    kind: BlockKind
    text: str


def chunk_document(doc: ParsedDocument, *, max_tokens: int = 350, overlap_tokens: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in doc.sections:
        for text in _chunk_section(section.text, max_tokens, overlap_tokens):
            chunks.append(Chunk(index=len(chunks), text=text, section_path=section.heading_path, page=section.page))
    return chunks


def _chunk_section(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    # Oversized prose is split a little smaller than the budget so the overlap still fits.
    prose_budget = max(max_tokens // 2, max_tokens - overlap_tokens)
    pieces: list[_Block] = []
    for block in _split_blocks(text):
        pieces.extend(_fit_block(block, max_tokens, prose_budget))

    chunks: list[str] = []
    current: list[_Block] = []
    current_tokens = 0
    for piece in pieces:
        size = estimate_tokens(piece.text)
        if current and current_tokens + size > max_tokens:
            chunks.append(_join(current))
            overlap = _overlap(current[-1], overlap_tokens)
            current = [overlap] if overlap and estimate_tokens(overlap.text) + size <= max_tokens else []
            current_tokens = sum(estimate_tokens(b.text) for b in current)
        current.append(piece)
        current_tokens += size
    if current:
        chunks.append(_join(current))
    return chunks


def _split_blocks(text: str) -> list[_Block]:
    blocks: list[_Block] = []
    lines: list[str] = []
    kind: BlockKind = "prose"

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            blocks.append(_Block(kind, body))
        lines.clear()

    for line in text.splitlines():
        if kind == "code":
            lines.append(line)
            if _FENCE_RE.match(line):
                flush()
                kind = "prose"
            continue
        if _FENCE_RE.match(line):
            flush()
            kind = "code"
            lines.append(line)
            continue
        is_table_row = line.lstrip().startswith("|")
        if is_table_row and kind != "table":
            flush()
            kind = "table"
        elif not is_table_row and kind == "table":
            flush()
            kind = "prose"
        if kind == "prose" and not line.strip():
            flush()
            continue
        lines.append(line)
    flush()
    return blocks


def _fit_block(block: _Block, max_tokens: int, prose_budget: int) -> list[_Block]:
    if estimate_tokens(block.text) <= max_tokens:
        return [block]
    if block.kind == "table":
        return [_Block("table", t) for t in _split_table(block.text, max_tokens)]
    if block.kind == "code":
        return [_Block("code", t) for t in _split_lines(block.text, max_tokens)]
    return [_Block("prose", t) for t in _split_prose(block.text, prose_budget)]


def _split_table(text: str, max_tokens: int) -> list[str]:
    rows = text.splitlines()
    header_len = 2 if len(rows) > 1 and set(rows[1].replace("|", "").strip()) <= set("-: ") else 1
    header, body = rows[:header_len], rows[header_len:]
    out: list[str] = []
    current: list[str] = []
    for row in body:
        candidate = "\n".join(header + current + [row])
        if current and estimate_tokens(candidate) > max_tokens:
            out.append("\n".join(header + current))
            current = []
        current.append(row)
    if current:
        out.append("\n".join(header + current))
    return out


def _split_lines(text: str, max_tokens: int) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if current and estimate_tokens("\n".join(current + [line])) > max_tokens:
            out.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        out.append("\n".join(current))
    return out


def _split_prose(text: str, max_tokens: int) -> list[str]:
    out: list[str] = []
    current = ""
    for sentence in _sentences(text):
        for part in _split_words(sentence, max_tokens):
            candidate = f"{current} {part}".strip()
            if current and estimate_tokens(candidate) > max_tokens:
                out.append(current)
                current = part
            else:
                current = candidate
    if current:
        out.append(current)
    return out


def _split_words(sentence: str, max_tokens: int) -> list[str]:
    if estimate_tokens(sentence) <= max_tokens:
        return [sentence]
    out: list[str] = []
    current: list[str] = []
    for word in sentence.split():
        if current and estimate_tokens(" ".join(current + [word])) > max_tokens:
            out.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        out.append(" ".join(current))
    return out


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _overlap(block: _Block, overlap_tokens: int) -> _Block | None:
    if overlap_tokens <= 0 or block.kind != "prose":
        return None
    picked: list[str] = []
    for sentence in reversed(_sentences(block.text)):
        if estimate_tokens(" ".join([sentence, *picked])) > overlap_tokens:
            break
        picked.insert(0, sentence)
    return _Block("prose", " ".join(picked)) if picked else None


def _join(blocks: list[_Block]) -> str:
    return "\n\n".join(b.text for b in blocks)
