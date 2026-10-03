"""Markdown is the canonical format: every other parser converts to it and lands here."""

from __future__ import annotations

import re

from ingest_service.parsers.base import ParsedDocument, Section

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def parse_markdown(text: str, *, fallback_title: str, page: int | None = None) -> ParsedDocument:
    title: str | None = None
    stack: list[tuple[int, str]] = []
    sections: list[Section] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(Section(heading_path=[h for _, h in stack], text=body, page=page))
        buffer.clear()

    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue
        match = None if in_fence else _HEADING_RE.match(line)
        if match is None:
            buffer.append(line)
            continue
        flush()
        level, heading = len(match.group(1)), match.group(2).strip()
        if level == 1 and title is None:
            title = heading
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
    flush()

    return ParsedDocument(title=title or fallback_title, sections=sections)
