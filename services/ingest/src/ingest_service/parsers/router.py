"""Pick a parser by file type. Every parser returns a ParsedDocument of markdown sections."""

from __future__ import annotations

import logging
from pathlib import Path

from ingest_service.parsers.base import ParsedDocument, ParseError, Section, default_title
from ingest_service.parsers.markdown import parse_markdown

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}
MIN_CHARS = 20


def parse_file(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        doc = parse_markdown(_read_text(path), fallback_title=default_title(path))
    elif suffix == ".txt":
        doc = ParsedDocument(title=default_title(path), sections=[Section(heading_path=[], text=_read_text(path))])
    elif suffix in {".html", ".htm"}:
        doc = parse_html(_read_text(path), fallback_title=default_title(path))
    elif suffix == ".pdf":
        doc = parse_pdf(path)
    else:
        raise ParseError(f"unsupported file type: {suffix or '<none>'}")

    if doc.char_count < MIN_CHARS:
        raise ParseError("no extractable text (empty or scanned document?)")
    return doc


def parse_html(html: str, *, fallback_title: str) -> ParsedDocument:
    import trafilatura

    markdown = trafilatura.extract(
        html, output_format="markdown", include_tables=True, include_links=False, favor_recall=True
    )
    if not markdown:
        raise ParseError("no main content found in HTML")
    metadata = trafilatura.extract_metadata(html)
    title = (metadata.title if metadata and metadata.title else None) or fallback_title
    doc = parse_markdown(markdown, fallback_title=title)
    doc.title = title
    return doc


def parse_pdf(path: Path) -> ParsedDocument:
    """Use Docling when installed (layout and tables); otherwise fall back to pypdf text."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        return _parse_pdf_basic(path)

    result = DocumentConverter().convert(str(path))
    return parse_markdown(result.document.export_to_markdown(), fallback_title=default_title(path))


def _parse_pdf_basic(path: Path) -> ParsedDocument:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pypdf raises many types for corrupt files
        raise ParseError(f"unreadable PDF: {exc}") from exc
    title = default_title(path)
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title)
    sections = [
        Section(heading_path=[], text=text.strip(), page=number)
        for number, page in enumerate(reader.pages, start=1)
        if (text := page.extract_text() or "").strip()
    ]
    return ParsedDocument(title=title, sections=sections)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        log.warning("%s is not UTF-8; decoding as latin-1", path)
        return raw.decode("latin-1")
