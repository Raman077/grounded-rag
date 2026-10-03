from pathlib import Path

import pytest
from ingest_service.parsers import ParseError, parse_file
from ingest_service.parsers.markdown import parse_markdown

CORPUS = Path(__file__).parents[3] / "eval" / "fixtures" / "corpus"


def test_markdown_sections_follow_heading_hierarchy() -> None:
    doc = parse_markdown("# Title\nintro\n## A\none\n### A.1\ntwo\n## B\nthree\n", fallback_title="fallback")
    assert doc.title == "Title"
    assert [(s.heading_path, s.text) for s in doc.sections] == [
        (["Title"], "intro"),
        (["Title", "A"], "one"),
        (["Title", "A", "A.1"], "two"),
        (["Title", "B"], "three"),
    ]


def test_markdown_ignores_headings_inside_code_fences() -> None:
    doc = parse_markdown("# T\n```bash\n# not a heading\necho hi\n```\n", fallback_title="x")
    assert len(doc.sections) == 1
    assert "# not a heading" in doc.sections[0].text


def test_markdown_without_h1_uses_fallback_title() -> None:
    assert parse_markdown("## Only h2\ntext", fallback_title="From File").title == "From File"


def test_html_extracts_main_content_and_drops_boilerplate() -> None:
    doc = parse_file(CORPUS / "release-notes.html")
    text = "\n".join(s.text for s in doc.sections)
    assert doc.title == "Nimbus Release Notes"
    assert "$0.004 per GB-month" in text
    assert "1 March 2027" in text
    assert "Cookie settings" not in text


def test_unsupported_and_empty_files_raise(tmp_path: Path) -> None:
    (tmp_path / "data.bin").write_bytes(b"\x00\x01")
    with pytest.raises(ParseError, match="unsupported"):
        parse_file(tmp_path / "data.bin")
    (tmp_path / "empty.md").write_text("# \n\n", encoding="utf-8")
    with pytest.raises(ParseError, match="no extractable text"):
        parse_file(tmp_path / "empty.md")


def test_plain_text_is_one_section(tmp_path: Path) -> None:
    (tmp_path / "notes-file.txt").write_text("Plain notes about the archive tier.", encoding="utf-8")
    doc = parse_file(tmp_path / "notes-file.txt")
    assert doc.title == "Notes File"
    assert len(doc.sections) == 1
