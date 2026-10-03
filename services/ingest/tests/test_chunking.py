from ingest_service.chunking import chunk_document, estimate_tokens
from ingest_service.parsers.base import ParsedDocument, Section


def _doc(text: str, heading: list[str] | None = None) -> ParsedDocument:
    return ParsedDocument(title="T", sections=[Section(heading_path=heading or ["T"], text=text)])


def test_small_section_is_one_chunk_with_its_heading_path() -> None:
    chunks = chunk_document(_doc("Short paragraph.", ["T", "Plans"]), max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].section_path == ["T", "Plans"]


def test_chunks_respect_max_tokens_and_never_cross_sections() -> None:
    para = " ".join(f"Sentence number {i} talks about storage." for i in range(60))
    doc = ParsedDocument(title="T", sections=[Section(["T", "A"], para), Section(["T", "B"], "Tiny.")])
    chunks = chunk_document(doc, max_tokens=80, overlap_tokens=15)
    assert all(estimate_tokens(c.text) <= 80 for c in chunks)
    assert chunks[-1].section_path == ["T", "B"] and chunks[-1].text == "Tiny."
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_consecutive_prose_chunks_overlap() -> None:
    para = " ".join(f"Fact {i} is here." for i in range(80))
    chunks = chunk_document(_doc(para), max_tokens=60, overlap_tokens=12)
    assert len(chunks) > 1
    last_sentence = chunks[0].text.rsplit(". ", 1)[-1]
    overlap = chunks[1].text[: chunks[1].text.index(last_sentence) + len(last_sentence)]
    assert chunks[0].text.endswith(overlap)  # chunk 2 opens with the tail of chunk 1
    assert 0 < estimate_tokens(overlap) <= 12


def test_table_kept_whole_when_it_fits() -> None:
    table = "| Plan | Price |\n|---|---|\n| Pro | $49 |\n| Business | $299 |"
    chunks = chunk_document(_doc(f"Intro text.\n\n{table}\n\nOutro."), max_tokens=200)
    assert any(table in c.text for c in chunks)


def test_oversized_table_split_by_rows_with_header_repeated() -> None:
    rows = "\n".join(f"| Row {i} | value {i} |" for i in range(60))
    table = f"| Name | Value |\n|---|---|\n{rows}"
    chunks = chunk_document(_doc(table), max_tokens=80)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith("| Name | Value |\n|---|---|")
    body_rows = [line for c in chunks for line in c.text.splitlines()[2:]]
    assert len(body_rows) == 60


def test_code_block_not_split_or_overlapped() -> None:
    code = "```http\nHTTP/1.1 429 Too Many Requests\nRetry-After: 2\n```"
    prose = " ".join(f"Line {i} of explanation." for i in range(30))
    chunks = chunk_document(_doc(f"{prose}\n\n{code}"), max_tokens=60, overlap_tokens=10)
    assert sum(code in c.text for c in chunks) == 1
