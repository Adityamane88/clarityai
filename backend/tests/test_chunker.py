"""Elite contract tests for ``app.services.chunker.chunk_sections``.

These tests are intentionally broader than the original single smoke test.
They validate the behavior a production-ready chunker should provide:

- deterministic output
- stable page/source/heading metadata propagation
- non-empty chunks only
- reasonable chunk sizing
- overlap continuity between adjacent chunks
- resilience to blank sections and long unbroken text

The suite is flexible about the concrete return type. ``chunk_sections`` may
return dicts, dataclasses, pydantic models, or simple objects, as long as the
expected fields are exposed either directly or through a ``metadata`` mapping.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from app.services.chunker import chunk_sections


TEXT_FIELDS = ("text", "content", "chunk_text", "body")
PAGE_FIELDS = ("page_label", "page", "page_number")
SOURCE_FIELDS = ("source", "source_name", "document_id", "doc_id")
HEADING_FIELDS = ("heading", "title", "section_heading", "section_title")


def _call_chunker(
    sections: list[dict[str, Any]],
    *,
    max_chars: int = 180,
    overlap: int = 30,
) -> list[Any]:
    result = chunk_sections(sections, max_chars=max_chars, overlap=overlap)
    if isinstance(result, list):
        return result
    if isinstance(result, Iterable):
        return list(result)
    raise AssertionError(
        f"chunk_sections should return an iterable of chunks, got {type(result)!r}"
    )


def _from_metadata(chunk: Any, key: str) -> Any:
    if isinstance(chunk, Mapping):
        meta = chunk.get("metadata")
    else:
        meta = getattr(chunk, "metadata", None)
    if isinstance(meta, Mapping) and key in meta:
        return meta[key]
    return None


def _get(chunk: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(chunk, Mapping) and key in chunk:
            return chunk[key]
        if hasattr(chunk, key):
            return getattr(chunk, key)
        value = _from_metadata(chunk, key)
        if value is not None:
            return value
    return None


def _text(chunk: Any) -> str:
    value = _get(chunk, *TEXT_FIELDS)
    assert value is not None, f"chunk is missing text/content field: {chunk!r}"
    assert isinstance(value, str), f"chunk text should be str, got {type(value)!r}"
    return value


def _page_label(chunk: Any) -> str | None:
    value = _get(chunk, *PAGE_FIELDS)
    return None if value is None else str(value)


def _source(chunk: Any) -> str | None:
    value = _get(chunk, *SOURCE_FIELDS)
    return None if value is None else str(value)


def _heading(chunk: Any) -> str | None:
    value = _get(chunk, *HEADING_FIELDS)
    return None if value is None else str(value)


def _normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _boundary_overlap(a: str, b: str, *, min_chars: int = 12) -> bool:
    """Return True when adjacent chunks clearly share overlap.

    We check both exact suffix/prefix character overlap and short token overlap
    so the assertion stays robust across slightly different chunkers.
    """

    left = _normalize_ws(a)
    right = _normalize_ws(b)
    max_chars = min(len(left), len(right), 160)
    for size in range(max_chars, min_chars - 1, -1):
        if left[-size:] == right[:size]:
            return True

    left_words = left.split()
    right_words = right.split()
    for size in range(min(len(left_words), len(right_words), 10), 1, -1):
        if left_words[-size:] == right_words[:size]:
            return True
    return False


def _snapshot(chunks: list[Any]) -> list[tuple[str, str | None, str | None, str | None]]:
    return [
        (
            _normalize_ws(_text(chunk)),
            _page_label(chunk),
            _source(chunk),
            _heading(chunk),
        )
        for chunk in chunks
    ]


def test_chunk_sections_returns_empty_for_empty_input() -> None:
    assert _call_chunker([]) == []


def test_chunk_sections_preserves_page_labels_across_multiple_sections() -> None:
    sections = [
        {"text": "First paragraph. " * 80, "page_label": "1", "source": "guide.pdf"},
        {"text": "Second page text. " * 60, "page_label": "2", "source": "guide.pdf"},
    ]

    chunks = _call_chunker(sections, max_chars=180, overlap=20)

    assert len(chunks) >= 3
    assert _page_label(chunks[0]) == "1"
    assert _page_label(chunks[-1]) == "2"

    page_labels = [_page_label(chunk) for chunk in chunks]
    assert page_labels.count("1") >= 1
    assert page_labels.count("2") >= 1
    assert page_labels == sorted(page_labels, key=lambda v: (v is None, v))


def test_chunk_sections_skips_blank_or_missing_text_sections() -> None:
    sections = [
        {"text": "   ", "page_label": "1", "source": "guide.pdf"},
        {"page_label": "1", "source": "guide.pdf"},
        {"text": None, "page_label": "1", "source": "guide.pdf"},
        {"text": "Useful content. " * 40, "page_label": "2", "source": "guide.pdf"},
    ]

    chunks = _call_chunker(sections, max_chars=160, overlap=20)

    assert chunks, "real content should still produce chunks"
    assert all(_normalize_ws(_text(chunk)) for chunk in chunks)
    assert all("Useful content." in _text(chunk) for chunk in chunks)
    assert {_page_label(chunk) for chunk in chunks} == {"2"}


def test_chunk_sections_never_emits_empty_text_chunks() -> None:
    sections = [{"text": "Alpha beta gamma. " * 60, "page_label": "4"}]

    chunks = _call_chunker(sections, max_chars=170, overlap=25)

    assert chunks
    assert all(_normalize_ws(_text(chunk)) for chunk in chunks)


def test_chunk_sections_respects_max_chars_with_reasonable_tolerance() -> None:
    text = " ".join(f"Sentence {i:03d}." for i in range(1, 120))
    max_chars = 180
    overlap = 35

    chunks = _call_chunker(
        [{"text": text, "page_label": "5", "source": "sizing.pdf"}],
        max_chars=max_chars,
        overlap=overlap,
    )

    assert len(chunks) >= 4
    # Small tolerance allows sentence-aware splitters to finish a clause cleanly.
    assert max(len(_text(chunk)) for chunk in chunks) <= max_chars + 40


def test_chunk_sections_applies_overlap_between_adjacent_chunks() -> None:
    text = " ".join(f"Clause {i:03d}." for i in range(1, 90))

    chunks = _call_chunker(
        [{"text": text, "page_label": "6", "source": "overlap.pdf"}],
        max_chars=140,
        overlap=40,
    )

    assert len(chunks) >= 2
    assert any(
        _boundary_overlap(_text(left), _text(right))
        for left, right in zip(chunks, chunks[1:])
    ), "expected at least one overlapping boundary when overlap > 0"


def test_chunk_sections_preserves_source_metadata() -> None:
    sections = [
        {
            "text": "Deployment notes. " * 50,
            "page_label": "7",
            "source": "handbook.pdf",
            "document_id": "doc-123",
        }
    ]

    chunks = _call_chunker(sections, max_chars=165, overlap=20)

    assert chunks
    assert all(_source(chunk) in {"handbook.pdf", "doc-123"} for chunk in chunks)


def test_chunk_sections_preserves_heading_context_in_text_or_metadata() -> None:
    sections = [
        {
            "heading": "Installation",
            "text": "Install the CLI. Configure the API key. Verify the login. " * 24,
            "page_label": "8",
            "source": "setup.docx",
        }
    ]

    chunks = _call_chunker(sections, max_chars=175, overlap=25)

    assert chunks
    first = chunks[0]
    heading = _heading(first)
    text = _text(first)
    assert heading == "Installation" or "Installation" in text


def test_chunk_sections_handles_long_unbroken_text_without_infinite_loop() -> None:
    sections = [
        {
            "text": "A" * 1200,
            "page_label": "9",
            "source": "ocr.txt",
        }
    ]

    chunks = _call_chunker(sections, max_chars=180, overlap=30)

    assert len(chunks) >= 2
    assert all(_text(chunk) for chunk in chunks)
    assert max(len(_text(chunk)) for chunk in chunks) <= 220


def test_chunk_sections_is_deterministic_for_identical_input() -> None:
    sections = [
        {
            "heading": "Overview",
            "text": "Consistent results matter. " * 60,
            "page_label": "10",
            "source": "determinism.pdf",
        }
    ]

    first = _call_chunker(sections, max_chars=170, overlap=20)
    second = _call_chunker(sections, max_chars=170, overlap=20)

    assert _snapshot(first) == _snapshot(second)
