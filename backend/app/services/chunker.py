from __future__ import annotations

"""
Heading-aware semantic chunker for ClarityAI.

Goals:
- Split on natural boundaries (headings, blank lines, sentences) rather than
  blind character windows.
- Carry a small overlap so a sentence on a chunk boundary is still findable.
- Track page labels coming in from the PDF extractor so retrieval results can
  cite the right page.
- Avoid producing tiny dribble chunks (< ~80 chars) that have no real content.
"""

import re
from dataclasses import dataclass


@dataclass(slots=True)
class ChunkRecord:
    content: str
    page_label: str | None
    meta: dict


# Markdown / RST-ish heading detector
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6}\s+\S|[A-Z][A-Z0-9 _\-]{3,}$)", re.MULTILINE)


def normalize_whitespace(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Sentence-level splitter (small + dependency free)
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    # Conservative regex: split on sentence-ending punctuation followed by space + capital,
    # but keep the punctuation attached.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\d])", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Splitter that respects max_chars + overlap
# ---------------------------------------------------------------------------


def split_large_block(text: str, max_chars: int, overlap: int) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    pieces: list[str] = []
    buffer = ""
    step_overlap = max(0, overlap)

    for sentence in sentences:
        if len(sentence) > max_chars:
            # The sentence itself is bigger than the chunk size — hard split.
            if buffer:
                pieces.append(buffer.strip())
                buffer = ""
            start = 0
            step = max(1, max_chars - step_overlap)
            while start < len(sentence):
                pieces.append(sentence[start : start + max_chars].strip())
                start += step
            continue

        candidate = sentence if not buffer else f"{buffer} {sentence}".strip()
        if len(candidate) <= max_chars:
            buffer = candidate
            continue

        # Flush current buffer with sentence-aware overlap
        pieces.append(buffer.strip())
        carry = buffer[-step_overlap:].strip() if step_overlap else ""
        buffer = f"{carry} {sentence}".strip() if carry else sentence

    if buffer:
        pieces.append(buffer.strip())
    return [p for p in pieces if p]


# ---------------------------------------------------------------------------
# Heading-aware section splitter
# ---------------------------------------------------------------------------


def _split_by_headings(text: str) -> list[str]:
    """If the text has obvious headings, split there. Otherwise return [text]."""
    if not _HEADING_RE.search(text):
        return [text]
    # Find heading positions and slice between them
    positions = [m.start() for m in _HEADING_RE.finditer(text)]
    if not positions:
        return [text]
    if positions[0] != 0:
        positions = [0, *positions]
    pieces: list[str] = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            pieces.append(chunk)
    return pieces


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_sections(sections: list[dict], max_chars: int, overlap: int) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for index, section in enumerate(sections):
        page_label = section.get("page_label")
        text = normalize_whitespace(section.get("text", ""))
        if not text:
            continue

        # First, prefer heading-based splitting; fall back to paragraph splits.
        macro_blocks = _split_by_headings(text)
        # Then split each macro block on blank lines (paragraphs).
        paragraphs: list[str] = []
        for block in macro_blocks:
            paragraphs.extend(p.strip() for p in re.split(r"\n\n+", block) if p.strip())

        buffer = ""
        for paragraph in paragraphs:
            if len(paragraph) > max_chars:
                if buffer:
                    chunks.append(
                        ChunkRecord(
                            content=buffer.strip(),
                            page_label=page_label,
                            meta={"section_index": index},
                        )
                    )
                    buffer = ""
                for piece in split_large_block(paragraph, max_chars=max_chars, overlap=overlap):
                    chunks.append(
                        ChunkRecord(
                            content=piece,
                            page_label=page_label,
                            meta={"section_index": index},
                        )
                    )
                continue

            candidate = paragraph if not buffer else f"{buffer}\n\n{paragraph}"
            if len(candidate) <= max_chars:
                buffer = candidate
                continue

            chunks.append(
                ChunkRecord(
                    content=buffer.strip(),
                    page_label=page_label,
                    meta={"section_index": index},
                )
            )
            carry = buffer[-overlap:].strip() if overlap else ""
            buffer = f"{carry}\n\n{paragraph}".strip() if carry else paragraph

        if buffer:
            chunks.append(
                ChunkRecord(
                    content=buffer.strip(),
                    page_label=page_label,
                    meta={"section_index": index},
                )
            )

    # Drop the dribble — chunks too short to ever be informative.
    return [c for c in chunks if len(c.content) >= 40]