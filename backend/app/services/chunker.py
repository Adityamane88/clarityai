from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class ChunkRecord:
    content: str
    page_label: str | None
    meta: dict


def normalize_whitespace(text: str) -> str:
    text = text.replace('\u00a0', ' ')
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def split_large_block(text: str, max_chars: int, overlap: int) -> list[str]:
    sentences = [segment.strip() for segment in re.split(r'(?<=[.!?])\s+', text) if segment.strip()]
    if not sentences:
        return []
    pieces: list[str] = []
    buffer = ''
    step_overlap = max(0, overlap)
    for sentence in sentences:
        if len(sentence) > max_chars:
            if buffer:
                pieces.append(buffer.strip())
                buffer = ''
            start = 0
            step = max(1, max_chars - step_overlap)
            while start < len(sentence):
                pieces.append(sentence[start:start + max_chars].strip())
                start += step
            continue
        candidate = sentence if not buffer else f'{buffer} {sentence}'.strip()
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        pieces.append(buffer.strip())
        carry = buffer[-step_overlap:].strip() if step_overlap else ''
        buffer = f'{carry} {sentence}'.strip() if carry else sentence
    if buffer:
        pieces.append(buffer.strip())
    return [piece for piece in pieces if piece]


def chunk_sections(sections: list[dict], max_chars: int, overlap: int) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for index, section in enumerate(sections):
        page_label = section.get('page_label')
        text = normalize_whitespace(section.get('text', ''))
        if not text:
            continue
        paragraphs = [part.strip() for part in re.split(r'\n\n+', text) if part.strip()]
        buffer = ''
        for paragraph in paragraphs:
            if len(paragraph) > max_chars:
                if buffer:
                    chunks.append(ChunkRecord(content=buffer.strip(), page_label=page_label, meta={'section_index': index}))
                    buffer = ''
                for piece in split_large_block(paragraph, max_chars=max_chars, overlap=overlap):
                    chunks.append(ChunkRecord(content=piece, page_label=page_label, meta={'section_index': index}))
                continue
            candidate = paragraph if not buffer else f'{buffer}\n\n{paragraph}'
            if len(candidate) <= max_chars:
                buffer = candidate
                continue
            chunks.append(ChunkRecord(content=buffer.strip(), page_label=page_label, meta={'section_index': index}))
            carry = buffer[-overlap:].strip() if overlap else ''
            buffer = f'{carry}\n\n{paragraph}'.strip() if carry else paragraph
        if buffer:
            chunks.append(ChunkRecord(content=buffer.strip(), page_label=page_label, meta={'section_index': index}))
    return chunks
