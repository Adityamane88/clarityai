from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9._-]+', '-', name.strip())
    cleaned = cleaned.strip('.-')
    return cleaned or 'document'


@dataclass(slots=True)
class ExtractedDocument:
    title: str
    source_name: str
    mime_type: str
    raw_text: str
    sections: list[dict]


def guess_mime_type(path: Path, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or 'text/plain'


def compute_checksum(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _extract_pdf(path: Path, source_name: str, mime_type: str) -> ExtractedDocument:
    reader = PdfReader(str(path))
    sections: list[dict] = []
    all_parts: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or '').strip()
        except Exception:  # noqa: BLE001 - bad pages should not kill the upload
            text = ''
        if not text:
            continue
        sections.append({'text': text, 'page_label': str(page_number)})
        all_parts.append(f'Page {page_number}\n{text}')
    raw_text = '\n\n'.join(all_parts).strip()
    return ExtractedDocument(title=path.stem, source_name=source_name, mime_type=mime_type, raw_text=raw_text, sections=sections)


def _extract_csv(path: Path, source_name: str, mime_type: str) -> ExtractedDocument:
    rows: list[str] = []
    with path.open('r', encoding='utf-8', errors='ignore', newline='') as handle:
        reader = csv.reader(handle)
        for row in reader:
            rows.append(' | '.join(cell.strip() for cell in row))
    raw_text = '\n'.join(rows)
    return ExtractedDocument(
        title=path.stem,
        source_name=source_name,
        mime_type=mime_type,
        raw_text=raw_text,
        sections=[{'text': raw_text, 'page_label': None}],
    )


def _extract_json(path: Path, source_name: str, mime_type: str) -> ExtractedDocument:
    with path.open('r', encoding='utf-8', errors='ignore') as handle:
        data = json.load(handle)
    raw_text = json.dumps(data, indent=2, ensure_ascii=False)
    return ExtractedDocument(
        title=path.stem,
        source_name=source_name,
        mime_type=mime_type,
        raw_text=raw_text,
        sections=[{'text': raw_text, 'page_label': None}],
    )


def _extract_text_like(path: Path, source_name: str, mime_type: str) -> ExtractedDocument:
    raw_text = path.read_text(encoding='utf-8', errors='ignore')
    return ExtractedDocument(
        title=path.stem,
        source_name=source_name,
        mime_type=mime_type,
        raw_text=raw_text,
        sections=[{'text': raw_text, 'page_label': None}],
    )


def extract_document(path: Path, source_name: str, content_type: str | None = None) -> ExtractedDocument:
    mime_type = guess_mime_type(path, explicit=content_type)
    suffix = path.suffix.lower()
    if suffix == '.pdf':
        return _extract_pdf(path, source_name=source_name, mime_type=mime_type)
    if suffix == '.csv':
        return _extract_csv(path, source_name=source_name, mime_type=mime_type)
    if suffix == '.json':
        return _extract_json(path, source_name=source_name, mime_type=mime_type)
    return _extract_text_like(path, source_name=source_name, mime_type=mime_type)
