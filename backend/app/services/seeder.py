from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import KnowledgeChunk, KnowledgeDocument
from app.services.chunker import chunk_sections
from app.services.documents import compute_checksum, extract_document, sanitize_filename

logger = logging.getLogger(__name__)
settings = get_settings()


def seed_sample_knowledge(db: Session) -> int:
    """Seed the knowledge base with sample files on first run.

    Looks for files in <project_root>/sample_knowledge and ingests them
    if the database has no documents yet. Returns the number of documents seeded.
    Safe to call on every startup — does nothing if knowledge already exists.
    """
    existing_count = db.execute(select(KnowledgeDocument)).scalars().first()
    if existing_count is not None:
        return 0

    # The sample_knowledge folder lives at the project root, two levels up from app/services
    project_root = Path(__file__).resolve().parents[3]
    sample_dir = project_root / 'sample_knowledge'
    if not sample_dir.is_dir():
        logger.info('No sample_knowledge folder found at %s; skipping seed.', sample_dir)
        return 0

    seeded = 0
    for source_path in sorted(sample_dir.iterdir()):
        if not source_path.is_file():
            continue
        if source_path.name.startswith('.'):
            continue
        try:
            raw = source_path.read_bytes()
            if not raw or len(raw) > settings.max_upload_size_mb * 1024 * 1024:
                continue
            checksum = compute_checksum(raw)
            already = db.execute(
                select(KnowledgeDocument).where(KnowledgeDocument.checksum == checksum)
            ).scalar_one_or_none()
            if already is not None:
                continue
            safe_name = sanitize_filename(source_path.name)
            target_path = settings.upload_dir / f'{uuid4()}-{safe_name}'
            target_path.write_bytes(raw)
            extracted = extract_document(target_path, source_name=source_path.name)
            if len(extracted.raw_text.strip()) < 40:
                target_path.unlink(missing_ok=True)
                continue
            chunks = chunk_sections(
                extracted.sections,
                max_chars=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
            if not chunks:
                target_path.unlink(missing_ok=True)
                continue
            document = KnowledgeDocument(
                title=extracted.title,
                source_name=source_path.name,
                mime_type=extracted.mime_type,
                path=str(target_path),
                checksum=checksum,
                text_preview=extracted.raw_text[:320].strip(),
            )
            db.add(document)
            db.flush()
            for index, chunk in enumerate(chunks):
                db.add(
                    KnowledgeChunk(
                        document_id=document.id,
                        chunk_index=index,
                        content=chunk.content,
                        page_label=chunk.page_label,
                        meta_json=chunk.meta,
                    )
                )
            seeded += 1
            logger.info('Seeded sample document: %s (%d chunks)', source_path.name, len(chunks))
        except Exception as exc:  # noqa: BLE001
            logger.warning('Failed to seed %s: %s', source_path.name, exc)
            continue

    if seeded:
        db.commit()
        logger.info('Sample knowledge seeded: %d document(s).', seeded)
    return seeded
