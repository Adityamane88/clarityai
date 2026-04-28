from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.config import get_settings
from app.db.models import KnowledgeChunk, KnowledgeDocument
from app.db.session import get_db
from app.services.chunker import chunk_sections
from app.services.documents import compute_checksum, extract_document, sanitize_filename
from app.services.retrieval import retrieval_index
from app.utils.serializers import serialize_document

settings = get_settings()
router = APIRouter()


@router.get('/documents')
def list_documents(db: Session = Depends(get_db)) -> list[dict]:
    documents = db.execute(
        select(KnowledgeDocument)
        .options(selectinload(KnowledgeDocument.chunks))
        .order_by(KnowledgeDocument.updated_at.desc())
    ).scalars().all()
    return [serialize_document(document) for document in documents]


@router.post('/upload')
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail='Empty upload')
    if len(raw) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail='File is too large for this starter configuration')
    checksum = compute_checksum(raw)
    existing = db.execute(
        select(KnowledgeDocument)
        .where(KnowledgeDocument.checksum == checksum)
        .options(selectinload(KnowledgeDocument.chunks))
    ).scalar_one_or_none()
    if existing is not None:
        return serialize_document(existing)
    original_name = file.filename or 'document'
    safe_name = sanitize_filename(original_name)
    target_name = f'{uuid4()}-{safe_name}'
    target_path = settings.upload_dir / target_name
    target_path.write_bytes(raw)
    extracted = extract_document(target_path, source_name=original_name, content_type=file.content_type)
    if len(extracted.raw_text.strip()) < 40:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail='Not enough readable text could be extracted from this file')
    chunks = chunk_sections(extracted.sections, max_chars=settings.chunk_size, overlap=settings.chunk_overlap)
    if not chunks:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail='Text extraction succeeded but no chunks were generated')
    document = KnowledgeDocument(
        title=extracted.title,
        source_name=original_name,
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
    db.commit()
    document = db.execute(
        select(KnowledgeDocument)
        .where(KnowledgeDocument.id == document.id)
        .options(selectinload(KnowledgeDocument.chunks))
    ).scalar_one()
    retrieval_index.rebuild(db)
    return serialize_document(document)


@router.get('/search')
def search_knowledge(q: str = Query(..., min_length=2), db: Session = Depends(get_db)) -> dict:
    results = retrieval_index.search(query=q)
    return {
        'query': q,
        'count': len(results['results']),
        'confidence': results['confidence'],
        'results': results['results'],
    }


@router.post('/reindex')
def reindex_knowledge(db: Session = Depends(get_db)) -> dict:
    return retrieval_index.rebuild(db)


@router.delete('/documents/{document_id}')
def delete_document(document_id: str, db: Session = Depends(get_db)) -> dict:
    document = db.get(KnowledgeDocument, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail='Document not found')
    path = Path(document.path)
    db.delete(document)
    db.commit()
    path.unlink(missing_ok=True)
    retrieval_index.rebuild(db)
    return {'status': 'deleted', 'document_id': document_id}
