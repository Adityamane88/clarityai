from __future__ import annotations

from app.db.models import ChatMessage, ChatSession, KnowledgeDocument


def serialize_message(message: ChatMessage) -> dict:
    return {
        'id': message.id,
        'session_id': message.session_id,
        'role': message.role,
        'content': message.content,
        'citations': message.citations_json or [],
        # `images_json` may not exist on older DBs; fall back gracefully.
        'images': getattr(message, 'images_json', None) or [],
        'feedback_rating': message.feedback_rating,
        'feedback_note': message.feedback_note,
        'created_at': message.created_at.isoformat(),
    }


def serialize_session(session: ChatSession, include_messages: bool = False) -> dict:
    payload = {
        'id': session.id,
        'title': session.title,
        'summary': session.summary or '',
        'created_at': session.created_at.isoformat(),
        'updated_at': session.updated_at.isoformat(),
    }
    if include_messages:
        payload['messages'] = [serialize_message(message) for message in session.messages]
    return payload


def serialize_document(document: KnowledgeDocument) -> dict:
    return {
        'id': document.id,
        'title': document.title,
        'source_name': document.source_name,
        'mime_type': document.mime_type,
        'text_preview': document.text_preview,
        'chunk_count': len(document.chunks),
        'created_at': document.created_at.isoformat(),
        'updated_at': document.updated_at.isoformat(),
    }
