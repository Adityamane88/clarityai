from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChatSession(Base):
    __tablename__ = 'chat_sessions'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    title: Mapped[str] = mapped_column(String(200), default='New conversation')
    summary: Mapped[str] = mapped_column(Text, default='')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    messages: Mapped[list['ChatMessage']] = relationship(
        back_populates='session',
        cascade='all, delete-orphan',
        order_by='ChatMessage.created_at',
    )


class ChatMessage(Base):
    __tablename__ = 'chat_messages'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    session_id: Mapped[str] = mapped_column(ForeignKey('chat_sessions.id', ondelete='CASCADE'), index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    citations_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    # Elite addition: persist any images attached to the assistant turn so
    # they survive a page refresh / session reload.
    images_json: Mapped[list[dict]] = mapped_column(JSON, default=list, nullable=True)
    feedback_rating: Mapped[str | None] = mapped_column(String(12), nullable=True)
    feedback_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    session: Mapped[ChatSession] = relationship(back_populates='messages')


class KnowledgeDocument(Base):
    __tablename__ = 'knowledge_documents'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    title: Mapped[str] = mapped_column(String(240))
    source_name: Mapped[str] = mapped_column(String(240))
    mime_type: Mapped[str] = mapped_column(String(120), default='text/plain')
    path: Mapped[str] = mapped_column(String(500))
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    text_preview: Mapped[str] = mapped_column(Text, default='')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    chunks: Mapped[list['KnowledgeChunk']] = relationship(
        back_populates='document',
        cascade='all, delete-orphan',
        order_by='KnowledgeChunk.chunk_index',
    )


class KnowledgeChunk(Base):
    __tablename__ = 'knowledge_chunks'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey('knowledge_documents.id', ondelete='CASCADE'), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, index=True)
    content: Mapped[str] = mapped_column(Text)
    page_label: Mapped[str | None] = mapped_column(String(40), nullable=True)
    meta_json: Mapped[dict] = mapped_column(JSON, default=dict)
    document: Mapped[KnowledgeDocument] = relationship(back_populates='chunks')
