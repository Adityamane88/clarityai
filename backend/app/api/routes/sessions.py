from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models import ChatSession
from app.db.session import get_db
from app.schemas.session import CreateSessionRequest
from app.utils.serializers import serialize_session

router = APIRouter()


@router.get('')
def list_sessions(db: Session = Depends(get_db)) -> list[dict]:
    sessions = db.execute(select(ChatSession).order_by(ChatSession.updated_at.desc())).scalars().all()
    return [serialize_session(session) for session in sessions]


@router.post('')
def create_session(payload: CreateSessionRequest | None = None, db: Session = Depends(get_db)) -> dict:
    title = payload.title if payload and payload.title else 'New conversation'
    session = ChatSession(title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return serialize_session(session)


@router.get('/{session_id}')
def get_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    session = db.execute(
        select(ChatSession)
        .where(ChatSession.id == session_id)
        .options(selectinload(ChatSession.messages))
    ).scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found')
    return serialize_session(session, include_messages=True)


@router.delete('/{session_id}')
def delete_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found')
    db.delete(session)
    db.commit()
    return {'status': 'deleted', 'session_id': session_id}
