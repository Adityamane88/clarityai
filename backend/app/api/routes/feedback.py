from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import ChatMessage
from app.db.session import get_db
from app.schemas.feedback import FeedbackRequest
from app.utils.serializers import serialize_message

router = APIRouter()


@router.post('/messages/{message_id}')
def store_feedback(message_id: str, payload: FeedbackRequest, db: Session = Depends(get_db)) -> dict:
    message = db.get(ChatMessage, message_id)
    if message is None:
        raise HTTPException(status_code=404, detail='Message not found')
    message.feedback_rating = payload.rating
    message.feedback_note = payload.note or None
    db.commit()
    db.refresh(message)
    return serialize_message(message)
