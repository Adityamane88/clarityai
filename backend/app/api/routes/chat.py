from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.chat import ChatTurnRequest
from app.services.chat_engine import sse_event, stream_chat_reply

router = APIRouter()


@router.post('/stream')
async def stream_chat(payload: ChatTurnRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    async def event_generator():
        async for event in stream_chat_reply(
            db=db,
            session_id=payload.session_id,
            user_message=payload.message,
            mode=payload.mode,
            research_mode=payload.research_mode,
        ):
            event_type = event.pop('type')
            yield sse_event(event_type, event).encode('utf-8')

    headers = {
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
    }
    return StreamingResponse(event_generator(), media_type='text/event-stream', headers=headers)
