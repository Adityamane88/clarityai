from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatTurnRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(..., min_length=1, max_length=12000)
    mode: Literal['balanced', 'concise', 'deep'] = 'balanced'
    research_mode: Literal['auto', 'off', 'force'] = 'auto'
