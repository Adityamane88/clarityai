from __future__ import annotations

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)
