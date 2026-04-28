from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    rating: Literal['up', 'down']
    note: str | None = Field(default=None, max_length=1000)
