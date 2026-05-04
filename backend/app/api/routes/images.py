from __future__ import annotations

"""
Standalone images endpoint.

Useful for:
- The "Show me images of X" button in the UI without going through the chat.
- Manual debugging.
"""

from fastapi import APIRouter, HTTPException, Query

from app.services.image_search import image_search

router = APIRouter()


@router.get("/search")
async def search_images(
    q: str = Query(..., min_length=2, description="Image search query"),
    limit: int = Query(6, ge=1, le=20),
) -> dict:
    try:
        results = await image_search.search(q, max_results=limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Image search failed: {exc}") from exc

    return {
        "query": q,
        "count": len(results),
        "results": [r.to_dict() for r in results],
    }
