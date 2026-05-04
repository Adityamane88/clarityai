from __future__ import annotations

"""
Image search for ClarityAI.

Strategy (in order):
1. Tavily image results - if Tavily is configured (via TAVILY_API_KEY) we ask it
   to include images in the same call we're already paying for. This is the most
   reliable path.
2. DuckDuckGo image scraper - free, no key, but unofficial. Used when Tavily is
   not configured OR as a complement when Tavily returns fewer images than
   requested.

Both paths return a normalized list of `ImageResult` objects so the rest of the
app doesn't care which provider produced them.

Failure semantics: this module never raises in the happy path - if the provider
is unreachable, malformed, or returns no images, we just return [] so the
chat engine can degrade gracefully. The caller can choose to log the empty
result.
"""

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ImageResult:
    url: str               # the image itself
    thumbnail: str         # thumbnail (often same as url for DDG)
    source_url: str        # the page the image came from
    title: str             # human-friendly title / alt text
    width: int | None = None
    height: int | None = None
    domain: str | None = None
    provider: str = "ddg"  # "tavily" | "ddg"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_BANNED_DOMAIN_HINTS = (
    "porn", "xxx", "xvideos", "redtube", "pornhub", "xhamster", "onlyfans",
)
_BANNED_KEYWORD_HINTS = (
    "nude", "naked", "nsfw", "porn", "explicit",
)
_INVALID_IMAGE_EXT = (
    ".svg",  # often inline UI artifacts
)


def _is_banned(url: str, title: str) -> bool:
    text = f"{url} {title}".lower()
    if any(hint in text for hint in _BANNED_DOMAIN_HINTS):
        return True
    if any(hint in text for hint in _BANNED_KEYWORD_HINTS):
        return True
    return False


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _is_http(url: str) -> bool:
    return url.startswith(("http://", "https://"))


def _looks_like_image(url: str) -> bool:
    if not _is_http(url):
        return False
    if url.lower().endswith(_INVALID_IMAGE_EXT):
        return False
    return True


# ---------------------------------------------------------------------------
# DuckDuckGo image scraper (free, unofficial)
# ---------------------------------------------------------------------------


_DDG_VQD_RE = re.compile(r'vqd=["\']?([\d-]+)["\']?')

_DDG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://duckduckgo.com/",
}


async def _fetch_ddg_vqd(client: httpx.AsyncClient, query: str) -> str | None:
    """Hit the DDG HTML endpoint to extract the short-lived vqd token."""
    try:
        response = await client.get(
            "https://duckduckgo.com/",
            params={"q": query, "iax": "images", "ia": "images"},
            headers=_DDG_HEADERS,
        )
        response.raise_for_status()
        match = _DDG_VQD_RE.search(response.text)
        if match:
            return match.group(1)
    except Exception as exc:  # noqa: BLE001
        logger.debug("DDG vqd fetch failed: %s", exc)
    return None


async def _ddg_image_search(query: str, max_results: int) -> list[ImageResult]:
    """Free DuckDuckGo image search via the unofficial JSON endpoint."""
    timeout = httpx.Timeout(15.0, connect=8.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        vqd = await _fetch_ddg_vqd(client, query)
        if not vqd:
            return []

        params = {
            "l": "us-en",
            "o": "json",
            "q": query,
            "vqd": vqd,
            "f": ",,,,,",
            "p": "1",   # safe search on
            "v7exp": "a",
        }
        try:
            response = await client.get(
                "https://duckduckgo.com/i.js",
                params=params,
                headers=_DDG_HEADERS,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.debug("DDG image fetch failed: %s", exc)
            return []

    items = data.get("results") or []
    results: list[ImageResult] = []
    for item in items:
        url = (item.get("image") or "").strip()
        thumb = (item.get("thumbnail") or url).strip()
        source = (item.get("url") or "").strip()
        title = (item.get("title") or "").strip()

        if not _looks_like_image(url):
            continue
        if _is_banned(url, title) or _is_banned(source, title):
            continue

        results.append(
            ImageResult(
                url=url,
                thumbnail=thumb,
                source_url=source,
                title=title or _domain(source) or "Image",
                width=item.get("width"),
                height=item.get("height"),
                domain=_domain(source),
                provider="ddg",
            )
        )
        if len(results) >= max_results:
            break

    return results


# ---------------------------------------------------------------------------
# Tavily images (when available)
# ---------------------------------------------------------------------------


async def _tavily_image_search(query: str, max_results: int) -> list[ImageResult]:
    if not settings.web_research_configured:
        return []

    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": max(max_results, 5),
        "include_images": True,
        "include_image_descriptions": True,
        "search_depth": "basic",
    }
    timeout = httpx.Timeout(20.0, connect=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post("https://api.tavily.com/search", json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Tavily image fetch failed: %s", exc)
        return []

    raw_images = data.get("images") or []
    results: list[ImageResult] = []
    for entry in raw_images:
        # Tavily returns either a string (just the URL) or a dict with
        # `url` + `description`. Handle both.
        if isinstance(entry, str):
            url = entry.strip()
            title = ""
        elif isinstance(entry, dict):
            url = (entry.get("url") or "").strip()
            title = (entry.get("description") or "").strip()
        else:
            continue

        if not _looks_like_image(url):
            continue
        if _is_banned(url, title):
            continue

        results.append(
            ImageResult(
                url=url,
                thumbnail=url,
                source_url=url,
                title=title or _domain(url) or "Image",
                domain=_domain(url),
                provider="tavily",
            )
        )
        if len(results) >= max_results:
            break

    return results


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------


class ImageSearchClient:
    """High-level entry point used by the chat engine."""

    async def search(self, query: str, max_results: int = 6) -> list[ImageResult]:
        query = (query or "").strip()
        if not query:
            return []

        # Run both providers concurrently when Tavily is configured; otherwise
        # just hit DDG. We dedupe on URL afterwards.
        tasks: list[asyncio.Task] = []
        if settings.web_research_configured:
            tasks.append(asyncio.create_task(_tavily_image_search(query, max_results)))
        tasks.append(asyncio.create_task(_ddg_image_search(query, max_results)))

        gathered: list[list[ImageResult]] = []
        try:
            gathered = await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Image search failed: %s", exc)
            return []

        seen_urls: set[str] = set()
        merged: list[ImageResult] = []
        # Tavily first if present -> generally more curated
        for batch in gathered:
            for result in batch:
                key = result.url.split("?", 1)[0]
                if key in seen_urls:
                    continue
                seen_urls.add(key)
                merged.append(result)
                if len(merged) >= max_results:
                    return merged
        return merged


image_search = ImageSearchClient()
