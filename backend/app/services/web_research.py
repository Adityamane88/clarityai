from __future__ import annotations

"""
Web research client for ClarityAI.

Improvements:
- Better domain trust scoring (allowlist + heuristic for official docs).
- Aggressive deduplication on URL and on near-duplicate snippets.
- Filters out content farms and very short snippets.
- Returns a clean ResearchSource that maps cleanly to a footnote citation.
"""

import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from app.config import get_settings

settings = get_settings()


# ---------------------------------------------------------------------------
# Errors and dataclass
# ---------------------------------------------------------------------------


class WebResearchUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class ResearchSource:
    title: str
    url: str
    snippet: str
    content: str
    domain: str
    score: float
    published_at: str | None = None

    def to_citation(self, label: str) -> dict:
        return {
            "id": label,
            "label": label,
            "display_name": self.title,
            "chunk_id": None,
            "document_id": f"web:{self.url}",
            "document_title": self.title,
            "source_name": self.domain or "web",
            "page_label": None,
            "snippet": self.snippet,
            "content": self.content,
            "score": round(self.score, 4),
            "score_band": "high" if self.score >= 0.7 else "medium" if self.score >= 0.35 else "low",
            "source_type": "web",
            "url": self.url,
            "published_at": self.published_at,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LOW_QUALITY_DOMAIN_HINTS = (
    "pinterest.", "quora.", "answers.com", "ehow.", "wikihow.",
    "tripadvisor.", "yelp.",
)


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Tavily client (default)
# ---------------------------------------------------------------------------


class TavilyResearchClient:
    endpoint = "https://api.tavily.com/search"

    @property
    def available(self) -> bool:
        return settings.web_research_configured

    def _domain_boost(self, domain: str) -> float:
        if not domain:
            return 0.0
        # User-supplied allowlist
        for rule in getattr(settings, "research_trust_allowlist", None) or []:
            rule = (rule or "").lower().strip()
            if rule and (domain == rule or domain.endswith(f".{rule}")):
                return 0.15
        # Strong defaults
        if domain.endswith(".gov") or domain.endswith(".edu"):
            return 0.15
        if any(seg in domain for seg in ("docs.", "developer.", "support.", "help.", "api.")):
            return 0.08
        if domain.endswith(("reuters.com", "apnews.com", "bbc.co.uk", "bbc.com", "nature.com", "science.org")):
            return 0.10
        # Penalize content-farm-ish domains
        for hint in _LOW_QUALITY_DOMAIN_HINTS:
            if hint in domain:
                return -0.08
        return 0.0

    async def search(self, query: str, max_results: int | None = None) -> list[ResearchSource]:
        if not self.available:
            raise WebResearchUnavailableError(
                "Web research is not configured. Set TAVILY_API_KEY to enable it."
            )

        # Slightly over-fetch then prune so dedupe + low-quality filtering doesn't
        # leave us short of useful sources.
        target = max_results or settings.research_max_results
        fetch_n = max(target * 2, target + 3)

        payload = {
            "api_key": settings.tavily_api_key,
            "query": query,
            "max_results": fetch_n,
            "search_depth": "advanced",
            "include_answer": False,
            "include_raw_content": False,
        }
        timeout = httpx.Timeout(35.0, connect=15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self.endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

        items = data.get("results") or []
        parsed: list[ResearchSource] = []
        seen_urls: set[str] = set()
        seen_snippets: set[str] = set()

        for item in items:
            url = (item.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            domain = _domain(url)
            title = (item.get("title") or domain or "Web source").strip()
            content = (item.get("content") or "").strip()
            if len(content) < 60 and not item.get("score"):
                # Skip near-empty snippets — they almost never help the model.
                continue

            snippet = (content[:340].strip() if content else title)[:340]
            snippet_key = _normalize_for_dedupe(snippet[:140])
            if snippet_key in seen_snippets:
                continue
            seen_snippets.add(snippet_key)

            base_score = float(item.get("score") or 0.0)
            score = max(0.0, base_score + self._domain_boost(domain))

            parsed.append(
                ResearchSource(
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=(content or snippet)[:1800],
                    domain=domain,
                    score=score,
                    published_at=item.get("published_date"),
                )
            )

        parsed.sort(key=lambda s: s.score, reverse=True)
        return parsed[:target]


web_research = TavilyResearchClient()