from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from app.config import get_settings

settings = get_settings()


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
            'id': label,
            'label': label,
            'chunk_id': None,
            'document_id': f'web:{self.url}',
            'document_title': self.title,
            'source_name': self.domain or 'web',
            'page_label': None,
            'snippet': self.snippet,
            'content': self.content,
            'score': round(self.score, 4),
            'source_type': 'web',
            'url': self.url,
            'published_at': self.published_at,
        }


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix('www.')
    except Exception:
        return ''


class TavilyResearchClient:
    endpoint = 'https://api.tavily.com/search'

    @property
    def available(self) -> bool:
        return settings.web_research_configured

    def _score_domain(self, domain: str) -> float:
        if not domain:
            return 0.0
        trusted_rules = settings.research_trust_allowlist
        for rule in trusted_rules:
            rule = rule.lower().strip()
            if not rule:
                continue
            if domain == rule or domain.endswith(f'.{rule}'):
                return 0.12
        if domain.endswith('.gov') or domain.endswith('.edu'):
            return 0.12
        if any(term in domain for term in ('docs.', 'developer.', 'support.', 'help.')):
            return 0.06
        return 0.0

    async def search(self, query: str, max_results: int | None = None) -> list[ResearchSource]:
        if not self.available:
            raise WebResearchUnavailableError('Web research is not configured. Set TAVILY_API_KEY to enable it.')
        payload = {
            'api_key': settings.tavily_api_key,
            'query': query,
            'max_results': max_results or settings.research_max_results,
            'search_depth': 'advanced',
            'include_answer': False,
            'include_raw_content': False,
        }
        timeout = httpx.Timeout(35.0, connect=15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self.endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
        items = data.get('results') or []
        parsed: list[ResearchSource] = []
        for item in items:
            url = (item.get('url') or '').strip()
            domain = _domain(url)
            title = (item.get('title') or domain or 'Web source').strip()
            content = (item.get('content') or '').strip()
            snippet = content[:320].strip() if content else title
            score = float(item.get('score') or 0.0) + self._score_domain(domain)
            parsed.append(
                ResearchSource(
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=content or snippet,
                    domain=domain,
                    score=score,
                    published_at=item.get('published_date'),
                )
            )
        parsed.sort(key=lambda source: source.score, reverse=True)
        return parsed[: settings.research_max_results]


web_research = TavilyResearchClient()
