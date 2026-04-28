from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from app.config import get_settings

settings = get_settings()

CURRENT_INFO_PATTERNS = [
    r'\btoday\b',
    r'\blatest\b',
    r'\brecent\b',
    r'\bcurrent\b',
    r'\bnews\b',
    r'\bupdate\b',
    r'\bprice\b',
    r'\bschedule\b',
    r'\bweather\b',
    r'\bscore\b',
    r'\bversion\b',
    r'\bstatus\b',
]

RESEARCH_INTENT_PATTERNS = [
    r'\bresearch\b',
    r'\binvestigate\b',
    r'\bcompare\b',
    r'\bpros and cons\b',
    r'\bbest option\b',
    r'\bwhat should i choose\b',
    r'\bfind sources\b',
    r'\bcite\b',
]


@dataclass(slots=True)
class RouteDecision:
    route: str
    needs_web_research: bool
    needs_local_knowledge: bool
    reason: str
    query_is_time_sensitive: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def choose_route(user_message: str, local_confidence: float, local_hits: int, research_mode: str) -> RouteDecision:
    normalized = user_message.strip().lower()
    asks_current_info = _matches_any(normalized, CURRENT_INFO_PATTERNS)
    explicit_research = _matches_any(normalized, RESEARCH_INTENT_PATTERNS)
    weak_local = local_hits == 0 or local_confidence < settings.auto_research_threshold

    if research_mode == 'off':
        return RouteDecision(
            route='local',
            needs_web_research=False,
            needs_local_knowledge=True,
            reason='research_disabled_by_user',
            query_is_time_sensitive=asks_current_info,
        )

    if research_mode == 'force':
        route = 'hybrid' if local_hits > 0 else 'research'
        return RouteDecision(
            route=route,
            needs_web_research=True,
            needs_local_knowledge=local_hits > 0,
            reason='research_forced_by_user',
            query_is_time_sensitive=asks_current_info,
        )

    if asks_current_info and local_hits > 0:
        return RouteDecision(
            route='hybrid',
            needs_web_research=True,
            needs_local_knowledge=True,
            reason='time_sensitive_query_with_local_context',
            query_is_time_sensitive=True,
        )

    if asks_current_info and local_hits == 0:
        return RouteDecision(
            route='research',
            needs_web_research=True,
            needs_local_knowledge=False,
            reason='time_sensitive_query_needs_research',
            query_is_time_sensitive=True,
        )

    if explicit_research and local_hits > 0:
        return RouteDecision(
            route='hybrid',
            needs_web_research=True,
            needs_local_knowledge=True,
            reason='explicit_research_request',
            query_is_time_sensitive=asks_current_info,
        )

    if explicit_research and local_hits == 0:
        return RouteDecision(
            route='research',
            needs_web_research=True,
            needs_local_knowledge=False,
            reason='explicit_research_request',
            query_is_time_sensitive=asks_current_info,
        )

    if weak_local and local_hits > 0:
        return RouteDecision(
            route='hybrid',
            needs_web_research=True,
            needs_local_knowledge=True,
            reason='local_confidence_low',
            query_is_time_sensitive=asks_current_info,
        )

    if weak_local and local_hits == 0:
        return RouteDecision(
            route='research',
            needs_web_research=True,
            needs_local_knowledge=False,
            reason='no_local_match',
            query_is_time_sensitive=asks_current_info,
        )

    return RouteDecision(
        route='local',
        needs_web_research=False,
        needs_local_knowledge=True,
        reason='local_knowledge_sufficient',
        query_is_time_sensitive=asks_current_info,
    )
