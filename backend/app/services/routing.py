from __future__ import annotations

"""
Route selection for ClarityAI.

Decides whether a user turn should be answered from local knowledge, web
research, hybrid, or treated as casual chat. This version adds:
- Explicit intent buckets: chat / code / math / how-to / research / current-info.
- A "no_research_needed" fast path for math, code, and casual messages.
- Better handling when local knowledge is technically present but irrelevant.
"""

import re
from dataclasses import asdict, dataclass

from app.config import get_settings

settings = get_settings()


# ---------------------------------------------------------------------------
# Pattern banks
# ---------------------------------------------------------------------------

CURRENT_INFO_PATTERNS = [
    r"\btoday\b", r"\btonight\b", r"\byesterday\b",
    r"\blatest\b", r"\brecent\b", r"\bcurrent\b", r"\bright now\b",
    r"\bnews\b", r"\bupdate(?:s|d)?\b", r"\bbreaking\b",
    r"\bprice(?:s)?\b", r"\bstock\b", r"\bschedule\b",
    r"\bweather\b", r"\bscore(?:s)?\b", r"\bversion\b",
    r"\bstatus\b", r"\brelease(?:d)?\b", r"\blaunch(?:ed)?\b",
    r"\b(?:in|for|since)\s+202[0-9]\b", r"\b202[3-9]\b",
]

RESEARCH_INTENT_PATTERNS = [
    r"\bresearch\b", r"\binvestigate\b", r"\blook (?:it|that|this) up\b",
    r"\bcompare\b", r"\bvs\b", r"\bversus\b",
    r"\bpros and cons\b", r"\bbest option\b", r"\bbest (?:way|tool|library)\b",
    r"\bwhat should i (?:choose|pick|use)\b",
    r"\bfind sources\b", r"\bcite (?:sources|something)\b", r"\bverify\b",
    r"\bevidence\b", r"\bstudies\b", r"\bbenchmark(?:s)?\b",
]

CASUAL_CHAT_PATTERNS = [
    r"^\s*(?:hi|hello|hey|yo|sup|hiya|howdy)\b",
    r"\bhow are you\b", r"\bwho are you\b",
    r"\bwhat can you do\b", r"\bwhat are you\b",
    r"\bgood (?:morning|afternoon|evening|night)\b",
    r"^\s*thanks?\b", r"^\s*thank you\b", r"^\s*ok(?:ay)?\b",
    r"^\s*nice\b", r"^\s*cool\b",
]

# Code/math intent — these should never trigger web research by themselves.
CODE_INTENT_PATTERNS = [
    r"```", r"\bdebug\b", r"\bstack\s*trace\b", r"\btraceback\b",
    r"\bregex\b", r"\bfunction\b", r"\bclass\b", r"\bmethod\b",
    r"\bsyntax error\b", r"\bcompile\b", r"\brefactor\b",
    r"\bunit test\b",
    r"\b(?:python|javascript|typescript|java|c\+\+|rust|go|kotlin|swift)\b.*\b(?:code|function|script|error)\b",
]

MATH_INTENT_PATTERNS = [
    r"\bsolve\b.*=", r"\bderivative\b", r"\bintegral\b",
    r"\bequation\b", r"\bcalculate\b", r"^\s*\d+(\s*[\+\-\*/\^]\s*\d+)+",
    r"\bprobability\b", r"\bfactorial\b",
]


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RouteDecision:
    route: str
    needs_web_research: bool
    needs_local_knowledge: bool
    reason: str
    query_is_time_sensitive: bool
    intent: str = "general"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _classify_intent(normalized: str) -> str:
    if _matches_any(normalized, CASUAL_CHAT_PATTERNS) and len(normalized.split()) <= 8:
        return "chat"
    if _matches_any(normalized, CODE_INTENT_PATTERNS):
        return "code"
    if _matches_any(normalized, MATH_INTENT_PATTERNS):
        return "math"
    if _matches_any(normalized, CURRENT_INFO_PATTERNS):
        return "current_info"
    if _matches_any(normalized, RESEARCH_INTENT_PATTERNS):
        return "research"
    return "general"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def choose_route(
    user_message: str,
    local_confidence: float,
    local_hits: int,
    research_mode: str,
) -> RouteDecision:
    normalized = user_message.strip().lower()
    intent = _classify_intent(normalized)

    asks_current_info = intent == "current_info"
    explicit_research = intent == "research"
    casual = intent == "chat"
    code_or_math = intent in {"code", "math"}

    weak_local = local_hits == 0 or local_confidence < settings.auto_research_threshold
    strong_local = local_hits > 0 and local_confidence >= max(settings.auto_research_threshold, 0.28)

    # User-controlled overrides
    if research_mode == "off":
        return RouteDecision(
            route="local" if local_hits > 0 else "chat",
            needs_web_research=False,
            needs_local_knowledge=local_hits > 0,
            reason="research_disabled_by_user",
            query_is_time_sensitive=asks_current_info,
            intent=intent,
        )

    if research_mode == "force":
        return RouteDecision(
            route="hybrid" if local_hits > 0 else "research",
            needs_web_research=True,
            needs_local_knowledge=local_hits > 0,
            reason="research_forced_by_user",
            query_is_time_sensitive=asks_current_info,
            intent=intent,
        )

    # Casual chat short-circuit
    if casual:
        return RouteDecision(
            route="chat",
            needs_web_research=False,
            needs_local_knowledge=False,
            reason="casual_chat_no_research_needed",
            query_is_time_sensitive=False,
            intent=intent,
        )

    # Code/math is almost always answerable without research; only fall back if
    # the user explicitly asked for verification or current info.
    if code_or_math and not asks_current_info and not explicit_research:
        return RouteDecision(
            route="local" if strong_local else "chat",
            needs_web_research=False,
            needs_local_knowledge=local_hits > 0,
            reason=f"{intent}_intent_no_research_needed",
            query_is_time_sensitive=False,
            intent=intent,
        )

    # Time-sensitive
    if asks_current_info and strong_local:
        return RouteDecision(
            route="hybrid",
            needs_web_research=True,
            needs_local_knowledge=True,
            reason="time_sensitive_with_local_support",
            query_is_time_sensitive=True,
            intent=intent,
        )
    if asks_current_info and not strong_local:
        return RouteDecision(
            route="research",
            needs_web_research=True,
            needs_local_knowledge=local_hits > 0,
            reason="time_sensitive_prefers_research",
            query_is_time_sensitive=True,
            intent=intent,
        )

    # Explicit research
    if explicit_research and local_hits > 0:
        return RouteDecision(
            route="hybrid",
            needs_web_research=True,
            needs_local_knowledge=True,
            reason="explicit_research_with_local_context",
            query_is_time_sensitive=False,
            intent=intent,
        )
    if explicit_research and local_hits == 0:
        return RouteDecision(
            route="research",
            needs_web_research=True,
            needs_local_knowledge=False,
            reason="explicit_research_without_local_hits",
            query_is_time_sensitive=False,
            intent=intent,
        )

    # Default branches
    if strong_local:
        return RouteDecision(
            route="local",
            needs_web_research=False,
            needs_local_knowledge=True,
            reason="local_knowledge_sufficient",
            query_is_time_sensitive=False,
            intent=intent,
        )
    if weak_local and local_hits > 0:
        return RouteDecision(
            route="hybrid",
            needs_web_research=True,
            needs_local_knowledge=True,
            reason="local_present_but_not_strong_enough",
            query_is_time_sensitive=False,
            intent=intent,
        )

    # Nothing local, not casual, not code/math — fall back to research only if
    # the user seems to be asking something factual; otherwise just chat.
    is_factual = bool(re.search(r"\b(what|when|where|who|how|why|which)\b", normalized))
    if is_factual:
        return RouteDecision(
            route="research",
            needs_web_research=True,
            needs_local_knowledge=False,
            reason="no_local_match_factual_query",
            query_is_time_sensitive=False,
            intent=intent,
        )
    return RouteDecision(
        route="chat",
        needs_web_research=False,
        needs_local_knowledge=False,
        reason="no_local_match_conversational",
        query_is_time_sensitive=False,
        intent=intent,
    )