from __future__ import annotations

"""
Route selection for ClarityAI - Elite version.

What's new vs the previous version:
- New intents: `image`, `self_identity`, `creative`.
- `image` intent triggers image search even on the "off" research mode (image
  search and web research are decoupled).
- `self_identity` short-circuits research entirely - the assistant answers from
  its own description; we never burn web calls on "who are you".
- Keeps the previous decision shape so the rest of the app keeps working.
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
    r"\bwho\s+is\s+the\s+(?:current\s+)?(?:ceo|president|prime\s+minister|leader|king|queen)\b",
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
    r"\bhow are you\b",
    r"\bgood (?:morning|afternoon|evening|night)\b",
    r"^\s*thanks?\b", r"^\s*thank you\b", r"^\s*ok(?:ay)?\b",
    r"^\s*nice\b", r"^\s*cool\b",
]

# Self-identity questions - "who are you", "what are you", "your name", etc.
# These never need web research; ClarityAI should answer from its own system
# prompt directly.
SELF_IDENTITY_PATTERNS = [
    r"\bwho\s+are\s+you\b",
    r"\bwhat\s+are\s+you\b",
    r"\bwhat\s+(?:can|do)\s+you\s+do\b",
    r"\bwhat\s+is\s+your\s+name\b",
    r"\btell\s+me\s+about\s+yourself\b",
    r"\bare\s+you\s+(?:gpt|chatgpt|claude|gemini|llama|an\s+ai|a\s+human|a\s+bot)\b",
    r"\bwhat\s+(?:llm|model|ai)\s+(?:are|is)\s+(?:you|this)\b",
    r"\bdifference\s+between\s+(?:you|this)\s+and\s+(?:claude|chatgpt|gpt|gemini|copilot)\b",
    r"\bare\s+you\s+made\s+by\b",
    r"\bwho\s+(?:made|built|created|trained)\s+you\b",
]

# Code/math intent - these should never trigger web research by themselves.
CODE_INTENT_PATTERNS = [
    r"```", r"\bdebug\b", r"\bstack\s*trace\b", r"\btraceback\b",
    r"\bregex\b", r"\bfunction\b", r"\bclass\b", r"\bmethod\b",
    r"\bsyntax error\b", r"\bcompile\b", r"\brefactor\b",
    r"\bunit test\b", r"\bnull\s*pointer\b",
    r"\b(?:python|javascript|typescript|java|c\+\+|rust|go(?:lang)?|kotlin|swift|ruby|php|sql|bash|shell)\b"
    r".*\b(?:code|function|script|error|snippet|example|how)\b",
    r"\bwrite\s+(?:a|some)?\s*(?:code|function|script|program|class|module)\b",
]

MATH_INTENT_PATTERNS = [
    r"\bsolve\b.*=", r"\bderivative\b", r"\bintegral\b",
    r"\bequation\b", r"\bcalculate\b", r"^\s*\d+(\s*[\+\-\*/\^]\s*\d+)+",
    r"\bprobability\b", r"\bfactorial\b",
]

# Image / visual content intent.
IMAGE_INTENT_PATTERNS = [
    r"\b(?:show|find|get|fetch|give|grab)\s+(?:me\s+)?(?:some\s+|a\s+few\s+|several\s+)?(?:images?|pictures?|photos?|pics?|visuals?)\b",
    r"\b(?:images?|pictures?|photos?|pics?)\s+(?:of|about|for|on|showing)\b",
    r"\bwhat\s+(?:does|do|did)\s+(?:the\s+|a\s+)?[\w\s']+\s+look\s+like\b",
    r"\bvisuals?\s+of\b",
    r"\bshow\s+me\s+how\s+\w+\s+looks?\b",
    r"\bcan\s+you\s+show\s+me\b.*\b(?:images?|pictures?|photos?|pics?)\b",
    r"\bphotos?\s+(?:of|from|showing)\b",
]

# Creative writing intent - we don't research creative tasks.
CREATIVE_INTENT_PATTERNS = [
    r"\bwrite\s+(?:me\s+)?a\s+(?:poem|story|song|haiku|limerick|essay|tweet|email|letter)\b",
    r"\bdraft\s+(?:me\s+)?a\s+(?:poem|story|song|essay|tweet|email|letter)\b",
    r"\bcompose\s+(?:me\s+)?a\b",
    r"\bbrainstorm\b",
    r"\bcome\s+up\s+with\b",
]


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RouteDecision:
    route: str
    needs_web_research: bool
    needs_local_knowledge: bool
    needs_image_search: bool
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
    # Order matters: a question that *contains* the word "show" but is asking
    # about identity ("can you show me what you can do") should still route to
    # self_identity. So we check identity first.
    if _matches_any(normalized, SELF_IDENTITY_PATTERNS):
        return "self_identity"
    if _matches_any(normalized, IMAGE_INTENT_PATTERNS):
        return "image"
    if _matches_any(normalized, CASUAL_CHAT_PATTERNS) and len(normalized.split()) <= 8:
        return "chat"
    if _matches_any(normalized, CODE_INTENT_PATTERNS):
        return "code"
    if _matches_any(normalized, MATH_INTENT_PATTERNS):
        return "math"
    if _matches_any(normalized, CREATIVE_INTENT_PATTERNS):
        return "creative"
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
    creative = intent == "creative"
    self_id = intent == "self_identity"
    image_intent = intent == "image"

    weak_local = local_hits == 0 or local_confidence < settings.auto_research_threshold
    strong_local = local_hits > 0 and local_confidence >= max(settings.auto_research_threshold, 0.28)

    # Image searches are independent of the research mode. Even in "off" mode,
    # if the user explicitly asked for images we should still try to fetch them.
    needs_images = image_intent

    # ------------------------------------------------------------------
    # User-controlled overrides
    # ------------------------------------------------------------------
    if research_mode == "off":
        return RouteDecision(
            route="local" if local_hits > 0 else "chat",
            needs_web_research=False,
            needs_local_knowledge=local_hits > 0,
            needs_image_search=needs_images,
            reason="research_disabled_by_user",
            query_is_time_sensitive=asks_current_info,
            intent=intent,
        )

    if research_mode == "force":
        return RouteDecision(
            route="hybrid" if local_hits > 0 else "research",
            needs_web_research=True,
            needs_local_knowledge=local_hits > 0,
            needs_image_search=needs_images,
            reason="research_forced_by_user",
            query_is_time_sensitive=asks_current_info,
            intent=intent,
        )

    # ------------------------------------------------------------------
    # Self-identity short-circuit
    # ------------------------------------------------------------------
    if self_id:
        return RouteDecision(
            route="chat",
            needs_web_research=False,
            needs_local_knowledge=False,
            needs_image_search=False,
            reason="self_identity_no_research_needed",
            query_is_time_sensitive=False,
            intent=intent,
        )

    # ------------------------------------------------------------------
    # Casual chat short-circuit
    # ------------------------------------------------------------------
    if casual:
        return RouteDecision(
            route="chat",
            needs_web_research=False,
            needs_local_knowledge=False,
            needs_image_search=needs_images,
            reason="casual_chat_no_research_needed",
            query_is_time_sensitive=False,
            intent=intent,
        )

    # ------------------------------------------------------------------
    # Image-first queries: fetch images, but also research lightly so the
    # accompanying text isn't wrong.
    # ------------------------------------------------------------------
    if image_intent:
        return RouteDecision(
            route="research" if not strong_local else "hybrid",
            needs_web_research=True,
            needs_local_knowledge=local_hits > 0,
            needs_image_search=True,
            reason="image_request_with_supporting_research",
            query_is_time_sensitive=False,
            intent=intent,
        )

    # ------------------------------------------------------------------
    # Creative / Code / Math: answer without research unless asked otherwise.
    # ------------------------------------------------------------------
    if creative:
        return RouteDecision(
            route="chat",
            needs_web_research=False,
            needs_local_knowledge=local_hits > 0,
            needs_image_search=False,
            reason="creative_intent_no_research_needed",
            query_is_time_sensitive=False,
            intent=intent,
        )

    if code_or_math and not asks_current_info and not explicit_research:
        return RouteDecision(
            route="local" if strong_local else "chat",
            needs_web_research=False,
            needs_local_knowledge=local_hits > 0,
            needs_image_search=False,
            reason=f"{intent}_intent_no_research_needed",
            query_is_time_sensitive=False,
            intent=intent,
        )

    # ------------------------------------------------------------------
    # Time-sensitive
    # ------------------------------------------------------------------
    if asks_current_info and strong_local:
        return RouteDecision(
            route="hybrid",
            needs_web_research=True,
            needs_local_knowledge=True,
            needs_image_search=needs_images,
            reason="time_sensitive_with_local_support",
            query_is_time_sensitive=True,
            intent=intent,
        )
    if asks_current_info:
        return RouteDecision(
            route="research",
            needs_web_research=True,
            needs_local_knowledge=local_hits > 0,
            needs_image_search=needs_images,
            reason="time_sensitive_prefers_research",
            query_is_time_sensitive=True,
            intent=intent,
        )

    # ------------------------------------------------------------------
    # Explicit research
    # ------------------------------------------------------------------
    if explicit_research and local_hits > 0:
        return RouteDecision(
            route="hybrid",
            needs_web_research=True,
            needs_local_knowledge=True,
            needs_image_search=needs_images,
            reason="explicit_research_with_local_context",
            query_is_time_sensitive=False,
            intent=intent,
        )
    if explicit_research:
        return RouteDecision(
            route="research",
            needs_web_research=True,
            needs_local_knowledge=False,
            needs_image_search=needs_images,
            reason="explicit_research_without_local_hits",
            query_is_time_sensitive=False,
            intent=intent,
        )

    # ------------------------------------------------------------------
    # Default branches
    # ------------------------------------------------------------------
    if strong_local:
        return RouteDecision(
            route="local",
            needs_web_research=False,
            needs_local_knowledge=True,
            needs_image_search=needs_images,
            reason="local_knowledge_sufficient",
            query_is_time_sensitive=False,
            intent=intent,
        )
    if weak_local and local_hits > 0:
        return RouteDecision(
            route="hybrid",
            needs_web_research=True,
            needs_local_knowledge=True,
            needs_image_search=needs_images,
            reason="local_present_but_not_strong_enough",
            query_is_time_sensitive=False,
            intent=intent,
        )

    # Nothing local, not casual/code/math/creative - fall back to research
    # only if the user seems to be asking something factual; otherwise just chat.
    is_factual = bool(re.search(r"\b(what|when|where|who|how|why|which)\b", normalized))
    if is_factual:
        return RouteDecision(
            route="research",
            needs_web_research=True,
            needs_local_knowledge=False,
            needs_image_search=needs_images,
            reason="no_local_match_factual_query",
            query_is_time_sensitive=False,
            intent=intent,
        )
    return RouteDecision(
        route="chat",
        needs_web_research=False,
        needs_local_knowledge=False,
        needs_image_search=needs_images,
        reason="no_local_match_conversational",
        query_is_time_sensitive=False,
        intent=intent,
    )
