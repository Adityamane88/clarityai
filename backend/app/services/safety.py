from __future__ import annotations

"""
Lightweight safety classifier for ClarityAI.

Goals:
- Block prompt-injection attempts and direct, actionable harm requests.
- Detect crisis signals (self-harm, suicidal ideation, harm-to-others) and
  respond with a calm, human, non-clinical message that points to real help.
- For "medium-severity" emotional language, *don't* block — just adjust tone.
- Avoid false positives on benign words ("I'm dying for coffee").

This module returns a SafetyDecision; the chat engine decides what to do with it.
"""

import re
from dataclasses import asdict, dataclass


# ---------------------------------------------------------------------------
# Pattern banks
# ---------------------------------------------------------------------------

# Direct, unambiguous self-harm / suicidal ideation phrasing.
SELF_HARM_PATTERNS = [
    r"\bkill(?:ing)?\s+myself\b",
    r"\bend(?:ing)?\s+(?:my|this)\s+life\b",
    r"\bend\s+it\s+all\b",
    r"\btake\s+my\s+(?:own\s+)?life\b",
    r"\bcommit\s+suicide\b",
    r"\bi\s+want\s+to\s+die\b",
    r"\bi\s+wanna\s+die\b",
    r"\bi\s+(?:can'?t|cannot)\s+go\s+on\b",
    r"\bsuicid(?:e|al)\b",
    r"\b(?:cut|cutting)\s+myself\b",
    r"\bhurt(?:ing)?\s+myself\b",
    r"\boverdos(?:e|ing)\b",
    r"\bself[\s-]?harm\b",
    r"\bnot\s+safe\s+to\s+be\s+(?:home|alone)\b",
]

# Harm-to-others (kept narrow on purpose).
HARM_OTHERS_PATTERNS = [
    r"\bkill(?:ing)?\s+(?:someone|him|her|them|my\s+\w+)\b",
    r"\bhurt(?:ing)?\s+(?:someone|him|her|them|my\s+\w+)\b",
    r"\bshoot\s+up\b",
    r"\bmake\s+(?:a\s+)?(?:bomb|explosive|pipe\s*bomb)\b",
    r"\bhow\s+to\s+(?:make|build)\s+(?:a\s+)?(?:bomb|nerve\s*agent|biological\s*weapon)\b",
    r"\b(?:poison|attack)\s+(?:my|someone'?s)\s+(?:water|food)\b",
]

# Medium-severity emotional language — DO NOT block, just shift tone.
MEDIUM_RISK_PATTERNS = [
    r"\bhopeless\b",
    r"\bnothing\s+matters\b",
    r"\bpanic\s+attack\b",
    r"\bwant\s+to\s+disappear\b",
    r"\b(?:can'?t|cannot)\s+cope\b",
    r"\boverwhelm(?:ed|ing)\b",
    r"\banxious\b", r"\banxiety\b",
    r"\bburned?\s+out\b", r"\bburn[\s-]?out\b",
    r"\bdepress(?:ed|ion)\b",
    r"\bgrieving\b", r"\bgrief\b",
    r"\bexhausted\b", r"\bspiraling\b",
    r"\bisolated\b", r"\blonely\b",
]

# Prompt-injection / jailbreak attempts.
JAILBREAK_PATTERNS = [
    r"ignore\s+(?:all\s+)?previous\s+instructions",
    r"ignore\s+the\s+system\s+prompt",
    r"reveal\s+(?:your|the)\s+system\s+prompt",
    r"print\s+(?:your|the)\s+system\s+prompt",
    r"bypass\s+(?:your\s+)?safety",
    r"disable\s+(?:your\s+)?safety",
    r"pretend\s+you\s+have\s+no\s+rules",
    r"act\s+as\s+(?:dan|jailbroken)",
    r"developer\s+mode",
    r"you\s+are\s+now\s+(?:dan|free|unrestricted)",
]


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SafetyDecision:
    severity: str            # "low" | "medium" | "high"
    blocked: bool
    message: str | None
    reason: str | None = None
    category: str | None = None  # "self_harm" | "harm_others" | "prompt_injection" | None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Canonical responses (calm, human, not clinical)
# ---------------------------------------------------------------------------


SELF_HARM_MESSAGE = (
    "I'm really glad you said something. What you're describing sounds heavy, and I want to make "
    "sure you're safe right now.\n\n"
    "If you're in immediate danger, please reach out to someone who can be with you tonight — a "
    "person you trust or local emergency services. A few lines you can use any time:\n\n"
    "- **US & Canada:** call or text **988** (Suicide & Crisis Lifeline)\n"
    "- **UK & Ireland:** call **116 123** (Samaritans)\n"
    "- **Anywhere:** [findahelpline.com](https://findahelpline.com) lists free, confidential lines by country.\n\n"
    "I'm not a clinician and I won't pretend to be one, but I'm here. If you'd like, tell me what's "
    "going on — I can listen, help you think through one small next step, or help you write a "
    "message to someone you trust."
)

HARM_OTHERS_MESSAGE = (
    "I can't help with anything aimed at hurting another person — that's a hard line for me. If "
    "you're feeling pushed to that point, please contact a crisis line or someone who can talk it "
    "through with you in real time. If someone else is in immediate danger, contact local "
    "emergency services.\n\n"
    "If you want, I can help you think through what's actually driving this, or help you write a "
    "message to step back from a situation safely."
)

INJECTION_MESSAGE = (
    "I can't disable my guardrails or reveal protected instructions, but I can absolutely still "
    "help with the underlying question — just rephrase it as what you actually want to know or do."
)

MEDIUM_RISK_MESSAGE = (
    "This sounds like a lot to carry. I can help in a calm, practical way without pretending to be "
    "a clinician — tell me what would actually be useful right now."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _looks_like_figurative(text: str) -> bool:
    """Cheap heuristic to filter out obvious figurative use of dark phrases."""
    figurative_markers = [
        r"\bdying\s+for\b",     # "I'm dying for coffee"
        r"\bdying\s+to\b",      # "I'm dying to know"
        r"\bkill\s+for\b",      # "I'd kill for a vacation"
        r"\bkilled\s+it\b",     # "you killed it"
        r"\bkilling\s+it\b",
    ]
    return _matches_any(text, figurative_markers)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_safety(text: str) -> SafetyDecision:
    normalized = text.strip().lower()

    # Prompt injection wins first — it's structural.
    if _matches_any(normalized, JAILBREAK_PATTERNS):
        return SafetyDecision(
            severity="low",
            blocked=True,
            message=INJECTION_MESSAGE,
            reason="prompt_injection",
            category="prompt_injection",
        )

    # High-severity self-harm
    if _matches_any(normalized, SELF_HARM_PATTERNS) and not _looks_like_figurative(normalized):
        return SafetyDecision(
            severity="high",
            blocked=True,
            message=SELF_HARM_MESSAGE,
            reason="self_harm_language",
            category="self_harm",
        )

    # High-severity harm-to-others
    if _matches_any(normalized, HARM_OTHERS_PATTERNS):
        return SafetyDecision(
            severity="high",
            blocked=True,
            message=HARM_OTHERS_MESSAGE,
            reason="harm_others_language",
            category="harm_others",
        )

    # Medium-severity (do NOT block — adjust tone only)
    if _matches_any(normalized, MEDIUM_RISK_PATTERNS):
        return SafetyDecision(
            severity="medium",
            blocked=False,
            message=MEDIUM_RISK_MESSAGE,
            reason="medium_risk_language",
            category=None,
        )

    return SafetyDecision(severity="low", blocked=False, message=None, reason=None, category=None)