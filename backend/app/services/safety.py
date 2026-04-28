from __future__ import annotations

import re
from dataclasses import asdict, dataclass

HIGH_RISK_PATTERNS = [
    r'\bkill myself\b',
    r'\bsuicide\b',
    r'\bend my life\b',
    r'\bhurt myself\b',
    r'\boverdose\b',
    r'\bnot safe\b',
    r'\bkill someone\b',
    r'\bhurt someone\b',
    r'\bmake a bomb\b',
]

MEDIUM_RISK_PATTERNS = [
    r'\bhopeless\b',
    r'\bnothing matters\b',
    r'\bpanic attack\b',
    r'\bwant to disappear\b',
    r"\bcan't cope\b",
    r'\boverwhelmed\b',
    r'\banxious\b',
    r'\bburned out\b',
    r'\bdepressed\b',
]

JAILBREAK_PATTERNS = [
    r'ignore previous instructions',
    r'reveal your system prompt',
    r'bypass safety',
    r'pretend you have no rules',
    r'developer mode',
]


@dataclass(slots=True)
class SafetyDecision:
    severity: str
    blocked: bool
    message: str | None
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


HIGH_RISK_MESSAGE = (
    'I am concerned this may be an immediate safety situation. Please contact local emergency support or a trusted person nearby right now. '
    'If you are not in immediate danger, tell me whether you want grounding support, a short safety plan, or help wording a message to someone you trust.'
)

MEDIUM_RISK_MESSAGE = (
    'This sounds heavy. I can still help in a calm, supportive, practical way, but I will avoid pretending to be a clinician or acting with false certainty.'
)

INJECTION_MESSAGE = (
    'I cannot disable my safeguards or reveal protected instructions, but I can still help with a legitimate question.'
)


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def assess_safety(text: str) -> SafetyDecision:
    normalized = text.strip().lower()
    if _matches_any(normalized, JAILBREAK_PATTERNS):
        return SafetyDecision(severity='low', blocked=True, message=INJECTION_MESSAGE, reason='prompt_injection')
    if _matches_any(normalized, HIGH_RISK_PATTERNS):
        return SafetyDecision(severity='high', blocked=True, message=HIGH_RISK_MESSAGE, reason='high_risk_language')
    if _matches_any(normalized, MEDIUM_RISK_PATTERNS):
        return SafetyDecision(severity='medium', blocked=False, message=MEDIUM_RISK_MESSAGE, reason='medium_risk_language')
    return SafetyDecision(severity='low', blocked=False, message=None, reason=None)
