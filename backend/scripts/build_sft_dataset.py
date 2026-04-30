from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

DEFAULT_SYSTEM_PROMPT = (
    "You are ClarityAI. Lead with the answer. Be accurate, grounded, and practical. "
    "When evidence is available, rely on it. If something is uncertain or missing, say so plainly."
)

ALLOWED_ROLES = {"system", "user", "assistant"}
LEGACY_CITATION_PATTERNS = [
    re.compile(r"\[(?:S|W)\d+\]", re.IGNORECASE),
    re.compile(r"\(\s*source\s*\d+\s*\)", re.IGNORECASE),
    re.compile(r"\(\s*sources\s*:\s*[^)]*\)", re.IGNORECASE),
]
PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "<EMAIL>"),
    (
        re.compile(
            r"(?:(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d))"
        ),
        "<PHONE>",
    ),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<SSN>"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "<PAYMENT_CARD>"),
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), "<API_KEY>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<AWS_KEY>"),
    (
        re.compile(r"\b(?:Bearer|Token)\s+[A-Za-z0-9._-]{16,}\b", re.IGNORECASE),
        "<AUTH_TOKEN>",
    ),
]
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINE_RE = re.compile(r"\n{3,}")


@dataclass(slots=True)
class Example:
    messages: list[dict[str, str]]
    metadata: dict[str, Any]
    fingerprint: str

    def as_payload(self) -> dict[str, Any]:
        return {"messages": self.messages, "metadata": self.metadata}


@dataclass(slots=True)
class BuildStats:
    written_train: int = 0
    written_val: int = 0
    duplicates: int = 0
    skipped_empty: int = 0
    skipped_no_user_turn: int = 0
    skipped_short_user: int = 0
    skipped_short_assistant: int = 0
    skipped_long_assistant: int = 0
    skipped_low_value: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cleaned SFT-ready JSONL dataset from chat sessions."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path. If --val-out is set, this becomes the train split path.",
    )
    parser.add_argument(
        "--val-out",
        help="Optional validation JSONL path. When set, examples are split deterministically.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation fraction used when --val-out is provided.",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=8,
        help="Maximum number of preceding messages to keep before the target assistant answer.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt prepended to every training example.",
    )
    parser.add_argument(
        "--min-user-chars",
        type=int,
        default=6,
        help="Minimum character count for the latest user message.",
    )
    parser.add_argument(
        "--min-assistant-chars",
        type=int,
        default=24,
        help="Minimum character count for the assistant target answer.",
    )
    parser.add_argument(
        "--max-assistant-chars",
        type=int,
        default=12000,
        help="Maximum character count for the assistant target answer.",
    )
    parser.add_argument(
        "--scrub-pii",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Redact common PII and obvious secret formats before export.",
    )
    parser.add_argument(
        "--strip-legacy-citations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove old inline citation tags like [S1], [W2], or (source 3).",
    )
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()


def strip_legacy_citations(text: str) -> str:
    cleaned = text
    for pattern in LEGACY_CITATION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\[\s*\]", "", cleaned)
    return normalize_whitespace(cleaned)


def scrub_pii(text: str) -> str:
    cleaned = text
    for pattern, replacement in PII_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def sanitize_text(text: str, *, strip_citations: bool, redact_pii: bool) -> str:
    cleaned = normalize_whitespace(text or "")
    if not cleaned:
        return ""
    if strip_citations:
        cleaned = strip_legacy_citations(cleaned)
    if redact_pii:
        cleaned = scrub_pii(cleaned)
    return normalize_whitespace(cleaned)


def approx_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def low_value_text(text: str) -> bool:
    lowered = text.lower()
    if lowered in {"ok", "okay", "thanks", "thank you", "done", "cool"}:
        return True
    if len(text) < 40 and re.fullmatch(r"[\W_]+", text):
        return True
    return False


def normalize_role(role: Any) -> str:
    role_text = str(role or "").strip().lower()
    return role_text if role_text in ALLOWED_ROLES else "assistant"


def message_sort_key(message: Any) -> tuple[Any, Any, Any]:
    return (
        getattr(message, "created_at", None)
        or getattr(message, "inserted_at", None)
        or getattr(message, "updated_at", None),
        getattr(message, "sequence", None)
        or getattr(message, "position", None)
        or getattr(message, "sort_order", None),
        getattr(message, "id", 0),
    )


def fingerprint_messages(messages: Iterable[dict[str, str]]) -> str:
    serialized = json.dumps(
        [{"role": item["role"], "content": item["content"]} for item in messages],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def split_bucket(fingerprint: str) -> float:
    integer = int(hashlib.sha1(fingerprint.encode("utf-8")).hexdigest(), 16)
    return integer / float(16**40 - 1)


def iter_positive_examples(db: Any, args: argparse.Namespace) -> tuple[list[Example], BuildStats]:
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from app.db.models import ChatSession

    stats = BuildStats()
    seen: set[str] = set()
    examples: list[Example] = []

    sessions = db.execute(
        select(ChatSession).options(selectinload(ChatSession.messages))
    ).scalars().all()

    for session in sessions:
        messages = sorted(list(getattr(session, "messages", []) or []), key=message_sort_key)
        for index, message in enumerate(messages):
            if normalize_role(getattr(message, "role", "")) != "assistant":
                continue
            rating = str(getattr(message, "feedback_rating", "") or "").strip().lower()
            if rating != "up":
                continue

            raw_history = messages[max(0, index - args.max_history) : index + 1]
            cleaned_history: list[dict[str, str]] = []
            for item in raw_history:
                content = sanitize_text(
                    str(getattr(item, "content", "") or ""),
                    strip_citations=args.strip_legacy_citations,
                    redact_pii=args.scrub_pii,
                )
                if not content:
                    continue
                cleaned_history.append(
                    {
                        "role": normalize_role(getattr(item, "role", "assistant")),
                        "content": content,
                    }
                )

            if not cleaned_history:
                stats.skipped_empty += 1
                continue
            if cleaned_history[-1]["role"] != "assistant":
                stats.skipped_empty += 1
                continue

            latest_user = next(
                (item["content"] for item in reversed(cleaned_history[:-1]) if item["role"] == "user"),
                "",
            )
            assistant_text = cleaned_history[-1]["content"]

            if not latest_user:
                stats.skipped_no_user_turn += 1
                continue
            if len(latest_user) < args.min_user_chars:
                stats.skipped_short_user += 1
                continue
            if len(assistant_text) < args.min_assistant_chars:
                stats.skipped_short_assistant += 1
                continue
            if len(assistant_text) > args.max_assistant_chars:
                stats.skipped_long_assistant += 1
                continue
            if low_value_text(assistant_text):
                stats.skipped_low_value += 1
                continue

            payload_messages = [
                {"role": "system", "content": args.system_prompt},
                *cleaned_history,
            ]
            fingerprint = fingerprint_messages(payload_messages)
            if fingerprint in seen:
                stats.duplicates += 1
                continue
            seen.add(fingerprint)

            examples.append(
                Example(
                    messages=payload_messages,
                    metadata={
                        "session_id": getattr(session, "id", None),
                        "assistant_message_id": getattr(message, "id", None),
                        "title": getattr(session, "title", None),
                        "feedback_rating": rating,
                        "assistant_chars": len(assistant_text),
                        "assistant_tokens": approx_token_count(assistant_text),
                        "latest_user_chars": len(latest_user),
                        "latest_user_tokens": approx_token_count(latest_user),
                    },
                    fingerprint=fingerprint,
                )
            )

    return examples, stats


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    if args.val_out and not (0.0 < args.val_ratio < 1.0):
        raise SystemExit("--val-ratio must be between 0 and 1 when --val-out is provided.")

    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        examples, stats = iter_positive_examples(db, args)
    finally:
        db.close()

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []

    for example in examples:
        payload = example.as_payload()
        if args.val_out and split_bucket(example.fingerprint) < args.val_ratio:
            payload["metadata"]["split"] = "val"
            val_rows.append(payload)
        else:
            payload["metadata"]["split"] = "train"
            train_rows.append(payload)

    train_path = Path(args.out)
    train_count = write_jsonl(train_path, train_rows)
    stats.written_train = train_count

    val_count = 0
    if args.val_out:
        val_path = Path(args.val_out)
        val_count = write_jsonl(val_path, val_rows)
        stats.written_val = val_count
    else:
        stats.written_train = train_count

    summary = {
        "train_examples": stats.written_train,
        "val_examples": stats.written_val,
        "duplicates_skipped": stats.duplicates,
        "skipped_empty": stats.skipped_empty,
        "skipped_no_user_turn": stats.skipped_no_user_turn,
        "skipped_short_user": stats.skipped_short_user,
        "skipped_short_assistant": stats.skipped_short_assistant,
        "skipped_long_assistant": stats.skipped_long_assistant,
        "skipped_low_value": stats.skipped_low_value,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.val_out:
        print(f"Wrote {train_count} train examples to {train_path}")
        print(f"Wrote {val_count} validation examples to {Path(args.val_out)}")
    else:
        print(f"Wrote {train_count} SFT examples to {train_path}")


if __name__ == "__main__":
    main()
