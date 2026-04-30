from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from build_sft_dataset import (
    approx_token_count,
    fingerprint_messages,
    low_value_text,
    message_sort_key,
    normalize_role,
    sanitize_text,
    split_bucket,
)

RATING_ALIASES = {
    "thumbs_up": "up",
    "positive": "up",
    "like": "up",
    "good": "up",
    "thumbs_down": "down",
    "negative": "down",
    "dislike": "down",
    "bad": "down",
    "neutral": "neutral",
}
RATING_SCORE = {"up": 1, "down": -1, "neutral": 0}


@dataclass(slots=True)
class TurnRecord:
    prompt_messages: list[dict[str, str]]
    response: str
    label: str
    metadata: dict[str, Any]
    fingerprint: str
    prompt_fingerprint: str


@dataclass(slots=True)
class ExportStats:
    turns_emitted: int = 0
    pairwise_emitted: int = 0
    val_emitted: int = 0
    duplicates_skipped: int = 0
    skipped_empty_prompt: int = 0
    skipped_empty_response: int = 0
    skipped_no_user_turn: int = 0
    skipped_low_value: int = 0
    skipped_rating: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export cleaned feedback datasets from rated assistant turns."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path. If --val-out is set, this becomes the train split path.",
    )
    parser.add_argument(
        "--val-out",
        help="Optional validation JSONL path. When set, rows are split deterministically.",
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
        default=6,
        help="Maximum number of preceding messages to keep before each rated assistant turn.",
    )
    parser.add_argument(
        "--format",
        choices=("turn", "pairwise"),
        default="turn",
        help="turn: one row per rated response. pairwise: chosen/rejected pairs when both exist for the same prompt.",
    )
    parser.add_argument(
        "--ratings",
        default="up,down",
        help="Comma-separated feedback labels to include, e.g. 'up,down' or 'up'.",
    )
    parser.add_argument(
        "--min-response-chars",
        type=int,
        default=8,
        help="Minimum cleaned assistant response length to keep.",
    )
    parser.add_argument(
        "--max-response-chars",
        type=int,
        default=12000,
        help="Maximum cleaned assistant response length to keep.",
    )
    parser.add_argument(
        "--max-pairs-per-prompt",
        type=int,
        default=3,
        help="Maximum chosen/rejected pairs to emit for a single prompt in pairwise mode.",
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


def normalize_rating(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    return RATING_ALIASES.get(value, value)


def parse_rating_set(raw: str) -> set[str]:
    values = {normalize_rating(part) for part in raw.split(",") if part.strip()}
    return {value for value in values if value}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_turn_records(db: Any, args: argparse.Namespace) -> tuple[list[TurnRecord], ExportStats, Counter[str]]:
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from app.db.models import ChatSession

    desired_ratings = parse_rating_set(args.ratings)
    seen: set[str] = set()
    stats = ExportStats()
    rating_counts: Counter[str] = Counter()
    turns: list[TurnRecord] = []

    sessions = db.execute(
        select(ChatSession).options(selectinload(ChatSession.messages))
    ).scalars().all()

    for session in sessions:
        messages = sorted(list(getattr(session, "messages", []) or []), key=message_sort_key)
        for index, message in enumerate(messages):
            if normalize_role(getattr(message, "role", "")) != "assistant":
                continue

            rating = normalize_rating(getattr(message, "feedback_rating", ""))
            if desired_ratings and rating not in desired_ratings:
                stats.skipped_rating += 1
                continue

            prompt_window = messages[max(0, index - args.max_history) : index]
            prompt_messages: list[dict[str, str]] = []
            for item in prompt_window:
                content = sanitize_text(
                    str(getattr(item, "content", "") or ""),
                    strip_citations=args.strip_legacy_citations,
                    redact_pii=args.scrub_pii,
                )
                if not content:
                    continue
                role = normalize_role(getattr(item, "role", "assistant"))
                if role not in {"system", "user", "assistant"}:
                    continue
                prompt_messages.append({"role": role, "content": content})

            if not prompt_messages:
                stats.skipped_empty_prompt += 1
                continue

            latest_user = next(
                (item["content"] for item in reversed(prompt_messages) if item["role"] == "user"),
                "",
            )
            if not latest_user:
                stats.skipped_no_user_turn += 1
                continue

            response = sanitize_text(
                str(getattr(message, "content", "") or ""),
                strip_citations=args.strip_legacy_citations,
                redact_pii=args.scrub_pii,
            )
            if not response:
                stats.skipped_empty_response += 1
                continue
            if len(response) < args.min_response_chars:
                stats.skipped_empty_response += 1
                continue
            if len(response) > args.max_response_chars:
                stats.skipped_empty_response += 1
                continue
            if rating == "up" and low_value_text(response):
                stats.skipped_low_value += 1
                continue

            fingerprint = fingerprint_messages(
                [*prompt_messages, {"role": "assistant", "content": response}]
            )
            if fingerprint in seen:
                stats.duplicates_skipped += 1
                continue
            seen.add(fingerprint)

            turns.append(
                TurnRecord(
                    prompt_messages=prompt_messages,
                    response=response,
                    label=rating,
                    metadata={
                        "session_id": getattr(session, "id", None),
                        "assistant_message_id": getattr(message, "id", None),
                        "title": getattr(session, "title", None),
                        "response_chars": len(response),
                        "response_tokens": approx_token_count(response),
                        "latest_user_chars": len(latest_user),
                        "latest_user_tokens": approx_token_count(latest_user),
                    },
                    fingerprint=fingerprint,
                    prompt_fingerprint=fingerprint_messages(prompt_messages),
                )
            )
            rating_counts[rating] += 1

    return turns, stats, rating_counts


def emit_turn_rows(turns: list[TurnRecord]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for turn in turns:
        row = {
            "messages": turn.prompt_messages,
            "response": turn.response,
            "label": turn.label,
            "score": RATING_SCORE.get(turn.label, 0),
            "metadata": turn.metadata,
        }
        rows.append((turn.fingerprint, row))
    return rows


def emit_pairwise_rows(turns: list[TurnRecord], max_pairs_per_prompt: int) -> list[tuple[str, dict[str, Any]]]:
    by_prompt: dict[str, list[TurnRecord]] = defaultdict(list)
    rows: list[tuple[str, dict[str, Any]]] = []

    for turn in turns:
        by_prompt[turn.prompt_fingerprint].append(turn)

    for group in by_prompt.values():
        positives = [item for item in group if item.label == "up"]
        negatives = [item for item in group if item.label == "down"]
        if not positives or not negatives:
            continue

        positives.sort(key=lambda item: (-item.metadata.get("response_chars", 0), item.fingerprint))
        negatives.sort(key=lambda item: (item.metadata.get("response_chars", 0), item.fingerprint))

        emitted = 0
        for chosen in positives:
            for rejected in negatives:
                if emitted >= max_pairs_per_prompt:
                    break
                pair_fingerprint = f"{chosen.prompt_fingerprint}:{chosen.fingerprint}:{rejected.fingerprint}"
                row = {
                    "messages": chosen.prompt_messages,
                    "chosen": chosen.response,
                    "rejected": rejected.response,
                    "metadata": {
                        "session_id": chosen.metadata.get("session_id"),
                        "title": chosen.metadata.get("title"),
                        "chosen_message_id": chosen.metadata.get("assistant_message_id"),
                        "rejected_message_id": rejected.metadata.get("assistant_message_id"),
                    },
                }
                rows.append((pair_fingerprint, row))
                emitted += 1
            if emitted >= max_pairs_per_prompt:
                break
    return rows


def main() -> None:
    args = parse_args()
    if args.val_out and not (0.0 < args.val_ratio < 1.0):
        raise SystemExit("--val-ratio must be between 0 and 1 when --val-out is provided.")

    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        turns, stats, rating_counts = iter_turn_records(db, args)
    finally:
        db.close()

    if args.format == "pairwise":
        emitted_rows = emit_pairwise_rows(turns, max_pairs_per_prompt=args.max_pairs_per_prompt)
    else:
        emitted_rows = emit_turn_rows(turns)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    seen_rows: set[str] = set()

    for fingerprint, row in emitted_rows:
        if fingerprint in seen_rows:
            stats.duplicates_skipped += 1
            continue
        seen_rows.add(fingerprint)
        if args.val_out and split_bucket(fingerprint) < args.val_ratio:
            row.setdefault("metadata", {})["split"] = "val"
            val_rows.append(row)
        else:
            row.setdefault("metadata", {})["split"] = "train"
            train_rows.append(row)

    train_count = write_jsonl(Path(args.out), train_rows)
    val_count = 0
    if args.val_out:
        val_count = write_jsonl(Path(args.val_out), val_rows)

    if args.format == "pairwise":
        stats.pairwise_emitted = train_count + val_count
        stats.val_emitted = val_count
    else:
        stats.turns_emitted = train_count + val_count
        stats.val_emitted = val_count

    summary = {
        "format": args.format,
        "ratings": sorted(parse_rating_set(args.ratings)),
        "train_rows": train_count,
        "val_rows": val_count,
        "source_turns_by_rating": dict(rating_counts),
        "duplicates_skipped": stats.duplicates_skipped,
        "skipped_empty_prompt": stats.skipped_empty_prompt,
        "skipped_empty_response": stats.skipped_empty_response,
        "skipped_no_user_turn": stats.skipped_no_user_turn,
        "skipped_low_value": stats.skipped_low_value,
        "skipped_rating": stats.skipped_rating,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.val_out:
        print(f"Wrote {train_count} train rows to {Path(args.out)}")
        print(f"Wrote {val_count} validation rows to {Path(args.val_out)}")
    else:
        print(f"Wrote {train_count} feedback rows to {Path(args.out)}")


if __name__ == "__main__":
    main()
