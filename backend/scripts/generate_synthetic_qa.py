from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

DEFAULT_SYSTEM_PROMPT = (
    "You are ClarityAI. Lead with the answer. Be accurate, grounded, and practical. "
    "Use only what the provided source supports. If something is not supported, do not invent it."
)

GENERATOR_SYSTEM_PROMPT = (
    "You create high-quality supervised fine-tuning examples for a grounded assistant. "
    "Return valid JSON only and never wrap it in prose."
)

PROMPT_VARIANTS: tuple[dict[str, str], ...] = (
    {
        "name": "faq",
        "instruction": (
            "Generate realistic FAQ-style questions from the excerpt. Favor direct, practical user phrasing. "
            "Avoid vague prompts like 'summarize this document'."
        ),
    },
    {
        "name": "troubleshooting",
        "instruction": (
            "Generate practical troubleshooting or how-to questions someone would ask while using this information. "
            "Prioritize step-by-step, edge-case, or decision-making scenarios when supported."
        ),
    },
    {
        "name": "policy",
        "instruction": (
            "Generate policy, requirement, deadline, eligibility, or compliance questions when the excerpt supports them. "
            "Surface limits, exceptions, and conditions."
        ),
    },
    {
        "name": "decision",
        "instruction": (
            "Generate comparison, tradeoff, or decision questions when the excerpt supports them. "
            "Answers should be specific, balanced, and grounded in the source."
        ),
    },
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINE_RE = re.compile(r"\n{3,}")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(slots=True)
class ChunkTask:
    document_id: Any
    document_title: str
    chunk_id: Any
    heading: str | None
    content: str
    prompt_variant: str
    prompt: str


@dataclass(slots=True)
class CandidateExample:
    question: str
    answer: str
    metadata: dict[str, Any]
    quality_score: float

    @property
    def question_key(self) -> str:
        return question_fingerprint(self.question)

    @property
    def qa_key(self) -> str:
        digest = hashlib.sha1(
            f"{normalize_text(self.question)}\n{normalize_text(self.answer)}".encode("utf-8")
        ).hexdigest()
        return digest

    def as_payload(self) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": self.question},
                {"role": "assistant", "content": self.answer},
            ],
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class GenerationStats:
    chunks_considered: int = 0
    chunks_submitted: int = 0
    chunks_succeeded: int = 0
    chunks_failed: int = 0
    api_retries: int = 0
    raw_rows: int = 0
    accepted_rows: int = 0
    deduped_rows: int = 0
    rejected_empty: int = 0
    rejected_generic: int = 0
    rejected_similarity: int = 0
    rejected_grounding: int = 0
    rejected_length: int = 0
    train_rows: int = 0
    val_rows: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a high-quality synthetic QA dataset from knowledge documents."
    )
    parser.add_argument("--out", required=True, help="Output JSONL path for train rows or all rows.")
    parser.add_argument("--val-out", help="Optional validation JSONL path.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.03,
        help="Validation fraction used when --val-out is provided.",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=30,
        help="Maximum number of knowledge documents to sample.",
    )
    parser.add_argument(
        "--chunks-per-doc",
        type=int,
        default=4,
        help="Maximum number of diversified chunks to sample per document.",
    )
    parser.add_argument(
        "--qa-count",
        type=int,
        default=4,
        help="Requested QA pairs per chunk.",
    )
    parser.add_argument(
        "--chunk-char-limit",
        type=int,
        default=2200,
        help="Maximum chunk size sent to the model.",
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=220,
        help="Ignore chunks shorter than this after cleaning.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Maximum concurrent LLM requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum retry attempts per request on timeout or 5xx/429 errors.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.35,
        help="Sampling temperature for synthetic generation.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=("mixed", "faq", "troubleshooting", "policy", "decision"),
        default="mixed",
        help="Prompt family to use. mixed rotates across diverse templates.",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.55,
        help="Minimum heuristic quality score required to keep a QA pair.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for chunk and prompt selection.",
    )
    parser.add_argument("--base-url", help="Optional override for LLM base URL.")
    parser.add_argument("--api-key", help="Optional override for LLM API key.")
    parser.add_argument("--model", help="Optional override for chat model.")
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()


def normalize_text(text: str) -> str:
    return normalize_whitespace(text).lower()


def content_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    ]


def question_fingerprint(text: str) -> str:
    tokens = content_tokens(text)
    base = " ".join(tokens[:18]) or normalize_text(text)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def split_bucket(fingerprint: str) -> float:
    integer = int(hashlib.sha1(fingerprint.encode("utf-8")).hexdigest(), 16)
    return integer / float(16**40 - 1)


def chunk_sort_key(chunk: Any) -> tuple[Any, Any, Any]:
    return (
        getattr(chunk, "chunk_index", None)
        or getattr(chunk, "position", None)
        or getattr(chunk, "sequence", None),
        getattr(chunk, "created_at", None) or getattr(chunk, "updated_at", None),
        getattr(chunk, "id", 0),
    )


def clean_chunk_text(text: str) -> str:
    cleaned = normalize_whitespace(text or "")
    cleaned = re.sub(r"\n?Page \d+(?: of \d+)?\n?", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?Confidential\n?", "\n", cleaned, flags=re.IGNORECASE)
    return normalize_whitespace(cleaned)


def choose_diverse_chunks(chunks: Iterable[Any], limit: int, min_chunk_chars: int) -> list[Any]:
    ordered = sorted(list(chunks or []), key=chunk_sort_key)
    filtered = [chunk for chunk in ordered if len(clean_chunk_text(str(getattr(chunk, "content", "") or ""))) >= min_chunk_chars]
    if len(filtered) <= limit:
        return filtered
    if limit <= 1:
        return [filtered[len(filtered) // 2]]

    picks: list[Any] = []
    seen_indices: set[int] = set()
    last = len(filtered) - 1
    for slot in range(limit):
        index = round(slot * last / (limit - 1))
        if index in seen_indices:
            continue
        seen_indices.add(index)
        picks.append(filtered[index])
    if len(picks) < limit:
        for index, chunk in enumerate(filtered):
            if index in seen_indices:
                continue
            picks.append(chunk)
            if len(picks) >= limit:
                break
    return picks[:limit]


def prompt_variant_by_name(name: str, rng: random.Random, ordinal: int) -> dict[str, str]:
    if name == "mixed":
        return PROMPT_VARIANTS[ordinal % len(PROMPT_VARIANTS)]
    for variant in PROMPT_VARIANTS:
        if variant["name"] == name:
            return variant
    return rng.choice(list(PROMPT_VARIANTS))


def build_generation_prompt(
    *,
    title: str,
    heading: str | None,
    content: str,
    qa_count: int,
    variant: dict[str, str],
) -> str:
    heading_line = f"Section heading: {heading}\n" if heading else ""
    return (
        f"Create {qa_count} diverse, grounded QA pairs from the source excerpt below.\n"
        f"{variant['instruction']}\n"
        "Rules:\n"
        "- Questions must sound like real user requests, not dataset prompts.\n"
        "- Answers must be directly supported by the excerpt.\n"
        "- Do not invent facts, policies, deadlines, or steps.\n"
        "- Avoid duplicate questions or trivial rephrasings.\n"
        "- Avoid questions whose answer is just the question repeated.\n"
        "- Prefer concrete details, tradeoffs, conditions, and exceptions when supported.\n"
        "Return valid JSON only in this exact schema:\n"
        "[\n"
        '  {"question": "...", "answer": "..."}\n'
        "]\n\n"
        f"Document title: {title}\n"
        f"Prompt style: {variant['name']}\n"
        f"{heading_line}"
        f"Excerpt:\n{content}"
    )


def extract_balanced_json(text: str, open_char: str, close_char: str) -> str | None:
    start = text.find(open_char)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates: list[str] = []
    fence_match = CODE_FENCE_RE.search(stripped)
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    candidates.append(stripped)
    array_blob = extract_balanced_json(stripped, "[", "]")
    if array_blob:
        candidates.append(array_blob)
    object_blob = extract_balanced_json(stripped, "{", "}")
    if object_blob:
        candidates.append(object_blob)
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", candidate.strip())
        if cleaned and cleaned not in seen:
            unique.append(cleaned)
            seen.add(cleaned)
    return unique


def parse_generation_rows(raw_text: str) -> list[dict[str, Any]]:
    for candidate in json_candidates(raw_text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        rows: Any = payload
        if isinstance(payload, dict):
            for key in ("items", "qas", "qa_pairs", "data", "rows"):
                if isinstance(payload.get(key), list):
                    rows = payload[key]
                    break
            else:
                if all(key in payload for key in ("question", "answer")):
                    rows = [payload]
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    parsed_lines: list[dict[str, Any]] = []
    for line in raw_text.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and {"question", "answer"}.issubset(payload):
            parsed_lines.append(payload)
    return parsed_lines


def is_generic_question(question: str) -> bool:
    lowered = normalize_text(question)
    generic_patterns = (
        "what is this document about",
        "summarize this",
        "summarize the document",
        "what does this say",
        "what is the main idea",
        "can you summarize",
        "what is discussed",
    )
    return any(pattern in lowered for pattern in generic_patterns)


def qa_quality_score(question: str, answer: str, source_text: str, title: str) -> tuple[float, str]:
    question_clean = normalize_whitespace(question)
    answer_clean = normalize_whitespace(answer)
    if not question_clean or not answer_clean:
        return 0.0, "empty"

    q_tokens = content_tokens(question_clean)
    a_tokens = content_tokens(answer_clean)
    source_tokens = set(content_tokens(title + " " + source_text))
    shared_with_source = len(set(a_tokens) & source_tokens)
    question_shared = len(set(q_tokens) & source_tokens)
    qa_similarity = SequenceMatcher(None, normalize_text(question_clean), normalize_text(answer_clean)).ratio()

    score = 0.0
    reason = "ok"

    if is_generic_question(question_clean):
        return 0.25, "generic"
    if len(question_clean) < 12 or len(answer_clean) < 40:
        return 0.2, "length"
    if len(answer_clean) > 1800 or len(question_clean) > 260:
        return 0.2, "length"
    if qa_similarity >= 0.72:
        return 0.2, "similarity"
    if any(phrase in normalize_text(answer_clean) for phrase in ("not mentioned", "not specified", "not provided", "cannot be determined")):
        return 0.25, "grounding"

    if 4 <= len(q_tokens) <= 24:
        score += 0.16
    else:
        score += 0.06
        reason = "length"

    if 12 <= len(a_tokens) <= 180:
        score += 0.18
    else:
        score += 0.08
        reason = "length"

    if question_clean.endswith("?"):
        score += 0.08

    if question_shared >= 2:
        score += 0.16
    elif question_shared == 1:
        score += 0.08
    else:
        reason = "grounding"

    if shared_with_source >= 5:
        score += 0.32
    elif shared_with_source >= 3:
        score += 0.22
    elif shared_with_source >= 2:
        score += 0.14
        reason = "grounding"
    else:
        return 0.3, "grounding"

    if qa_similarity <= 0.48:
        score += 0.10
    elif qa_similarity <= 0.60:
        score += 0.05
    else:
        reason = "similarity"

    return min(score, 1.0), reason


def dedupe_candidates(candidates: Iterable[CandidateExample]) -> tuple[list[CandidateExample], int]:
    best_by_question: dict[str, CandidateExample] = {}
    seen_qa: set[str] = set()
    dropped = 0

    for candidate in candidates:
        if candidate.qa_key in seen_qa:
            dropped += 1
            continue
        seen_qa.add(candidate.qa_key)
        existing = best_by_question.get(candidate.question_key)
        if existing is None or (candidate.quality_score, len(candidate.answer)) > (
            existing.quality_score,
            len(existing.answer),
        ):
            if existing is not None:
                dropped += 1
            best_by_question[candidate.question_key] = candidate
        else:
            dropped += 1
    kept = sorted(
        best_by_question.values(),
        key=lambda item: (
            item.metadata.get("document_title", ""),
            item.metadata.get("chunk_id", 0),
            -item.quality_score,
        ),
    )
    return kept, dropped


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def resolve_llm_config(args: argparse.Namespace) -> tuple[str, str | None, str]:
    from app.config import get_settings

    settings = get_settings()
    base_url = args.base_url or getattr(settings, "llm_base_url", None)
    api_key = args.api_key if args.api_key is not None else getattr(settings, "llm_api_key", None)
    model = args.model or getattr(settings, "chat_model", None)
    if not base_url or not model:
        raise RuntimeError(
            "Synthetic QA generation requires an LLM provider. Set --base-url/--model or configure LLM_BASE_URL and CHAT_MODEL."
        )
    return str(base_url).rstrip("/"), api_key, str(model)


def load_chunk_tasks(args: argparse.Namespace) -> tuple[list[ChunkTask], GenerationStats]:
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from app.db.models import KnowledgeDocument
    from app.db.session import SessionLocal

    rng = random.Random(args.seed)
    stats = GenerationStats()
    tasks: list[ChunkTask] = []

    db = SessionLocal()
    try:
        documents = (
            db.execute(
                select(KnowledgeDocument)
                .options(selectinload(KnowledgeDocument.chunks))
                .order_by(KnowledgeDocument.updated_at.desc())
                .limit(args.limit_docs)
            )
            .scalars()
            .all()
        )
    finally:
        db.close()

    ordinal = 0
    for document in documents:
        chosen_chunks = choose_diverse_chunks(
            getattr(document, "chunks", []) or [],
            limit=args.chunks_per_doc,
            min_chunk_chars=args.min_chunk_chars,
        )
        stats.chunks_considered += len(getattr(document, "chunks", []) or [])
        for chunk in chosen_chunks:
            content = clean_chunk_text(str(getattr(chunk, "content", "") or ""))
            if len(content) < args.min_chunk_chars:
                continue
            content = content[: args.chunk_char_limit]
            variant = prompt_variant_by_name(args.prompt_style, rng, ordinal)
            ordinal += 1
            heading = (
                getattr(chunk, "heading", None)
                or getattr(chunk, "section_title", None)
                or getattr(chunk, "title", None)
            )
            tasks.append(
                ChunkTask(
                    document_id=getattr(document, "id", None),
                    document_title=str(getattr(document, "title", "Untitled document") or "Untitled document"),
                    chunk_id=getattr(chunk, "id", None),
                    heading=str(heading).strip() if heading else None,
                    content=content,
                    prompt_variant=variant["name"],
                    prompt=build_generation_prompt(
                        title=str(getattr(document, "title", "Untitled document") or "Untitled document"),
                        heading=str(heading).strip() if heading else None,
                        content=content,
                        qa_count=args.qa_count,
                        variant=variant,
                    ),
                )
            )
    stats.chunks_submitted = len(tasks)
    return tasks, stats


async def request_completion(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_retries: int,
    stats: GenerationStats,
) -> str:
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }

    attempt = 0
    while True:
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.NetworkError) as exc:
            retryable = True
            if isinstance(exc, httpx.HTTPStatusError):
                retryable = exc.response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
            if attempt >= max_retries or not retryable:
                raise
            attempt += 1
            stats.api_retries += 1
            await asyncio.sleep(min(8.0, 0.6 * (2 ** (attempt - 1))) + random.random() * 0.15)


async def generate_for_chunk(
    task: ChunkTask,
    *,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str | None,
    model: str,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    stats: GenerationStats,
) -> list[CandidateExample]:
    async with semaphore:
        content = await request_completion(
            client,
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": task.prompt},
            ],
            temperature=args.temperature,
            max_retries=args.max_retries,
            stats=stats,
        )
    rows = parse_generation_rows(content)
    stats.raw_rows += len(rows)
    accepted: list[CandidateExample] = []

    for row in rows:
        question = normalize_whitespace(str(row.get("question", "") or ""))
        answer = normalize_whitespace(str(row.get("answer", "") or ""))
        quality, reason = qa_quality_score(question, answer, task.content, task.document_title)
        if not question or not answer:
            stats.rejected_empty += 1
            continue
        if quality < args.min_quality_score:
            if reason == "generic":
                stats.rejected_generic += 1
            elif reason == "similarity":
                stats.rejected_similarity += 1
            elif reason == "grounding":
                stats.rejected_grounding += 1
            else:
                stats.rejected_length += 1
            continue

        accepted.append(
            CandidateExample(
                question=question if question.endswith("?") else f"{question}?",
                answer=answer,
                quality_score=quality,
                metadata={
                    "document_id": task.document_id,
                    "document_title": task.document_title,
                    "chunk_id": task.chunk_id,
                    "chunk_heading": task.heading,
                    "prompt_variant": task.prompt_variant,
                    "quality_score": round(quality, 4),
                    "model": model,
                },
            )
        )

    return accepted


async def async_main(args: argparse.Namespace) -> None:
    if args.val_out and not (0.0 < args.val_ratio < 1.0):
        raise SystemExit("--val-ratio must be between 0 and 1 when --val-out is provided.")

    base_url, api_key, model = resolve_llm_config(args)
    tasks, stats = load_chunk_tasks(args)
    if not tasks:
        raise SystemExit("No eligible document chunks found. Try lowering --min-chunk-chars or increasing --limit-docs.")

    timeout = httpx.Timeout(90.0, connect=25.0)
    limits = httpx.Limits(max_connections=max(args.max_workers * 2, 4), max_keepalive_connections=max(args.max_workers, 2))
    semaphore = asyncio.Semaphore(max(args.max_workers, 1))

    all_candidates: list[CandidateExample] = []
    started = time.perf_counter()

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        coroutines = [
            generate_for_chunk(
                task,
                client=client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                args=args,
                semaphore=semaphore,
                stats=stats,
            )
            for task in tasks
        ]

        results = await asyncio.gather(*coroutines, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            stats.chunks_failed += 1
            continue
        stats.chunks_succeeded += 1
        all_candidates.extend(result)

    deduped, dropped = dedupe_candidates(all_candidates)
    stats.accepted_rows = len(deduped)
    stats.deduped_rows = dropped

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for candidate in deduped:
        payload = candidate.as_payload()
        if args.val_out and split_bucket(candidate.question_key) < args.val_ratio:
            payload["metadata"]["split"] = "val"
            val_rows.append(payload)
        else:
            payload["metadata"]["split"] = "train"
            train_rows.append(payload)

    stats.train_rows = write_jsonl(Path(args.out), train_rows)
    if args.val_out:
        stats.val_rows = write_jsonl(Path(args.val_out), val_rows)

    elapsed = time.perf_counter() - started
    summary = {
        "documents_limit": args.limit_docs,
        "chunks_considered": stats.chunks_considered,
        "chunks_submitted": stats.chunks_submitted,
        "chunks_succeeded": stats.chunks_succeeded,
        "chunks_failed": stats.chunks_failed,
        "api_retries": stats.api_retries,
        "raw_rows": stats.raw_rows,
        "accepted_rows": stats.accepted_rows,
        "deduped_rows": stats.deduped_rows,
        "train_rows": stats.train_rows,
        "val_rows": stats.val_rows,
        "rejected": {
            "empty": stats.rejected_empty,
            "generic": stats.rejected_generic,
            "similarity": stats.rejected_similarity,
            "grounding": stats.rejected_grounding,
            "length": stats.rejected_length,
        },
        "seconds": round(elapsed, 2),
        "rows_per_second": round((stats.train_rows + stats.val_rows) / elapsed, 2) if elapsed else 0.0,
        "model": model,
        "prompt_style": args.prompt_style,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.val_out:
        print(f"Wrote {stats.train_rows} train rows to {Path(args.out)}")
        print(f"Wrote {stats.val_rows} validation rows to {Path(args.val_out)}")
    else:
        print(f"Wrote {stats.train_rows} synthetic QA rows to {Path(args.out)}")


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
