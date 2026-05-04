from __future__ import annotations

"""
Chat orchestration for ClarityAI - Elite version.

What's new vs the previous version:
- Concurrent web research + image search using asyncio.gather, so adding
  images doesn't slow down answers.
- Image search is triggered when the user explicitly asks for images and
  also (optionally) when the answer is being researched and the topic is
  visual ("how does X look", "what is X").
- A new SSE event `images` carries the image gallery to the frontend before
  the text streams.
- Self-identity questions are answered with a hand-written, paraphrased
  template when no LLM is configured - never with web search results that
  describe a different assistant.
- Slightly tighter token streaming with newline-preserving fallback.
- Cleaner ProviderUnavailable handling - the user always gets a useful
  reply even when the LLM is down.
"""

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import ChatMessage, ChatSession
from app.services.image_search import ImageResult, image_search
from app.services.prompts import build_system_prompt, build_user_prompt
from app.services.providers import ProviderUnavailableError, pick_provider, provider
from app.services.retrieval import retrieval_index
from app.services.routing import choose_route
from app.services.safety import assess_safety
from app.services.web_research import WebResearchUnavailableError, web_research
from app.utils.serializers import serialize_message, serialize_session

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Patterns for citation cleanup
# ---------------------------------------------------------------------------

INTERNAL_TAG_RE = re.compile(
    r"""
    \[\s*(?:S|W)\d+(?:\s*,\s*(?:S|W)\d+)*\s*\]
    | \(\s*[Ss]ource[s]?\s*[:#]?\s*\d+(?:\s*,\s*\d+)*\s*\)
    | \[\s*[Ss]ource\s*\d+\s*\]
    """,
    re.VERBOSE,
)
FOOTNOTE_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def generate_title(message: str) -> str:
    words = [w for w in message.replace("\n", " ").split() if w.strip()]
    if not words:
        return "New conversation"
    title = " ".join(words[:8]).strip()
    return title[:60] + ("..." if len(title) > 60 else "")


def sse_event(event_name: str, payload: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _build_history(messages: list[ChatMessage], char_budget: int) -> list[dict]:
    selected: list[dict] = []
    total = 0
    for message in reversed(messages[-6:]):
        content = (message.content or "").strip()
        if not content:
            continue
        length = len(content)
        if total + length > char_budget and selected:
            break
        selected.append({"role": message.role, "content": content})
        total += length
    return list(reversed(selected))


def _update_summary(existing_summary: str, user_text: str, assistant_text: str) -> str:
    existing_lines = [line.strip() for line in existing_summary.splitlines() if line.strip()]
    merged = existing_lines[-4:] + [
        f"- User: {user_text[:180]}",
        f"- Assistant: {assistant_text[:220]}",
    ]
    return "\n".join(merged[-6:])


def _source_name(source: dict) -> str:
    title = (source.get("document_title") or source.get("source_name") or "Source").strip()
    if source.get("page_label"):
        return f"{title} (page {source['page_label']})"
    return title


def _resolve_route_label(
    route_decision_label: str, local_sources: list[dict], web_sources: list[dict]
) -> str:
    if route_decision_label == "chat":
        return "chat"
    if local_sources and web_sources:
        return "hybrid"
    if web_sources:
        return "research"
    if local_sources:
        return "local"
    return "chat"


# ---------------------------------------------------------------------------
# Citation post-processing
# ---------------------------------------------------------------------------


def _renumber_citations(citations: list[dict]) -> list[dict]:
    out = []
    for i, c in enumerate(citations, start=1):
        item = dict(c)
        item["number"] = i
        item["display_name"] = _source_name(item)
        item["label"] = f"[{i}]"
        out.append(item)
    return out


def _strip_internal_tags(text: str) -> str:
    cleaned = INTERNAL_TAG_RE.sub("", text)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _filter_footnotes(text: str, valid_numbers: set[int]) -> str:
    def repl(match: re.Match) -> str:
        nums = [int(n.strip()) for n in match.group(1).split(",") if n.strip().isdigit()]
        valid = [n for n in nums if n in valid_numbers]
        if not valid:
            return ""
        if len(valid) == len(nums):
            return match.group(0)
        return "[" + ", ".join(str(n) for n in valid) + "]"

    return FOOTNOTE_RE.sub(repl, text)


def _build_sources_section(citations: list[dict]) -> str:
    if not citations:
        return ""
    lines = ["", "---", "**Sources**"]
    for c in citations:
        n = c.get("number") or 0
        name = c.get("display_name") or "Source"
        url = c.get("url")
        kind = "web" if c.get("source_type") == "web" else "file"
        if url:
            lines.append(f"{n}. [{name}]({url}) - {kind}")
        else:
            lines.append(f"{n}. {name} - {kind}")
    return "\n".join(lines)


def _polish_answer(raw_text: str, citations: list[dict]) -> str:
    text = _strip_internal_tags(raw_text)
    valid_numbers = {c["number"] for c in citations}
    text = _filter_footnotes(text, valid_numbers)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    if citations:
        text += "\n" + _build_sources_section(citations)
    return text.strip()


# ---------------------------------------------------------------------------
# Self-identity hand-written answer (used when LLM is offline)
# ---------------------------------------------------------------------------


SELF_ID_FALLBACK = (
    "I'm **ClarityAI**, a research-grade conversational assistant. I'm built on "
    "an open-weights LLM and grounded in three things at once:\n\n"
    "1. **Your uploaded knowledge** - PDFs, docs, notes, manuals you drop into the right panel.\n"
    "2. **Conversation memory** - I remember the recent turns so follow-ups make sense.\n"
    "3. **Optional web research** - when you ask for current or external facts, I can pull from trusted sources and cite them.\n\n"
    "I'm not Claude, ChatGPT, Gemini, or any other named commercial assistant - just my own thing, "
    "designed around grounded answers with citations you can actually inspect."
)


# ---------------------------------------------------------------------------
# Fallback synthesis when the LLM is unavailable
# ---------------------------------------------------------------------------


def _summarize_snippet(snippet: str, max_len: int = 260) -> str:
    snippet = re.sub(r"\s+", " ", snippet or "").strip()
    if len(snippet) <= max_len:
        return snippet
    cut = snippet[:max_len]
    last = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if last > 80:
        cut = cut[: last + 1]
    return cut.rstrip() + "..."


def _fallback_answer(
    user_message: str,
    citations: list[dict],
    retrieval_confidence: float,
    medium_risk: bool,
    research_error: str | None,
    llm_error_hint: str | None,
    intent: str,
    image_count: int,
) -> str:
    # Self-identity: never use random web sources to answer "who are you".
    if intent == "self_identity":
        return SELF_ID_FALLBACK

    if not citations:
        if image_count > 0:
            lines = [
                f"I've pulled {image_count} image(s) for you - they're in the gallery above.",
                "I don't have strong text evidence beyond that yet.",
            ]
        else:
            lines = [
                "I don't have strong evidence for this yet, so I'd rather not guess.",
            ]
        if llm_error_hint:
            lines.append(llm_error_hint)
        if research_error:
            lines.append(f"Web research note: {research_error}")
        lines.append(
            "Best next step: upload a more relevant document, or rephrase the "
            "question with one or two specific names, products, or terms."
        )
        if medium_risk:
            lines.append("I'll keep the tone calm and practical - happy to help work through it.")
        return "\n\n".join(lines)

    locals_ = [c for c in citations if c.get("source_type") != "web"]
    webs = [c for c in citations if c.get("source_type") == "web"]

    header = (
        "I'm running without my full model right now, so here's the best grounded synthesis I can "
        "give from the retrieved evidence:"
    )
    if llm_error_hint:
        header += f" ({llm_error_hint})"
    paras: list[str] = [header]

    if image_count > 0:
        paras.append(f"I've also attached {image_count} image(s) in the gallery above.")

    if locals_:
        paras.append("**From your uploaded knowledge:**")
        for c in locals_[:3]:
            paras.append(f"- *{c['display_name']}* - {_summarize_snippet(c.get('snippet') or '')}")

    if webs:
        paras.append("**From web research:**")
        for c in webs[:3]:
            url = c.get("url")
            link = f"[{c['display_name']}]({url})" if url else c["display_name"]
            paras.append(f"- {link} - {_summarize_snippet(c.get('snippet') or '')}")

    if retrieval_confidence < settings.low_confidence_threshold:
        paras.append(
            "Confidence is on the lower side - a narrower question or a more specific keyword "
            "will improve this."
        )

    if research_error:
        paras.append(f"Research note: {research_error}")

    if medium_risk:
        paras.append("I'm keeping this calm and practical because your message sounded heavy.")

    return "\n\n".join(paras)


async def _stream_text(text: str) -> AsyncIterator[str]:
    """Stream a string word-by-word so the UI feels alive on fallback paths."""
    for token in text.split(" "):
        yield token + " "
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Concurrent fetch helper
# ---------------------------------------------------------------------------


async def _gather_external(
    user_message: str,
    needs_research: bool,
    needs_images: bool,
    research_max: int,
    image_max: int,
) -> tuple[list[dict], list[ImageResult], str | None, str | None]:
    """
    Run web research and image search concurrently. Each side fails
    independently - if research blows up, images can still arrive, and vice
    versa.
    Returns: (web_citations, images, research_error, image_error)
    """
    research_error: str | None = None
    image_error: str | None = None

    async def _research() -> list[dict]:
        nonlocal research_error
        if not needs_research:
            return []
        try:
            sources = await web_research.search(user_message, max_results=research_max)
            return [
                source.to_citation(label=f"W{i}")
                for i, source in enumerate(sources, start=1)
            ]
        except WebResearchUnavailableError as exc:
            research_error = str(exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Web research failed: %s", exc)
            research_error = "Web research failed for this turn."
        return []

    async def _images() -> list[ImageResult]:
        nonlocal image_error
        if not needs_images:
            return []
        try:
            return await image_search.search(user_message, max_results=image_max)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Image search failed: %s", exc)
            image_error = "Image search failed for this turn."
            return []

    web_citations, images = await asyncio.gather(_research(), _images())
    return web_citations, images, research_error, image_error


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def stream_chat_reply(
    db: Session,
    session_id: str | None,
    user_message: str,
    mode: str,
    research_mode: str,
) -> AsyncIterator[dict]:
    # ----- Session bootstrap --------------------------------------------------
    session = db.get(ChatSession, session_id) if session_id else None
    if session is None:
        session = ChatSession(title=generate_title(user_message))
        db.add(session)
        db.flush()
    elif session.title == "New conversation":
        session.title = generate_title(user_message)

    user_record = ChatMessage(
        session_id=session.id, role="user", content=user_message, citations_json=[]
    )
    db.add(user_record)
    db.flush()

    yield {"type": "status", "stage": "retrieving", "message": "Searching your knowledge base"}

    # ----- Safety + retrieval ------------------------------------------------
    safety = assess_safety(user_message)

    recent_messages = db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id, ChatMessage.id != user_record.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(min(settings.max_history_messages, 6))
    ).scalars().all()
    recent_messages = list(reversed(recent_messages))

    recent_user_context = " ".join(
        m.content for m in recent_messages if m.role == "user"
    )[-600:]

    retrieval = retrieval_index.search(query=user_message, conversation_context=recent_user_context)
    local_citations = retrieval["results"]
    if retrieval["confidence"] < 0.12:
        local_citations = []

    # ----- Routing -----------------------------------------------------------
    route_decision = choose_route(
        user_message=user_message,
        local_confidence=retrieval["confidence"],
        local_hits=len(local_citations),
        research_mode=research_mode,
    )

    web_citations: list[dict] = []
    images: list[ImageResult] = []
    research_error: str | None = None
    image_error: str | None = None

    if not safety.blocked and (route_decision.needs_web_research or route_decision.needs_image_search):
        if route_decision.needs_image_search:
            yield {"type": "status", "stage": "researching", "message": "Researching and pulling images"}
        else:
            yield {"type": "status", "stage": "researching", "message": "Researching trusted web sources"}

        web_citations, images, research_error, image_error = await _gather_external(
            user_message=user_message,
            needs_research=route_decision.needs_web_research,
            needs_images=route_decision.needs_image_search,
            research_max=settings.research_max_results,
            image_max=6,
        )

    raw_citations = [*local_citations, *web_citations]
    citations = _renumber_citations(raw_citations)

    resolved_route = _resolve_route_label(route_decision.route, local_citations, web_citations)
    image_dicts = [img.to_dict() for img in images]

    # ----- Initial meta event -------------------------------------------------
    yield {
        "type": "meta",
        "session": serialize_session(session),
        "citations": citations,
        "search": {
            "confidence": retrieval["confidence"],
            "hits": len(local_citations),
            "dense_used": retrieval.get("dense_used", False),
        },
        "safety": safety.to_dict(),
        "route": {**route_decision.to_dict(), "resolved_route": resolved_route},
        "research": {
            "attempted": route_decision.needs_web_research,
            "count": len(web_citations),
            "error": research_error,
        },
        "images": {
            "attempted": route_decision.needs_image_search,
            "count": len(image_dicts),
            "error": image_error,
            "results": image_dicts,
        },
    }

    # Emit a separate `images` event so the frontend can render the gallery
    # *before* text starts streaming - feels much faster.
    if image_dicts:
        yield {"type": "images", "results": image_dicts}

    # ----- Safety block early-return -----------------------------------------
    if safety.blocked:
        assistant_text = safety.message or "I cannot continue with that request."
        async for token in _stream_text(assistant_text):
            yield {"type": "token", "content": token}
        assistant_record = ChatMessage(
            session_id=session.id,
            role="assistant",
            content=assistant_text,
            citations_json=[],
        )
        if image_dicts:
            assistant_record.citations_json = []  # keep clean
        db.add(assistant_record)
        session.summary = _update_summary(session.summary or "", user_message, assistant_text)
        db.commit()
        db.refresh(session)
        db.refresh(assistant_record)

        message_payload = serialize_message(assistant_record)
        # Attach images on the wire so the UI can re-render historical turns.
        message_payload["images"] = image_dicts

        yield {
            "type": "done",
            "message": message_payload,
            "citations": [],
            "session": serialize_session(session),
            "safety": safety.to_dict(),
            "route": {"resolved_route": "safety_block"},
            "research": {"attempted": False, "count": 0, "error": None},
            "images": {
                "attempted": route_decision.needs_image_search,
                "count": len(image_dicts),
                "error": image_error,
                "results": image_dicts,
            },
        }
        return

    # ----- Build LLM messages -------------------------------------------------
    model_messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                mode=mode,
                route=resolved_route,
                medium_risk=safety.severity == "medium",
                has_images=bool(image_dicts),
                intent=route_decision.intent,
            ),
        }
    ]
    model_messages.extend(
        _build_history(recent_messages, char_budget=min(settings.history_char_budget, 1800))
    )
    model_messages.append(
        {
            "role": "user",
            "content": build_user_prompt(
                user_message=user_message,
                local_sources=local_citations,
                web_sources=web_citations,
                retrieval_confidence=retrieval["confidence"],
                session_summary=session.summary or "",
                route_reason=route_decision.reason,
                image_count=len(image_dicts),
            ),
        }
    )

    yield {"type": "status", "stage": "answering", "message": "Composing answer"}

    # ----- Stream from LLM ----------------------------------------------------
    response_parts: list[str] = []
    used_fallback = False
    llm_error_hint: str | None = None

    selected_provider = pick_provider(
        user_message=user_message,
        route_decision=route_decision,
        mode=mode,
    )

    # Surface which model/provider answered so the UI/logs can show it.
    yield {
        "type": "status",
        "stage": "answering",
        "message": f"Using {selected_provider.model}",
        "provider": selected_provider.name,
        "model": selected_provider.model,
    }

    try:
        async for token in selected_provider.stream_chat(model_messages):
            response_parts.append(token)
            yield {"type": "token", "content": token}
    except ProviderUnavailableError as exc:
        used_fallback = True
        llm_error_hint = str(exc)
        logger.warning("LLM provider unavailable: %s", exc)
    except httpx.HTTPStatusError as exc:
        used_fallback = True
        status = exc.response.status_code if exc.response is not None else None
        if status == 401:
            llm_error_hint = (
                f"The {selected_provider.name!r} provider rejected the API key (HTTP 401). "
                f"Check the relevant *_API_KEY in backend/.env."
            )
        elif status == 404:
            llm_error_hint = (
                f"The model \"{selected_provider.model}\" was not found at "
                f"{selected_provider.base_url}. Check the model name."
            )
        elif status == 429:
            llm_error_hint = (
                f"The {selected_provider.name!r} provider rate-limited this request. "
                f"On Gemini's free tier, Pro is capped at 50 req/day and Flash at 1,500/day. "
                f"Wait briefly and retry, or switch CHAT_MODEL/HEAVY_CHAT_MODEL to a less-loaded tier."
            )
        elif status and 500 <= status < 600:
            llm_error_hint = f"The {selected_provider.name!r} provider had a server error (HTTP {status}). Try again."
        else:
            llm_error_hint = f"The {selected_provider.name!r} provider returned HTTP {status}."
        logger.warning("LLM HTTP error (%s): %s", selected_provider.name, llm_error_hint)
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
        used_fallback = True
        llm_error_hint = (
            f"Could not reach the {selected_provider.name!r} provider at {selected_provider.base_url}. "
            "Check connectivity and the *_BASE_URL setting."
        )
    except Exception as exc:  # noqa: BLE001
        used_fallback = True
        llm_error_hint = f"Unexpected LLM error: {type(exc).__name__}."
        logger.exception("Unexpected LLM error during streaming")

    if used_fallback or not "".join(response_parts).strip():
        fallback = _fallback_answer(
            user_message=user_message,
            citations=citations,
            retrieval_confidence=retrieval["confidence"],
            medium_risk=safety.severity == "medium",
            research_error=research_error,
            llm_error_hint=llm_error_hint,
            intent=route_decision.intent,
            image_count=len(image_dicts),
        )
        response_parts = []
        async for token in _stream_text(fallback):
            response_parts.append(token)
            yield {"type": "token", "content": token}

    # ----- Polish, persist, and emit done ------------------------------------
    raw_answer = "".join(response_parts).strip()
    final_text = _polish_answer(raw_answer, citations)

    assistant_record = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=final_text,
        citations_json=citations,
    )
    # Persist images on the message so reloads still see them.
    if image_dicts and hasattr(assistant_record, "images_json"):
        assistant_record.images_json = image_dicts

    db.add(assistant_record)
    session.summary = _update_summary(session.summary or "", user_message, final_text)
    db.commit()
    db.refresh(session)
    db.refresh(assistant_record)

    message_payload = serialize_message(assistant_record)
    message_payload["images"] = image_dicts

    yield {
        "type": "done",
        "message": message_payload,
        "citations": citations,
        "session": serialize_session(session),
        "safety": safety.to_dict(),
        "route": {**route_decision.to_dict(), "resolved_route": resolved_route},
        "research": {
            "attempted": route_decision.needs_web_research,
            "count": len(web_citations),
            "error": research_error,
        },
        "images": {
            "attempted": route_decision.needs_image_search,
            "count": len(image_dicts),
            "error": image_error,
            "results": image_dicts,
        },
        "final_text": final_text,
    }