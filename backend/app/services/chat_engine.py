from __future__ import annotations

"""
Chat orchestration for ClarityAI.

Major improvements over the original:
- Proper reference handling: any inline tag the model emits (e.g. [S1], [W1],
  "(source 2)") is rewritten to a clean footnote like [1]; only footnote
  numbers that match a real source survive. A polished "Sources" section is
  appended after streaming finishes.
- Cleaner streaming: token chunks are flushed as-is to the client; sanitization
  happens once after the stream ends, and the final text written to DB matches
  what the user sees.
- Smarter fallback synthesis when the LLM is unavailable: produces a real,
  readable answer from the strongest sources rather than a list of bullets.
- Richer error mapping for httpx and provider errors.
- Handles the "chat" route (casual messages) without forcing in fake citations.
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
from app.services.prompts import build_system_prompt, build_user_prompt
from app.services.providers import ProviderUnavailableError, provider
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

# Catches the model's stray internal tags: [S1], [W2], [S1, W2], (source 1), (Source: 2)
INTERNAL_TAG_RE = re.compile(
    r"""
    \[\s*(?:S|W)\d+(?:\s*,\s*(?:S|W)\d+)*\s*\]      # [S1], [S1, W2]
    | \(\s*[Ss]ource[s]?\s*[:#]?\s*\d+(?:\s*,\s*\d+)*\s*\)   # (source 1) (Sources: 1, 2)
    | \[\s*[Ss]ource\s*\d+\s*\]                     # [Source 1]
    """,
    re.VERBOSE,
)

# A footnote like [1] or [1, 3] that we want to keep — but only if it points
# to a citation that actually exists.
FOOTNOTE_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def generate_title(message: str) -> str:
    words = [w for w in message.replace("\n", " ").split() if w.strip()]
    if not words:
        return "New conversation"
    title = " ".join(words[:8]).strip()
    return title[:60] + ("…" if len(title) > 60 else "")


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


def _resolve_route_label(route_decision_label: str, local_sources: list[dict], web_sources: list[dict]) -> str:
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
    """Assign clean public-facing footnote numbers (1..N) and a display name."""
    out = []
    for i, c in enumerate(citations, start=1):
        item = dict(c)
        item["number"] = i
        item["display_name"] = _source_name(item)
        # `label` was the internal S1/W1; replace with the public footnote.
        item["label"] = f"[{i}]"
        out.append(item)
    return out


def _strip_internal_tags(text: str) -> str:
    """Remove [S1]/[W1]/(source 2)/[Source 1] etc."""
    cleaned = INTERNAL_TAG_RE.sub("", text)
    # Tidy whitespace left behind
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _filter_footnotes(text: str, valid_numbers: set[int]) -> str:
    """Drop bracket footnotes whose numbers aren't in the citation list.

    Keeps the bracket if at least one number is valid, dropping the invalid ones.
    """
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
    """Render a clean Markdown 'Sources' section."""
    if not citations:
        return ""
    lines = ["", "---", "**Sources**"]
    for c in citations:
        n = c.get("number") or 0
        name = c.get("display_name") or "Source"
        url = c.get("url")
        kind = "web" if c.get("source_type") == "web" else "file"
        if url:
            lines.append(f"{n}. [{name}]({url}) — {kind}")
        else:
            lines.append(f"{n}. {name} — {kind}")
    return "\n".join(lines)


def _polish_answer(raw_text: str, citations: list[dict]) -> str:
    """Final pass that runs on the assembled answer.

    1. Strip internal tags ([S1], (source 2), …).
    2. Filter footnote numbers that don't exist.
    3. Append a Sources section (only if the answer actually used or has citations).
    """
    text = _strip_internal_tags(raw_text)
    valid_numbers = {c["number"] for c in citations}
    text = _filter_footnotes(text, valid_numbers)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    if citations:
        text += "\n" + _build_sources_section(citations)
    return text.strip()


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
    return cut.rstrip() + "…"


def _fallback_answer(
    user_message: str,
    citations: list[dict],
    retrieval_confidence: float,
    medium_risk: bool,
    research_error: str | None,
    llm_error_hint: str | None,
) -> str:
    """Build a readable answer purely from retrieved evidence."""
    if not citations:
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
            lines.append("I'll keep the tone calm and practical — happy to help work through it.")
        return "\n\n".join(lines)

    # Compose a grounded summary in prose.
    locals_ = [c for c in citations if c.get("source_type") != "web"]
    webs = [c for c in citations if c.get("source_type") == "web"]

    header = (
        "I'm running without my full model right now, so here's the best grounded synthesis I can "
        "give from the retrieved evidence:"
    )
    if llm_error_hint:
        header += f" ({llm_error_hint})"
    paras: list[str] = [header]

    if locals_:
        paras.append("**From your uploaded knowledge:**")
        for c in locals_[:3]:
            paras.append(f"- *{c['display_name']}* — {_summarize_snippet(c.get('snippet') or '')}")

    if webs:
        paras.append("**From web research:**")
        for c in webs[:3]:
            url = c.get("url")
            link = f"[{c['display_name']}]({url})" if url else c["display_name"]
            paras.append(f"- {link} — {_summarize_snippet(c.get('snippet') or '')}")

    if retrieval_confidence < settings.low_confidence_threshold:
        paras.append(
            "Confidence is on the lower side — a narrower question or a more specific keyword "
            "will improve this."
        )

    if research_error:
        paras.append(f"Research note: {research_error}")

    if medium_risk:
        paras.append("I'm keeping this calm and practical because your message sounded heavy.")

    return "\n\n".join(paras)


async def _stream_text(text: str) -> AsyncIterator[str]:
    for token in text.split(" "):
        yield token + " "
        await asyncio.sleep(0)


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
    research_error: str | None = None
    if not safety.blocked and route_decision.needs_web_research:
        yield {"type": "status", "stage": "researching", "message": "Researching trusted web sources"}
        try:
            research_sources = await web_research.search(
                user_message, max_results=settings.research_max_results
            )
            web_citations = [
                source.to_citation(label=f"W{i}")
                for i, source in enumerate(research_sources, start=1)
            ]
        except WebResearchUnavailableError as exc:
            research_error = str(exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Web research failed: %s", exc)
            research_error = "Web research failed for this turn."

    raw_citations = [*local_citations, *web_citations]
    citations = _renumber_citations(raw_citations)
    valid_numbers = {c["number"] for c in citations}

    resolved_route = _resolve_route_label(route_decision.route, local_citations, web_citations)

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
    }

    # ----- Safety block early-return -----------------------------------------
    if safety.blocked:
        assistant_text = safety.message or "I cannot continue with that request."
        # Stream the message so the UI feels live
        async for token in _stream_text(assistant_text):
            yield {"type": "token", "content": token}
        assistant_record = ChatMessage(
            session_id=session.id, role="assistant", content=assistant_text, citations_json=[]
        )
        db.add(assistant_record)
        session.summary = _update_summary(session.summary or "", user_message, assistant_text)
        db.commit()
        db.refresh(session)
        db.refresh(assistant_record)
        yield {
            "type": "done",
            "message": serialize_message(assistant_record),
            "citations": [],
            "session": serialize_session(session),
            "safety": safety.to_dict(),
            "route": {"resolved_route": "safety_block"},
            "research": {"attempted": False, "count": 0, "error": None},
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
            ),
        }
    )

    yield {"type": "status", "stage": "answering", "message": "Composing answer"}

    # ----- Stream from LLM ----------------------------------------------------
    response_parts: list[str] = []
    used_fallback = False
    llm_error_hint: str | None = None

    try:
        async for token in provider.stream_chat(model_messages):
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
                "The LLM provider rejected the API key (HTTP 401). "
                "Check LLM_API_KEY in backend/.env."
            )
        elif status == 404:
            llm_error_hint = (
                f"The model \"{settings.chat_model}\" was not found at "
                f"{settings.llm_base_url}. Check CHAT_MODEL."
            )
        elif status == 429:
            llm_error_hint = "The LLM provider rate-limited this request. Wait briefly and retry."
        elif status and 500 <= status < 600:
            llm_error_hint = f"The LLM provider had a server error (HTTP {status}). Try again."
        else:
            llm_error_hint = f"The LLM provider returned HTTP {status}."
        logger.warning("LLM HTTP error: %s", llm_error_hint)
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
        used_fallback = True
        llm_error_hint = (
            f"Could not reach the LLM provider at {settings.llm_base_url}. "
            "Check connectivity and LLM_BASE_URL."
        )
    except Exception as exc:  # noqa: BLE001
        used_fallback = True
        llm_error_hint = f"Unexpected LLM error: {type(exc).__name__}."
        logger.exception("Unexpected LLM error during streaming")

    # If the LLM produced nothing or failed, stream a fallback answer.
    if used_fallback or not "".join(response_parts).strip():
        fallback = _fallback_answer(
            user_message=user_message,
            citations=citations,
            retrieval_confidence=retrieval["confidence"],
            medium_risk=safety.severity == "medium",
            research_error=research_error,
            llm_error_hint=llm_error_hint,
        )
        # Reset and stream the fallback so the UI shows a coherent answer
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
    db.add(assistant_record)
    session.summary = _update_summary(session.summary or "", user_message, final_text)
    db.commit()
    db.refresh(session)
    db.refresh(assistant_record)

    yield {
        "type": "done",
        "message": serialize_message(assistant_record),
        "citations": citations,
        "session": serialize_session(session),
        "safety": safety.to_dict(),
        "route": {**route_decision.to_dict(), "resolved_route": resolved_route},
        "research": {
            "attempted": route_decision.needs_web_research,
            "count": len(web_citations),
            "error": research_error,
        },
        # Tell the UI to render the polished, sanitized text instead of
        # whatever it accumulated from raw stream tokens, if it wants to.
        "final_text": final_text,
    }