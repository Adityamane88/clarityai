from __future__ import annotations

import asyncio
import json
import logging
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


def generate_title(message: str) -> str:
    words = [word for word in message.replace('\n', ' ').split() if word.strip()]
    if not words:
        return 'New conversation'
    title = ' '.join(words[:8]).strip()
    return title[:60] + ('...' if len(title) > 60 else '')



def sse_event(event_name: str, payload: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"



def _build_history(messages: list[ChatMessage], char_budget: int) -> list[dict]:
    selected: list[dict] = []
    total = 0
    for message in reversed(messages):
        length = len(message.content)
        if total + length > char_budget and selected:
            break
        selected.append({'role': message.role, 'content': message.content})
        total += length
    return list(reversed(selected))



def _update_summary(existing_summary: str, user_text: str, assistant_text: str) -> str:
    existing_lines = [line.strip() for line in existing_summary.splitlines() if line.strip()]
    new_line = f'- User asked: {user_text[:160]}'
    assistant_line = f'- Assistant answered: {assistant_text[:200]}'
    merged = existing_lines[-6:] + [new_line, assistant_line]
    return '\n'.join(merged[-8:])



def _resolve_route_label(local_sources: list[dict], web_sources: list[dict]) -> str:
    if local_sources and web_sources:
        return 'hybrid'
    if web_sources:
        return 'research'
    return 'local'



def _fallback_answer(
    user_message: str,
    local_sources: list[dict],
    web_sources: list[dict],
    retrieval_confidence: float,
    medium_risk: bool,
    research_error: str | None,
    llm_error_hint: str | None = None,
) -> str:
    combined = [*local_sources[:3], *web_sources[:3]]
    if not combined:
        base = (
            'I cannot give a real answer right now because I do not have an LLM connected '
            'and there is no strong match in the uploaded knowledge.'
        )
        if llm_error_hint:
            base += f' {llm_error_hint}'
        if research_error:
            base += f' Web research was not available: {research_error}'
        if not llm_error_hint and not research_error:
            base += ' Add knowledge files in the right panel, or set LLM_API_KEY in backend/.env.'
        if medium_risk:
            base += ' I can still respond in a calm, supportive way once that is set up.'
        return base

    lines = ['Best grounded answer I can give right now (LLM is offline, this is direct from sources):']
    if local_sources:
        lines.append('')
        lines.append('From uploaded knowledge:')
        for source in local_sources[:3]:
            page_text = f" on page {source['page_label']}" if source.get('page_label') else ''
            lines.append(f"- [{source['label']}] {source['document_title']}{page_text}: {source['snippet']}")
    if web_sources:
        lines.append('')
        lines.append('From researched web sources:')
        for source in web_sources[:3]:
            domain = source.get('source_name') or 'web'
            lines.append(f"- [{source['label']}] {source['document_title']} ({domain}): {source['snippet']}")
    lines.append('')
    if llm_error_hint:
        lines.append(f'LLM note: {llm_error_hint}')
    if retrieval_confidence < settings.low_confidence_threshold:
        lines.append('Confidence note: the evidence is still weak; a narrower follow-up would improve the answer.')
    else:
        lines.append('Confidence note: the answer is grounded in the sources above.')
    if research_error:
        lines.append(f'Research note: {research_error}')
    if medium_risk:
        lines.append('Tone note: I will keep the conversation supportive and practical.')
    return '\n'.join(lines)


async def _stream_text(text: str) -> AsyncIterator[str]:
    for token in text.split(' '):
        yield token + ' '
        await asyncio.sleep(0)


async def stream_chat_reply(
    db: Session,
    session_id: str | None,
    user_message: str,
    mode: str,
    research_mode: str,
) -> AsyncIterator[dict]:
    session = db.get(ChatSession, session_id) if session_id else None
    if session is None:
        session = ChatSession(title=generate_title(user_message))
        db.add(session)
        db.flush()
    elif session.title == 'New conversation':
        session.title = generate_title(user_message)

    user_record = ChatMessage(session_id=session.id, role='user', content=user_message, citations_json=[])
    db.add(user_record)
    db.flush()

    yield {'type': 'status', 'stage': 'retrieving', 'message': 'Searching your knowledge base'}

    safety = assess_safety(user_message)
    recent_messages = db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id, ChatMessage.id != user_record.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(settings.max_history_messages)
    ).scalars().all()
    recent_messages = list(reversed(recent_messages))

    recent_user_context = ' '.join(message.content for message in recent_messages if message.role == 'user')[-1200:]
    retrieval = retrieval_index.search(query=user_message, conversation_context=recent_user_context)
    local_citations = retrieval['results']

    route_decision = choose_route(
        user_message=user_message,
        local_confidence=retrieval['confidence'],
        local_hits=len(local_citations),
        research_mode=research_mode,
    )

    web_citations: list[dict] = []
    research_error: str | None = None
    if not safety.blocked and route_decision.needs_web_research:
        yield {'type': 'status', 'stage': 'researching', 'message': 'Researching trusted web sources'}
        try:
            research_sources = await web_research.search(user_message, max_results=settings.research_max_results)
            web_citations = [source.to_citation(label=f'W{index}') for index, source in enumerate(research_sources, start=1)]
        except WebResearchUnavailableError as exc:
            research_error = str(exc)
        except Exception:
            research_error = 'Web research failed for this turn.'

    resolved_route = _resolve_route_label(local_citations, web_citations)
    citations = [*local_citations, *web_citations]

    yield {
        'type': 'meta',
        'session': serialize_session(session),
        'citations': citations,
        'search': {
            'confidence': retrieval['confidence'],
            'hits': len(local_citations),
            'dense_used': retrieval.get('dense_used', False),
        },
        'safety': safety.to_dict(),
        'route': {
            **route_decision.to_dict(),
            'resolved_route': resolved_route,
        },
        'research': {
            'attempted': route_decision.needs_web_research,
            'count': len(web_citations),
            'error': research_error,
        },
    }

    if safety.blocked:
        assistant_text = safety.message or 'I cannot continue with that request.'
        assistant_record = ChatMessage(
            session_id=session.id,
            role='assistant',
            content=assistant_text,
            citations_json=[],
        )
        db.add(assistant_record)
        session.summary = _update_summary(session.summary or '', user_message, assistant_text)
        db.commit()
        for token in assistant_text.split(' '):
            yield {'type': 'token', 'content': token + ' '}
        yield {
            'type': 'done',
            'message': serialize_message(assistant_record),
            'citations': [],
            'session': serialize_session(session),
            'safety': safety.to_dict(),
            'route': {'resolved_route': 'safety_block'},
            'research': {'attempted': False, 'count': 0, 'error': None},
        }
        return

    model_messages = [
        {
            'role': 'system',
            'content': build_system_prompt(mode=mode, route=resolved_route, medium_risk=safety.severity == 'medium'),
        }
    ]
    model_messages.extend(_build_history(recent_messages, char_budget=settings.history_char_budget))
    model_messages.append(
        {
            'role': 'user',
            'content': build_user_prompt(
                user_message=user_message,
                local_sources=local_citations,
                web_sources=web_citations,
                retrieval_confidence=retrieval['confidence'],
                session_summary=session.summary or '',
                route_reason=route_decision.reason,
            ),
        }
    )

    yield {'type': 'status', 'stage': 'answering', 'message': 'Composing answer'}

    response_parts: list[str] = []
    used_fallback = False
    llm_error_hint: str | None = None
    try:
        async for token in provider.stream_chat(model_messages):
            response_parts.append(token)
            yield {'type': 'token', 'content': token}
    except ProviderUnavailableError as exc:
        used_fallback = True
        llm_error_hint = str(exc)
        logger.warning('LLM provider unavailable: %s', exc)
    except httpx.HTTPStatusError as exc:
        used_fallback = True
        status = exc.response.status_code if exc.response is not None else None
        if status == 401:
            llm_error_hint = (
                'The LLM provider rejected the API key (HTTP 401). '
                'Check LLM_API_KEY in backend/.env.'
            )
        elif status == 404:
            llm_error_hint = (
                f'The LLM provider returned 404. The model "{settings.chat_model}" may not exist '
                f'at {settings.llm_base_url}. Check CHAT_MODEL in backend/.env.'
            )
        elif status == 429:
            llm_error_hint = 'The LLM provider rate-limited this request. Wait a moment and retry.'
        else:
            llm_error_hint = f'The LLM provider returned HTTP {status}.'
        logger.warning('LLM HTTP error: %s', llm_error_hint)
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
        used_fallback = True
        llm_error_hint = (
            f'Could not reach the LLM provider at {settings.llm_base_url}. '
            f'Check your internet connection and LLM_BASE_URL.'
        )
        logger.warning('LLM connection error: %s', exc)
    except Exception as exc:  # noqa: BLE001
        used_fallback = True
        llm_error_hint = f'Unexpected LLM error: {type(exc).__name__}.'
        logger.exception('Unexpected LLM error during streaming')

    if used_fallback or not ''.join(response_parts).strip():
        fallback = _fallback_answer(
            user_message=user_message,
            local_sources=local_citations,
            web_sources=web_citations,
            retrieval_confidence=retrieval['confidence'],
            medium_risk=safety.severity == 'medium',
            research_error=research_error,
            llm_error_hint=llm_error_hint,
        )
        response_parts = []
        async for token in _stream_text(fallback):
            response_parts.append(token)
            yield {'type': 'token', 'content': token}

    assistant_text = ''.join(response_parts).strip()
    assistant_record = ChatMessage(
        session_id=session.id,
        role='assistant',
        content=assistant_text,
        citations_json=citations,
    )
    db.add(assistant_record)
    session.summary = _update_summary(session.summary or '', user_message, assistant_text)
    db.commit()
    db.refresh(session)
    db.refresh(assistant_record)

    yield {
        'type': 'done',
        'message': serialize_message(assistant_record),
        'citations': citations,
        'session': serialize_session(session),
        'safety': safety.to_dict(),
        'route': {
            **route_decision.to_dict(),
            'resolved_route': resolved_route,
        },
        'research': {
            'attempted': route_decision.needs_web_research,
            'count': len(web_citations),
            'error': research_error,
        },
    }
