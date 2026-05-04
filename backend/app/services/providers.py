from __future__ import annotations

"""
LLM and embedding providers for ClarityAI - dual-Gemini edition.

Two LLM providers run side by side:

  - `provider`         (Provider 1, "easy turns")     -> Gemini 2.5 Flash
                                                         1,500 requests/day free
  - `heavy_provider`   (Provider 2, "hard turns")     -> Gemini 2.5 Pro
                                                         50 requests/day free

Both use the same `OpenAICompatibleProvider` class because Gemini exposes an
OpenAI-compatible Chat Completions endpoint at
https://generativelanguage.googleapis.com/v1beta/openai. Same wire format as
Groq, OpenAI, Together, etc., so the existing provider class works as-is.

Per-turn selection happens in `pick_provider()`. The chat_engine should call
that function instead of importing `provider` directly. The legacy `provider`
export still exists so older callers (and `retrieval.py`) keep working.

Why two providers and not one? Because Gemini Pro on the free tier is rate
limited to 50 requests/day TOTAL (across all your users combined). If we
sent every chat turn to Pro, the app would die after the 50th message of
the day. Routing easy turns to Flash (which gets 1,500/day) buys you ~30x
more headroom while keeping Pro-class quality on the questions that matter.
"""

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol

import httpx

from app.config import get_settings

if TYPE_CHECKING:
    # Only used for type hints - chat_engine passes us the route decision.
    from app.services.routing import RouteDecision

settings = get_settings()
logger = logging.getLogger(__name__)


class ProviderUnavailableError(RuntimeError):
    def __init__(self, message: str, reason: str = "unavailable") -> None:
        super().__init__(message)
        self.reason = reason


class LLMProvider(Protocol):
    """Minimal interface every chat provider must implement."""

    name: str

    @property
    def available(self) -> bool: ...

    def stream_chat(self, messages: list[dict]) -> AsyncIterator[str]: ...


# ---------------------------------------------------------------------------
# OpenAI-compatible provider (used for Groq, OpenAI, Gemini-via-compat, etc.)
# ---------------------------------------------------------------------------


class OpenAICompatibleProvider:
    """A single instance can target any OpenAI-shape endpoint.

    The class no longer reads settings from globals at construction time -
    instead it takes its config as constructor args, so we can stand up two
    instances pointing at different endpoints / models / keys.
    """

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.3,
        configured: bool = True,
    ) -> None:
        self.name = name
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self._configured = configured

    @property
    def available(self) -> bool:
        return bool(self._configured and self.base_url and self.model)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _extract_delta(payload: dict) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        content = delta.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if item.get("type") == "text" and item.get("text"):
                    parts.append(item["text"])
            return "".join(parts)
        return ""

    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        if not self.available:
            raise ProviderUnavailableError(
                f"Provider {self.name!r} is not configured. "
                f"Check the relevant *_BASE_URL / *_API_KEY / *_MODEL env vars.",
                reason="not_configured",
            )

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        timeout = httpx.Timeout(connect=20.0, read=None, write=20.0, pool=20.0)

        # Retry only the *connect* phase; once we're streaming we can't safely retry.
        max_attempts = 3
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST", url, headers=self._headers(), json=payload
                    ) as response:
                        if response.status_code >= 400:
                            # Drain a snippet of the body for debugging then re-raise.
                            body = (await response.aread()).decode("utf-8", errors="replace")[:400]
                            logger.warning(
                                "Provider %r returned HTTP %s: %s",
                                self.name, response.status_code, body,
                            )
                            response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            if line.startswith("data:"):
                                data = line[5:].strip()
                            else:
                                continue
                            if not data:
                                continue
                            if data == "[DONE]":
                                return
                            try:
                                event = json.loads(data)
                            except json.JSONDecodeError:
                                # Some providers occasionally emit partial frames.
                                continue
                            delta = self._extract_delta(event)
                            if delta:
                                yield delta
                        return
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                last_exc = exc
                if attempt < max_attempts:
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
                    continue
                raise ProviderUnavailableError(
                    f"Could not reach {self.name!r} at {self.base_url}: {type(exc).__name__}.",
                    reason="connect_failed",
                ) from exc
            except httpx.HTTPStatusError:
                # Don't swallow - chat engine maps status codes to human hints.
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.exception("Unexpected error while streaming from %r", self.name)
                raise ProviderUnavailableError(
                    f"Unexpected error from {self.name!r}: {type(exc).__name__}.",
                    reason="unexpected",
                ) from exc

        if last_exc:
            raise ProviderUnavailableError(
                f"{self.name!r} unreachable after retries.", reason="connect_failed"
            ) from last_exc


# ---------------------------------------------------------------------------
# Embeddings (OpenAI-compatible) - unchanged from the original
# ---------------------------------------------------------------------------


class EmbeddingProvider:
    def __init__(self) -> None:
        # Embeddings reuse the LLM provider's base/key by default. Gemini's
        # embeddings endpoint is the same OpenAI-compat host, so this works.
        self.base_url = (settings.embedding_base_url or settings.llm_base_url).rstrip("/")
        self.api_key = settings.embedding_api_key or settings.llm_api_key
        self.model = settings.embedding_model
        self.batch_size = max(1, settings.embedding_batch_size)

    @property
    def available(self) -> bool:
        return settings.embedding_provider_configured

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.available:
            raise ProviderUnavailableError(
                "Embedding provider is not configured.",
                reason="not_configured",
            )
        if not texts:
            return []

        url = f"{self.base_url}/embeddings"
        timeout = httpx.Timeout(60.0, connect=20.0)
        vectors: list[list[float]] = []

        with httpx.Client(timeout=timeout) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                payload = {"model": self.model, "input": batch}

                # Retry transient failures up to 3 times.
                for attempt in range(1, 4):
                    try:
                        response = client.post(url, headers=self._headers(), json=payload)
                        response.raise_for_status()
                        break
                    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
                        if attempt == 3:
                            raise
                        import time
                        time.sleep(0.5 * (2 ** (attempt - 1)))
                    except httpx.HTTPStatusError as exc:
                        status = exc.response.status_code if exc.response is not None else None
                        if status in (429, 500, 502, 503, 504) and attempt < 3:
                            import time
                            time.sleep(0.7 * (2 ** (attempt - 1)))
                            continue
                        raise

                data = response.json().get("data") or []
                # Force ordering by 'index' since some providers don't guarantee it.
                ordered = sorted(data, key=lambda item: item.get("index", 0))
                for item in ordered:
                    vectors.append(item.get("embedding") or [])

        if len(vectors) != len(texts):
            raise RuntimeError("Embedding provider returned an unexpected number of vectors.")
        return vectors


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

# "Easy" provider - the everyday workhorse. Wired to whatever LLM_BASE_URL /
# LLM_API_KEY / CHAT_MODEL point at. Default config in .env.example points
# this at Gemini 2.5 Flash, which gets 1,500 requests/day free.
provider = OpenAICompatibleProvider(
    name="easy",
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    model=settings.chat_model,
    temperature=settings.temperature,
    configured=settings.remote_llm_configured,
)

# "Heavy" provider - reserved for hard turns. Wired to HEAVY_LLM_BASE_URL /
# HEAVY_LLM_API_KEY / HEAVY_CHAT_MODEL. Default config points this at Gemini
# 2.5 Pro, which gets 50 requests/day free. If you only configure the easy
# provider, hard turns simply route to it (no error).
heavy_provider = OpenAICompatibleProvider(
    name="heavy",
    base_url=settings.heavy_llm_base_url,
    api_key=settings.heavy_llm_api_key,
    model=settings.heavy_chat_model,
    temperature=settings.temperature,
    configured=settings.heavy_llm_configured,
)

embedding_provider = EmbeddingProvider()


# ---------------------------------------------------------------------------
# Per-turn provider selection ("Pro for hard turns, Flash for easy")
# ---------------------------------------------------------------------------

# Keywords that strongly imply the model needs to do real reasoning.
_HARD_KEYWORDS = re.compile(
    r"\b("
    r"explain why|prove|derive|step[- ]by[- ]step|walk me through|"
    r"refactor|review (?:my )?(?:code|design|architecture)|"
    r"debug|stack trace|traceback|"
    r"compare and contrast|tradeoffs?|pros and cons|"
    r"design a|architect|propose|"
    r"summarize this|critique|analyze this"
    r")\b",
    re.IGNORECASE,
)

# Intents that always go to the heavy model when it's available.
_HARD_INTENTS = {"code", "math", "creative", "research"}

# Intents that should always go to the cheap/fast model.
_EASY_INTENTS = {"chat", "self_identity"}


def _is_hard_turn(
    *,
    user_message: str,
    route_decision: "RouteDecision | None",
    mode: str | None,
) -> bool:
    """Heuristic: does this turn deserve a Pro-tier call?

    True signals (any one is enough):
      - User chose "deep" mode.
      - Route resolved to research/hybrid (we have to synthesize sources).
      - Intent is code/math/creative/research.
      - Message is long (>= 600 chars) and not a casual chat intent.
      - Message contains hard-reasoning keywords.

    False signals (force easy provider):
      - Intent is casual chat / self-identity (overrides everything).

    Why the bias toward False? On Gemini's free tier, Pro is capped at 50
    requests/day across the whole app. We want to use that budget for the
    questions where the quality jump actually matters.
    """
    if mode == "deep":
        return True

    intent = (route_decision.intent if route_decision else "general") or "general"
    if intent in _EASY_INTENTS:
        return False

    if intent in _HARD_INTENTS:
        return True

    if route_decision and route_decision.route in ("research", "hybrid"):
        return True

    text = user_message or ""
    if len(text) >= 600:
        return True

    if _HARD_KEYWORDS.search(text):
        return True

    return False


def pick_provider(
    *,
    user_message: str,
    route_decision: "RouteDecision | None" = None,
    mode: str | None = None,
) -> "LLMProvider":
    """Return the right LLM provider for this turn.

    Decision tree:
      1. If only one provider is configured → use that one.
      2. If both are configured → heavy for "hard" turns, easy otherwise.
      3. If neither → return the easy one and let its existing
         "not configured" error path run.
    """
    easy_ready = provider.available
    heavy_ready = heavy_provider.available

    if not heavy_ready and not easy_ready:
        return provider  # will raise its existing "not configured" error
    if not heavy_ready:
        return provider
    if not easy_ready:
        return heavy_provider

    hard = _is_hard_turn(
        user_message=user_message,
        route_decision=route_decision,
        mode=mode,
    )
    chosen = heavy_provider if hard else provider
    logger.info(
        "Provider routing: %s (model=%s, hard=%s, intent=%s, mode=%s, len=%d)",
        chosen.name,
        chosen.model,
        hard,
        getattr(route_decision, "intent", None),
        mode,
        len(user_message or ""),
    )
    return chosen