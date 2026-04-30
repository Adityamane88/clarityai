from __future__ import annotations

"""
LLM and embedding providers for ClarityAI.

Improvements over the original:
- Async retry with exponential backoff on transient errors (5xx, timeouts).
- Structured ProviderUnavailableError with a 'reason' code so the chat engine
  can give the user a precise hint instead of a generic message.
- Streaming JSON parser is tolerant of malformed lines.
- Embedding provider batches and returns a stable order even if the API
  doesn't preserve the input order.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator

import httpx

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class ProviderUnavailableError(RuntimeError):
    def __init__(self, message: str, reason: str = "unavailable") -> None:
        super().__init__(message)
        self.reason = reason


# ---------------------------------------------------------------------------
# Chat (OpenAI-compatible)
# ---------------------------------------------------------------------------


class OpenAICompatibleProvider:
    def __init__(self) -> None:
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.chat_model
        self.temperature = settings.temperature

    @property
    def available(self) -> bool:
        return settings.remote_llm_configured

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
                "Remote LLM is not configured. Set LLM_BASE_URL and LLM_API_KEY.",
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
                    async with client.stream("POST", url, headers=self._headers(), json=payload) as response:
                        if response.status_code >= 400:
                            body = (await response.aread()).decode("utf-8", errors="replace")[:400]
                            response.raise_for_status()  # raises HTTPStatusError with status info
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
                    f"Could not reach LLM at {self.base_url}: {type(exc).__name__}.",
                    reason="connect_failed",
                ) from exc
            except httpx.HTTPStatusError:
                # Don't swallow — chat engine maps status codes to human hints.
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.exception("Unexpected error while streaming from LLM")
                raise ProviderUnavailableError(
                    f"Unexpected LLM error: {type(exc).__name__}.",
                    reason="unexpected",
                ) from exc

        if last_exc:
            raise ProviderUnavailableError("LLM unreachable after retries.", reason="connect_failed") from last_exc


# ---------------------------------------------------------------------------
# Embeddings (OpenAI-compatible)
# ---------------------------------------------------------------------------


class EmbeddingProvider:
    def __init__(self) -> None:
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
                        # Backoff before retry
                        import time
                        time.sleep(0.5 * (2 ** (attempt - 1)))
                    except httpx.HTTPStatusError as exc:
                        # Retry only on 5xx / 429
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


provider = OpenAICompatibleProvider()
embedding_provider = EmbeddingProvider()