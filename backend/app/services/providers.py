from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from app.config import get_settings

settings = get_settings()


class ProviderUnavailableError(RuntimeError):
    pass


class OpenAICompatibleProvider:
    def __init__(self) -> None:
        self.base_url = settings.llm_base_url
        self.api_key = settings.llm_api_key
        self.model = settings.chat_model
        self.temperature = settings.temperature

    @property
    def available(self) -> bool:
        return settings.remote_llm_configured

    def _headers(self) -> dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    @staticmethod
    def _extract_delta(payload: dict) -> str:
        choices = payload.get('choices') or []
        if not choices:
            return ''
        delta = choices[0].get('delta') or {}
        content = delta.get('content', '')
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if item.get('type') == 'text' and item.get('text'):
                    parts.append(item['text'])
            return ''.join(parts)
        return ''

    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        if not self.available:
            raise ProviderUnavailableError('Remote LLM is not configured.')
        url = f'{self.base_url}/chat/completions'
        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': self.temperature,
            'stream': True,
        }
        timeout = httpx.Timeout(connect=20.0, read=None, write=20.0, pool=20.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream('POST', url, headers=self._headers(), json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == '[DONE]':
                        break
                    payload = json.loads(data)
                    delta = self._extract_delta(payload)
                    if delta:
                        yield delta


class EmbeddingProvider:
    def __init__(self) -> None:
        self.base_url = settings.embedding_base_url or settings.llm_base_url
        self.api_key = settings.embedding_api_key or settings.llm_api_key
        self.model = settings.embedding_model
        self.batch_size = settings.embedding_batch_size

    @property
    def available(self) -> bool:
        return settings.embedding_provider_configured

    def _headers(self) -> dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.available:
            raise ProviderUnavailableError('Embedding provider is not configured.')
        url = f'{self.base_url}/embeddings'
        timeout = httpx.Timeout(60.0, connect=20.0)
        vectors: list[list[float]] = []
        with httpx.Client(timeout=timeout) as client:
            for start in range(0, len(texts), max(1, self.batch_size)):
                batch = texts[start:start + max(1, self.batch_size)]
                payload = {
                    'model': self.model,
                    'input': batch,
                }
                response = client.post(url, headers=self._headers(), json=payload)
                response.raise_for_status()
                data = response.json().get('data') or []
                ordered = sorted(data, key=lambda item: item.get('index', 0))
                for item in ordered:
                    vectors.append(item.get('embedding') or [])
        if len(vectors) != len(texts):
            raise RuntimeError('Embedding provider returned an unexpected number of vectors.')
        return vectors


provider = OpenAICompatibleProvider()
embedding_provider = EmbeddingProvider()
