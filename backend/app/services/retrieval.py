from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import KnowledgeChunk, KnowledgeDocument
from app.services.providers import ProviderUnavailableError, embedding_provider

settings = get_settings()


@dataclass(slots=True)
class IndexedChunk:
    chunk_id: int
    document_id: str
    document_title: str
    source_name: str
    content: str
    page_label: str | None
    chunk_index: int


class RetrievalIndex:
    def __init__(self) -> None:
        self._lock = RLock()
        self._items: list[IndexedChunk] = []
        self._word_vectorizer: TfidfVectorizer | None = None
        self._word_matrix = None
        self._char_vectorizer: TfidfVectorizer | None = None
        self._char_matrix = None
        self._dense_matrix: np.ndarray | None = None
        self._dense_enabled = False

    def rebuild(self, db: Session) -> dict:
        rows = db.execute(
            select(KnowledgeChunk, KnowledgeDocument)
            .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
            .order_by(KnowledgeDocument.created_at.desc(), KnowledgeChunk.chunk_index.asc())
        ).all()
        with self._lock:
            self._items = [
                IndexedChunk(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    document_title=document.title,
                    source_name=document.source_name,
                    content=chunk.content,
                    page_label=chunk.page_label,
                    chunk_index=chunk.chunk_index,
                )
                for chunk, document in rows
            ]
            if not self._items:
                self._word_vectorizer = None
                self._word_matrix = None
                self._char_vectorizer = None
                self._char_matrix = None
                self._dense_matrix = None
                self._dense_enabled = False
                return {'status': 'empty', 'chunks': 0, 'dense': 'disabled'}
            texts = [item.content for item in self._items]
            self._word_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=50000)
            self._char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=40000)
            self._word_matrix = self._word_vectorizer.fit_transform(texts)
            self._char_matrix = self._char_vectorizer.fit_transform(texts)
            dense_status = 'disabled'
            self._dense_enabled = False
            self._dense_matrix = None
            if settings.enable_dense_retrieval and embedding_provider.available:
                try:
                    vectors = embedding_provider.embed_texts(texts)
                    dense = np.array(vectors, dtype=np.float32)
                    if dense.size:
                        dense = self._normalize(dense)
                        self._dense_matrix = dense
                        self._dense_enabled = True
                        dense_status = 'enabled'
                except (ProviderUnavailableError, Exception):
                    self._dense_matrix = None
                    self._dense_enabled = False
                    dense_status = 'unavailable'
            return {'status': 'ok', 'chunks': len(self._items), 'dense': dense_status}

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _dense_scores(self, query: str, expanded_query: str) -> np.ndarray | None:
        if not self._dense_enabled or self._dense_matrix is None or not embedding_provider.available:
            return None
        try:
            vector = np.array(embedding_provider.embed_texts([expanded_query or query])[0], dtype=np.float32)
            vector = self._normalize(vector.reshape(1, -1)).ravel()
            return self._dense_matrix @ vector
        except (ProviderUnavailableError, Exception):
            return None

    def search(self, query: str, conversation_context: str = '', k: int | None = None, candidate_pool: int | None = None) -> dict:
        with self._lock:
            if not self._items or self._word_vectorizer is None or self._char_vectorizer is None:
                return {'results': [], 'confidence': 0.0, 'dense_used': False}
            k = k or settings.retrieval_top_k
            candidate_pool = candidate_pool or settings.retrieval_candidate_pool
            expanded_query = ' '.join(part for part in [query, conversation_context] if part).strip() or query
            word_query = self._word_vectorizer.transform([expanded_query])
            char_query = self._char_vectorizer.transform([query])
            word_scores = cosine_similarity(word_query, self._word_matrix).ravel()
            char_scores = cosine_similarity(char_query, self._char_matrix).ravel()
            dense_scores = self._dense_scores(query=query, expanded_query=expanded_query)
            query_terms = {term for term in query.lower().split() if len(term) > 2}
            combined_scores = []
            for idx, item in enumerate(self._items):
                title_terms = set(item.document_title.lower().split())
                title_overlap = len(query_terms & title_terms)
                title_boost = min(0.12, 0.03 * title_overlap)
                exact_phrase_boost = 0.08 if query.lower() in item.content.lower() else 0.0
                dense_component = float(dense_scores[idx]) if dense_scores is not None else 0.0
                score = (
                    (0.52 * float(word_scores[idx]))
                    + (0.14 * float(char_scores[idx]))
                    + (0.26 * dense_component)
                    + title_boost
                    + exact_phrase_boost
                )
                combined_scores.append(score)
            ranked_indices = np.argsort(np.array(combined_scores))[::-1][:candidate_pool]
            selected: list[int] = []
            doc_counts: dict[str, int] = {}
            for idx in ranked_indices:
                score = combined_scores[idx]
                if score <= 0:
                    continue
                item = self._items[idx]
                if doc_counts.get(item.document_id, 0) >= 2:
                    continue
                snippet_key = item.content[:120].strip().lower()
                if any(self._items[prev].content[:120].strip().lower() == snippet_key for prev in selected):
                    continue
                selected.append(int(idx))
                doc_counts[item.document_id] = doc_counts.get(item.document_id, 0) + 1
                if len(selected) >= k:
                    break
            results = []
            for rank, idx in enumerate(selected, start=1):
                item = self._items[idx]
                score = float(combined_scores[idx])
                results.append(
                    {
                        'id': f'S{rank}',
                        'label': f'S{rank}',
                        'chunk_id': item.chunk_id,
                        'document_id': item.document_id,
                        'document_title': item.document_title,
                        'source_name': item.source_name,
                        'page_label': item.page_label,
                        'snippet': item.content[:340].strip(),
                        'content': item.content,
                        'score': round(score, 4),
                        'source_type': 'knowledge',
                        'url': None,
                    }
                )
            confidence = round(float(max(combined_scores)) if combined_scores else 0.0, 4)
            return {'results': results, 'confidence': confidence, 'dense_used': dense_scores is not None}


retrieval_index = RetrievalIndex()
