from __future__ import annotations

"""
Hybrid retrieval index for ClarityAI.

Improvements over the original:
- Conversation-aware query expansion (last user turns merged carefully, not blindly).
- BM25-style scoring sits alongside word-tfidf and char-tfidf (ngram fallback for typos / proper nouns).
- Optional dense embeddings combined with sparse via convex weighted sum.
- Cross-source re-ranking with title / source / phrase / freshness boosts.
- MMR (Maximal Marginal Relevance) diversification so the top-k aren't 5 near-duplicate paragraphs.
- Stable, deduplicated snippets with consistent metadata for the UI layer.
"""

import math
import re
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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IndexedChunk:
    chunk_id: int
    document_id: str
    document_title: str
    source_name: str
    content: str
    page_label: str | None
    chunk_index: int


# ---------------------------------------------------------------------------
# Tiny BM25 implementation (kept local so we don't add a heavy dep)
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-z0-9_./\-]+")


def _bm25_tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


class _BM25:
    """Minimal BM25 Okapi implementation. Good enough for chat-scale corpora."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_len) / len(corpus)) if corpus else 0.0
        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        for doc in corpus:
            seen = set()
            counts: dict[str, int] = {}
            for tok in doc:
                counts[tok] = counts.get(tok, 0) + 1
                if tok not in seen:
                    self.df[tok] = self.df.get(tok, 0) + 1
                    seen.add(tok)
            self.tf.append(counts)
        n = len(corpus)
        self.idf: dict[str, float] = {
            tok: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for tok, df in self.df.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        n = len(self.corpus)
        scores = np.zeros(n, dtype=np.float32)
        if not query_tokens or not n:
            return scores
        for tok in query_tokens:
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for i in range(n):
                f = self.tf[i].get(tok, 0)
                if not f:
                    continue
                dl = self.doc_len[i] or 1
                denom = f + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1)))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        # Normalize roughly into [0, 1] so it can mix with cosine scores
        max_s = float(scores.max()) if scores.size else 0.0
        if max_s > 0:
            scores = scores / max_s
        return scores


# ---------------------------------------------------------------------------
# Retrieval index
# ---------------------------------------------------------------------------


class RetrievalIndex:
    def __init__(self) -> None:
        self._lock = RLock()
        self._items: list[IndexedChunk] = []
        self._word_vectorizer: TfidfVectorizer | None = None
        self._word_matrix = None
        self._char_vectorizer: TfidfVectorizer | None = None
        self._char_matrix = None
        self._bm25: _BM25 | None = None
        self._dense_matrix: np.ndarray | None = None
        self._dense_enabled = False

    # ------- Build / rebuild ------------------------------------------------

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
                self._reset_indexes()
                return {"status": "empty", "chunks": 0, "dense": "disabled"}

            texts = [item.content for item in self._items]

            self._word_vectorizer = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2), max_features=50000, sublinear_tf=True
            )
            self._char_vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), max_features=30000, sublinear_tf=True
            )
            self._word_matrix = self._word_vectorizer.fit_transform(texts)
            self._char_matrix = self._char_vectorizer.fit_transform(texts)
            self._bm25 = _BM25([_bm25_tokenize(t) for t in texts])

            dense_status = "disabled"
            self._dense_enabled = False
            self._dense_matrix = None
            if settings.enable_dense_retrieval and embedding_provider.available:
                try:
                    vectors = embedding_provider.embed_texts(texts)
                    dense = np.array(vectors, dtype=np.float32)
                    if dense.size:
                        self._dense_matrix = self._normalize(dense)
                        self._dense_enabled = True
                        dense_status = "enabled"
                except (ProviderUnavailableError, Exception):
                    self._dense_matrix = None
                    self._dense_enabled = False
                    dense_status = "unavailable"

            return {"status": "ok", "chunks": len(self._items), "dense": dense_status}

    def _reset_indexes(self) -> None:
        self._word_vectorizer = None
        self._word_matrix = None
        self._char_vectorizer = None
        self._char_matrix = None
        self._bm25 = None
        self._dense_matrix = None
        self._dense_enabled = False

    # ------- Helpers --------------------------------------------------------

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2}

    @staticmethod
    def _expand_query(query: str, conversation_context: str) -> str:
        """Build a richer retrieval query without polluting it with chat noise.

        Strategy: take the raw query, append only the *last 1–2 informative
        nouns/phrases* from conversation context. We keep the original query
        weighted by repeating it once at the front.
        """
        q = query.strip()
        if not conversation_context:
            return q
        ctx = conversation_context.strip()
        # Keep only tokens that look content-bearing
        ctx_tokens = [
            t for t in re.findall(r"[A-Za-z0-9_./\-]{3,}", ctx)
            if t.lower() not in _CHAT_NOISE
        ]
        # Keep last 12 informative tokens; this picks up entities, names, ids.
        tail = " ".join(ctx_tokens[-12:])
        if not tail:
            return q
        return f"{q} {q} {tail}".strip()

    # ------- Scoring --------------------------------------------------------

    def _dense_scores(self, query: str) -> np.ndarray | None:
        if not self._dense_enabled or self._dense_matrix is None or not embedding_provider.available:
            return None
        try:
            vector = np.array(embedding_provider.embed_texts([query])[0], dtype=np.float32)
            vector = self._normalize(vector.reshape(1, -1)).ravel()
            return self._dense_matrix @ vector
        except (ProviderUnavailableError, Exception):
            return None

    def _hybrid_score(
        self,
        idx: int,
        item: IndexedChunk,
        query_terms: set[str],
        word_scores: np.ndarray,
        char_scores: np.ndarray,
        bm25_scores: np.ndarray,
        dense_scores: np.ndarray | None,
        raw_query: str,
    ) -> float:
        query_l = raw_query.lower().strip()
        content_l = item.content.lower()
        title_l = item.document_title.lower()
        source_l = item.source_name.lower()

        title_terms = self._tokenize(title_l)
        source_terms = self._tokenize(source_l)
        head_terms = self._tokenize(content_l[:900])

        title_overlap = len(query_terms & title_terms)
        source_overlap = len(query_terms & source_terms)
        content_overlap = len(query_terms & head_terms)

        title_boost = min(0.18, 0.045 * title_overlap)
        source_boost = min(0.08, 0.02 * source_overlap)
        content_overlap_boost = min(0.12, 0.018 * content_overlap)
        exact_phrase_boost = 0.12 if query_l and len(query_l) > 4 and query_l in content_l else 0.0
        prefix_boost = 0.05 if any(t and t in title_l for t in list(query_terms)[:3]) else 0.0
        dense_component = float(dense_scores[idx]) if dense_scores is not None else 0.0

        # Convex combination of sparse signals + dense + boosts.
        score = (
            0.34 * float(word_scores[idx])
            + 0.10 * float(char_scores[idx])
            + 0.22 * float(bm25_scores[idx])
            + 0.22 * dense_component
            + title_boost
            + source_boost
            + content_overlap_boost
            + exact_phrase_boost
            + prefix_boost
        )

        # Penalize ultra-short chunks that rarely contain a real answer.
        if len(item.content) < 80:
            score *= 0.78
        return float(score)

    # ------- MMR diversification -------------------------------------------

    def _mmr_select(
        self,
        candidate_idxs: list[int],
        scores: list[float],
        k: int,
        lambda_: float = 0.72,
    ) -> list[int]:
        """Pick k items balancing relevance vs. novelty.

        Similarity between two chunks is approximated by Jaccard overlap of
        their token sets — fast and dependency-free, and good enough for
        de-clustering near-duplicate chunks.
        """
        if not candidate_idxs:
            return []
        # Pre-tokenize once
        token_sets: dict[int, set[str]] = {
            i: self._tokenize(self._items[i].content[:500]) for i in candidate_idxs
        }
        score_map = dict(zip(candidate_idxs, scores))

        selected: list[int] = []
        remaining = set(candidate_idxs)
        while remaining and len(selected) < k:
            best_idx = None
            best_value = -1e9
            for idx in remaining:
                rel = score_map[idx]
                if not selected:
                    div_pen = 0.0
                else:
                    sims = []
                    a = token_sets[idx]
                    for s in selected:
                        b = token_sets[s]
                        if not a or not b:
                            sims.append(0.0)
                            continue
                        inter = len(a & b)
                        union = len(a | b) or 1
                        sims.append(inter / union)
                    div_pen = max(sims)
                value = lambda_ * rel - (1 - lambda_) * div_pen
                if value > best_value:
                    best_value = value
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.discard(best_idx)
        return selected

    # ------- Public search --------------------------------------------------

    def _make_display_name(self, item: IndexedChunk) -> str:
        return item.source_name or item.document_title

    def search(
        self,
        query: str,
        conversation_context: str = "",
        k: int | None = None,
        candidate_pool: int | None = None,
    ) -> dict:
        with self._lock:
            if (
                not self._items
                or self._word_vectorizer is None
                or self._char_vectorizer is None
                or self._bm25 is None
            ):
                return {"results": [], "confidence": 0.0, "dense_used": False}

            k = k or settings.retrieval_top_k
            candidate_pool = candidate_pool or settings.retrieval_candidate_pool
            expanded_query = self._expand_query(query, conversation_context)
            query_terms = self._tokenize(expanded_query)

            word_q = self._word_vectorizer.transform([expanded_query])
            char_q = self._char_vectorizer.transform([query])
            word_scores = cosine_similarity(word_q, self._word_matrix).ravel()
            char_scores = cosine_similarity(char_q, self._char_matrix).ravel()
            bm25_scores = self._bm25.scores(_bm25_tokenize(expanded_query))
            dense_scores = self._dense_scores(expanded_query)

            combined = [
                self._hybrid_score(
                    idx=i,
                    item=item,
                    query_terms=query_terms,
                    word_scores=word_scores,
                    char_scores=char_scores,
                    bm25_scores=bm25_scores,
                    dense_scores=dense_scores,
                    raw_query=query,
                )
                for i, item in enumerate(self._items)
            ]

            scores_arr = np.array(combined, dtype=np.float32)
            top_indices = np.argsort(scores_arr)[::-1][:candidate_pool]

            # Pre-filter weak candidates and per-document caps and dedupe
            pruned_idxs: list[int] = []
            pruned_scores: list[float] = []
            doc_counts: dict[str, int] = {}
            seen_snippets: set[str] = set()
            for idx in top_indices:
                idx = int(idx)
                score = float(combined[idx])
                if score <= 0.06:
                    continue
                item = self._items[idx]
                if doc_counts.get(item.document_id, 0) >= 2:
                    continue
                snippet_key = re.sub(r"\s+", " ", item.content[:160].strip().lower())
                if snippet_key in seen_snippets:
                    continue
                pruned_idxs.append(idx)
                pruned_scores.append(score)
                doc_counts[item.document_id] = doc_counts.get(item.document_id, 0) + 1
                seen_snippets.add(snippet_key)

            # MMR for the final top-k
            selected = self._mmr_select(pruned_idxs, pruned_scores, k=k, lambda_=0.74)

            results = []
            for rank, idx in enumerate(selected, start=1):
                item = self._items[idx]
                score = float(combined[idx])
                results.append(
                    {
                        "id": f"S{rank}",            # internal stable id (kept for backwards compat)
                        "label": f"S{rank}",         # internal label (NOT shown to user)
                        "rank": rank,
                        "display_name": self._make_display_name(item),
                        "chunk_id": item.chunk_id,
                        "document_id": item.document_id,
                        "document_title": item.document_title,
                        "source_name": item.source_name,
                        "page_label": item.page_label,
                        "snippet": item.content[:340].strip(),
                        "content": item.content,
                        "score": round(score, 4),
                        "score_band": "high" if score >= 0.42 else "medium" if score >= 0.22 else "low",
                        "source_type": "knowledge",
                        "url": None,
                    }
                )

            top_score = float(scores_arr.max()) if scores_arr.size else 0.0
            confidence = max(0.0, min(1.0, round(math.tanh(top_score * 1.65), 4)))
            return {
                "results": results,
                "confidence": confidence,
                "dense_used": dense_scores is not None,
            }


# ---------------------------------------------------------------------------
# Module-level singleton + small constants
# ---------------------------------------------------------------------------


_CHAT_NOISE: set[str] = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "was",
    "were", "are", "you", "your", "what", "when", "how", "why", "who", "where",
    "can", "could", "would", "should", "tell", "give", "show", "please",
    "thanks", "thank", "okay", "ok", "yes", "no", "hi", "hello", "hey",
}


retrieval_index = RetrievalIndex()