# src/cache/semantic_cache.py
"""
Redis-backed semantic cache.

Stores (query_embedding, answer, citations) in Redis.
On each incoming query, embeds it and searches for a
semantically similar cached query using cosine similarity.

If similarity > threshold (0.92): return cached answer.
If similarity < threshold: run the full RAG pipeline.

Why 0.92: lower threshold = more cache hits but risk of
returning an answer to a slightly different question.
0.92 means the queries must be 92% similar — same question
with minor wording differences qualifies, genuinely different
questions do not.

Cost saving: a cache hit costs $0.00 vs ~$0.002 for a full
RAG pipeline call.
"""

from __future__ import annotations

import json
import time

import numpy as np
import redis

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.retrieval.dense import embed_query

logger = get_logger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a)
    vb = np.array(b)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


class SemanticCache:
    """
    Redis-backed semantic cache for RAG responses.

    Keys: cache:embedding:{index} → serialised embedding vector
          cache:answer:{index}    → JSON with answer and citations
          cache:count             → total number of cached entries
    """

    def __init__(self) -> None:
        self._redis: redis.Redis | None = None  # type: ignore[type-arg]

    def _get_redis(self) -> redis.Redis:  # type: ignore[type-arg]
        """Lazy-connect to Redis."""
        if self._redis is None:
            self._redis = redis.from_url(
                settings.redis_url,
                decode_responses=False,
            )
        return self._redis

    def get(self, query: str) -> dict[str, object] | None:
        """
        Look up a query in the semantic cache.

        Args:
            query: The user's question.

        Returns:
            Cached result dict with 'answer' and 'citations' keys,
            or None if no similar query is cached.
        """
        try:
            r = self._get_redis()
            count_raw = r.get("cache:count")
            if count_raw is None:
                return None

            count = int(count_raw)
            if count == 0:
                return None

            query_embedding = embed_query(query)
            best_similarity = 0.0
            best_index = -1

            for i in range(count):
                cached_emb_raw = r.get(f"cache:embedding:{i}")
                if cached_emb_raw is None:
                    continue
                cached_emb: list[float] = json.loads(cached_emb_raw)
                sim = _cosine_similarity(query_embedding, cached_emb)
                if sim > best_similarity:
                    best_similarity = sim
                    best_index = i

            if best_similarity >= settings.cache_similarity_threshold:
                cached_answer_raw = r.get(f"cache:answer:{best_index}")
                if cached_answer_raw:
                    result: dict[str, object] = json.loads(cached_answer_raw)
                    logger.info(
                        f"Cache hit: similarity={best_similarity:.4f} "
                        f"(threshold={settings.cache_similarity_threshold})"
                    )
                    return result

            return None

        except Exception as e:
            logger.warning(f"Cache get failed (non-blocking): {e}")
            return None

    def set(
        self,
        query: str,
        answer: str,
        citations: list[dict[str, object]],
    ) -> None:
        """
        Store a query-answer pair in the cache.

        Args:
            query:     The user's question.
            answer:    The generated answer.
            citations: Source citations from the retrieved chunks.
        """
        try:
            r = self._get_redis()
            count_raw = r.get("cache:count")
            count = int(count_raw) if count_raw else 0

            query_embedding = embed_query(query)

            r.set(
                f"cache:embedding:{count}",
                json.dumps(query_embedding),
            )
            r.set(
                f"cache:answer:{count}",
                json.dumps(
                    {
                        "answer": answer,
                        "citations": citations,
                        "cached_at": time.time(),
                    }
                ),
            )
            r.set("cache:count", count + 1)

            logger.info(f"Cached query #{count}: '{query[:50]}'")

        except Exception as e:
            logger.warning(f"Cache set failed (non-blocking): {e}")

    def clear(self) -> None:
        """Clear all cached entries."""
        try:
            r = self._get_redis()
            count_raw = r.get("cache:count")
            if count_raw:
                count = int(count_raw)
                for i in range(count):
                    r.delete(f"cache:embedding:{i}")
                    r.delete(f"cache:answer:{i}")
                r.delete("cache:count")
            logger.info("Semantic cache cleared")
        except Exception as e:
            logger.warning(f"Cache clear failed: {e}")
