# src/retrieval/reranker.py
"""
Local cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

Why reranking improves precision:
Initial retrieval ranks chunks by embedding similarity — a global
measure of how similar the chunk is to the query in vector space.
A cross-encoder reads the (query, chunk) pair together and scores
their relevance directly. This is slower but more accurate.

Typical workflow:
1. Retrieve top-20 candidates (fast, approximate)
2. Rerank to top-5 (slower, precise)
3. Pass top-5 to generation

The reranker sees both the query and the chunk simultaneously,
allowing it to detect subtle relevance signals that embedding
similarity misses.

Cost: $0.00 — runs locally on CPU.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.retrieval.dense import RetrievedChunk

logger = get_logger(__name__)

_reranker: CrossEncoder | None = None  # type: ignore[type-arg]


def _get_reranker() -> CrossEncoder:  # type: ignore[type-arg]
    """Lazy-load the cross-encoder model."""
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker: {settings.reranker_model}")
        _reranker = CrossEncoder(settings.reranker_model)
        logger.info("Reranker loaded")
    return _reranker


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int | None = None,
) -> list[RetrievedChunk]:
    """
    Rerank retrieved chunks using the cross-encoder.

    Args:
        query:  The user's question.
        chunks: Candidate chunks from hybrid search.
        top_n:  Number of chunks to return after reranking.
                Defaults to settings.rerank_top_n.

    Returns:
        Top-n chunks sorted by rerank score descending.
    """
    if top_n is None:
        top_n = settings.rerank_top_n

    if not chunks:
        return []

    reranker = _get_reranker()

    pairs = [(query, chunk.content) for chunk in chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores, strict=False):
        chunk.rerank_score = float(score)

    reranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
    top = reranked[:top_n]

    logger.info(f"Reranking: {len(chunks)} candidates → {len(top)} selected (top_n={top_n})")
    return top
