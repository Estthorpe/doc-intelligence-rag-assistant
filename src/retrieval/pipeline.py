# src/retrieval/pipeline.py
"""
Assembled end-to-end retrieval pipeline.

Orchestrates: hybrid search → reranking → return top-n chunks.
This is the single entry point for all retrieval operations.
The serving layer calls this — it never calls dense/sparse/rerank directly.
"""

from __future__ import annotations

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.retrieval.dense import RetrievedChunk
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank

logger = get_logger(__name__)


def retrieve(
    query: str,
    top_k: int | None = None,
    top_n: int | None = None,
) -> list[RetrievedChunk]:
    """
    Full retrieval pipeline: hybrid search + reranking.

    Args:
        query:  The user's question.
        top_k:  Candidates to retrieve before reranking.
        top_n:  Final chunks to return after reranking.

    Returns:
        Top-n reranked chunks ready for generation.
    """
    if top_k is None:
        top_k = settings.retrieval_top_k
    if top_n is None:
        top_n = settings.rerank_top_n

    candidates = hybrid_search(query, top_k=top_k)

    if not candidates:
        logger.warning(f"No candidates retrieved for query: '{query[:60]}'")
        return []

    final = rerank(query, candidates, top_n=top_n)

    logger.info(f"Pipeline complete: '{query[:50]}...' → {len(final)} chunks for generation")
    return final
