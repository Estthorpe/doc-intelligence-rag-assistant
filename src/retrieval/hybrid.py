# src/retrieval/hybrid.py
"""
Hybrid retrieval combining dense and sparse search via
Reciprocal Rank Fusion (RRF).

RRF combines ranked lists without requiring score calibration.
It uses rank positions, not raw scores — so dense similarity scores
(0.0 to 1.0) and BM25 scores (unbounded) can be combined cleanly.

RRF formula: score(d) = sum(1 / (k + rank(d, list_i)))
where k=60 is a smoothing constant (standard value).

A document ranked 1st in both lists scores: 1/61 + 1/61 = 0.0328
A document ranked 1st in only one list scores: 1/61 = 0.0164

This naturally boosts documents that appear high in both lists —
semantically relevant AND keyword-matching. Those are the most
likely to contain the answer.
"""

from __future__ import annotations

from src.config.logging_config import get_logger
from src.retrieval.dense import RetrievedChunk, dense_search
from src.retrieval.sparse import sparse_search

logger = get_logger(__name__)

RRF_K = 60  # Standard RRF smoothing constant


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    k: int = RRF_K,
) -> list[RetrievedChunk]:
    """
    Combine dense and sparse ranked lists using Reciprocal Rank Fusion.

    Args:
        dense_results:  Chunks ranked by vector similarity.
        sparse_results: Chunks ranked by BM25 score.
        k:              Smoothing constant. 60 is standard.

    Returns:
        Combined list sorted by RRF score descending.
    """
    chunk_map: dict[str, RetrievedChunk] = {}
    rrf_scores: dict[str, float] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        chunk.dense_rank = rank
        chunk_map[chunk.id] = chunk
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (k + rank))

    for rank, chunk in enumerate(sparse_results, start=1):
        chunk.sparse_rank = rank
        if chunk.id not in chunk_map:
            chunk_map[chunk.id] = chunk
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (k + rank))

    for chunk_id, score in rrf_scores.items():
        chunk_map[chunk_id].rrf_score = score

    fused = sorted(
        chunk_map.values(),
        key=lambda c: c.rrf_score,
        reverse=True,
    )

    logger.info(
        f"RRF fusion: {len(dense_results)} dense + "
        f"{len(sparse_results)} sparse → "
        f"{len(fused)} unique chunks"
    )
    return fused


def hybrid_search(
    query: str,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """
    Full hybrid search: dense + sparse + RRF fusion.

    Args:
        query:  The user's question.
        top_k:  Number of candidates to retrieve from each method.
                The fused list may contain up to 2*top_k unique chunks.

    Returns:
        RRF-fused list of RetrievedChunk objects.
    """
    dense_results = dense_search(query, top_k=top_k)
    sparse_results = sparse_search(query, top_k=top_k)
    return reciprocal_rank_fusion(dense_results, sparse_results)
