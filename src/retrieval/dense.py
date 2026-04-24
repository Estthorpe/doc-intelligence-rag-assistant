# src/retrieval/dense.py
"""
Dense retrieval using pgvector cosine similarity.

Embeds the query using the same BGE-small model used at ingestion time.
Queries pgvector using the <=> cosine distance operator.
Returns top-k chunks ranked by similarity.

Why the same model matters: if you index with model A and query with
model B, the vectors live in different spaces and retrieval is garbage.
Model consistency is a hard constraint, not a preference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import psycopg2

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.ingestion.embedder import _get_model as get_embedding_model

logger = get_logger(__name__)
model = get_embedding_model()


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval with ranking metadata."""

    id: str
    content: str
    source_doc_id: str
    source_path: str
    chunk_index: int
    token_count: int
    similarity_score: float = 0.0
    dense_rank: int = 0
    sparse_rank: int = 0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


def embed_query(query: str) -> list[float]:
    """
    Embed a query string using the same model as ingestion.

    Uses normalize_embeddings=True to match ingestion behaviour.
    Cosine similarity requires normalised vectors.
    """
    model = get_embedding_model()
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embedding[0].tolist()  # type: ignore[no-any-return]


def dense_search(
    query: str,
    top_k: int | None = None,
    min_similarity: float = 0.0,
) -> list[RetrievedChunk]:
    """
    Search pgvector for the top-k most similar chunks to the query.

    Uses cosine distance operator (<=>). Lower distance = higher similarity.
    Converts distance to similarity: similarity = 1 - distance.

    Args:
        query:          The user's question.
        top_k:          Number of results to return. Defaults to settings value.
        min_similarity: Minimum similarity threshold (0.0 to 1.0).

    Returns:
        List of RetrievedChunk objects sorted by similarity descending.
    """
    if top_k is None:
        top_k = settings.retrieval_top_k

    query_embedding = embed_query(query)

    sql = """
        SELECT
            id,
            content,
            source_doc_id,
            source_path,
            chunk_index,
            token_count,
            metadata,
            1 - (embedding <=> %s::vector) AS similarity
        FROM document_chunks
        WHERE 1 - (embedding <=> %s::vector) >= %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    conn = psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )

    results: list[RetrievedChunk] = []

    try:
        with conn.cursor() as cur:
            embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
            cur.execute(
                sql,
                (
                    embedding_str,
                    embedding_str,
                    min_similarity,
                    embedding_str,
                    top_k,
                ),
            )
            rows = cur.fetchall()

        for rank, row in enumerate(rows, start=1):
            (
                chunk_id,
                content,
                source_doc_id,
                source_path,
                chunk_index,
                token_count,
                metadata_raw,
                similarity,
            ) = row

            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

            results.append(
                RetrievedChunk(
                    id=str(chunk_id),
                    content=content,
                    source_doc_id=source_doc_id,
                    source_path=source_path,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    similarity_score=float(similarity),
                    dense_rank=rank,
                    metadata=metadata,
                )
            )

        logger.info(f"Dense search: '{query[:50]}...' → {len(results)} results (top_k={top_k})")

    except Exception as e:
        logger.error(f"Dense search failed: {e}")
        raise
    finally:
        conn.close()

    return results
