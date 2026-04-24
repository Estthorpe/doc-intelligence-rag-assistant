# src/monitoring/retrieval_monitor.py
"""
Retrieval quality monitoring.

Tracks hit-rate, latency, and failure rate by reading
from the retrieval_log table in pgvector.
"""

from __future__ import annotations

import psycopg2

from src.config.logging_config import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


def log_retrieval(
    query_hash: str,
    retrieved_count: int,
    reranked_count: int,
    latency_ms: float,
    cache_hit: bool = False,
) -> None:
    """Log a retrieval event to the retrieval_log table."""
    conn = psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO retrieval_log
                        (query_hash, retrieved_count, reranked_count,
                         latency_ms, cache_hit)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (query_hash, retrieved_count, reranked_count, latency_ms, cache_hit),
                )
    finally:
        conn.close()


def get_retrieval_stats() -> dict[str, float]:
    """Return retrieval statistics from the log table."""
    conn = psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    AVG(latency_ms) as avg_latency,
                    SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits,
                    AVG(retrieved_count) as avg_retrieved,
                    AVG(reranked_count) as avg_reranked
                FROM retrieval_log
            """)
            row = cur.fetchone()
            if row is None:
                return {}
            total, avg_latency, cache_hits, avg_retrieved, avg_reranked = row
            return {
                "total_queries": float(total or 0),
                "avg_latency_ms": float(avg_latency or 0),
                "cache_hit_rate": float(cache_hits or 0) / max(float(total or 1), 1),
                "avg_chunks_retrieved": float(avg_retrieved or 0),
                "avg_chunks_after_rerank": float(avg_reranked or 0),
            }
    finally:
        conn.close()
