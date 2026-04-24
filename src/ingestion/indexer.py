# src/ingestion/indexer.py
"""
pgvector indexer — stores validated, embedded chunks.

Uses SQLAlchemy for connection management and parameterised queries.
Parameterised queries prevent SQL injection — never use f-strings
to build SQL with user-provided data.
"""

from __future__ import annotations

import json

import psycopg2
from psycopg2.extras import execute_values

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.ingestion.contracts import DocumentChunk, DocumentMetadata

logger = get_logger(__name__)


def get_connection() -> psycopg2.extensions.connection:
    """Create a new database connection."""
    return psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )


def upsert_chunks(
    chunk_embedding_pairs: list[tuple[DocumentChunk, list[float]]],
) -> int:
    """
    Upsert a batch of (chunk, embedding) pairs into pgvector.

    Uses INSERT ... ON CONFLICT DO UPDATE so re-ingestion is safe.
    All pairs are inserted in a single transaction — either all
    succeed or none do. No partial ingestion states.
    """
    if not chunk_embedding_pairs:
        return 0

    rows = []
    for chunk, embedding in chunk_embedding_pairs:
        rows.append(
            (
                chunk.id,
                chunk.content,
                embedding,  # pgvector accepts Python lists directly
                chunk.source_doc_id,
                chunk.source_path,
                chunk.chunk_index,
                chunk.token_count,
                chunk.ingested_at,
                json.dumps(chunk.metadata),
            )
        )

    sql = """
        INSERT INTO document_chunks
            (id, content, embedding, source_doc_id, source_path,
             chunk_index, token_count, ingested_at, metadata)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            content       = EXCLUDED.content,
            embedding     = EXCLUDED.embedding,
            token_count   = EXCLUDED.token_count,
            ingested_at   = EXCLUDED.ingested_at,
            metadata      = EXCLUDED.metadata
    """

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    sql,
                    rows,
                    template="(%s, %s, %s::vector, %s, %s, %s, %s, %s, %s::jsonb)",
                )
        logger.info(f"Upserted {len(rows)} chunks into pgvector")
        return len(rows)
    except Exception as e:
        logger.error(f"Failed to upsert chunks: {e}")
        raise
    finally:
        conn.close()


def upsert_document_metadata(metadata: DocumentMetadata) -> None:
    """
    Store document-level metadata in the documents table.
    Called once per document, after all its chunks are upserted.
    """
    sql = """
        INSERT INTO documents
            (doc_id, filename, file_path, file_type,
             page_count, total_chunks, ingested_at, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (doc_id) DO UPDATE SET
            total_chunks = EXCLUDED.total_chunks,
            ingested_at  = EXCLUDED.ingested_at
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        metadata.doc_id,
                        metadata.filename,
                        metadata.file_path,
                        metadata.file_type,
                        metadata.page_count,
                        metadata.total_chunks,
                        metadata.ingested_at,
                        json.dumps({}),
                    ),
                )
        logger.info(f"Stored document metadata: {metadata.filename}")
    except Exception as e:
        logger.error(f"Failed to store document metadata: {e}")
        raise
    finally:
        conn.close()


def get_chunk_count() -> int:
    """Return total number of chunks currently in pgvector."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_chunks")
            result = cur.fetchone()
            return result[0] if result else 0
    finally:
        conn.close()
