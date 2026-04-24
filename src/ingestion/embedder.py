"""
Embedding pipeline using BAAI/bge-small-en-v1.5 locally"""

from __future__ import annotations

import time

from langfuse import Langfuse
from sentence_transformers import SentenceTransformer

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.ingestion.contracts import DocumentChunk

logger = get_logger(__name__)

# Batch size for embedding calls
BATCH_SIZE = 64

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load the BAAI/bge-small-en-v1.5 model"""
    global _model
    if _model is None:
        logger.info(f"Load embedding model: {settings.embedding_model}")
        start_time = time.time()
        _model = SentenceTransformer(settings.embedding_model)
        elapsed_time = time.time() - start_time
        logger.info(f"Model loaded in {elapsed_time:.2f} seconds.")
    return _model


def embed_chunks(chunks: list[DocumentChunk]) -> list[tuple[DocumentChunk, list[float]]]:
    """Embed document chunks using the BAAI/bge-small-en-v1.5 model"""
    if not chunks:
        logger.warning("embed_chunks called with empty chunk list.")
        return []
    model = get_model()
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )

    results: list[tuple[DocumentChunk, list[float]]] = []
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, batch_start in enumerate(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [chunk.content for chunk in batch]

        trace = langfuse.trace(
            name="embed_batch",
            metadata={
                "batch_num": batch_num + 1,
                "total_batches": total_batches,
                "batch_size": len(batch),
                "model": settings.embedding_model,
                "source_docs": list({c.source_doc_id for c in batch}),
            },
        )

        start_time = time.perf_counter()
        try:
            embeddings = model.encode(
                texts,
                batch_size=len(texts),
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            elapsed_time = time.perf_counter() - start_time

            for chunk, embedding in zip(batch, embeddings, strict=False):
                results.append((chunk, embedding.tolist()))

            trace.update(
                output={
                    "embeddings_produced": len(batch),
                    "embedding_dimensions": len(embeddings[0]),
                    "elapsed_seconds": round(elapsed_time, 3),
                    "cost_usd": 0.0,  # No cost for local embedding
                }
            )

            logger.info(
                f"Batch {batch_num + 1}/{total_batches}: "
                f"Embedded {len(batch)} chunks in {elapsed_time:.2f}s "
                f"(cost: $0.00)"
            )

        except Exception as e:
            logger.error(f"Embedding batch {batch_num + 1} failed: {e}")
            trace.update(metadata={"error": str(e)})
            raise

    logger.info(f" Embedding complete: {len(results)} chunks embedded in {total_batches} batches.")
    return results
