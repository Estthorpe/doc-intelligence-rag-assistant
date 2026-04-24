# scripts/ingest.py
"""
One-command document ingestion entry point

Processes every .txt, .md, and .pdf file in the data directory.
Skips files that fail loading (logs the error and continues).
Reports total chunks ingested and total cost at the end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so src/ imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import get_logger
from src.ingestion.contracts import DocumentMetadata
from src.ingestion.embedder import embed_chunks
from src.ingestion.indexer import get_chunk_count, upsert_chunks, upsert_document_metadata
from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_document

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def ingest_directory(data_dir: str, batch_size: int = 64) -> None:
    """
    Ingest all supported documents in a directory.

    Args:
        data_dir:   Path to directory containing documents.
        batch_size: Number of chunks to embed per batch.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    files = [
        f for f in data_path.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.error(f"No supported files found in {data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(files)} documents to ingest")

    chunks_before = get_chunk_count()
    total_chunks_added = 0
    failed_docs = 0

    for i, file_path in enumerate(sorted(files), start=1):
        logger.info(f"Processing [{i}/{len(files)}]: {file_path.name}")

        # Step 1: Load
        try:
            text, file_type = load_document(file_path)
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e} — skipping")
            failed_docs += 1
            continue

        if not text.strip():
            logger.warning(f"{file_path.name}: no text extracted — skipping")
            failed_docs += 1
            continue

        # Step 2: Build document metadata
        doc_meta = DocumentMetadata(
            filename=file_path.name,
            file_path=str(file_path),
            file_type=file_type,
        )

        # Step 3: Chunk with contract validation
        chunks = chunk_document(
            text=text,
            source_doc_id=doc_meta.doc_id,
            source_path=str(file_path),
        )

        if not chunks:
            logger.warning(f"{file_path.name}: produced 0 valid chunks — skipping")
            failed_docs += 1
            continue

        # Step 4: Embed (local, $0.00)
        chunk_embedding_pairs = embed_chunks(chunks)

        # Step 5: Store in pgvector
        doc_meta.total_chunks = len(chunk_embedding_pairs)
        upsert_chunks(chunk_embedding_pairs)
        upsert_document_metadata(doc_meta)

        total_chunks_added += len(chunk_embedding_pairs)
        logger.info(f"Ingested {file_path.name}: {len(chunk_embedding_pairs)} chunks stored")

    chunks_after = get_chunk_count()

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info(f"  Documents processed: {len(files) - failed_docs}/{len(files)}")
    logger.info(f"  Documents failed:    {failed_docs}")
    logger.info(f"  Chunks added:        {total_chunks_added}")
    logger.info(f"  Total in pgvector:   {chunks_after} (was {chunks_before})")
    logger.info(f"  Embedding cost:      $0.00 (local model)")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into pgvector")
    parser.add_argument("--data-dir", default="data/raw/", help="Directory of documents")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    ingest_directory(args.data_dir, args.batch_size)
