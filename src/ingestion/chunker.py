"""Document chunking pipeline
chunk_size=500 tokens: small enough for precise embeddings,
    large enough for a meaningful answer. 2000-token chunks produce
    noisy embeddings where only 20 tokens may be relevant to a query.
chunk_overlap=50 tokens: prevents answers being split across
    chunk boundaries. A sentence starting at the end of chunk N
    continues at the start of chunk N+1.

Every chunk is validated against DocumentChunk contracts after splitting.
Invalid chunks are logged and discarded — never silently passed through.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.ingestion.contracts import DocumentChunk

logger = get_logger(__name__)

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a string using tiktoken."""
    return len(_encoder.encode(text))


def chunk_document(
    text: str,
    source_doc_id: str,
    source_path: str,
    metadata: dict | None = None,
) -> list[DocumentChunk]:
    """
    Split a document into validated chunks ready for embedding
    Processing order:
    1.  Check for empty input, return early if nothing to chunk
    2. Split with REcursiveCharacterTextSplitter
    3. Validate every raw chunk against DocumentChunk contract
    4. Discard invalid chunks with a warning log
    5. Return only valid chunks.
    """
    if not text or not text.strip():
        logger.warning(f"Empty document provided: {source_path}. Skipping.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", " ", ""],
    )

    raw_chunks = splitter.split_text(text)
    logger.info(f"Split '{Path(source_path).name}' into {len(raw_chunks)} raw chunks")
    validated: list[DocumentChunk] = []
    discarded = 0

    for i, chunk_text in enumerate(raw_chunks):
        token_count = count_tokens(chunk_text)
        try:
            chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                content=chunk_text,
                source_doc_id=source_doc_id,
                source_path=source_path,
                chunk_index=i,
                metadata=metadata or {},
                token_count=token_count,
            )
            validated.append(chunk)
        except ValueError as e:
            logger.warning(
                f"Chunk {i} from '{Path(source_path).name}' "
                f"failed contract: {e}"
                f"(tokens={token_count}) - discarding"
            )
            discarded += 1
    logger.info(
        f"Chunking complete for '{Path(source_path).name}': "
        f"{len(validated)} valid chunks, {discarded} discarded"
    )
    return validated
