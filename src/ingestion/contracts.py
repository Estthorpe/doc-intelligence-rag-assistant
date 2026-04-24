"""Pydantic data contracts for document chunks

Every chunk must pass these contracts before it is embedded or stored.
A chunk that fails is logged and discarded — it never reaches pgvector.
Silent failures in chunking produce silent failures in retrieval.
These contracts make every failure explicit and traceable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
import uuid

from pydantic import BaseModel, Field, field_validator


class DocumentChunk(BaseModel):
    """
    A single validated chunk ready for embedding and storage,
    id:            UUID — unique across all chunks in the index.
                   Duplicate IDs corrupt vector store retrieval.
    content:       The actual text. Non-empty, within token bounds.
    source_doc_id: ID of the parent document. Required for citations.
    source_path:   Original file path. Required for citations.
    chunk_index:   Position within the source document.
                   Enables document order reconstruction.
    token_count:   Tokens in this chunk. Enforced between 10 and 600.
    ingested_at:   UTC timestamp. Enables recency-aware retrieval.
    metadata:      Optional key-value pairs from the source document.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., min_length=1)
    source_doc_id: str = Field(..., description="Parent document ID")
    source_path: str = Field(..., description="Original file path")
    chunk_index: int = Field(..., ge=0)
    token_count: int = Field(..., gt=0)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def content_not_whitespace(cls, v: str) -> str:
        """Empty or whitespace-only chunks produce zero-information embeddings."""
        if not v.strip():
            raise ValueError("Chunk content is empty or whitespace only")
        return v.strip()

    @field_validator("token_count")
    @classmethod
    def token_count_within_bounds(cls, v: int) -> int:
        """
        Oversized chunks (>600) indicate a parsing failure.
        Undersized chunks (<10) add noise to the retrieval index.
        Both are discarded, not silently accepted.
        """
        if v < 10:
            raise ValueError(f"Chunk too small: {v} tokens (minimum: 10)")
        if v > 600:
            raise ValueError(f"Chunk too large: {v} tokens (maximum: 600)")
        return v


class DocumentMetadata(BaseModel):
    """
    Metadata for an ingested source document.
    Validated before ingestion begins — a document with
    an invalid file type is rejected at the door.
    """

    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_path: str
    file_type: str = Field(..., pattern=r"^(pdf|txt|md)$")
    page_count: Optional[int] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    total_chunks: Optional[int] = None
