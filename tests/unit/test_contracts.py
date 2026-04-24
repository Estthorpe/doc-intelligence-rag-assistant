# tests/unit/test_contracts.py
"""
Data contract tests — run in CI on every push (Gate G-001).
These tests define what a valid chunk looks like.
A chunk that fails these contracts never reaches pgvector.
"""

import pytest

from src.ingestion.contracts import DocumentChunk, DocumentMetadata


class TestDocumentChunkContracts:
    def test_valid_chunk_passes(self) -> None:
        chunk = DocumentChunk(
            content="This is a valid legal clause about termination rights.",
            source_doc_id="doc_001",
            source_path="contracts/agreement.txt",
            chunk_index=0,
            token_count=12,
        )
        assert chunk.id is not None
        assert chunk.content == "This is a valid legal clause about termination rights."

    def test_empty_content_fails(self) -> None:
        with pytest.raises(ValueError, match="empty or whitespace"):
            DocumentChunk(
                content="   ",
                source_doc_id="doc_001",
                source_path="test.txt",
                chunk_index=0,
                token_count=1,
            )

    def test_oversized_chunk_fails(self) -> None:
        with pytest.raises(ValueError, match="too large"):
            DocumentChunk(
                content="x" * 100,
                source_doc_id="doc_001",
                source_path="test.txt",
                chunk_index=0,
                token_count=601,
            )

    def test_undersized_chunk_fails(self) -> None:
        with pytest.raises(ValueError, match="too small"):
            DocumentChunk(
                content="Hi",
                source_doc_id="doc_001",
                source_path="test.txt",
                chunk_index=0,
                token_count=2,
            )

    def test_all_ids_unique(self) -> None:
        chunks = [
            DocumentChunk(
                content=f"Clause {i} — this is valid content with enough tokens.",
                source_doc_id="doc_001",
                source_path="test.txt",
                chunk_index=i,
                token_count=15,
            )
            for i in range(10)
        ]
        ids = [c.id for c in chunks]
        assert len(set(ids)) == 10

    def test_whitespace_stripped_from_content(self) -> None:
        chunk = DocumentChunk(
            content="  Valid content with leading and trailing spaces.  ",
            source_doc_id="doc_001",
            source_path="test.txt",
            chunk_index=0,
            token_count=10,
        )
        assert not chunk.content.startswith(" ")
        assert not chunk.content.endswith(" ")


class TestDocumentMetadataContracts:
    def test_valid_metadata_passes(self) -> None:
        meta = DocumentMetadata(
            filename="contract.txt",
            file_path="data/raw/contract.txt",
            file_type="txt",
        )
        assert meta.doc_id is not None

    def test_invalid_file_type_fails(self) -> None:
        with pytest.raises(ValueError):
            DocumentMetadata(
                filename="contract.docx",
                file_path="data/raw/contract.docx",
                file_type="docx",
            )
