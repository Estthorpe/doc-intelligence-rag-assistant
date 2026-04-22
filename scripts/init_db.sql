CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding VECTOR(384),
    source_doc_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::JSONB,
    CONSTRAINT content_not_empty CHECK (length(trim(content)) > 0),
    CONSTRAINT token_count_positive CHECK (token_count > 0),
    CONSTRAINT chunk_index_nonneg CHECK (chunk_index >= 0)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON document_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS chunks_source_idx
    ON document_chunks (source_doc_id);

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL CHECK (file_type IN ('pdf', 'txt', 'md')),
    page_count INTEGER,
    total_chunks INTEGER,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::JSONB
);

CREATE TABLE IF NOT EXISTS retrieval_log (
    id BIGSERIAL PRIMARY KEY,
    query_hash TEXT NOT NULL,
    retrieved_count INTEGER NOT NULL,
    reranked_count INTEGER NOT NULL,
    latency_ms FLOAT NOT NULL,
    cache_hit BOOLEAN NOT NULL DEFAULT FALSE,
    logged_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);