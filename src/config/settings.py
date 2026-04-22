# src/config/settings.py
"""
Typed configuration for doc-intelligence-rag-assistant.

All settings loaded from environment variables via pydantic-settings.
Every configurable value lives here — no magic numbers in application code.
Field(...) means required — missing env var = clear startup failure.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Project ───────────────────────────────────────────────────────
    project_name: str = "doc-intelligence-rag-assistant"
    project_version: str = "0.1.0"
    environment: str = Field(default="development")

    # ── Anthropic ─────────────────────────────────────────────────────
    anthropic_api_key: str = Field(..., description="Anthropic API key")
    generation_model: str = "claude-haiku-4-5"
    max_output_tokens: int = 1024

    # ── Langfuse ──────────────────────────────────────────────────────
    langfuse_public_key: str = Field(..., description="Langfuse public key")
    langfuse_secret_key: str = Field(..., description="Langfuse secret key")
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── pgvector ──────────────────────────────────────────────────────
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="ragdb")
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(..., description="Postgres password")

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ─────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379")
    # 0.92 = query must be 92% similar to a cached query for a cache hit
    cache_similarity_threshold: float = 0.92

    # ── Embeddings (locked after first ingestion) ─────────────────────
    # BAAI/bge-small-en-v1.5: optimised for retrieval, 384 dims, free
    # Changing this requires dropping the pgvector table and re-indexing
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384

    # ── Reranking ─────────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Chunking (binding values from spec) ───────────────────────────
    # 500 tokens: precise embeddings, meaningful answers
    # 50 overlap: prevents answers split at chunk boundaries
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_chunk_tokens: int = 600  # above = parsing failure
    min_chunk_tokens: int = 10  # below = noise

    # ── Retrieval ─────────────────────────────────────────────────────
    retrieval_top_k: int = 20  # candidates before reranking
    rerank_top_n: int = 5  # chunks passed to generation

    # ── Agent ─────────────────────────────────────────────────────────
    confidence_threshold: float = 0.7  # below = escalate to human

    # ── Cost governance ───────────────────────────────────────────────
    budget_ceiling_usd: float = 5.00
    circuit_breaker_threshold_usd: float = 4.00
    hard_stop_threshold_usd: float = 4.75

    # ── Paths ─────────────────────────────────────────────────────────
    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    logs_dir: Path = Path("logs")

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}


settings = Settings()
