# src/serving/schemas.py
"""
Request and response schemas for the FastAPI serving layer.

Typed schemas mean:
- Invalid requests are rejected at the boundary with a clear error
- The API contract is documented in code, not in a README
- OpenAPI docs are generated automatically
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to answer from the document corpus.",
        examples=["What is the notice period for contract termination?"],
    )
    use_hyde: bool = Field(
        default=False,
        description=(
            "Enable HyDE (Hypothetical Document Embedding). "
            "Generates a hypothetical answer first, then uses it "
            "for retrieval. Improves recall for knowledge-intensive queries."
        ),
    )
    prompt_version: str = Field(
        default="v1",
        description="Prompt template version to use for generation.",
    )


class Citation(BaseModel):
    """A source citation from a retrieved chunk."""

    source: str = Field(..., description="Source document filename.")
    chunk_index: int = Field(..., description="Chunk position within the document.")
    content_preview: str = Field(..., description="First 100 characters of the chunk content.")


class AskResponse(BaseModel):
    """Final response payload sent after streaming completes."""

    answer: str
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
    cost_usd: float
    cached: bool
    chunks_retrieved: int
    latency_ms: float


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    version: str
    environment: str
    pgvector: str
    redis: str


class MetricsResponse(BaseModel):
    """Response for GET /metrics."""

    total_requests: int
    cache_hits: int
    cache_hit_rate: float
    avg_latency_ms: float
    total_cost_usd: float
    budget_remaining_usd: float
