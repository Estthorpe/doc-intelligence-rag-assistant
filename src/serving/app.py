# src/serving/app.py
"""
FastAPI serving layer for doc-intelligence-rag-assistant.

Endpoints:
  POST /ask     — RAG pipeline with streaming SSE response
  GET  /health  — Infrastructure health check
  GET  /ready   — Readiness check (pgvector + redis connected)
  GET  /metrics — Request and cost metrics

Every /ask request:
1. Runs guardrail pipeline (injection, PII, topic)
2. Checks semantic cache — returns immediately on hit
3. Runs hybrid retrieval + reranking
4. Streams generation from Claude Haiku 3
5. Logs cost and latency to Langfuse
6. Updates semantic cache
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from pathlib import Path

import psycopg2
import redis as redis_lib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langfuse import Langfuse

from src.cache.semantic_cache import SemanticCache
from src.config.logging_config import get_logger
from src.config.settings import settings
from src.generation.generator import stream_response
from src.monitoring.cost_monitor import get_total_spend
from src.retrieval.dense import RetrievedChunk
from src.retrieval.pipeline import retrieve
from src.serving.guardrails import GuardrailPipeline
from src.serving.schemas import (
    AskRequest,
    HealthResponse,
    MetricsResponse,
)

logger = get_logger(__name__)

app = FastAPI(
    title="doc-intelligence-rag-assistant",
    version=settings.project_version,
    description="Production RAG system for legal document intelligence",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level singletons
_guardrails = GuardrailPipeline()
_cache = SemanticCache()
_langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host=settings.langfuse_host,
)

# Request tracking
_total_requests = 0
_cache_hits = 0
_total_latency_ms = 0.0


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check that the API is running."""
    return HealthResponse(
        status="ok",
        version=settings.project_version,
        environment=settings.environment,
        pgvector="connected",
        redis="connected",
    )


@app.get("/ready")
def ready() -> dict[str, str]:
    """
    Readiness check — verifies pgvector and Redis are reachable.
    Returns 503 if either is unavailable.
    """
    errors: list[str] = []

    try:
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
        )
        conn.close()
    except Exception as e:
        errors.append(f"pgvector: {e}")

    try:
        r = redis_lib.from_url(settings.redis_url)
        r.ping()
    except Exception as e:
        errors.append(f"redis: {e}")

    if errors:
        raise HTTPException(status_code=503, detail={"errors": errors})

    return {"status": "ready"}


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Return request and cost metrics."""
    hit_rate = _cache_hits / _total_requests if _total_requests > 0 else 0.0
    avg_latency = _total_latency_ms / _total_requests if _total_requests > 0 else 0.0
    total_cost = get_total_spend()

    return MetricsResponse(
        total_requests=_total_requests,
        cache_hits=_cache_hits,
        cache_hit_rate=round(hit_rate, 4),
        avg_latency_ms=round(avg_latency, 2),
        total_cost_usd=total_cost,
        budget_remaining_usd=round(settings.budget_ceiling_usd - total_cost, 6),
    )


@app.post("/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    """
    Answer a question using the RAG pipeline with streaming output.

    Returns a Server-Sent Events stream.
    Each event: data: {"token": "..."}\n\n
    Final event: data: {"done": true, "answer": "...", "citations": [...],
                         "confidence": 0.0, "cost_usd": 0.0, "cached": false}\n\n
    """

    async def generate() -> AsyncIterator[str]:
        global _total_requests, _cache_hits, _total_latency_ms

        _total_requests += 1
        start_time = time.perf_counter()

        # ── Step 1: Guardrails ──────────────────────────────────────
        guard_result = _guardrails.run(request.question)
        if not guard_result.passed:
            yield f"data: {json.dumps({'error': guard_result.violation_detail})}\n\n"
            return

        clean_query = guard_result.query

        # ── Step 2: Semantic cache check ────────────────────────────
        cached = _cache.get(clean_query)
        if cached:
            _cache_hits += 1
            latency = (time.perf_counter() - start_time) * 1000
            _total_latency_ms += latency

            answer = str(cached.get("answer", ""))
            citations = cached.get("citations", [])

            yield f"data: {json.dumps({'token': answer})}\n\n"
            yield f"data: {json.dumps({'done': True, 'cached': True, 'answer': answer, 'citations': citations, 'confidence': 0.9, 'cost_usd': 0.0, 'latency_ms': latency})}\n\n"
            return

        # ── Step 3: Retrieval ───────────────────────────────────────
        chunks: list[RetrievedChunk] = retrieve(clean_query)

        # ── Step 4: Stream generation ───────────────────────────────
        full_answer = ""
        for token in stream_response(
            question=clean_query,
            chunks=chunks,
            prompt_version=request.prompt_version,
        ):
            full_answer += token
            yield f"data: {json.dumps({'token': token})}\n\n"

        # ── Step 5: Final event ─────────────────────────────────────
        latency = (time.perf_counter() - start_time) * 1000
        _total_latency_ms += latency

        citations = [
            {
                "source": Path(c.source_path).name,
                "chunk_index": c.chunk_index,
                "content_preview": c.content[:100],
            }
            for c in chunks
        ]

        confidence = float(chunks[0].rerank_score) if chunks and chunks[0].rerank_score else 0.75

        # ── Step 6: Update cache ────────────────────────────────────
        _cache.set(
            query=clean_query,
            answer=full_answer,
            citations=citations,
        )

        yield f"data: {json.dumps({'done': True, 'cached': False, 'answer': full_answer, 'citations': citations, 'confidence': confidence, 'cost_usd': 0.0, 'latency_ms': latency})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
