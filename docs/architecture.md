# doc-intelligence-rag-assistant — System Architecture

**Version:** 0.1.0 | **Author:** Esther Uzor | **Status:** Phase 2 (Template)

## Stack
Claude Haiku 3 | BAAI/bge-small-en-v1.5 | pgvector | Redis | LangGraph | Langfuse

## Architecture Decisions

| Component      | Spec Default                 | This Build             | Reason                    |
|----------------|------------------------------|-------------------------|--------------------------|
| Vector store   | pgvector (Docker)            | pgvector (Docker) ✅    | Docker available         |
| Semantic cache | Redis (Docker)               | Redis 7 (Docker) ✅     | Docker available         |
| Embeddings     | text-embedding-3-small       | BAAI/bge-small-en-v1.5  | $5 Anthropic-only budget  |
| Generation     | gpt-4o-mini                  | Claude Haiku 3          | $5 Anthropic-only budget |
| Reranking      | Cohere Rerank                | cross-encoder/ms-marco  | No Cohere API access     |

## System Diagram

CLIENT LAYER
Streamlit UI │ POST /ask │ GET /health │ GET /metrics
SERVING LAYER (FastAPI)
GuardrailPipeline → [injection | PII | topic] → 400 or continue
SemanticCache.get() → Redis → hit: return immediately
RAG Pipeline invocation → StreamingResponse (SSE)
Langfuse trace every request
RAG PIPELINE

HyDE (optional)   question → Haiku → hypothetical answer → embed
Hybrid Retrieval  dense (pgvector) + sparse (BM25) → RRF → top-20
Reranking         cross-encoder → top-5
Generation        Claude Haiku 3 + versioned prompt → stream tokens
Cache update      SemanticCache.set() → Redis

VECTOR STORE
pgvector: document_chunks (id, content, embedding VECTOR(384),
source_doc_id, source_path, chunk_index, token_count,
ingested_at, metadata)
IVFFlat index: vector_cosine_ops, lists=100
INGESTION LAYER
loader.py    PDF/text parsing (pdfplumber)
chunker.py   RecursiveCharacterTextSplitter (500t, 50 overlap)
contracts.py Pydantic validation (no empty, no oversized, unique IDs)
embedder.py  BAAI/bge-small-en-v1.5 (local CPU, batched, Langfuse)
indexer.py   pgvector batch upsert via SQLAlchemy
AGENT LAYER (LangGraph)
route_question → retrieve → generate → assess_confidence
│
┌───────────────────────────────┤
▼               ▼               ▼
escalate       summarise    trigger_reindex
(conf<0.7)   (summary req)   (new docs)
MONITORING
cost_monitor.py      per-query cost, circuit breaker at $4.00
retrieval_monitor.py hit-rate@k, latency, failure rate
query_drift.py       query type distribution shift

## Cost Governance
- Budget: $5.00 total (Anthropic Claude Haiku 3)
- Circuit breaker: $4.00 — suspend non-essential calls
- Hard stop: $4.75 — block all LLM calls
- Embeddings: $0.00 (local)
- Reranking: $0.00 (local)