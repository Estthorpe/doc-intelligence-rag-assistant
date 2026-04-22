# doc-intelligence-rag-assistant

**Portfolio Project P6 — AI Engineering Master Plan V3**


---

## What This Is

A production-grade RAG system for legal document intelligence.
Ask questions about legal contracts and receive grounded, cited answers
with streaming output, semantic caching, and full LLM observability.



## Stack

Claude Haiku 3 | BAAI/bge-small-en-v1.5 | pgvector | Redis | LangGraph | Langfuse | FastAPI | Streamlit

## Quick Start

```powershell
git clone https://github.com/YOUR-USERNAME/doc-intelligence-rag-assistant.git
cd doc-intelligence-rag-assistant
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.template .env  
docker-compose up -d pgvector redis
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
streamlit run src/ui/streamlit_app.py
```

## Engineering Lifecycle

| Stage | Status | Deliverable |
|-------|--------|-------------|
| 1. Template | In progress | Scaffold, CI, typed config |
| 2. Ingestion | Pending | 20+ docs chunked in pgvector |
| 3. Evaluation | Pending | RAGAS Faithfulness > 0.85 |
| 4. Serving | Pending | Streaming /ask endpoint |
| 5. Monitoring | Pending | Cost + drift monitoring |
| 6. GenAI | Pending | Grounded generation + UI |
| 7. Agentic | Pending | Knowledge Ops Agent |

## Architecture Trade-offs

| Spec Default | This Build | Reason |
|-------------|-----------|--------|
| text-embedding-3-small | BGE-small-en-v1.5 (local) | $5 Anthropic budget |
| gpt-4o-mini | Claude Haiku 3 | $5 Anthropic budget |
| Cohere Rerank | cross-encoder/ms-marco (local) | No Cohere API access |