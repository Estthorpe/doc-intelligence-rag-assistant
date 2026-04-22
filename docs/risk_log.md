# Risk Log — doc-intelligence-rag-assistant (P6)

**Last updated:** Phase 2 (Template)

## Cost Governance
| Item | Value |
|------|-------|
| Total budget | $5.00 |
| Circuit breaker | $4.00 |
| Hard stop | $4.75 |
| Current spend | $0.00 |
| Remaining | $5.00 |

## Active Risks
| ID    | Risk                                     | Severity  | Mitigation                                            |
|-------|------------------------------------------|-----------|-------------------------------------------------------|
| R-001 | API costs spiral                         | High      | Circuit breaker at $4.00; Langfuse cost logging       |
| R-002 | RAG without evaluation is tutorial-level | High      | RAGAS harness built in Phase 4 before Phase 5         |
| R-003 | BGE-small quality insufficient           | Low       | Validate Hit-rate@5; upgrade to BGE-base if <0.7      |
| R-004 | LangGraph complexity step-up             | Medium    | Hand-rolled understanding before LangGraph in Phase 7 |

## Resolved
| ID | Resolution |
|----|-----------|
| ADR-001 | Local embeddings: BAAI/bge-small-en-v1.5 |
| ADR-002 | Generation: Claude Haiku 3 |
| ADR-004 | pgvector (Docker) restored |
| ADR-005 | Redis (Docker) restored |
| ADR-006 | Local cross-encoder (no Cohere access) |