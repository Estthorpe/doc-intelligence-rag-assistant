# Model Card — doc-intelligence-rag-assistant

**Status:** Phase 2 — to be completed after Phase 4 (Evaluation)

## Models
| Model                                   | Purpose                    | Cost          |
|-----------------------------------------|----------------------------|---------------|
| BAAI/bge-small-en-v1.5                  | Document + query embedding | $0.00 (local) |
| cross-encoder/ms-marco-MiniLM-L-6-v2    | Reranking                  | $0.00 (local) |
| Claude Haiku 3 | Answer generation      | $0.80/MTok input           |
| Claude Haiku 3 | RAGAS evaluation judge | $0.80/MTok input           |

## Evaluation Results (to be completed after Phase 4)
| Metric            | Score | Gate   | Status |
|-------------------|-------|--------|--------|
| Faithfulness      | TBD   | ≥ 0.85 | Pending |
| Answer Relevancy  | TBD   | ≥ 0.80 | Pending |
| Context Recall    | TBD   | ≥ 0.70 | Pending |
| Context Precision | TBD   | ≥ 0.70 | Pending |

## Known Limitations
- BGE-small (384 dims) may miss subtle distinctions in highly specialized
  legal terminology. Production would use Voyage Law-2.
- Confidence scoring is heuristic, not LLM-based. Deliberate — avoids
  circular LLM self-evaluation.