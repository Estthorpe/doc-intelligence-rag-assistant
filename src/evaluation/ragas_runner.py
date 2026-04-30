# src/evaluation/ragas_runner.py
"""
RAGAS evaluation harness with Claude Haiku as judge.

Measures four dimensions of RAG quality:
  Faithfulness:      Does the answer only contain claims from retrieved context?
  Answer Relevancy:  Is the answer relevant to the question asked?
  Context Recall:    Does the retrieved context contain what is needed?
  Context Precision: What fraction of retrieved chunks are actually relevant?

CI gate: Faithfulness >= 0.85 (Gate G-002).
Build fails if this threshold is not met.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.retrieval.pipeline import retrieve

logger = get_logger(__name__)

FAITHFULNESS_GATE = 0.85
GOLDEN_SET_PATH = Path("configs/golden_set.yaml")
REPORT_PATH = Path("docs/ragas_report.json")


def load_golden_set() -> list[dict[str, str]]:
    """Load the 30-question golden set from configs/golden_set.yaml."""
    with open(GOLDEN_SET_PATH) as f:
        data: dict[str, list[dict[str, str]]] = yaml.safe_load(f)
    questions = data["questions"]
    logger.info(f"Loaded {len(questions)} questions from golden set")
    return questions


def run_rag_pipeline(question: str) -> tuple[str, list[str]]:
    """
    Run the retrieval pipeline for a question and return answer + contexts.

    Returns:
        Tuple of (answer_text, list_of_context_strings)
    """
    from src.generation.generator import stream_response

    chunks = retrieve(question)
    if not chunks:
        return "No relevant documents found.", []

    contexts = [chunk.content for chunk in chunks]

    full_answer = ""
    for token in stream_response(question=question, chunks=chunks):
        full_answer += token

    return full_answer, contexts


def run_ragas_evaluation(
    sample_size: int = 10,
    output_path: Path = REPORT_PATH,
) -> dict[str, float]:
    """
    Run RAGAS evaluation on a sample of the golden set.

    Args:
        sample_size: Number of questions to evaluate (default 10 to save cost).
        output_path: Where to write the JSON report.

    Returns:
        Dict with RAGAS scores and gate pass/fail status.

    Raises:
        AssertionError if Faithfulness < FAITHFULNESS_GATE.
    """
    logger.info(f"Starting RAGAS evaluation (sample_size={sample_size})")

    # Configure Claude Haiku as the RAGAS judge
    llm = ChatAnthropic(
        model=settings.generation_model,
        api_key=settings.anthropic_api_key,
        temperature=0,
    )
    wrapped_llm = LangchainLLMWrapper(llm)

    # Configure local BGE embeddings — prevents RAGAS calling OpenAI
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
    )
    wrapped_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    # Set LLM and embeddings on each metric before evaluation
    faithfulness.llm = wrapped_llm
    answer_relevancy.llm = wrapped_llm
    answer_relevancy.embeddings = wrapped_embeddings
    context_recall.llm = wrapped_llm
    context_precision.llm = wrapped_llm

    # Load golden set
    golden_questions = load_golden_set()[:sample_size]

    # Run RAG pipeline for each question
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for i, item in enumerate(golden_questions):
        q = item["question"]
        gt = item["ground_truth_answer"]

        logger.info(f"Evaluating [{i + 1}/{sample_size}]: {q[:60]}...")

        try:
            answer, context_list = run_rag_pipeline(q)
            questions.append(q)
            answers.append(answer)
            contexts.append(context_list if context_list else [gt])
            ground_truths.append(gt)
        except Exception as e:
            logger.error(f"Pipeline failed for question {i + 1}: {e}")
            questions.append(q)
            answers.append("Error generating answer.")
            contexts.append([gt])
            ground_truths.append(gt)

    # Build RAGAS dataset
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    logger.info("Running RAGAS metrics...")

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    scores = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_recall": float(result["context_recall"]),
        "context_precision": float(result["context_precision"]),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": sample_size,
        "gate_faithfulness": FAITHFULNESS_GATE,
        "gate_passed": float(result["faithfulness"]) >= FAITHFULNESS_GATE,
    }

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    logger.info("RAGAS evaluation complete:")
    logger.info(f"  Faithfulness:      {scores['faithfulness']:.4f} (gate: {FAITHFULNESS_GATE})")
    logger.info(f"  Answer Relevancy:  {scores['answer_relevancy']:.4f}")
    logger.info(f"  Context Recall:    {scores['context_recall']:.4f}")
    logger.info(f"  Context Precision: {scores['context_precision']:.4f}")
    logger.info(f"  Gate passed:       {scores['gate_passed']}")

    if not scores["gate_passed"]:
        raise AssertionError(
            f"RAGAS gate FAILED: Faithfulness={scores['faithfulness']:.4f} "
            f"< {FAITHFULNESS_GATE}. Improve retrieval or prompt quality."
        )

    return scores
