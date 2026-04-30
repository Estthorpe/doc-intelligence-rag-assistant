# scripts/evaluate.py
"""
RAGAS evaluation entry point.

Usage:
    python scripts/evaluate.py                    # evaluate 10 questions
    python scripts/evaluate.py --sample-size 30   # full 30-question evaluation
    python scripts/evaluate.py --ci               # CI mode: exit 1 if gate fails

In CI, this script is called by the GitHub Actions workflow.
Exit code 0 = gate passed. Exit code 1 = gate failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import get_logger
from src.evaluation.ragas_runner import run_ragas_evaluation

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of golden set questions to evaluate (default: 10)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: exit with code 1 if gate fails",
    )
    args = parser.parse_args()

    try:
        scores = run_ragas_evaluation(sample_size=args.sample_size)
        print("\n" + "=" * 55)
        print("RAGAS EVALUATION REPORT")
        print("=" * 55)
        print(f"  Faithfulness:      {scores['faithfulness']:.4f}  (gate: >= 0.85)")
        print(f"  Answer Relevancy:  {scores['answer_relevancy']:.4f}")
        print(f"  Context Recall:    {scores['context_recall']:.4f}")
        print(f"  Context Precision: {scores['context_precision']:.4f}")
        print(f"  Gate passed:       {scores['gate_passed']}")
        print(f"  Report saved to:   docs/ragas_report.json")
        print("=" * 55 + "\n")

        if args.ci and not scores["gate_passed"]:
            sys.exit(1)

    except AssertionError as e:
        logger.error(str(e))
        if args.ci:
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
