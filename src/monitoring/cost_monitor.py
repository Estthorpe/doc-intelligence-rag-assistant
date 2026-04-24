"""
Cost monitor with circuit breaker for OpenAI API calls.
Tracks token usage and costs, and trips a circuit breaker if limits are exceeded."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.config.logging_config import get_logger
from src.config.settings import settings

logger = get_logger(__name__)

COST_LOG_PATH = Path("logs/cost_log.jsonl")

# Claude Haiku 3 pricing (per million tokens)
HAIKU_INPUT_COST_PER_M = 0.80
HAIKU_OUTPUT_COST_PER_M = 4.00


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate the USD cost of an Anthropic API call."""
    input_cost = (input_tokens / 1_000_000) * HAIKU_INPUT_COST_PER_M
    output_cost = (output_tokens / 1_000_000) * HAIKU_OUTPUT_COST_PER_M
    return round(input_cost + output_cost, 8)


def get_total_spend() -> float:
    """Read the total spend from the cost log."""
    if not COST_LOG_PATH.exists():
        return 0.0
    total = 0.0
    with open(COST_LOG_PATH) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                total += entry.get("cost_usd", 0.0)
            except json.JSONDecodeError:
                continue
    return round(total, 8)


def check_budget() -> tuple[bool, str]:
    """
    Check whether the budget allows another API call.

    Returns:
        (True, "") if call is permitted.
        (False, reason) if circuit breaker is active.
    """
    total = get_total_spend()

    if total >= settings.hard_stop_threshold_usd:
        msg = (
            f"HARD STOP: Total spend ${total:.4f} has reached "
            f"the hard stop threshold ${settings.hard_stop_threshold_usd}. "
            f"All LLM calls are blocked."
        )
        logger.error(msg)
        return False, msg

    if total >= settings.circuit_breaker_threshold_usd:
        msg = (
            f"CIRCUIT BREAKER: Total spend ${total:.4f} has reached "
            f"${settings.circuit_breaker_threshold_usd} (80% of budget). "
            f"Non-essential LLM calls suspended."
        )
        logger.warning(msg)
        return False, msg

    return True, ""


def log_api_call(
    operation: str,
    input_tokens: int,
    output_tokens: int,
    model: str = "claude-haiku-4-5",
) -> float:
    """
    Log an API call and return the cost.

    Args:
        operation:     Description of the call (e.g., "rag_generation").
        input_tokens:  Tokens in the prompt.
        output_tokens: Tokens in the response.
        model:         Model used.

    Returns:
        Cost in USD for this call.
    """
    cost = calculate_cost(input_tokens, output_tokens)
    total = get_total_spend() + cost

    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "operation": operation,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "running_total_usd": total,
        "budget_remaining_usd": round(settings.budget_ceiling_usd - total, 8),
    }

    COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COST_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info(
        f"API call logged: {operation} | "
        f"tokens={input_tokens}+{output_tokens} | "
        f"cost=${cost:.6f} | "
        f"total=${total:.6f} | "
        f"remaining=${settings.budget_ceiling_usd - total:.6f}"
    )
    return cost


def report() -> None:
    """Print a cost summary to the terminal."""
    total = get_total_spend()
    remaining = settings.budget_ceiling_usd - total
    pct = (total / settings.budget_ceiling_usd) * 100

    print("\n" + "=" * 50)
    print("COST GOVERNANCE REPORT")
    print("=" * 50)
    print(f"  Budget ceiling:    ${settings.budget_ceiling_usd:.2f}")
    print(f"  Total spend:       ${total:.6f}")
    print(f"  Remaining:         ${remaining:.6f}")
    print(f"  Budget used:       {pct:.2f}%")
    print(f"  Circuit breaker:   ${settings.circuit_breaker_threshold_usd:.2f}")
    print(f"  Hard stop:         ${settings.hard_stop_threshold_usd:.2f}")
    print("=" * 50 + "\n")
