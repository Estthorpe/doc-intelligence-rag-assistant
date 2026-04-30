# src/agents/tools.py
"""
HTTP tool wrappers for the Knowledge Ops Agent.

The agent calls the serving layer via HTTP only.
It never imports model weights or calls generation functions directly.
"""

from __future__ import annotations

import httpx

from src.config.logging_config import get_logger

logger = get_logger(__name__)

API_BASE_URL = "http://localhost:8000"
TIMEOUT_SECONDS = 60


def call_ask_endpoint(question: str, use_hyde: bool = False) -> dict[str, object]:
    """
    Call the /ask endpoint and return the full response.

    Used by the agent to get answers without importing
    the generation pipeline directly.

    Args:
        question: The user's question.
        use_hyde: Whether to enable HyDE query transformation.

    Returns:
        Dict with answer, citations, confidence, cost_usd, cached.
    """
    try:
        full_answer = ""
        final_data: dict[str, object] = {}

        with httpx.stream(
            "POST",
            f"{API_BASE_URL}/ask",
            json={"question": question, "use_hyde": use_hyde},
            timeout=TIMEOUT_SECONDS,
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    import json

                    data = json.loads(line[6:])
                    if "token" in data and not data.get("done"):
                        full_answer += str(data["token"])
                    elif data.get("done"):
                        final_data = data

        return {
            "answer": final_data.get("answer", full_answer),
            "citations": final_data.get("citations", []),
            "confidence": final_data.get("confidence", 0.0),
            "cost_usd": final_data.get("cost_usd", 0.0),
            "cached": final_data.get("cached", False),
        }

    except Exception as e:
        logger.error(f"HTTP tool call failed: {e}")
        return {
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "cost_usd": 0.0,
            "cached": False,
            "error": str(e),
        }
