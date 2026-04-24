# src/generation/prompts.py
"""Versioned prompt loader from configs/prompts/."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.config.logging_config import get_logger

logger = get_logger(__name__)

PROMPTS_DIR = Path("configs/prompts")


def load_prompt(version: str = "v1") -> dict[str, str]:
    """Load a versioned prompt template from configs/prompts/."""
    prompt_path = PROMPTS_DIR / f"rag_{version}.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path) as f:
        prompt: dict[str, str] = yaml.safe_load(f)

    logger.info(f"Loaded prompt: {prompt.get('name')} (version={version})")
    return prompt


def format_prompt(
    question: str,
    context: str,
    version: str = "v1",
) -> tuple[str, str]:
    """
    Format system and user prompts with question and context.

    Returns:
        Tuple of (system_prompt, user_message).
    """
    prompt = load_prompt(version)
    system_prompt: str = prompt["system_prompt"]
    user_message: str = prompt["user_template"].format(
        question=question,
        context=context,
    )
    return system_prompt, user_message
