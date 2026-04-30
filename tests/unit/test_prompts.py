# tests/unit/test_prompts.py
"""
Prompt regression tests — Gate G-010.
5 tests that verify the prompt template loads correctly
and formats as expected. These catch prompt changes that
break the expected structure without requiring API calls.
"""

from __future__ import annotations

import pytest
from src.generation.prompts import format_prompt, load_prompt


class TestPromptLoading:
    def test_v1_prompt_loads(self) -> None:
        prompt = load_prompt("v1")
        assert prompt["version"] == "v1"
        assert "system_prompt" in prompt
        assert "user_template" in prompt

    def test_nonexistent_version_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_prompt("v99")

    def test_system_prompt_contains_legal_instruction(self) -> None:
        prompt = load_prompt("v1")
        assert "legal document assistant" in prompt["system_prompt"].lower()

    def test_format_prompt_inserts_question(self) -> None:
        system, user = format_prompt(
            question="What is the notice period?",
            context="Either party may terminate upon 30 days notice.",
        )
        assert "What is the notice period?" in user

    def test_format_prompt_inserts_context(self) -> None:
        context = "Either party may terminate upon 30 days notice."
        system, user = format_prompt(
            question="What is the notice period?",
            context=context,
        )
        assert context in user
