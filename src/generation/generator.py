# src/generation/generator.py
"""
Claude Haiku 3 generation with streaming and cost tracking.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import anthropic
from langfuse import Langfuse

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.generation.prompts import format_prompt
from src.monitoring.cost_monitor import check_budget, log_api_call
from src.retrieval.dense import RetrievedChunk

logger = get_logger(__name__)

_client: anthropic.Anthropic | None = None
_langfuse: Langfuse | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _get_langfuse() -> Langfuse:
    global _langfuse
    if _langfuse is None:
        _langfuse = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    return _langfuse


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a context string for the prompt."""
    parts = []
    for chunk in chunks:
        source = Path(chunk.source_path).name if chunk.source_path else "unknown"
        parts.append(f"[Source: {source}, Chunk {chunk.chunk_index}]\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


def stream_response(
    question: str,
    chunks: list[RetrievedChunk],
    prompt_version: str = "v1",
) -> Iterator[str]:
    """
    Stream a response from Claude Haiku token by token.

    Yields string tokens as they arrive from the API.
    Never raises — yields a fallback message on any failure.
    """
    allowed, reason = check_budget()
    if not allowed:
        yield f"[Budget limit reached: {reason}]"
        return

    if not chunks:
        yield "The provided documents do not contain sufficient information to answer this question."
        return

    context = format_context(chunks)
    system_prompt, user_message = format_prompt(
        question=question,
        context=context,
        version=prompt_version,
    )

    langfuse = _get_langfuse()
    trace = langfuse.trace(
        name="rag_generation",
        input={"question": question, "chunks_count": len(chunks)},
    )

    client = _get_client()
    full_response = ""
    input_tokens = 0
    output_tokens = 0

    try:
        with client.messages.stream(
            model=settings.generation_model,
            max_tokens=settings.max_output_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                yield text

            final_msg = stream.get_final_message()
            input_tokens = final_msg.usage.input_tokens
            output_tokens = final_msg.usage.output_tokens

        cost = log_api_call(
            operation="rag_generation",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        trace.update(
            output={"answer": full_response},
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "prompt_version": prompt_version,
                "chunks_used": len(chunks),
            },
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        trace.update(metadata={"error": str(e)})
        yield "\n[Generation failed. Please try again.]"
