# src/agents/schemas.py
"""
AgentState schema for the Knowledge Ops Agent.

Every node in the LangGraph graph reads from and writes to this
state object. State is passed between nodes — no global variables,
no side effects through function arguments.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.retrieval.dense import RetrievedChunk


class AgentState(BaseModel):
    """
    Complete state of the Knowledge Ops Agent at any point
    in the LangGraph execution.

    Every node receives this object and returns an updated version.
    Langfuse traces the state at each node transition.
    """

    # Input
    question: str

    # Set by route_question node
    query_type: Literal["factual", "summary", "comparison"] | None = None

    # Set by retrieve node
    retrieved_docs: list[RetrievedChunk] = Field(default_factory=list)

    # Set by generate node
    answer: str = ""
    citations: list[dict[str, object]] = Field(default_factory=list)
    cost_usd: float = 0.0

    # Set by assess_confidence node
    confidence: float = 0.0

    # Set by routing logic
    action: Literal["answer", "escalate", "summarise", "trigger_reindex"] = "answer"
    escalation_reason: str | None = None
    reindex_trigger: bool = False

    # Observability
    trace_id: str | None = None

    model_config = {"arbitrary_types_allowed": True}
