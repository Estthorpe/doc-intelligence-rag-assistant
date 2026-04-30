# src/agents/knowledge_ops_agent.py
"""
Knowledge Ops Agent — LangGraph implementation.

Graph structure:
  route_question → retrieve → generate → assess_confidence
                                               │
                           ┌───────────────────┼───────────────────┐
                           ▼                   ▼                   ▼
                        escalate           summarise         trigger_reindex
                      (conf<0.7)       (summary req)       (new docs)

Design rules:
  - Agent never imports model weights directly
  - Every state transition logged in Langfuse
  - Audit log is append-only JSONL
  - run() never raises — always returns a valid AgentState
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langfuse import Langfuse
from langgraph.graph import END, StateGraph

from src.agents.schemas import AgentState
from src.config.logging_config import get_logger
from src.config.settings import settings
from src.generation.generator import stream_response
from src.retrieval.pipeline import retrieve

logger = get_logger(__name__)

AUDIT_LOG_PATH = Path("logs/agent_audit.jsonl")
ESCALATION_QUEUE_PATH = Path("logs/escalation_queue.jsonl")


class KnowledgeOpsAgent:
    """Knowledge Ops Agent with conditional LangGraph routing."""

    def __init__(self) -> None:
        self._langfuse = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Build the LangGraph state machine."""
        workflow: StateGraph = StateGraph(AgentState)  # type: ignore[type-arg]

        workflow.add_node("route_question", self._route_question)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        workflow.add_node("assess_confidence", self._assess_confidence)
        workflow.add_node("escalate", self._escalate)
        workflow.add_node("summarise", self._summarise)
        workflow.add_node("trigger_reindex", self._trigger_reindex)

        workflow.set_entry_point("route_question")
        workflow.add_edge("route_question", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "assess_confidence")

        workflow.add_conditional_edges(
            "assess_confidence",
            self._route_from_confidence,
            {
                "escalate": "escalate",
                "summarise": "summarise",
                "trigger_reindex": "trigger_reindex",
                "done": END,
            },
        )

        workflow.add_edge("escalate", END)
        workflow.add_edge("summarise", END)
        workflow.add_edge("trigger_reindex", END)

        return workflow.compile()

    def _route_from_confidence(self, state: AgentState) -> str:
        """Conditional routing based on state after assess_confidence."""
        if state.confidence < settings.confidence_threshold:
            return "escalate"
        if state.query_type == "summary":
            return "summarise"
        if state.reindex_trigger:
            return "trigger_reindex"
        return "done"

    def _route_question(self, state: AgentState) -> AgentState:
        """Classify the query type for routing."""
        question_lower = state.question.lower()
        if any(w in question_lower for w in ["summarise", "summarize", "summary", "overview"]):
            state.query_type = "summary"
        elif any(w in question_lower for w in ["compare", "difference", "versus", "vs"]):
            state.query_type = "comparison"
        else:
            state.query_type = "factual"
        return state

    def _retrieve(self, state: AgentState) -> AgentState:
        """Retrieve relevant chunks. Failure returns empty list."""
        try:
            chunks = retrieve(state.question)
            state.retrieved_docs = chunks
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            state.retrieved_docs = []
        return state

    def _generate(self, state: AgentState) -> AgentState:
        """Generate an answer using retrieved chunks."""
        if not state.retrieved_docs:
            state.answer = "The provided documents do not contain sufficient information."
            state.citations = []
            state.cost_usd = 0.0
            return state

        try:
            full_answer = ""
            for token in stream_response(
                question=state.question,
                chunks=state.retrieved_docs,
            ):
                full_answer += token

            state.answer = full_answer
            state.citations = [
                {
                    "source": Path(c.source_path).name,
                    "chunk_index": c.chunk_index,
                }
                for c in state.retrieved_docs
            ]
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            state.answer = "Generation failed. Please try again."
            state.citations = []

        return state

    def _assess_confidence(self, state: AgentState) -> AgentState:
        """
        Score confidence using heuristics — not LLM-as-judge.
        LLM scoring its own output is circular. Heuristics are
        deterministic, free, and based on measurable signals.
        """
        low_confidence_phrases = [
            "i don't know",
            "not sure",
            "cannot find",
            "no information",
            "insufficient",
            "unclear",
            "does not contain",
            "unable to",
        ]

        answer_lower = state.answer.lower()

        if any(phrase in answer_lower for phrase in low_confidence_phrases):
            state.confidence = 0.4
        elif not state.citations:
            state.confidence = 0.5
        elif not state.retrieved_docs:
            state.confidence = 0.3
        else:
            scores = [c.rerank_score for c in state.retrieved_docs if c.rerank_score != 0.0]
            if scores:
                avg_raw = sum(scores) / len(scores)
                state.confidence = round(1.0 / (1.0 + math.exp(-avg_raw)), 4)
            else:
                state.confidence = 0.75

        return state

    def _escalate(self, state: AgentState) -> AgentState:
        """Flag this query for human review."""
        state.action = "escalate"
        state.escalation_reason = (
            f"Confidence {state.confidence:.2f} below threshold {settings.confidence_threshold}"
        )

        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "question": state.question,
            "answer": state.answer[:200],
            "confidence": state.confidence,
            "reason": state.escalation_reason,
        }

        ESCALATION_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ESCALATION_QUEUE_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

        logger.warning(f"Escalated: '{state.question[:60]}' (confidence={state.confidence:.2f})")
        return state

    def _summarise(self, state: AgentState) -> AgentState:
        """Handle summary requests."""
        state.action = "summarise"
        return state

    def _trigger_reindex(self, state: AgentState) -> AgentState:
        """Flag that re-indexing is required."""
        state.action = "trigger_reindex"
        logger.info("Re-indexing triggered by agent")
        return state

    def _write_audit_log(self, state: AgentState) -> None:
        """Append-only audit log. Never overwrite historical entries."""
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "question": state.question,
            "answer": state.answer[:200],
            "confidence": state.confidence,
            "action": state.action,
            "query_type": state.query_type,
            "citations_count": len(state.citations),
            "cost_usd": state.cost_usd,
            "trace_id": state.trace_id,
        }
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def run(self, question: str, reindex_trigger: bool = False) -> AgentState:
        """
        Run the Knowledge Ops Agent for a question.

        Always returns a valid AgentState. Never raises.
        Writes to audit log regardless of outcome.
        """
        langfuse_trace = self._langfuse.trace(
            name="knowledge_ops_agent",
            input={"question": question},
        )

        initial_state = AgentState(
            question=question,
            trace_id=langfuse_trace.id,
            reindex_trigger=reindex_trigger,
        )

        try:
            result_dict = self._graph.invoke(initial_state)
            # LangGraph returns AddableValuesDict — convert back to AgentState
            if hasattr(result_dict, "answer"):
                final_state = result_dict
            else:
                final_state = AgentState(
                    **{k: v for k, v in result_dict.items() if k in AgentState.model_fields}
                )

            langfuse_trace.update(
                output={
                    "answer": final_state.answer[:200],
                    "action": final_state.action,
                    "confidence": final_state.confidence,
                }
            )

        except Exception as e:
            logger.error(f"Agent graph failed: {e}")
            final_state = initial_state
            final_state.answer = "Agent encountered an error. Please try again."
            final_state.action = "escalate"
            final_state.confidence = 0.0

        self._write_audit_log(final_state)
        return final_state
