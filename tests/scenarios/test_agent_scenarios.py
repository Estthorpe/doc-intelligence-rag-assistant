# tests/scenarios/test_agent_scenarios.py
"""
Knowledge Ops Agent scenario tests — Gate G-007.
5/5 must pass in CI.

All retrieval and generation are mocked — tests are deterministic
and cost $0.00 to run.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agents.knowledge_ops_agent import KnowledgeOpsAgent
from src.agents.schemas import AgentState
from src.retrieval.dense import RetrievedChunk


def make_chunk(score: float = 2.0) -> RetrievedChunk:
    """Create a mock RetrievedChunk for testing."""
    return RetrievedChunk(
        id="test-id",
        content="Either party may terminate upon thirty days written notice.",
        source_doc_id="doc_001",
        source_path="data/raw/contract_01_SOFTWARE_LICENSE.txt",
        chunk_index=0,
        token_count=12,
        rerank_score=score,
    )


class TestKnowledgeOpsScenarios:
    @patch("src.agents.knowledge_ops_agent.retrieve")
    @patch("src.agents.knowledge_ops_agent.stream_response")
    def test_scenario_1_high_confidence_answer(
        self, mock_stream: MagicMock, mock_retrieve: MagicMock
    ) -> None:
        """
        Scenario 1: Normal factual Q&A with high-confidence retrieval.
        Expected: action == 'answer', confidence >= 0.7
        """
        mock_retrieve.return_value = [make_chunk(score=3.0)]
        mock_stream.return_value = iter(["The notice period is 30 days."])

        agent = KnowledgeOpsAgent()
        result = agent.run("What is the notice period for termination?")

        assert isinstance(result, AgentState)
        assert result.action == "answer"
        assert result.confidence >= 0.7

    @patch("src.agents.knowledge_ops_agent.retrieve")
    @patch("src.agents.knowledge_ops_agent.stream_response")
    def test_scenario_2_low_confidence_escalates(
        self, mock_stream: MagicMock, mock_retrieve: MagicMock
    ) -> None:
        """
        Scenario 2: Low confidence answer routes to escalation.
        Expected: action == 'escalate', confidence < 0.7
        """
        mock_retrieve.return_value = [make_chunk(score=0.1)]
        mock_stream.return_value = iter(
            ["I cannot find sufficient information to answer this question."]
        )

        agent = KnowledgeOpsAgent()
        result = agent.run("What is the quantum entanglement clause?")

        assert isinstance(result, AgentState)
        assert result.action == "escalate"
        assert result.confidence < 0.7

    @patch("src.agents.knowledge_ops_agent.retrieve")
    @patch("src.agents.knowledge_ops_agent.stream_response")
    def test_scenario_3_summary_request_routes_correctly(
        self, mock_stream: MagicMock, mock_retrieve: MagicMock
    ) -> None:
        """
        Scenario 3: Summary request routes to summarise node.
        Expected: action == 'summarise', query_type == 'summary'
        """
        mock_retrieve.return_value = [make_chunk(score=3.0)]
        mock_stream.return_value = iter(["Here is a summary of the contract."])

        agent = KnowledgeOpsAgent()
        result = agent.run("Please summarise the termination clauses")

        assert isinstance(result, AgentState)
        assert result.action == "summarise"
        assert result.query_type == "summary"

    @patch("src.agents.knowledge_ops_agent.retrieve")
    @patch("src.agents.knowledge_ops_agent.stream_response")
    def test_scenario_4_reindex_trigger(
        self, mock_stream: MagicMock, mock_retrieve: MagicMock
    ) -> None:
        """
        Scenario 4: New document flag triggers reindex action.
        Expected: action == 'trigger_reindex'
        """
        mock_retrieve.return_value = [make_chunk(score=3.0)]
        mock_stream.return_value = iter(["Answer based on current index."])

        agent = KnowledgeOpsAgent()
        result = agent.run(
            "What are the payment terms?",
            reindex_trigger=True,
        )

        assert isinstance(result, AgentState)
        assert result.action == "trigger_reindex"

    @patch("src.agents.knowledge_ops_agent.retrieve")
    def test_scenario_5_agent_never_raises(self, mock_retrieve: MagicMock) -> None:
        """
        Scenario 5: Agent handles all failures gracefully.
        Even when retrieval raises an exception, agent returns
        a valid AgentState and never raises itself.
        """
        mock_retrieve.side_effect = Exception("Database connection failed")

        agent = KnowledgeOpsAgent()
        result = agent.run("What are the termination clauses?")

        assert isinstance(result, AgentState)
        assert result.action in ["answer", "escalate", "summarise", "trigger_reindex"]
