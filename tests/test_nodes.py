from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from agenticrag.clients import get_retriever
from agenticrag.graph import create_agent
from agenticrag.nodes import (
    decide_after_kb_grade,
    grade_kb_evidence,
    grade_web_evidence,
    retrieve_kb,
    rewrite_query,
    route_question,
    search_web,
)


@pytest.mark.asyncio
async def test_route_question_fallback_on_exception():
    state = {"messages": [HumanMessage(content="Hello!")]}
    with patch("agenticrag.nodes.get_router_llm") as mock_llm:
        mock_llm.return_value.ainvoke = AsyncMock(
            side_effect=RuntimeError("Groq rate limit or timeout")
        )
        result = await route_question(state)
        assert result["route"] == "kb"
        assert result["current_query"] == "Hello!"
        assert result["kb_retrieval_failed"] is False


@pytest.mark.asyncio
async def test_retrieve_kb_failure_sets_flag():
    state = {"current_query": "return policy"}
    with (
        patch("agenticrag.nodes.get_vectorstore", return_value=None),
        patch("agenticrag.nodes.get_retriever") as mock_retriever,
    ):
        mock_client = MagicMock()
        mock_client.invoke.side_effect = ConnectionError("Pinecone unreachable")
        mock_retriever.return_value = mock_client
        result = await retrieve_kb(state)
        assert result["kb_retrieval_failed"] is True
        assert result["kb_docs"] == []


@pytest.mark.asyncio
async def test_retrieve_kb_none_retriever_sets_flag():
    state = {"current_query": "return policy"}
    with (
        patch("agenticrag.nodes.get_vectorstore", return_value=None),
        patch("agenticrag.nodes.get_retriever", return_value=None),
    ):
        result = await retrieve_kb(state)
        assert result["kb_retrieval_failed"] is True
        assert result["kb_docs"] == []


@pytest.mark.asyncio
async def test_grade_kb_evidence_fallback_on_exception():
    state = {
        "messages": [HumanMessage(content="What is the shipping cost?")],
        "kb_docs": [Document(page_content="Free shipping over $50", metadata={"source": "faq"})],
    }
    with patch("agenticrag.nodes.get_grader_llm") as mock_grader:
        mock_grader.return_value.ainvoke = AsyncMock(side_effect=TimeoutError("LLM call timed out"))
        result = await grade_kb_evidence(state)
        assert result["kb_grade"] == "weak"


@pytest.mark.asyncio
async def test_grade_kb_evidence_skip_grader_score():
    state = {
        "messages": [HumanMessage(content="What is the shipping cost?")],
        "kb_docs": [
            Document(
                page_content="Free shipping over $50", metadata={"source": "faq", "score": 0.92}
            )
        ],
        "top_similarity_score": 0.92,
    }
    with (
        patch("agenticrag.nodes.get_settings") as mock_settings,
        patch("agenticrag.nodes.get_grader_llm") as mock_grader,
    ):
        mock_settings.return_value.skip_grader_score = 0.85
        result = await grade_kb_evidence(state)
        assert result["kb_grade"] == "good"
        mock_grader.assert_not_called()


@pytest.mark.asyncio
async def test_grade_web_evidence_fallback_on_exception():
    state = {
        "messages": [HumanMessage(content="Is item in stock?")],
        "web_results": "Item is in stock at local stores.",
        "web_search_failed": False,
    }
    with patch("agenticrag.nodes.get_grader_llm") as mock_grader:
        mock_grader.return_value.ainvoke = AsyncMock(side_effect=Exception("Service unavailable"))
        result = await grade_web_evidence(state)
        assert result["web_grade"] == "weak"


@pytest.mark.asyncio
async def test_rewrite_query_fallback_on_exception():
    state = {
        "messages": [HumanMessage(content="track my parcel")],
        "retry_count": 0,
    }
    with patch("agenticrag.nodes.get_llm") as mock_llm:
        mock_llm.return_value.ainvoke = AsyncMock(side_effect=Exception("LLM failure"))
        result = await rewrite_query(state)
        assert result["current_query"] == "track my parcel"
        assert result["retry_count"] == 1


@pytest.mark.asyncio
async def test_rewrite_query_success():
    state = {
        "messages": [HumanMessage(content="track my parcel")],
        "retry_count": 0,
    }
    with patch("agenticrag.nodes.get_llm") as mock_llm:
        mock_llm.return_value.ainvoke = AsyncMock(
            return_value=AIMessage(content="how to track shipment parcel")
        )
        result = await rewrite_query(state)
        assert result["current_query"] == "how to track shipment parcel"
        assert result["retry_count"] == 1


@pytest.mark.asyncio
async def test_search_web_failure_sets_flag():
    state = {"current_query": "shipping tracking"}
    with patch("agenticrag.nodes.get_web_search") as mock_search:
        mock_search.return_value.invoke.side_effect = Exception("Tavily API 500")
        result = await search_web(state)
        assert result["web_search_failed"] is True
        assert result["web_results"] == ""


def test_decide_after_kb_grade_web_fallback_disabled():
    state = {"kb_grade": "weak", "messages": [HumanMessage(content="What is the weather?")]}
    with patch("agenticrag.nodes.get_settings") as mock_settings:
        mock_settings.return_value.web_fallback_enabled = False
        assert decide_after_kb_grade(state) == "answer_insufficient"


def test_decide_after_kb_grade_proprietary_policy_skips_web():
    state = {"kb_grade": "weak", "messages": [HumanMessage(content="What is your return policy?")]}
    assert decide_after_kb_grade(state) == "answer_insufficient"


def test_checkpoint_db_sqlite(tmp_path):
    db_file = str(tmp_path / "test_checkpoints.sqlite")
    with patch("agenticrag.graph.get_settings") as mock_settings:
        mock_settings.return_value.checkpoint_db = db_file
        agent = create_agent()
        assert agent is not None
        assert hasattr(agent, "checkpointer")
        assert agent.checkpointer.__class__.__name__ == "SqliteSaver"


def test_retriever_backoff_on_failure():
    with patch("agenticrag.clients.get_settings") as mock_settings:
        mock_settings.return_value.pinecone_api_key = "dummy-key"
        mock_settings.return_value.index_name = "test-index"
        mock_settings.return_value.namespace = "test-ns"

        import agenticrag.clients as cfg

        cfg._vectorstore_client = None
        cfg._retriever_client = None
        cfg._last_retriever_failure_time = 0.0

        with patch("pinecone.Pinecone", side_effect=Exception("Pinecone network down")):
            first = get_retriever()
            assert first is None
            assert cfg._last_retriever_failure_time > 0

        with patch("pinecone.Pinecone") as mock_pc:
            second = get_retriever()
            assert second is None
            mock_pc.assert_not_called()

        cfg._vectorstore_client = None
        cfg._retriever_client = None
        cfg._last_retriever_failure_time = 0.0
