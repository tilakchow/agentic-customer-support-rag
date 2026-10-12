from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from agenticrag.clients import get_retriever
from agenticrag.nodes import (
    decide_after_kb_grade,
    grade_kb_evidence,
    grade_web_evidence,
    retrieve_kb,
    rewrite_query,
    route_question,
    search_web,
)


def test_route_question_fallback_on_exception():
    state = {"messages": [HumanMessage(content="Hello!")]}
    with patch("agenticrag.nodes.get_router_llm") as mock_llm:
        mock_llm.return_value.invoke.side_effect = RuntimeError("Groq rate limit or timeout")
        result = route_question(state)
        assert result["route"] == "kb"
        assert result["current_query"] == "Hello!"
        assert result["kb_retrieval_failed"] is False


def test_retrieve_kb_failure_sets_flag():
    state = {"current_query": "return policy"}
    with patch("agenticrag.nodes.get_retriever") as mock_retriever:
        mock_client = MagicMock()
        mock_client.invoke.side_effect = ConnectionError("Pinecone unreachable")
        mock_retriever.return_value = mock_client
        result = retrieve_kb(state)
        assert result["kb_retrieval_failed"] is True
        assert result["kb_docs"] == []


def test_retrieve_kb_none_retriever_sets_flag():
    state = {"current_query": "return policy"}
    with patch("agenticrag.nodes.get_retriever", return_value=None):
        result = retrieve_kb(state)
        assert result["kb_retrieval_failed"] is True
        assert result["kb_docs"] == []


def test_grade_kb_evidence_fallback_on_exception():
    state = {
        "messages": [HumanMessage(content="What is the shipping cost?")],
        "kb_docs": [Document(page_content="Free shipping over $50", metadata={"source": "faq"})],
    }
    with patch("agenticrag.nodes.get_grader_llm") as mock_grader:
        mock_grader.return_value.invoke.side_effect = TimeoutError("LLM call timed out")
        result = grade_kb_evidence(state)
        assert result["kb_grade"] == "weak"


def test_grade_web_evidence_fallback_on_exception():
    state = {
        "messages": [HumanMessage(content="Is item in stock?")],
        "web_results": "Item is in stock at local stores.",
        "web_search_failed": False,
    }
    with patch("agenticrag.nodes.get_grader_llm") as mock_grader:
        mock_grader.return_value.invoke.side_effect = Exception("Service unavailable")
        result = grade_web_evidence(state)
        assert result["web_grade"] == "weak"


def test_rewrite_query_fallback_on_exception():
    state = {
        "messages": [HumanMessage(content="track my parcel")],
        "retry_count": 0,
    }
    with patch("agenticrag.nodes.get_llm") as mock_llm:
        mock_llm.return_value.invoke.side_effect = Exception("LLM failure")
        result = rewrite_query(state)
        assert result["current_query"] == "track my parcel"
        assert result["retry_count"] == 1


def test_search_web_failure_sets_flag():
    state = {"current_query": "shipping tracking"}
    with patch("agenticrag.nodes.get_web_search") as mock_search:
        mock_search.return_value.invoke.side_effect = Exception("Tavily API 500")
        result = search_web(state)
        assert result["web_search_failed"] is True
        assert result["web_results"] == ""


def test_decide_after_kb_grade_web_fallback_disabled():
    state = {
        "kb_grade": "weak",
        "messages": [HumanMessage(content="What is the weather?")]
    }
    with patch("agenticrag.nodes.get_settings") as mock_settings:
        mock_settings.return_value.web_fallback_enabled = False
        assert decide_after_kb_grade(state) == "answer_insufficient"


def test_decide_after_kb_grade_proprietary_policy_skips_web():
    state = {
        "kb_grade": "weak",
        "messages": [HumanMessage(content="What is your return policy?")]
    }
    assert decide_after_kb_grade(state) == "answer_insufficient"


def test_retriever_backoff_on_failure():
    with patch("agenticrag.clients.get_settings") as mock_settings:
        mock_settings.return_value.pinecone_api_key = "dummy-key"
        mock_settings.return_value.index_name = "test-index"
        mock_settings.return_value.namespace = "test-ns"

        import agenticrag.clients as cfg
        cfg._retriever_client = None
        cfg._last_retriever_failure_time = 0.0

        with patch("pinecone.Pinecone", side_effect=Exception("Pinecone network down")):
            # First call fails and records backoff timestamp
            first = get_retriever()
            assert first is None
            assert cfg._last_retriever_failure_time > 0

        # Second call immediately after should back off without trying Pinecone
        with patch("pinecone.Pinecone") as mock_pc:
            second = get_retriever()
            assert second is None
            mock_pc.assert_not_called()

        # Reset global state
        cfg._retriever_client = None
        cfg._last_retriever_failure_time = 0.0
