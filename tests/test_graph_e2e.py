"""End-to-end integration tests for LangGraph state machine with all external clients stubbed."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from agenticrag.graph import create_agent
from agenticrag.schemas import EvidenceGrade, RouteDecision


@pytest.fixture
def agent_app():
    return create_agent()


@pytest.mark.asyncio
async def test_e2e_direct_route(agent_app):
    """Direct route: greetings/chit-chat bypasses retrieval and generates direct answer."""
    with (
        patch("agenticrag.nodes.get_router_llm") as mock_router,
        patch("agenticrag.nodes.get_llm") as mock_llm,
    ):
        mock_router.return_value.ainvoke = AsyncMock(return_value=RouteDecision(route="direct"))
        mock_llm.return_value.ainvoke = AsyncMock(
            return_value=AIMessage(content="Hello! How can I help you today?")
        )

        result = await agent_app.ainvoke(
            {"messages": [HumanMessage(content="Hello there!")]},
            config={"configurable": {"thread_id": "test-e2e-direct"}},
        )

        assert result["source_used"] == "direct"
        assert result["route"] == "direct"
        assert len(result["messages"]) == 2
        assert "Hello!" in str(result["messages"][-1].content)


@pytest.mark.asyncio
async def test_e2e_kb_good(agent_app):
    """KB-good: query routes to KB, retriever finds docs, grader marks good, synthesizes from KB."""
    sample_doc = Document(page_content="Our return window is 30 days.", metadata={"source": "faq"})

    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = [(sample_doc, 0.88)]

    with (
        patch("agenticrag.nodes.get_router_llm") as mock_router,
        patch("agenticrag.nodes.get_vectorstore", return_value=mock_vs),
        patch("agenticrag.nodes.get_grader_llm") as mock_grader,
        patch("agenticrag.nodes.get_llm") as mock_llm,
    ):
        mock_router.return_value.ainvoke = AsyncMock(return_value=RouteDecision(route="kb"))
        mock_grader.return_value.ainvoke = AsyncMock(return_value=EvidenceGrade(grade="good"))
        mock_llm.return_value.ainvoke = AsyncMock(
            return_value=AIMessage(content="You have 30 days to return items.")
        )

        result = await agent_app.ainvoke(
            {"messages": [HumanMessage(content="What is the return window?")]},
            config={"configurable": {"thread_id": "test-e2e-kb-good"}},
        )

        assert result["source_used"] == "private_kb"
        assert result["kb_grade"] == "good"
        assert "30 days" in str(result["messages"][-1].content)


@pytest.mark.asyncio
async def test_e2e_kb_weak_to_web_good(agent_app):
    """KB-weak -> Web-good: KB grade is weak, searches web, web grader marks good, synthesizes from web."""
    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = []

    mock_search = MagicMock()
    mock_search.invoke.return_value = {
        "answer": "Order status is currently available on the dashboard."
    }

    with (
        patch("agenticrag.nodes.get_router_llm") as mock_router,
        patch("agenticrag.nodes.get_vectorstore", return_value=mock_vs),
        patch("agenticrag.nodes.get_grader_llm") as mock_grader,
        patch("agenticrag.nodes.get_web_search", return_value=mock_search),
        patch("agenticrag.nodes.get_llm") as mock_llm,
    ):
        mock_router.return_value.ainvoke = AsyncMock(return_value=RouteDecision(route="kb"))
        # First call is KB grader (weak), second is Web grader (good)
        mock_grader.return_value.ainvoke = AsyncMock(
            side_effect=[
                EvidenceGrade(grade="weak"),
                EvidenceGrade(grade="good"),
            ]
        )
        mock_llm.return_value.ainvoke = AsyncMock(
            return_value=AIMessage(content="You can view your status on the dashboard.")
        )

        result = await agent_app.ainvoke(
            {"messages": [HumanMessage(content="Where can I see live status?")]},
            config={"configurable": {"thread_id": "test-e2e-web-good"}},
        )

        assert result["source_used"] == "web_search"
        assert result["kb_grade"] == "weak"
        assert result["web_grade"] == "good"
        assert "dashboard" in str(result["messages"][-1].content)


@pytest.mark.asyncio
async def test_e2e_weak_rewrite_to_insufficient(agent_app):
    """Weak -> rewrite -> insufficient: both KB and web weak, rewrites once, still weak -> insufficient."""
    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = []

    mock_search = MagicMock()
    mock_search.invoke.return_value = {"answer": ""}

    with (
        patch("agenticrag.nodes.get_router_llm") as mock_router,
        patch("agenticrag.nodes.get_vectorstore", return_value=mock_vs),
        patch("agenticrag.nodes.get_grader_llm") as mock_grader,
        patch("agenticrag.nodes.get_web_search", return_value=mock_search),
        patch("agenticrag.nodes.get_llm") as mock_llm,
    ):
        mock_router.return_value.ainvoke = AsyncMock(return_value=RouteDecision(route="kb"))
        # All grading calls return weak
        mock_grader.return_value.ainvoke = AsyncMock(return_value=EvidenceGrade(grade="weak"))
        # First llm call is query rewrite, second is answer_insufficient
        mock_llm.return_value.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content="rewritten specific query"),
                AIMessage(
                    content="I could not find sufficient information to answer your request."
                ),
            ]
        )

        result = await agent_app.ainvoke(
            {"messages": [HumanMessage(content="Unknown obscure inquiry")]},
            config={"configurable": {"thread_id": "test-e2e-insufficient"}},
        )

        assert result["source_used"] == "insufficient_evidence"
        assert result["retry_count"] >= 1
        assert "could not find sufficient information" in str(result["messages"][-1].content)


@pytest.mark.asyncio
async def test_e2e_contact_query_skips_web(agent_app):
    """Contact inquiry without KB evidence immediately routes to insufficient notice without searching web."""
    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = []

    with (
        patch("agenticrag.nodes.get_router_llm") as mock_router,
        patch("agenticrag.nodes.get_vectorstore", return_value=mock_vs),
        patch("agenticrag.nodes.get_grader_llm") as mock_grader,
        patch("agenticrag.nodes.get_web_search") as mock_web_search,
        patch("agenticrag.nodes.get_llm") as mock_llm,
    ):
        mock_router.return_value.ainvoke = AsyncMock(return_value=RouteDecision(route="kb"))
        mock_grader.return_value.ainvoke = AsyncMock(return_value=EvidenceGrade(grade="weak"))
        mock_llm.return_value.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="Direct phone support is not listed in our records. Please contact us via Live Chat."
            )
        )

        result = await agent_app.ainvoke(
            {"messages": [HumanMessage(content="Can I have your customer support phone number?")]},
            config={"configurable": {"thread_id": "test-e2e-contact"}},
        )

        # Ensure web search was completely skipped
        mock_web_search.assert_not_called()
        assert result["source_used"] == "insufficient_evidence"
        assert "phone support is not listed" in str(result["messages"][-1].content)
