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
async def test_retrieve_kb_runs_in_worker_thread():
    """Verifies Pinecone calls run in asyncio.to_thread and not on the main event-loop thread."""
    import asyncio
    import threading

    state = {"current_query": "shipping info"}
    caller_threads = []

    def fake_search(*args, **kwargs):
        caller_threads.append(threading.current_thread())
        return [(Document(page_content="Free shipping", metadata={"source": "faq"}), 0.9)]

    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.side_effect = fake_search

    with patch("agenticrag.nodes.get_vectorstore", return_value=mock_vs):
        # Concurrently run retrieve_kb and an async event loop task
        async def concurrent_task():
            await asyncio.sleep(0.01)
            return "done"

        kb_result, task_result = await asyncio.gather(
            retrieve_kb(state),
            concurrent_task(),
        )

        assert task_result == "done"
        assert len(kb_result["kb_docs"]) == 1
        assert len(caller_threads) == 1
        # The blocking search executed on a worker thread, not the main thread
        assert caller_threads[0] != threading.main_thread()


def test_warm_up_clients_handles_missing_keys_gracefully():
    """warm_up_clients runs safely without throwing exceptions when keys are absent."""
    from agenticrag.clients import warm_up_clients

    # Should not raise exception
    warm_up_clients()


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


@pytest.mark.asyncio
async def test_search_web_timeout_sets_flag():
    state = {"current_query": "shipping tracking"}
    with patch("agenticrag.nodes.get_web_search") as mock_search:
        mock_search.return_value.invoke.side_effect = TimeoutError("Tavily timed out")
        result = await search_web(state)
        assert result["web_search_failed"] is True
        assert result["web_results"] == ""


@pytest.mark.asyncio
async def test_search_web_tavily_error_sets_flag():
    from agenticrag.nodes import TavilyClientError

    state = {"current_query": "shipping tracking"}
    with patch("agenticrag.nodes.get_web_search") as mock_search:
        mock_search.return_value.invoke.side_effect = TavilyClientError("Invalid API key")
        result = await search_web(state)
        assert result["web_search_failed"] is True
        assert result["web_results"] == ""


@pytest.mark.asyncio
async def test_search_web_concurrency_semaphore():
    import asyncio

    from agenticrag.nodes import get_tavily_semaphore

    sem = get_tavily_semaphore()
    assert isinstance(sem, asyncio.Semaphore)


def test_decide_after_kb_grade_web_fallback_disabled():
    state = {"kb_grade": "weak", "messages": [HumanMessage(content="What is the weather?")]}
    with patch("agenticrag.nodes.get_settings") as mock_settings:
        mock_settings.return_value.web_fallback_enabled = False
        assert decide_after_kb_grade(state) == "answer_insufficient"


def test_decide_after_kb_grade_proprietary_policy_skips_web():
    state = {"kb_grade": "weak", "messages": [HumanMessage(content="What is your return policy?")]}
    assert decide_after_kb_grade(state) == "answer_insufficient"


@pytest.mark.asyncio
async def test_checkpoint_db_sqlite(tmp_path):
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from agenticrag.schemas import RouteDecision

    db_file = str(tmp_path / "test_checkpoints.sqlite")

    # Use async with to ensure the connection is properly opened and closed
    async with aiosqlite.connect(db_file) as conn:
        saver = AsyncSqliteSaver(conn)
        agent = create_agent(checkpointer=saver)
        assert agent is not None

        with (
            patch("agenticrag.nodes.get_router_llm") as mock_router,
            patch("agenticrag.nodes.get_llm") as mock_llm,
        ):
            mock_router.return_value.ainvoke = AsyncMock(return_value=RouteDecision(route="direct"))
            mock_llm.return_value.ainvoke = AsyncMock(
                return_value=AIMessage(content="Hello! Support answer saved to sqlite.")
            )

            config = {"configurable": {"thread_id": "test-sqlite-thread-1"}}
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="Hello from sqlite test")]}, config=config
            )
            assert result["source_used"] == "direct"
            assert "sqlite" in str(result["messages"][-1].content)

            # Verify state was persisted and can be retrieved asynchronously
            state = await agent.aget_state(config)
            assert len(state.values["messages"]) == 2


@pytest.mark.asyncio
async def test_checkpoint_db_sqlite_persistence_across_sessions(tmp_path):
    """Messages persisted in SQLite survive agent re-instantiation (simulating restart)."""
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from agenticrag.schemas import RouteDecision

    db_file = str(tmp_path / "persist_checkpoints.sqlite")
    thread_config = {"configurable": {"thread_id": "persist-thread-abc"}}

    # Session 1: write a message
    async with aiosqlite.connect(db_file) as conn:
        agent = create_agent(checkpointer=AsyncSqliteSaver(conn))
        with (
            patch("agenticrag.nodes.get_router_llm") as mock_router,
            patch("agenticrag.nodes.get_llm") as mock_llm,
        ):
            mock_router.return_value.ainvoke = AsyncMock(return_value=RouteDecision(route="direct"))
            mock_llm.return_value.ainvoke = AsyncMock(
                return_value=AIMessage(content="Session 1 reply.")
            )
            await agent.ainvoke(
                {"messages": [HumanMessage(content="Session 1 message")]}, config=thread_config
            )

    # Session 2: open same DB file, state should still be there
    async with aiosqlite.connect(db_file) as conn2:
        agent2 = create_agent(checkpointer=AsyncSqliteSaver(conn2))
        state = await agent2.aget_state(thread_config)
        assert len(state.values["messages"]) >= 2  # human + ai from session 1


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


@pytest.mark.asyncio
async def test_generate_from_kb_success():
    from agenticrag.nodes import generate_from_kb

    state = {
        "messages": [HumanMessage(content="What is the return window?")],
        "kb_docs": [
            Document(
                page_content="Returns are accepted within 30 days.", metadata={"source": "faq"}
            )
        ],
    }
    with patch("agenticrag.nodes.get_llm") as mock_llm:
        mock_llm.return_value.ainvoke = AsyncMock(
            return_value=AIMessage(content="You have 30 days to return your purchase.")
        )
        result = await generate_from_kb(state)
        assert result["source_used"] == "private_kb"
        assert "30 days" in result["messages"][0].content


@pytest.mark.asyncio
async def test_generate_from_kb_llm_failure_containment():
    from agenticrag.nodes import generate_from_kb

    state = {
        "messages": [HumanMessage(content="What is the return window?")],
        "kb_docs": [
            Document(
                page_content="Returns are accepted within 30 days.", metadata={"source": "faq"}
            )
        ],
    }
    with patch("agenticrag.nodes.get_llm") as mock_llm:
        mock_llm.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("Groq inference 503"))
        result = await generate_from_kb(state)
        assert result["source_used"] == "insufficient_evidence"
        assert "temporary service issue" in result["messages"][0].content


@pytest.mark.asyncio
async def test_generate_from_web_success():
    from agenticrag.nodes import generate_from_web

    state = {
        "messages": [HumanMessage(content="When does the next sale start?")],
        "web_results": "The annual sale starts on November 15th.",
    }
    with patch("agenticrag.nodes.get_llm") as mock_llm:
        mock_llm.return_value.ainvoke = AsyncMock(
            return_value=AIMessage(content="The upcoming sale begins on November 15th.")
        )
        result = await generate_from_web(state)
        assert result["source_used"] == "web_search"
        assert "November 15th" in result["messages"][0].content


@pytest.mark.asyncio
async def test_generate_from_web_llm_failure_containment():
    from agenticrag.nodes import generate_from_web

    state = {
        "messages": [HumanMessage(content="When does the next sale start?")],
        "web_results": "The annual sale starts on November 15th.",
    }
    with patch("agenticrag.nodes.get_llm") as mock_llm:
        mock_llm.return_value.ainvoke = AsyncMock(
            side_effect=TimeoutError("Groq request timed out")
        )
        result = await generate_from_web(state)
        assert result["source_used"] == "insufficient_evidence"
        assert "temporary service issue" in result["messages"][0].content


@pytest.mark.asyncio
async def test_direct_answer_success():
    from agenticrag.nodes import direct_answer

    state = {"messages": [HumanMessage(content="Hi there!")]}
    with patch("agenticrag.nodes.get_llm") as mock_llm:
        mock_llm.return_value.ainvoke = AsyncMock(
            return_value=AIMessage(content="Hello! How can I help you today?")
        )
        result = await direct_answer(state)
        assert result["source_used"] == "direct"
        assert "Hello!" in result["messages"][0].content


@pytest.mark.asyncio
async def test_direct_answer_llm_failure_containment():
    from agenticrag.nodes import direct_answer

    state = {"messages": [HumanMessage(content="Hi there!")]}
    with patch("agenticrag.nodes.get_llm") as mock_llm:
        mock_llm.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("Groq rate limit"))
        result = await direct_answer(state)
        assert result["source_used"] == "direct"
        assert "Hello!" in result["messages"][0].content


@pytest.mark.asyncio
async def test_answer_insufficient_success():
    from agenticrag.nodes import answer_insufficient

    state = {"messages": [HumanMessage(content="Tell me proprietary secrets")]}
    with patch("agenticrag.nodes.get_llm") as mock_llm:
        mock_llm.return_value.ainvoke = AsyncMock(
            return_value=AIMessage(content="I do not have sufficient information to answer that.")
        )
        result = await answer_insufficient(state)
        assert result["source_used"] == "insufficient_evidence"
        assert "sufficient information" in result["messages"][0].content


@pytest.mark.asyncio
async def test_answer_insufficient_llm_failure_containment():
    from agenticrag.nodes import answer_insufficient

    state = {"messages": [HumanMessage(content="Unknown issue")]}
    with patch("agenticrag.nodes.get_llm") as mock_llm:
        mock_llm.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("Network error"))
        result = await answer_insufficient(state)
        assert result["source_used"] == "insufficient_evidence"
        assert "Live Chat" in result["messages"][0].content
