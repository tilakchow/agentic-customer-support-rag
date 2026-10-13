"""Unit tests for clients lazy factories, caching, reset_clients, and failure backoff."""

import time
from unittest.mock import MagicMock, patch

import pytest

from agenticrag.clients import (
    RETRIEVER_BACKOFF_SECONDS,
    get_embeddings,
    get_grader_llm,
    get_llm,
    get_retriever,
    get_router_llm,
    get_vectorstore,
    get_web_search,
    reset_clients,
    warm_up_clients,
)
from agenticrag.metrics import (
    get_provider_metrics,
    record_provider_failure,
    reset_metrics,
)


@pytest.fixture(autouse=True)
def clean_client_state():
    """Ensure every test begins and ends with clean factory cache state."""
    reset_clients()
    reset_metrics()
    yield
    reset_clients()
    reset_metrics()


def test_get_llm_cached():
    """get_llm uses LRU cache to return the same client instance on repeated calls."""
    with patch("langchain_groq.ChatGroq") as mock_groq:
        mock_instance = MagicMock()
        mock_groq.return_value = mock_instance

        llm1 = get_llm()
        llm2 = get_llm()

        assert llm1 is llm2
        assert mock_groq.call_count == 1


def test_reset_clients_clears_cache():
    """reset_clients clears the cache so subsequent calls reinitialize the client."""
    with patch("langchain_groq.ChatGroq") as mock_groq:
        mock_groq.return_value = MagicMock()

        _ = get_llm()
        assert mock_groq.call_count == 1

        reset_clients()

        _ = get_llm()
        assert mock_groq.call_count == 2


def test_get_router_and_grader_llm():
    """get_router_llm and get_grader_llm configure structured output correctly."""
    with patch("langchain_groq.ChatGroq") as mock_groq:
        mock_instance = MagicMock()
        mock_groq.return_value = mock_instance

        router = get_router_llm()
        mock_instance.with_structured_output.assert_called()
        assert router is not None

        grader = get_grader_llm()
        assert grader is not None


def test_get_web_search_cached():
    """get_web_search returns cached TavilySearch instance."""
    with patch("langchain_tavily.TavilySearch") as mock_tavily:
        mock_instance = MagicMock()
        mock_tavily.return_value = mock_instance

        ws1 = get_web_search()
        ws2 = get_web_search()

        assert ws1 is ws2
        assert mock_tavily.call_count == 1


def test_get_embeddings_cached():
    """get_embeddings returns cached FastEmbedEmbeddings instance."""
    with patch("langchain_community.embeddings.fastembed.FastEmbedEmbeddings") as mock_fastembed:
        mock_instance = MagicMock()
        mock_fastembed.return_value = mock_instance

        emb1 = get_embeddings()
        emb2 = get_embeddings()

        assert emb1 is emb2
        assert mock_fastembed.call_count == 1


def test_get_vectorstore_returns_none_when_api_key_empty():
    """get_vectorstore returns None when PINECONE_API_KEY is not configured."""
    with patch("agenticrag.clients.get_settings") as mock_settings:
        mock_settings.return_value.pinecone_api_key = ""
        vs = get_vectorstore()
        assert vs is None


def test_get_vectorstore_failure_backoff():
    """When Pinecone connection fails, backoff window suppresses retries for 30 seconds."""
    with patch("agenticrag.clients.get_settings") as mock_settings:
        mock_settings.return_value.pinecone_api_key = "test-key"
        mock_settings.return_value.index_name = "test-index"
        mock_settings.return_value.namespace = "test-ns"

        with patch("pinecone.Pinecone", side_effect=RuntimeError("Connection refused")):
            # First attempt fails and sets backoff timestamp
            vs1 = get_vectorstore()
            assert vs1 is None

            # Immediate second attempt within 30s returns None without calling Pinecone
            with patch("pinecone.Pinecone") as mock_pc:
                vs2 = get_vectorstore()
                assert vs2 is None
                mock_pc.assert_not_called()


def test_get_vectorstore_backoff_expiry():
    """When 30s backoff expires, subsequent get_vectorstore call retries connection."""
    import agenticrag.clients as clients_mod

    with patch("agenticrag.clients.get_settings") as mock_settings:
        mock_settings.return_value.pinecone_api_key = "test-key"
        mock_settings.return_value.index_name = "test-index"
        mock_settings.return_value.namespace = "test-ns"

        # Simulate a past failure that occurred 35 seconds ago
        clients_mod._last_retriever_failure_time = time.time() - (RETRIEVER_BACKOFF_SECONDS + 5)

        with patch("pinecone.Pinecone") as mock_pc:
            mock_index = MagicMock()
            mock_pc.return_value.Index.return_value = mock_index
            with (
                patch("langchain_pinecone.PineconeVectorStore") as mock_pvs,
                patch("agenticrag.clients.get_embeddings") as mock_emb,
            ):
                mock_pvs.return_value = MagicMock()
                mock_emb.return_value = MagicMock()
                vs = get_vectorstore()
                assert vs is not None
                assert mock_pc.called


def test_get_retriever_none_when_vectorstore_unavailable():
    """get_retriever returns None if get_vectorstore returns None."""
    with patch("agenticrag.clients.get_vectorstore", return_value=None):
        retriever = get_retriever()
        assert retriever is None


def test_warm_up_clients_tolerates_missing_keys():
    """warm_up_clients executes cleanly even when external credentials are empty."""
    with patch("agenticrag.clients.get_settings") as mock_settings:
        mock_settings.return_value.groq_api_key = ""
        mock_settings.return_value.pinecone_api_key = ""
        # Should not raise
        warm_up_clients()


def test_metrics_recording_and_reset():
    """Provider failure telemetry increments accurately and resets cleanly."""
    assert get_provider_metrics()["provider_failures"] == 0

    record_provider_failure(ValueError("LLM rate limit reached"))
    metrics = get_provider_metrics()
    assert metrics["provider_failures"] == 1
    assert "LLM rate limit reached" in metrics["last_error"]

    reset_metrics()
    assert get_provider_metrics()["provider_failures"] == 0
