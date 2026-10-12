"""Unit tests for the knowledge base ingestion pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from agenticrag.config import Settings
from agenticrag.ingest import run_ingestion


def test_ingest_fails_without_pinecone_api_key(monkeypatch):
    """Ingestion fails fast if PINECONE_API_KEY is not configured."""
    import agenticrag.ingest as ingest_module

    no_key_settings = Settings(pinecone_api_key="")
    monkeypatch.setattr(ingest_module, "get_settings", lambda: no_key_settings)

    with pytest.raises(RuntimeError, match="PINECONE_API_KEY environment variable is not set"):
        run_ingestion()


def test_ingest_success_creates_index_if_missing(monkeypatch):
    """Ingestion pipeline loads dataset, chunks documents, creates index if absent, and upserts."""
    import agenticrag.ingest as ingest_module

    settings = Settings(
        pinecone_api_key="mock-pinecone-key",
        index_name="test-index",
        namespace="test-ns",
        kb_sample_size=2,
    )
    monkeypatch.setattr(ingest_module, "get_settings", lambda: settings)

    fake_data = [
        {
            "instruction": "How do I return?",
            "response": "You have 30 days.",
            "category": "returns",
            "intent": "return_inquiry",
        },
        {
            "instruction": "What is shipping fee?",
            "response": "Shipping is free.",
            "category": "shipping",
            "intent": "shipping_fee",
        },
    ]

    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 2
    mock_shuffled = MagicMock()
    mock_selected = MagicMock()
    mock_selected.__iter__.return_value = iter(fake_data)
    mock_shuffled.select.return_value = mock_selected
    mock_dataset.shuffle.return_value = mock_shuffled

    mock_pc = MagicMock()
    mock_pc.list_indexes.return_value = []  # Index does not exist yet

    with (
        patch("datasets.load_dataset", return_value=mock_dataset),
        patch("agenticrag.ingest.Pinecone", return_value=mock_pc),
        patch("agenticrag.ingest.get_embeddings") as mock_get_embeddings,
        patch("agenticrag.ingest.PineconeVectorStore.from_documents") as mock_from_docs,
    ):
        mock_get_embeddings.return_value = MagicMock()
        run_ingestion()

        # Verifies index creation was triggered
        mock_pc.create_index.assert_called_once()
        call_kwargs = mock_pc.create_index.call_args.kwargs
        assert call_kwargs["name"] == "test-index"
        assert call_kwargs["metric"] == "cosine"

        # Verifies documents were upserted with deterministic IDs
        mock_from_docs.assert_called_once()
        docs_arg = mock_from_docs.call_args.kwargs["documents"]
        ids_arg = mock_from_docs.call_args.kwargs["ids"]
        assert len(docs_arg) >= 2
        assert len(ids_arg) == len(docs_arg)
        assert mock_from_docs.call_args.kwargs["index_name"] == "test-index"
        assert mock_from_docs.call_args.kwargs["namespace"] == "test-ns"


def test_ingest_success_skips_create_if_index_exists(monkeypatch):
    """Ingestion pipeline does not call create_index when index is already listed."""
    import agenticrag.ingest as ingest_module

    settings = Settings(
        pinecone_api_key="mock-pinecone-key",
        index_name="existing-index",
        namespace="test-ns",
        kb_sample_size=1,
    )
    monkeypatch.setattr(ingest_module, "get_settings", lambda: settings)

    fake_data = [
        {"instruction": "Hi", "response": "Hello", "category": "general", "intent": "greeting"},
    ]

    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 1
    mock_shuffled = MagicMock()
    mock_selected = MagicMock()
    mock_selected.__iter__.return_value = iter(fake_data)
    mock_shuffled.select.return_value = mock_selected
    mock_dataset.shuffle.return_value = mock_shuffled

    mock_pc = MagicMock()
    mock_pc.list_indexes.return_value = [{"name": "existing-index"}]

    with (
        patch("datasets.load_dataset", return_value=mock_dataset),
        patch("agenticrag.ingest.Pinecone", return_value=mock_pc),
        patch("agenticrag.ingest.get_embeddings"),
        patch("agenticrag.ingest.PineconeVectorStore.from_documents"),
    ):
        run_ingestion()
        mock_pc.create_index.assert_not_called()
