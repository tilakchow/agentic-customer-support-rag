"""Unit and integration tests for M6 Response Cache and Data Lifecycle."""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from starlette.testclient import TestClient

import agenticrag.api as api_module
from agenticrag.api import app
from agenticrag.cache import ResponseCache, get_response_cache
from agenticrag.ingest import diff_chunks_against_manifest, enrich_chunks_with_lifecycle_metadata


@pytest.fixture(autouse=True)
def clean_cache():
    """Clears global response cache before and after each test."""
    get_response_cache().clear()
    yield
    get_response_cache().clear()


def test_response_cache_set_and_get():
    """Caches response data and retrieves it by query and tenant."""
    cache = ResponseCache()
    query = "What is the return policy?"
    data = {"response": "You can return items within 30 days.", "source_used": "private_kb"}

    cache.set(query=query, response_data=data, tenant_id="tenant-1", kb_version="v2.1.0")

    # Hit same query & tenant & version
    hit = cache.get(query=query, tenant_id="tenant-1", kb_version="v2.1.0")
    assert hit is not None
    assert hit["response"] == "You can return items within 30 days."

    # Miss on different tenant
    miss_tenant = cache.get(query=query, tenant_id="tenant-2", kb_version="v2.1.0")
    assert miss_tenant is None

    # Miss on different KB version
    miss_version = cache.get(query=query, tenant_id="tenant-1", kb_version="v2.2.0")
    assert miss_version is None


def test_response_cache_invalidation_on_version_bump():
    """Invalidates old cache entries when KB version changes."""
    cache = ResponseCache()
    cache.set("order status", {"response": "Shipped"}, tenant_id="default", kb_version="v1.0")
    cache.set("return policy", {"response": "30 days"}, tenant_id="default", kb_version="v2.0")

    assert cache.get("order status", tenant_id="default", kb_version="v1.0") is not None

    # Invalidate v1.0
    evicted = cache.invalidate_version("v1.0")
    assert evicted == 1
    assert cache.get("order status", tenant_id="default", kb_version="v1.0") is None
    assert cache.get("return policy", tenant_id="default", kb_version="v2.0") is not None


def test_enrich_chunks_with_lifecycle_metadata():
    """Enriches chunk metadata with version, effective_date, and content_hash."""
    doc = Document(page_content="Return window is 30 days.", metadata={"intent": "returns"})
    chunks, ids = enrich_chunks_with_lifecycle_metadata(
        [doc], kb_version="v3.0.0", effective_date="2026-10-27"
    )

    assert len(chunks) == 1
    assert len(ids) == 1
    meta = chunks[0].metadata
    assert meta["version"] == "v3.0.0"
    assert meta["effective_date"] == "2026-10-27"
    assert "content_hash" in meta
    assert len(meta["content_hash"]) == 64


def test_diff_chunks_against_manifest_detects_changes():
    """Compares current chunks against prior manifest to skip unchanged chunks."""
    doc1 = Document(
        page_content="Policy A", metadata={"intent": "policy", "content_hash": "hash_a"}
    )
    doc2 = Document(
        page_content="Policy B", metadata={"intent": "policy", "content_hash": "hash_b"}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = os.path.join(tmpdir, "manifest.json")

        # Initial run: both docs are new
        chunks_upsert, ids_upsert, manifest = diff_chunks_against_manifest(
            [doc1, doc2], ["id1", "id2"], manifest_path
        )
        assert len(chunks_upsert) == 2
        assert len(ids_upsert) == 2

        # Save manifest
        import json

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        # Second run: identical chunks -> 0 chunks to upsert
        chunks_upsert2, ids_upsert2, _ = diff_chunks_against_manifest(
            [doc1, doc2], ["id1", "id2"], manifest_path
        )
        assert len(chunks_upsert2) == 0

        # Third run: doc2 content modified -> 1 chunk to upsert
        doc2_modified = Document(
            page_content="Policy B updated",
            metadata={"intent": "policy", "content_hash": "hash_b_modified"},
        )
        chunks_upsert3, ids_upsert3, _ = diff_chunks_against_manifest(
            [doc1, doc2_modified], ["id1", "id2"], manifest_path
        )
        assert len(chunks_upsert3) == 1
        assert ids_upsert3 == ["id2"]


def test_chat_endpoint_serves_from_cache(monkeypatch):
    """End-to-end /chat serves repeated query from cache on second call."""
    client = TestClient(app)
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(
        return_value={
            "messages": [AIMessage(content="First compute response")],
            "source_used": "private_kb",
        }
    )
    monkeypatch.setattr(api_module, "get_current_agent", lambda: mock_agent)

    # First call: computes via agent
    resp1 = client.post(
        "/chat", json={"message": "How do I cancel my order?", "thread_id": "cache-t1"}
    )
    assert resp1.status_code == 200
    assert resp1.json()["response"] == "First compute response"
    assert mock_agent.ainvoke.call_count == 1

    # Second call: served from cache without invoking agent
    resp2 = client.post(
        "/chat", json={"message": "How do I cancel my order?", "thread_id": "cache-t1"}
    )
    assert resp2.status_code == 200
    assert resp2.json()["response"] == "First compute response"
    assert resp2.json()["source_used"] == "cached_kb"
    # Agent should not have been called a second time
    assert mock_agent.ainvoke.call_count == 1
