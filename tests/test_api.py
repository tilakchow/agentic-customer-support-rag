"""Integration tests for FastAPI endpoints, input caps, and thread security."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from agenticrag.api import app, clean_thread_id
import agenticrag.api as api_module


@pytest.fixture(autouse=True)
def mock_retriever(monkeypatch):
    """Mocks vector retriever to keep test execution 100% offline and sub-second."""
    mock_ret = MagicMock()
    mock_ret.invoke.return_value = []
    monkeypatch.setattr(api_module, "get_retriever", lambda: mock_ret)
    return mock_ret


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "service" in data


def test_cors_vercel_allowed(client):
    response = client.options("/health", headers={
        "Origin": "https://agentic-customer-support-rag.vercel.app",
        "Access-Control-Request-Method": "GET"
    })
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://agentic-customer-support-rag.vercel.app"


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_kb_info_endpoint(client):
    response = client.get("/kb/info")
    assert response.status_code == 200
    data = response.json()
    assert "index_name" in data
    assert "namespace" in data
    assert "embedding_model" in data


def test_chat_empty_message_validation(client):
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_whitespace_message_validation(client):
    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 422


def test_chat_message_length_limit(client):
    too_long = "x" * 1501
    response = client.post("/chat", json={"message": too_long})
    assert response.status_code == 422


def test_clean_thread_id():
    # Whitespace bug verification
    assert clean_thread_id("   ") != ""
    assert len(clean_thread_id("   ")) > 10

    # Normal thread ID preserved
    assert clean_thread_id("user-session-123") == "user-session-123"

    # None thread ID generates a unique UUID
    val1 = clean_thread_id(None)
    val2 = clean_thread_id(None)
    assert val1 != val2
    assert len(val1) > 10
