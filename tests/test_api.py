"""Integration tests for FastAPI endpoints, input caps, and thread security."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import agenticrag.api as api_module
from agenticrag.api import app, clean_thread_id


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


def test_cors_allowed_origin(client):
    response = client.options("/health", headers={
        "Origin": "https://agentic-customer-support-rag.vercel.app",
        "Access-Control-Request-Method": "GET"
    })
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://agentic-customer-support-rag.vercel.app"


def test_cors_arbitrary_vercel_origin_rejected(client):
    response = client.options("/health", headers={
        "Origin": "https://arbitrary-attacker.vercel.app",
        "Access-Control-Request-Method": "GET"
    })
    # Since allow_origin_regex was removed, arbitrary vercel.app domains must not be allowed
    assert response.status_code == 400 or response.headers.get("access-control-allow-origin") is None


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
    assert data["dimension"] == 384
    assert data["sample_size"] == 500
    assert data["retriever_k"] == 4
    assert "bitext" in data["dataset"]


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


def test_require_api_key_refuses_startup(monkeypatch):
    """When REQUIRE_API_KEY is true and API_KEY is empty, lifespan refuses to start."""
    import asyncio

    from agenticrag.api import app, lifespan
    from agenticrag.config import Settings

    fake_settings = Settings(require_api_key=True, api_key="")
    monkeypatch.setattr(api_module, "get_settings", lambda: fake_settings)

    async def run_lifespan():
        async with lifespan(app):
            pass

    with pytest.raises(RuntimeError, match="REQUIRE_API_KEY is true, but API_KEY is empty"):
        asyncio.run(run_lifespan())


def test_verify_api_key_auth(client, monkeypatch):
    """Verifies that when API_KEY is configured, requests require matching X-API-Key header."""
    from agenticrag.config import Settings
    fake_settings = Settings(api_key="secret-token-123")
    monkeypatch.setattr(api_module, "get_settings", lambda: fake_settings)

    # Missing API key -> 401
    res_missing = client.get("/kb/info")
    assert res_missing.status_code == 401

    # Wrong API key -> 401
    res_wrong = client.get("/kb/info", headers={"X-API-Key": "wrong-token"})
    assert res_wrong.status_code == 401

    # Correct API key -> 200
    res_correct = client.get("/kb/info", headers={"X-API-Key": "secret-token-123"})
    assert res_correct.status_code == 200

