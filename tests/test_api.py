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
    response = client.options(
        "/health",
        headers={
            "Origin": "https://agentic-customer-support-rag.vercel.app",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert (
        response.headers.get("access-control-allow-origin")
        == "https://agentic-customer-support-rag.vercel.app"
    )


def test_cors_arbitrary_vercel_origin_rejected(client):
    response = client.options(
        "/health",
        headers={
            "Origin": "https://arbitrary-attacker.vercel.app",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Since allow_origin_regex was removed, arbitrary vercel.app domains must not be allowed
    assert (
        response.status_code == 400 or response.headers.get("access-control-allow-origin") is None
    )


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
    monkeypatch.setattr("agenticrag.auth.get_settings", lambda: fake_settings)
    monkeypatch.setattr("agenticrag.api.get_settings", lambda: fake_settings)

    # Missing API key -> 401
    res_missing = client.get("/kb/info")
    assert res_missing.status_code == 401

    # Wrong API key -> 403
    res_wrong = client.get("/kb/info", headers={"X-API-Key": "wrong-token"})
    assert res_wrong.status_code == 403

    # Correct API key -> 200
    res_correct = client.get("/kb/info", headers={"X-API-Key": "secret-token-123"})
    assert res_correct.status_code == 200


def test_verify_api_key_unset_is_public(client, monkeypatch):
    """When API_KEY is empty/unset, endpoints are public and succeed without headers."""
    from agenticrag.config import Settings

    fake_settings = Settings(api_key="")
    monkeypatch.setattr("agenticrag.auth.get_settings", lambda: fake_settings)
    monkeypatch.setattr("agenticrag.api.get_settings", lambda: fake_settings)

    response = client.get("/kb/info")
    assert response.status_code == 200


def test_chat_success_endpoint(client, monkeypatch):
    """Verifies that /chat returns valid JSON response with thread continuity."""
    from unittest.mock import AsyncMock

    from langchain_core.messages import AIMessage

    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(
        return_value={
            "messages": [AIMessage(content="Your package was delivered yesterday.")],
            "source_used": "private_kb",
        }
    )
    monkeypatch.setattr(api_module, "get_current_agent", lambda: mock_agent)

    response = client.post(
        "/chat", json={"message": "Did my order arrive?", "thread_id": "cust-session-999"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Your package was delivered yesterday."
    assert data["thread_id"] == "cust-session-999"
    assert data["source_used"] == "private_kb"


def test_chat_error_path_sanitized(client, monkeypatch):
    """Verifies that backend exceptions on /chat yield sanitized 500 error responses without leak."""
    from unittest.mock import AsyncMock

    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(side_effect=RuntimeError("INTERNAL_DATABASE_CREDENTIALS_LEAK"))
    monkeypatch.setattr(api_module, "get_current_agent", lambda: mock_agent)

    response = client.post("/chat", json={"message": "Trigger failure"})
    assert response.status_code == 500
    assert "INTERNAL_DATABASE_CREDENTIALS_LEAK" not in response.text
    data = response.json()
    assert "An error occurred while processing your request" in data["detail"]


def test_kb_search_retriever_unavailable(client, monkeypatch):
    """Verifies that /kb/search returns 503 when the retriever is None."""
    monkeypatch.setattr(api_module, "get_retriever", lambda: None)
    response = client.post("/kb/search", json={"query": "return window"})
    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]


def test_clean_thread_id_hostile_input():
    """Verifies that malicious or special-character thread IDs are sanitized to safe slugs."""
    # Path traversal attack
    sanitized_path = clean_thread_id("../../../etc/passwd")
    assert "/" not in sanitized_path and "." not in sanitized_path
    assert "etcpasswd" in sanitized_path

    # XSS script injection
    sanitized_xss = clean_thread_id("<script>alert('xss')</script>")
    assert "<" not in sanitized_xss and ">" not in sanitized_xss and "'" not in sanitized_xss
    assert "scriptalertxssscript" in sanitized_xss

    # Pure special characters fallback to valid uuid
    sanitized_symbols = clean_thread_id("!@#$%^&*()")
    assert len(sanitized_symbols) > 10


def test_production_environment_refuses_startup_without_api_key(monkeypatch):
    """When ENVIRONMENT is production and API_KEY is empty, lifespan refuses to start."""
    import asyncio

    from agenticrag.api import app, lifespan
    from agenticrag.config import Settings

    fake_settings = Settings(environment="production", api_key="")
    monkeypatch.setattr(api_module, "get_settings", lambda: fake_settings)

    async def run_lifespan():
        async with lifespan(app):
            pass

    with pytest.raises(RuntimeError, match="Production mode .* requires API_KEY"):
        asyncio.run(run_lifespan())


def test_get_client_rate_limit_key_resolution(monkeypatch):
    """Verifies that rate limit key properly verifies API keys against spoofing and resolves client IP."""
    from starlette.requests import Request

    from agenticrag.api import get_client_rate_limit_key
    from agenticrag.config import Settings

    # Configure server with genuine secret
    monkeypatch.setattr(api_module, "get_settings", lambda: Settings(api_key="secret-token-xyz"))

    # 1. Genuine API key receives dedicated bucket
    req_valid_key = Request(
        scope={
            "type": "http",
            "headers": [(b"x-api-key", b"secret-token-xyz")],
            "client": ("192.168.1.1", 12345),
        }
    )
    assert get_client_rate_limit_key(req_valid_key) == "key:secret-token-xyz"

    # 2. Fake / spoofed API key is NOT trusted and falls back to client IP
    req_fake_key = Request(
        scope={
            "type": "http",
            "headers": [(b"x-api-key", b"fake-random-key")],
            "client": ("192.168.1.1", 12345),
        }
    )
    assert get_client_rate_limit_key(req_fake_key) == "192.168.1.1"

    # 3. Direct client host resolution
    req_direct = Request(
        scope={
            "type": "http",
            "headers": [],
            "client": ("192.0.2.1", 12345),
        }
    )
    assert get_client_rate_limit_key(req_direct) == "192.0.2.1"


def test_settings_allowed_origins_from_env(monkeypatch):
    """Verifies that comma-separated ALLOWED_ORIGINS string in env parses cleanly without NoDecode errors."""
    from agenticrag.config import Settings

    monkeypatch.setenv("ALLOWED_ORIGINS", "http://app.example.com, https://portal.example.com")
    settings = Settings()
    assert settings.allowed_origins == ["http://app.example.com", "https://portal.example.com"]


def test_kb_search_endpoint_success(client, monkeypatch):
    """Verifies /kb/search returns formatted results when retriever finds documents."""
    from langchain_core.documents import Document

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="Our return window is 30 days.", metadata={"source": "kb-policy"})
    ]
    monkeypatch.setattr(api_module, "get_retriever", lambda: mock_retriever)

    response = client.post("/kb/search", json={"query": "return window"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "return window"
    assert len(data["results"]) == 1
    assert "return window is 30 days" in data["results"][0]["content"]
    assert data["results"][0]["metadata"]["source"] == "kb-policy"
