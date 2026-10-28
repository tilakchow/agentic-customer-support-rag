"""Unit and integration tests for M5 Auth, Multi-Tenancy, and Thread Isolation."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage
from starlette.testclient import TestClient

import agenticrag.api as api_module
from agenticrag.api import app
from agenticrag.auth import (
    create_jwt,
    decode_jwt,
    get_tenancy_manager,
    get_tenant_namespace,
)


@pytest.fixture(autouse=True)
def mock_agent_app(monkeypatch):
    """Mocks agent app to allow fast offline testing of auth and tenancy."""
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(
        return_value={
            "messages": [AIMessage(content="OK response")],
            "source_used": "direct",
        }
    )
    monkeypatch.setattr(api_module, "get_current_agent", lambda: mock_agent)
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    monkeypatch.setattr(api_module, "get_retriever", lambda: mock_retriever)
    return mock_agent


@pytest.fixture(autouse=True)
def clean_tenancy_state():
    """Resets the thread tenancy manager before and after each test."""
    get_tenancy_manager().reset()
    yield
    get_tenancy_manager().reset()


def test_jwt_encode_and_decode_cycle():
    """Generates a valid HMAC-SHA256 JWT and verifies claims."""
    payload = {"sub": "user-42", "tenant_id": "tenant-corp", "role": "admin"}
    token = create_jwt(payload)

    decoded = decode_jwt(token)
    assert decoded["sub"] == "user-42"
    assert decoded["tenant_id"] == "tenant-corp"
    assert decoded["role"] == "admin"


def test_jwt_tampered_signature_rejection():
    """Rejects JWTs where signature or payload has been tampered with."""
    payload = {"sub": "user-evil", "tenant_id": "tenant-corp"}
    token = create_jwt(payload)
    tampered = token[:-4] + "fake"

    with pytest.raises(ValueError, match="signature verification failed"):
        decode_jwt(tampered)


def test_tenant_namespace_resolution():
    """Calculates tenant-isolated Pinecone namespaces correctly."""
    assert get_tenant_namespace("default") == "bitext-support-v2"
    assert get_tenant_namespace("") == "bitext-support-v2"
    assert get_tenant_namespace("tenant-acme") == "tenant-acme-bitext-support-v2"


def test_cross_user_thread_hijacking_rejection():
    """Ensures User B cannot send messages to User A's thread."""
    client = TestClient(app)

    token_user_a = create_jwt({"sub": "user-a", "tenant_id": "tenant-alpha"})
    token_user_b = create_jwt({"sub": "user-b", "tenant_id": "tenant-alpha"})

    # User A starts thread-shared-123
    resp_a = client.post(
        "/chat",
        json={"message": "Hello from user A", "thread_id": "thread-shared-123"},
        headers={"Authorization": f"Bearer {token_user_a}"},
    )
    assert resp_a.status_code == 200

    # User B attempts to hijack thread-shared-123 -> 403 Forbidden
    resp_b = client.post(
        "/chat",
        json={"message": "Hijack attempt from user B", "thread_id": "thread-shared-123"},
        headers={"Authorization": f"Bearer {token_user_b}"},
    )
    assert resp_b.status_code == 403
    assert "Cross-user" in resp_b.json()["detail"]


def test_cross_tenant_thread_access_rejection():
    """Ensures Tenant Beta cannot access Tenant Alpha's thread."""
    client = TestClient(app)

    token_tenant_alpha = create_jwt({"sub": "user-corp", "tenant_id": "tenant-alpha"})
    token_tenant_beta = create_jwt({"sub": "user-corp", "tenant_id": "tenant-beta"})

    # Tenant Alpha starts thread-tenant-test
    resp_alpha = client.post(
        "/chat",
        json={"message": "Hello from alpha", "thread_id": "thread-tenant-test"},
        headers={"Authorization": f"Bearer {token_tenant_alpha}"},
    )
    assert resp_alpha.status_code == 200

    # Tenant Beta attempts to access thread-tenant-test -> 403 Forbidden
    resp_beta = client.post(
        "/chat",
        json={"message": "Access from beta", "thread_id": "thread-tenant-test"},
        headers={"Authorization": f"Bearer {token_tenant_beta}"},
    )
    assert resp_beta.status_code == 403
    assert "Cross-tenant" in resp_beta.json()["detail"]


def test_scoped_kb_search_tenant_isolation():
    """Direct /kb/search endpoint reflects caller tenant namespace."""
    client = TestClient(app)

    token = create_jwt({"sub": "analyst-1", "tenant_id": "tenant-gamma"})
    response = client.post(
        "/kb/search",
        json={"query": "refund policy"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["tenant_id"] == "tenant-gamma"
    assert data["namespace"] == "tenant-gamma-bitext-support-v2"


@pytest.mark.asyncio
async def test_retrieve_kb_node_tenant_namespace_isolation(monkeypatch):
    """Verifies nodes.retrieve_kb queries vectorstore using caller tenant namespace."""
    from agenticrag.nodes import retrieve_kb

    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = []
    monkeypatch.setattr("agenticrag.nodes.get_vectorstore", lambda: mock_vs)

    # State with Tenant Alpha
    state_alpha = {
        "current_query": "how do I cancel?",
        "tenant_id": "tenant-alpha",
    }
    await retrieve_kb(state_alpha)
    assert mock_vs.similarity_search_with_score.call_args is not None
    assert mock_vs.similarity_search_with_score.call_args.kwargs.get(
        "namespace"
    ) == get_tenant_namespace("tenant-alpha")

    # State with Tenant Beta
    state_beta = {
        "current_query": "how do I cancel?",
        "tenant_id": "tenant-beta",
    }
    await retrieve_kb(state_beta)
    assert mock_vs.similarity_search_with_score.call_args.kwargs.get(
        "namespace"
    ) == get_tenant_namespace("tenant-beta")


def test_chat_and_stream_pass_tenant_id_to_graph(monkeypatch):
    """Asserts /chat and /chat/stream forward caller tenant_id to initial graph input."""
    client = TestClient(app)
    recorded_inputs = []

    mock_agent = MagicMock()

    async def fake_ainvoke(input_data, config=None):
        recorded_inputs.append(("chat", input_data, config))
        return {
            "messages": [AIMessage(content="Response for tenant")],
            "source_used": "direct",
        }

    async def fake_astream_events(input_data, config=None, version="v2"):
        recorded_inputs.append(("stream", input_data, config))
        if False:
            yield {}

    async def fake_aget_state(config=None):
        mock_snapshot = MagicMock()
        mock_snapshot.values = {
            "messages": [AIMessage(content="Response for tenant")],
            "source_used": "direct",
        }
        return mock_snapshot

    mock_agent.ainvoke = fake_ainvoke
    mock_agent.astream_events = fake_astream_events
    mock_agent.aget_state = fake_aget_state
    monkeypatch.setattr(api_module, "get_current_agent", lambda: mock_agent)

    token_alpha = create_jwt({"sub": "user-1", "tenant_id": "tenant-alpha"})
    token_beta = create_jwt({"sub": "user-2", "tenant_id": "tenant-beta"})

    # 1. /chat with tenant-alpha
    resp_chat = client.post(
        "/chat",
        json={"message": "Query alpha", "thread_id": "th-chat-alpha"},
        headers={"Authorization": f"Bearer {token_alpha}"},
    )
    assert resp_chat.status_code == 200
    assert any(
        kind == "chat" and data.get("tenant_id") == "tenant-alpha"
        for kind, data, _ in recorded_inputs
    )

    # 2. /chat/stream with tenant-beta
    resp_stream = client.post(
        "/chat/stream",
        json={"message": "Query beta", "thread_id": "th-stream-beta"},
        headers={"Authorization": f"Bearer {token_beta}"},
    )
    assert resp_stream.status_code == 200
    # Consume stream
    _ = resp_stream.text
    assert any(
        kind == "stream" and data.get("tenant_id") == "tenant-beta"
        for kind, data, _ in recorded_inputs
    )
