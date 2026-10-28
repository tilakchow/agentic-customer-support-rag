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

    with pytest.raises(ValueError, match="JWT validation error:"):
        decode_jwt(tampered)


def test_jwt_expired_token_rejection():
    """Expired tokens must fail validation and return 401 on endpoints."""
    from datetime import UTC, datetime, timedelta

    import jwt

    from agenticrag.config import get_settings

    expired_payload = {
        "sub": "user-expired",
        "tenant_id": "tenant-corp",
        "exp": datetime.now(UTC) - timedelta(hours=2),
    }
    expired_token = jwt.encode(expired_payload, get_settings().jwt_secret, algorithm="HS256")

    with pytest.raises(ValueError, match="JWT validation error:.*expired"):
        decode_jwt(expired_token)

    client = TestClient(app)
    resp = client.post(
        "/chat",
        json={"message": "hello", "thread_id": "exp-th"},
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert resp.status_code == 401
    assert "Invalid or expired" in resp.json()["detail"]


def test_jwt_wrong_algorithm_none_rejection():
    """Rejects JWTs forged with 'none' or unauthorized algorithms."""
    import jwt

    # Token with alg="none"
    unsecured_token = jwt.encode(
        {"sub": "attacker", "tenant_id": "tenant-corp", "exp": 9999999999},
        key="",
        algorithm="none",
    )
    with pytest.raises(ValueError, match="JWT validation error:"):
        decode_jwt(unsecured_token)


def test_jwt_wrong_secret_rejection():
    """Rejects JWTs signed with a different secret."""
    import jwt

    token = jwt.encode(
        {"sub": "user-diff", "tenant_id": "tenant-corp", "exp": 9999999999},
        key="completely-different-secret-key-32chars!",
        algorithm="HS256",
    )
    with pytest.raises(ValueError, match="JWT validation error:"):
        decode_jwt(token)


def test_jwt_missing_exp_rejection():
    """Rejects JWTs missing required 'exp' claim."""
    import jwt

    from agenticrag.config import get_settings

    token = jwt.encode(
        {"sub": "user-no-exp", "tenant_id": "tenant-corp"},
        key=get_settings().jwt_secret,
        algorithm="HS256",
    )
    with pytest.raises(ValueError, match="JWT validation error:.*exp"):
        decode_jwt(token)


def test_header_spoofing_attempt():
    """Tenant and user must come from JWT, spoofed headers must be ignored."""
    client = TestClient(app)

    # Valid token for tenant-alpha / user-legit
    token = create_jwt({"sub": "user-legit", "tenant_id": "tenant-alpha"})

    # Attacker attempts to spoof X-Tenant-ID and X-User-ID headers
    resp = client.post(
        "/chat",
        json={"message": "hello", "thread_id": "spoof-th-1"},
        headers={
            "Authorization": f"Bearer {token}",
            "X-Tenant-ID": "malicious-tenant",
            "X-User-ID": "malicious-user",
        },
    )
    assert resp.status_code == 200

    # Thread should be bound to tenant-alpha, not malicious-tenant
    owner = get_tenancy_manager().storage.get_thread_owner("spoof-th-1")
    assert owner is not None
    tenant, user = owner
    assert tenant == "tenant-alpha"
    assert user == "user-legit"


def test_production_weak_secret_startup_failure():
    """In production, settings validation must raise ValueError if JWT_SECRET is unset or < 32 chars."""
    from agenticrag.config import Settings

    with pytest.raises(ValueError, match="JWT_SECRET must be set and at least 32 characters"):
        Settings(environment="production", jwt_secret="")

    with pytest.raises(ValueError, match="JWT_SECRET must be set and at least 32 characters"):
        Settings(environment="production", jwt_secret="too-short")

    # Valid 32-char secret passes
    valid_settings = Settings(
        environment="production", jwt_secret="a" * 32, allow_in_memory_checkpoints=True
    )
    assert valid_settings.jwt_secret == "a" * 32


def test_redis_thread_tenancy_storage(monkeypatch):
    """Verifies RedisThreadTenancyStorage binds and reads thread owners."""
    from agenticrag.auth import RedisThreadTenancyStorage

    mock_redis_client = MagicMock()
    mock_store = {}

    def fake_hsetnx(name, key, value):
        if name not in mock_store:
            mock_store[name] = {}
        if key not in mock_store[name]:
            mock_store[name][key] = value
            return 1
        return 0

    def fake_hgetall(name):
        return mock_store.get(name, {})

    mock_redis_client.hsetnx = fake_hsetnx
    mock_redis_client.hgetall = fake_hgetall
    mock_redis_client.keys.return_value = []

    storage = RedisThreadTenancyStorage(redis_url="redis://localhost:6379/0")
    monkeypatch.setattr(storage, "_get_client", lambda: mock_redis_client)

    storage.bind_thread("th-redis-1", "tenant-x", "user-y")
    owner = storage.get_thread_owner("th-redis-1")
    assert owner == ("tenant-x", "user-y")


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


def test_anonymous_multi_turn_thread_access_with_shifting_proxy_ips():
    """Verifies that an anonymous user asking follow-up queries on the same thread is not blocked by shifting edge proxy IPs."""
    client = TestClient(app)
    thread_id = "anon-thread-followup-test"

    # Turn 1: request comes via proxy IP 1
    resp1 = client.post(
        "/chat",
        json={"message": "First query", "thread_id": thread_id},
        headers={"X-Forwarded-For": "76.76.21.21"},
    )
    assert resp1.status_code == 200, f"Turn 1 failed: {resp1.text}"

    # Turn 2: second query on same thread via different proxy IP 2
    resp2 = client.post(
        "/chat",
        json={"message": "Second query", "thread_id": thread_id},
        headers={"X-Forwarded-For": "76.76.21.99"},
    )
    assert resp2.status_code == 200, f"Turn 2 failed with cross-user error: {resp2.text}"

    # Turn 3: third query with persistent X-Session-ID header
    resp3 = client.post(
        "/chat",
        json={"message": "Third query", "thread_id": thread_id},
        headers={"X-Session-ID": "test-session-12345", "X-Forwarded-For": "76.76.21.150"},
    )
    assert resp3.status_code == 200, f"Turn 3 failed: {resp3.text}"
