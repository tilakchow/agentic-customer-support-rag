"""Unit tests for the frontend API client."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from frontend.api_client import ApiClient, ApiClientError


def test_api_client_headers() -> None:
    """Verifies that headers are correctly constructed with auth and session info."""
    client = ApiClient(base_url="http://localhost:8000", api_key="secret-key")
    headers = client._get_headers(session_id="sess-123")

    assert headers["Accept"] == "application/json"
    assert headers["Content-Type"] == "application/json"
    assert headers["X-API-Key"] == "secret-key"
    assert headers["X-Session-ID"] == "sess-123"


def test_api_client_check_health_success() -> None:
    """Verifies health check success parsing."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://localhost:8000/health"
        return httpx.Response(200, json={"status": "ok", "service": "Agentic-RAG Backend"})

    client = ApiClient(
        base_url="http://localhost:8000",
        transport=httpx.MockTransport(mock_handler),
    )
    res = client.check_health()
    assert res["status"] == "ok"
    assert res["service"] == "Agentic-RAG Backend"


def test_api_client_check_health_error() -> None:
    """Verifies health check raises ApiClientError on non-200 or network failure."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    client = ApiClient(
        base_url="http://localhost:8000",
        transport=httpx.MockTransport(mock_handler),
    )
    with pytest.raises(ApiClientError, match="Backend unreachable"):
        client.check_health()


def test_api_client_get_kb_info() -> None:
    """Verifies fetching KB info."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://localhost:8000/kb/info"
        return httpx.Response(
            200,
            json={
                "index_name": "test-index",
                "namespace": "test-ns",
                "dimension": 384,
                "sample_size": 100,
            },
        )

    client = ApiClient(
        base_url="http://localhost:8000",
        transport=httpx.MockTransport(mock_handler),
    )
    info = client.get_kb_info()
    assert info["index_name"] == "test-index"
    assert info["dimension"] == 384


def test_api_client_search_kb() -> None:
    """Verifies direct KB search."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://localhost:8000/kb/search"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["query"] == "how to refund"
        return httpx.Response(200, json={"query": "how to refund", "results": [{"content": "Doc"}]})

    client = ApiClient(
        base_url="http://localhost:8000",
        transport=httpx.MockTransport(mock_handler),
    )
    result = client.search_kb("how to refund")
    assert len(result["results"]) == 1


def test_api_client_chat_sync() -> None:
    """Verifies synchronous non-streaming chat request."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://localhost:8000/chat"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["message"] == "hello"
        assert payload["thread_id"] == "t-123"
        return httpx.Response(
            200,
            json={
                "response": "Hello there!",
                "thread_id": "t-123",
                "source_used": "direct",
                "ticket_payload": None,
                "sources": None,
            },
        )

    client = ApiClient(
        base_url="http://localhost:8000",
        transport=httpx.MockTransport(mock_handler),
    )
    res = client.chat_sync("hello", thread_id="t-123")
    assert res["response"] == "Hello there!"
    assert res["source_used"] == "direct"


def test_api_client_chat_stream_success() -> None:
    """Verifies SSE line parsing and streaming event generator."""
    sse_lines = [
        b'data: {"type": "status", "node": "route_question", "message": "Analyzing question..."}\n\n',
        b'data: {"type": "token", "token": "Hello "}\n\n',
        b'data: {"type": "token", "token": "world!"}\n\n',
        b'data: {"type": "done", "thread_id": "t-99", "source_used": "direct", "full_content": "Hello world!", "sources": null}\n\n',
    ]

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://localhost:8000/chat/stream"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["message"] == "stream test"
        assert request.headers["Accept"] == "text/event-stream"
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=b"".join(sse_lines),
        )

    client = ApiClient(
        base_url="http://localhost:8000",
        transport=httpx.MockTransport(mock_handler),
    )
    events: list[dict[str, Any]] = list(client.chat_stream("stream test", thread_id="t-99"))

    assert len(events) == 4
    assert events[0]["type"] == "status"
    assert events[0]["message"] == "Analyzing question..."
    assert events[1]["type"] == "token"
    assert events[1]["token"] == "Hello "
    assert events[2]["type"] == "token"
    assert events[2]["token"] == "world!"
    assert events[3]["type"] == "done"
    assert events[3]["source_used"] == "direct"


def test_api_client_chat_stream_backend_error() -> None:
    """Verifies that non-200 streaming response yields a typed error event."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    client = ApiClient(
        base_url="http://localhost:8000",
        transport=httpx.MockTransport(mock_handler),
    )
    events = list(client.chat_stream("fail test"))
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "503" in events[0]["error"]
