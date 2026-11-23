"""Tests for the Streamlit app using AppTest."""

from __future__ import annotations

from pathlib import Path

import pytest

from frontend.api_client import ApiClient, ApiClientError

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent.parent / "frontend" / "app.py")


@pytest.fixture(autouse=True)
def mock_default_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mocks default ApiClient methods to prevent any network calls."""
    monkeypatch.setattr(
        ApiClient,
        "check_health",
        lambda self: {"status": "ok", "service": "Agentic-RAG Backend"},
    )
    monkeypatch.setattr(
        ApiClient,
        "get_kb_info",
        lambda self: {
            "index_name": "test-index",
            "namespace": "test-ns",
            "dimension": 384,
            "sample_size": 500,
        },
    )


def test_app_first_render() -> None:
    """Verifies initial render of title, sidebar, and status without errors."""
    at = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not at.exception
    assert len(at.title) >= 1
    assert "Agentic Customer Support" in at.title[0].value


def test_app_offline_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an offline backend renders an error banner in sidebar without crashing."""

    def mock_fail(self: ApiClient) -> dict:
        raise ApiClientError("Backend unreachable")

    monkeypatch.setattr(ApiClient, "check_health", mock_fail)
    monkeypatch.setattr(ApiClient, "get_kb_info", mock_fail)

    at = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not at.exception
    assert len(at.sidebar.error) >= 1
    assert "Backend: Offline" in at.sidebar.error[0].value


def test_app_send_message_and_render_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies sending a message, streaming response, and rendering sources."""

    def mock_stream(
        self: ApiClient, message: str, thread_id: str | None = None, session_id: str | None = None
    ):
        yield {"type": "status", "message": "Retrieving documents..."}
        yield {"type": "token", "token": "You can update payment "}
        yield {"type": "token", "token": "in your billing settings."}
        yield {
            "type": "done",
            "thread_id": "thread-456",
            "source_used": "private_kb",
            "full_content": "You can update payment in your billing settings.",
            "sources": [
                {
                    "doc_id": "doc-billing-01",
                    "intent": "billing",
                    "score": 0.88,
                    "content": "Navigate to settings > billing to update card.",
                }
            ],
            "ticket_payload": None,
        }

    monkeypatch.setattr(ApiClient, "chat_stream", mock_stream)

    at = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert len(at.chat_input) >= 1

    at.chat_input[0].set_value("How do I update payment?").run()
    assert not at.exception

    # Verify messages in session state
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "How do I update payment?"
    assert messages[1]["role"] == "assistant"
    assert "You can update payment" in messages[1]["content"]
    assert messages[1]["sources"] is not None
    assert messages[1]["sources"][0]["doc_id"] == "doc-billing-01"


def test_app_clear_conversation() -> None:
    """Verifies clicking clear conversation clears messages and resets state."""
    at = AppTest.from_file(APP_PATH, default_timeout=10).run()
    clear_btns = [b for b in at.sidebar.button if "Clear Conversation" in b.label]
    assert len(clear_btns) == 1
    clear_btns[0].click().run()
    assert not at.exception
    assert at.session_state["messages"] == []
