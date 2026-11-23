"""HTTP API client for communicating with the Agentic-RAG backend.

This client has zero dependencies on Streamlit and zero dependencies on backend internals.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ApiClientError(Exception):
    """Raised when an API request fails."""


class ApiClient:
    """HTTP client for Agentic-RAG backend API endpoints."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "",
        request_timeout: float = 15.0,
        stream_timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.stream_timeout = stream_timeout
        self.transport = transport

    def _get_headers(self, session_id: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if session_id:
            headers["X-Session-ID"] = session_id
        return headers

    def check_health(self) -> dict[str, Any]:
        """Probes the backend health check endpoint.

        Returns:
            Health status dictionary.

        Raises:
            ApiClientError: If the backend is unreachable or returns non-200.
        """
        url = f"{self.base_url}/health"
        try:
            with httpx.Client(transport=self.transport, timeout=self.request_timeout) as client:
                response = client.get(url, headers=self._get_headers())
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            logger.warning("Health check failed for %s: %s", url, exc)
            raise ApiClientError(f"Backend unreachable at {self.base_url}: {exc}") from exc

    def get_kb_info(self) -> dict[str, Any]:
        """Fetches metadata about the knowledge base vector index.

        Returns:
            Knowledge base info dictionary.

        Raises:
            ApiClientError: If request fails.
        """
        url = f"{self.base_url}/kb/info"
        try:
            with httpx.Client(transport=self.transport, timeout=self.request_timeout) as client:
                response = client.get(url, headers=self._get_headers())
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            logger.warning("KB info request failed for %s: %s", url, exc)
            raise ApiClientError(f"Failed to fetch KB info: {exc}") from exc

    def search_kb(self, query: str) -> dict[str, Any]:
        """Performs a direct vector search against the knowledge base.

        Args:
            query: User search query.

        Returns:
            Search results payload.

        Raises:
            ApiClientError: If request fails.
        """
        url = f"{self.base_url}/kb/search"
        payload = {"query": query}
        try:
            with httpx.Client(transport=self.transport, timeout=self.request_timeout) as client:
                response = client.post(url, json=payload, headers=self._get_headers())
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            logger.warning("KB search failed for %s: %s", url, exc)
            raise ApiClientError(f"Failed to search knowledge base: {exc}") from exc

    def chat_sync(
        self,
        message: str,
        thread_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Sends a non-streaming chat request.

        Args:
            message: User message.
            thread_id: Optional conversation thread identifier.
            session_id: Optional session identifier.

        Returns:
            ChatResponse dictionary.

        Raises:
            ApiClientError: If request fails.
        """
        url = f"{self.base_url}/chat"
        payload: dict[str, Any] = {"message": message}
        if thread_id:
            payload["thread_id"] = thread_id

        try:
            with httpx.Client(transport=self.transport, timeout=self.request_timeout) as client:
                response = client.post(
                    url, json=payload, headers=self._get_headers(session_id=session_id)
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            logger.warning("Chat request failed for %s: %s", url, exc)
            raise ApiClientError(f"Chat request failed: {exc}") from exc

    def chat_stream(
        self,
        message: str,
        thread_id: str | None = None,
        session_id: str | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Sends a streaming chat request via Server-Sent Events (SSE).

        Yields dictionaries representing typed SSE frames:
        - {"type": "status", "node": str, "message": str}
        - {"type": "token", "token": str}
        - {"type": "done", "thread_id": str, "source_used": str, "full_content": str, "ticket_payload": dict | None, "sources": list[dict] | None}
        - {"type": "error", "error": str}

        Args:
            message: User message.
            thread_id: Optional thread ID.
            session_id: Optional session ID.

        Yields:
            Parsed event payload dictionaries.
        """
        url = f"{self.base_url}/chat/stream"
        payload: dict[str, Any] = {"message": message}
        if thread_id:
            payload["thread_id"] = thread_id

        headers = self._get_headers(session_id=session_id)
        headers["Accept"] = "text/event-stream"

        try:
            with httpx.Client(transport=self.transport, timeout=self.stream_timeout) as client:
                with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        error_detail = response.read().decode("utf-8", errors="replace")
                        yield {
                            "type": "error",
                            "error": f"Backend returned HTTP {response.status_code}: {error_detail}",
                        }
                        return

                    for line in response.iter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data: "):
                            raw_data = line[6:].strip()
                            try:
                                event = json.loads(raw_data)
                                if isinstance(event, dict):
                                    yield event
                            except json.JSONDecodeError as decode_err:
                                logger.debug("Failed to decode SSE line: %s (%s)", line, decode_err)
        except httpx.HTTPError as exc:
            logger.warning("Streaming request failed for %s: %s", url, exc)
            yield {
                "type": "error",
                "error": f"Connection error communicating with backend: {exc}",
            }
