"""API layer for Agentic Customer Support RAG."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "app":
        from agenticrag.api.routes import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app"]
