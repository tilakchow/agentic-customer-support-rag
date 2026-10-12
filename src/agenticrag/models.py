"""Pydantic API request and response data models."""

from pydantic import BaseModel, Field, field_validator

from agenticrag.config import get_settings


class ChatRequest(BaseModel):
    """Payload for conversational queries."""

    message: str = Field(..., min_length=1, description="User question or statement")
    thread_id: str | None = Field(
        None, description="Session/Thread identifier for conversation memory"
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message cannot be empty or pure whitespace.")
        max_len = get_settings().max_message_length
        if len(stripped) > max_len:
            raise ValueError(f"Message exceeds maximum allowed length of {max_len} characters.")
        return stripped


class ChatResponse(BaseModel):
    """Synchronous JSON response for /chat endpoint."""

    response: str = Field(..., description="Assistant's response text")
    thread_id: str = Field(..., description="Active thread ID for conversational continuity")
    source_used: str | None = Field(
        None, description="Data source used (private_kb, web_search, direct, etc.)"
    )


class KBSearchRequest(BaseModel):
    """Payload for direct knowledge base search queries."""

    query: str = Field(..., min_length=1, max_length=500)
