"""Unit tests for Pydantic schemas and API models."""

import pytest
from pydantic import ValidationError

from agenticrag.config import Settings
from agenticrag.models import ChatRequest, ChatResponse, KBSearchRequest
from agenticrag.schemas import EvidenceGrade, RouteDecision


def test_route_decision_defaults():
    rd = RouteDecision()
    assert rd.route == "kb"


def test_route_decision_aliases():
    assert RouteDecision.model_validate({"decision": "direct"}).route == "direct"
    assert RouteDecision.model_validate({"destination": "kb"}).route == "kb"
    assert RouteDecision.model_validate({"choice": "direct"}).route == "direct"
    assert RouteDecision.model_validate({"answer": "direct"}).route == "direct"


def test_route_decision_invalid_route():
    with pytest.raises(ValidationError):
        RouteDecision(route="invalid_route")  # type: ignore[arg-type]


def test_evidence_grade_defaults():
    eg = EvidenceGrade()
    assert eg.grade == "weak"


def test_evidence_grade_aliases():
    assert EvidenceGrade.model_validate({"answer": "good"}).grade == "good"
    assert EvidenceGrade.model_validate({"score": "weak"}).grade == "weak"
    assert EvidenceGrade.model_validate({"decision": "good"}).grade == "good"
    assert EvidenceGrade.model_validate({"result": "good"}).grade == "good"


def test_evidence_grade_invalid_grade():
    with pytest.raises(ValidationError):
        EvidenceGrade(grade="excellent")  # type: ignore[arg-type]


def test_chat_request_valid():
    req = ChatRequest(message="  What is the return window?  ", thread_id="t123")
    assert req.message == "What is the return window?"
    assert req.thread_id == "t123"


def test_chat_request_empty_message():
    with pytest.raises(ValidationError, match="Message cannot be empty or pure whitespace."):
        ChatRequest(message="   ")


def test_chat_request_too_long(monkeypatch):
    import agenticrag.models as models_module

    monkeypatch.setattr(models_module, "get_settings", lambda: Settings(max_message_length=10))
    with pytest.raises(ValidationError, match="Message exceeds maximum allowed length"):
        ChatRequest(message="12345678901")


def test_chat_response_serialization():
    resp = ChatResponse(response="Support answer", thread_id="tid", source_used="private_kb")
    data = resp.model_dump()
    assert data["response"] == "Support answer"
    assert data["thread_id"] == "tid"
    assert data["source_used"] == "private_kb"


def test_kb_search_request_validation():
    req = KBSearchRequest(query="valid query")
    assert req.query == "valid query"

    with pytest.raises(ValidationError):
        KBSearchRequest(query="")

    with pytest.raises(ValidationError):
        KBSearchRequest(query="a" * 501)
