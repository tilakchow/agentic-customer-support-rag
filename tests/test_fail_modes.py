"""Tests for fail-open vs fail-closed decisions and telemetry across graph nodes (Item 5)."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

import agenticrag.nodes as nodes_module
from agenticrag.config import Settings
from agenticrag.nodes import (
    check_claim_support,
    classify_scope,
    grade_kb_evidence,
    grade_web_evidence,
    route_question,
)
from agenticrag.routing import route_after_scope
from agenticrag.telemetry import get_telemetry_collector


@pytest.fixture(autouse=True)
def clean_telemetry():
    collector = get_telemetry_collector()
    collector.clear()
    yield
    collector.clear()


@pytest.mark.asyncio
async def test_scope_classifier_fail_closed_and_open(monkeypatch):
    """Scope classifier defaults to fail-closed on provider error, routing to answer_insufficient."""
    collector = get_telemetry_collector()

    with patch("agenticrag.nodes.get_classifier_llm") as mock_llm:
        mock_llm.return_value.ainvoke = AsyncMock(side_effect=TimeoutError("Groq timeout"))

        # 1. Closed mode (default)
        closed_settings = Settings(fail_mode_scope_classifier="closed")
        monkeypatch.setattr(nodes_module, "get_settings", lambda: closed_settings)
        res = await classify_scope({"messages": [HumanMessage(content="Where is my order?")]})
        assert res["scope"] == "error_insufficient"
        assert route_after_scope(res) == "answer_insufficient"

        # Check telemetry event recorded
        events = collector.get_fail_events()
        assert len(events) == 1
        assert events[0]["node_name"] == "scope_classifier"
        assert events[0]["mode"] == "closed"
        assert events[0]["error_type"] == "TimeoutError"

        # 2. Open mode
        collector.clear()
        open_settings = Settings(fail_mode_scope_classifier="open")
        monkeypatch.setattr(nodes_module, "get_settings", lambda: open_settings)
        res_open = await classify_scope({"messages": [HumanMessage(content="Where is my order?")]})
        assert res_open["scope"] == "in_scope"
        assert route_after_scope(res_open) == "route_question"

        events_open = collector.get_fail_events()
        assert len(events_open) == 1
        assert events_open[0]["node_name"] == "scope_classifier"
        assert events_open[0]["mode"] == "open"


@pytest.mark.asyncio
async def test_router_fail_open_and_closed(monkeypatch):
    """Router defaults to fail-open (route='kb') and falls back to 'direct' when closed."""
    collector = get_telemetry_collector()

    with patch("agenticrag.nodes.get_router_llm") as mock_llm:
        mock_llm.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("Groq 429"))

        # 1. Open mode (default)
        open_settings = Settings(fail_mode_router="open")
        monkeypatch.setattr(nodes_module, "get_settings", lambda: open_settings)
        res_open = await route_question({"messages": [HumanMessage(content="Help me")]})
        assert res_open["route"] == "kb"

        events = collector.get_fail_events()
        assert any(e["node_name"] == "router" and e["mode"] == "open" for e in events)

        # 2. Closed mode
        closed_settings = Settings(fail_mode_router="closed")
        monkeypatch.setattr(nodes_module, "get_settings", lambda: closed_settings)
        res_closed = await route_question({"messages": [HumanMessage(content="Help me")]})
        assert res_closed["route"] == "direct"

        events_closed = collector.get_fail_events()
        assert any(e["node_name"] == "router" and e["mode"] == "closed" for e in events_closed)


@pytest.mark.asyncio
async def test_grader_fail_closed_and_open(monkeypatch):
    """Grader defaults to fail-closed (weak evidence) and can be configured to fail-open (good)."""
    from langchain_core.documents import Document

    docs = [Document(page_content="Store returns accepted within 30 days.")]

    with patch("agenticrag.nodes.get_grader_llm") as mock_llm:
        mock_llm.return_value.ainvoke = AsyncMock(side_effect=ConnectionError("Network error"))

        # 1. Closed mode (default)
        closed_settings = Settings(fail_mode_grader="closed", skip_grader_score=None)
        monkeypatch.setattr(nodes_module, "get_settings", lambda: closed_settings)
        res_closed = await grade_kb_evidence(
            {"messages": [HumanMessage(content="returns?")], "kb_docs": docs}
        )
        assert res_closed["kb_grade"] == "weak"

        # Web grader closed
        res_web_closed = await grade_web_evidence(
            {
                "messages": [HumanMessage(content="returns?")],
                "web_results": "Some web result",
                "web_search_failed": False,
            }
        )
        assert res_web_closed["web_grade"] == "weak"

        # 2. Open mode
        open_settings = Settings(fail_mode_grader="open", skip_grader_score=None)
        monkeypatch.setattr(nodes_module, "get_settings", lambda: open_settings)
        res_open = await grade_kb_evidence(
            {"messages": [HumanMessage(content="returns?")], "kb_docs": docs}
        )
        assert res_open["kb_grade"] == "good"


@pytest.mark.asyncio
async def test_faithfulness_adaptive_fail_mode(monkeypatch):
    """Faithfulness defaults to adaptive: fail-closed for policy/billing, fail-open otherwise."""
    collector = get_telemetry_collector()

    with patch("agenticrag.nodes.get_faithfulness_llm") as mock_llm:
        mock_llm.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("Evaluator failure"))

        adaptive_settings = Settings(fail_mode_faithfulness="adaptive")
        monkeypatch.setattr(nodes_module, "get_settings", lambda: adaptive_settings)

        # Policy query -> fail closed (supported=False)
        policy_res = await check_claim_support(
            question="What is your refund policy?",
            context="Refunds are accepted within 30 days.",
            response="You can get a full refund.",
        )
        assert policy_res.supported is False
        assert "fail-closed" in policy_res.reason

        events = collector.get_fail_events()
        assert any(e["node_name"] == "faithfulness" and e["mode"] == "closed" for e in events)

        # Generic non-policy query -> fail open (supported=True)
        generic_res = await check_claim_support(
            question="Do you sell blue running shoes?",
            context="We carry running shoes in multiple colors.",
            response="Yes, blue running shoes are available.",
        )
        assert generic_res.supported is True
        assert "fail-open" in generic_res.reason

        events_after = collector.get_fail_events()
        assert any(e["node_name"] == "faithfulness" and e["mode"] == "open" for e in events_after)
