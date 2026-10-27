"""Unit and integration tests for scope control, out-of-scope refusal, and human escalation."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from agenticrag.graph import create_agent
from agenticrag.node_meta import NodeName
from agenticrag.nodes import classify_scope, escalate_to_human, refuse_out_of_scope
from agenticrag.routing import (
    decide_after_kb_grade,
    is_allowed_web_fallback_query,
    is_human_escalation_request,
    is_policy_sensitive_request,
    route_after_scope,
)
from agenticrag.schemas import ScopeDecision


def test_scope_decision_schema_defaults_and_aliases():
    """Validates ScopeDecision parsing, default values, and key alias normalization."""
    default_decision = ScopeDecision()
    assert default_decision.scope == "in_scope"
    assert default_decision.reason == ""

    alias_decision = ScopeDecision.model_validate(
        {"classification": "out_of_scope", "reason": "coding script"}
    )
    assert alias_decision.scope == "out_of_scope"

    human_decision = ScopeDecision.model_validate(
        {"category": "needs_human", "reason": "customer requested representative"}
    )
    assert human_decision.scope == "needs_human"

    policy_decision = ScopeDecision.model_validate(
        {"decision": "policy_sensitive", "reason": "threat of legal action"}
    )
    assert policy_decision.scope == "policy_sensitive"


def test_is_human_escalation_request():
    """Validates detection of explicit requests for human support."""
    assert is_human_escalation_request("I want to speak with a human agent please") is True
    assert is_human_escalation_request("Can you transfer me to a live representative?") is True
    assert is_human_escalation_request("I need to speak to your manager right now") is True
    assert is_human_escalation_request("Can I talk to someone from support?") is True
    assert is_human_escalation_request("Please connect me to an operator") is True

    # Standard customer queries
    assert is_human_escalation_request("How do I return a damaged pair of shoes?") is False
    assert is_human_escalation_request("What is the status of my order #1234?") is False
    assert is_human_escalation_request("Can I change my delivery address?") is False


def test_is_policy_sensitive_request():
    """Validates detection of legal threats and regulatory filings."""
    assert is_policy_sensitive_request("I will sue you and have my lawyer contact you") is True
    assert is_policy_sensitive_request("I am filing a formal lawsuit in small claims court") is True
    assert is_policy_sensitive_request("I will submit an FTC complaint regarding fraud") is True

    assert is_policy_sensitive_request("What is your refund policy?") is False
    assert is_policy_sensitive_request("Can I cancel my account?") is False


def test_is_allowed_web_fallback_query():
    """Validates that coding, scripting, and programming requests are blocked from web search fallback."""
    # Disallowed coding/script requests
    assert is_allowed_web_fallback_query("Write a python script to calculate Fibonacci") is False
    assert is_allowed_web_fallback_query("Write javascript code for binary search") is False
    assert is_allowed_web_fallback_query("Generate a SQL query to drop database tables") is False
    assert is_allowed_web_fallback_query("Debug this recursive algorithm in typescript") is False

    # Allowed support & general search queries
    assert is_allowed_web_fallback_query("What are the standard delivery options?") is True
    assert is_allowed_web_fallback_query("How do I exchange my damaged shoes?") is True
    assert is_allowed_web_fallback_query("Who is the president of France?") is True


def test_route_after_scope():
    """Validates graph routing logic following scope classification."""
    assert route_after_scope({"scope": "in_scope"}) == NodeName.ROUTE_QUESTION.value
    assert route_after_scope({"scope": "out_of_scope"}) == NodeName.REFUSE_OUT_OF_SCOPE.value
    assert route_after_scope({"scope": "needs_human"}) == NodeName.ESCALATE_TO_HUMAN.value
    assert route_after_scope({"scope": "policy_sensitive"}) == NodeName.ESCALATE_TO_HUMAN.value
    assert (
        route_after_scope({"scope": "in_scope", "escalated": True})
        == NodeName.ESCALATE_TO_HUMAN.value
    )


@pytest.mark.asyncio
async def test_refuse_out_of_scope_node():
    """Validates refuse_out_of_scope node output and metadata."""
    state = {"current_query": "Write a python script to calculate Fibonacci numbers"}
    result = await refuse_out_of_scope(state)  # type: ignore[arg-type]

    assert result["source_used"] == "out_of_scope_refusal"
    assert result["scope"] == "out_of_scope"
    assert len(result["messages"]) == 1
    msg_content = str(result["messages"][0].content)
    assert "cannot assist with general topics" in msg_content


@pytest.mark.asyncio
async def test_escalate_to_human_node():
    """Validates escalate_to_human node creates ticket payload with valid identifiers."""
    sample_doc = Document(
        page_content="Return policy is 30 days.",
        metadata={"id": "doc-return-1", "intent": "return_policy"},
    )
    state = {
        "messages": [HumanMessage(content="I demand to speak to a supervisor immediately")],
        "scope": "needs_human",
        "kb_docs": [sample_doc],
    }
    result = await escalate_to_human(state)  # type: ignore[arg-type]

    assert result["source_used"] == "human_escalation"
    assert result["escalated"] is True
    payload = result["ticket_payload"]
    assert payload is not None
    assert payload["ticket_id"].startswith("TICK-")
    assert payload["status"] == "escalated"
    assert payload["reason"] == "needs_human"
    assert "supervisor" in payload["customer_query"]
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["metadata"]["intent"] == "return_policy"

    msg_content = str(result["messages"][0].content)
    assert payload["ticket_id"] in msg_content


@pytest.mark.asyncio
async def test_classify_scope_heuristics():
    """Validates fast-track heuristics for obvious cases without LLM calls."""
    # Obvious coding request
    res_code = await classify_scope(
        {"messages": [HumanMessage(content="Write a python script for fibonacci")]}  # type: ignore[arg-type]
    )
    assert res_code["scope"] == "out_of_scope"

    # Human request
    res_human = await classify_scope(
        {"messages": [HumanMessage(content="I want to talk to a human agent")]}  # type: ignore[arg-type]
    )
    assert res_human["scope"] == "needs_human"

    # Legal threat
    res_legal = await classify_scope(
        {"messages": [HumanMessage(content="I will sue you, my lawyer will file a lawsuit")]}  # type: ignore[arg-type]
    )
    assert res_legal["scope"] == "policy_sensitive"


@pytest.mark.asyncio
async def test_classify_scope_llm_invocation():
    """Validates structured LLM classification when heuristics do not fast-track."""
    with patch("agenticrag.nodes.get_classifier_llm") as mock_classifier:
        mock_classifier.return_value.ainvoke = AsyncMock(
            return_value=ScopeDecision(scope="in_scope", reason="standard order inquiry")
        )
        res = await classify_scope(
            {"messages": [HumanMessage(content="Can you check where my package is?")]}  # type: ignore[arg-type]
        )
        assert res["scope"] == "in_scope"
        assert mock_classifier.return_value.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_classify_scope_fallback_on_llm_error():
    """Validates fail-safe error handling when classifier LLM fails."""
    with patch("agenticrag.nodes.get_classifier_llm") as mock_classifier:
        mock_classifier.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("LLM 503 error"))

        res = await classify_scope(
            {"messages": [HumanMessage(content="I need help tracking my order")]}  # type: ignore[arg-type]
        )
        assert res["scope"] == "in_scope"


def test_decide_after_kb_grade_blocks_oos_from_web():
    """Validates that out-of-domain queries with weak KB evidence are NOT sent to web search."""
    state = {
        "kb_grade": "weak",
        "messages": [HumanMessage(content="Write a python script to calculate Fibonacci")],
    }
    assert decide_after_kb_grade(state) == NodeName.ANSWER_INSUFFICIENT.value


@pytest.mark.asyncio
async def test_e2e_out_of_scope_refusal_bypasses_retrieval_and_web():
    """E2E graph execution for out-of-scope query refuses immediately without calling vectorstore or web search."""
    app = create_agent()

    with (
        patch("agenticrag.nodes.get_vectorstore") as mock_vs,
        patch("agenticrag.nodes.get_web_search") as mock_web,
    ):
        result = await app.ainvoke(
            {
                "messages": [
                    HumanMessage(content="Write a python script to calculate Fibonacci numbers")
                ]
            },
            config={"configurable": {"thread_id": "test-oos-e2e"}},
        )

        assert result["source_used"] == "out_of_scope_refusal"
        assert result["scope"] == "out_of_scope"
        assert mock_vs.call_count == 0
        assert mock_web.call_count == 0
        assert "cannot assist with general topics" in str(result["messages"][-1].content)


@pytest.mark.asyncio
async def test_e2e_human_escalation_generates_ticket_payload():
    """E2E graph execution for human request escalates and attaches structured ticket payload."""
    app = create_agent()

    with (
        patch("agenticrag.nodes.get_vectorstore") as mock_vs,
        patch("agenticrag.nodes.get_web_search") as mock_web,
    ):
        result = await app.ainvoke(
            {"messages": [HumanMessage(content="Please transfer me to a human representative")]},
            config={"configurable": {"thread_id": "test-escalate-e2e"}},
        )

        assert result["source_used"] == "human_escalation"
        assert result["escalated"] is True
        assert result["ticket_payload"] is not None
        assert result["ticket_payload"]["ticket_id"].startswith("TICK-")
        assert result["ticket_payload"]["status"] == "escalated"
        assert mock_vs.call_count == 0
        assert mock_web.call_count == 0
        assert result["ticket_payload"]["ticket_id"] in str(result["messages"][-1].content)
