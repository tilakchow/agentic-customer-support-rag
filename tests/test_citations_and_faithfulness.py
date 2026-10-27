"""Unit and integration tests for source citations, doc metadata attribution, and claim-support faithfulness checks."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from agenticrag.api import app
from agenticrag.nodes import check_claim_support, generate_from_kb
from agenticrag.schemas import ClaimSupportCheck


@pytest.fixture
def client():
    return TestClient(app)


def test_claim_support_schema():
    """Validates ClaimSupportCheck model parsing and fallback aliases."""
    valid_check = ClaimSupportCheck(supported=True, reason="All claims match reference context.")
    assert valid_check.supported is True
    assert valid_check.unsupported_claims == []

    alias_check = ClaimSupportCheck.model_validate(
        {"grounded": False, "reason": "Invented price $999"}
    )
    assert alias_check.supported is False

    unsupported_check = ClaimSupportCheck(
        supported=False,
        reason="Claimed 2-year warranty when context states 1-year",
        unsupported_claims=["2-year replacement warranty"],
    )
    assert unsupported_check.supported is False
    assert len(unsupported_check.unsupported_claims) == 1


@pytest.mark.asyncio
async def test_check_claim_support_grounded():
    """Validates check_claim_support returns supported=True for grounded answers."""
    with patch("agenticrag.nodes.get_faithfulness_llm") as mock_faith:
        mock_faith.return_value.ainvoke = AsyncMock(
            return_value=ClaimSupportCheck(supported=True, reason="Grounded in context")
        )

        result = await check_claim_support(
            question="What is the return window?",
            context="Items can be returned within 30 days of delivery.",
            response="You can return items within 30 days.",
        )
        assert result.supported is True


@pytest.mark.asyncio
async def test_check_claim_support_hallucination_flagged():
    """Validates check_claim_support flags hallucinated claims."""
    with patch("agenticrag.nodes.get_faithfulness_llm") as mock_faith:
        mock_faith.return_value.ainvoke = AsyncMock(
            return_value=ClaimSupportCheck(
                supported=False,
                reason="Context specifies 30 days, not 90 days",
                unsupported_claims=["90 day return period"],
            )
        )

        result = await check_claim_support(
            question="What is the return window?",
            context="Items can be returned within 30 days of delivery.",
            response="You can return items anytime within 90 days for a full cash refund.",
        )
        assert result.supported is False
        assert "90 day return period" in result.unsupported_claims


@pytest.mark.asyncio
async def test_generate_from_kb_annotates_sources_and_faithfulness():
    """generate_from_kb formats context with numbered citations and runs claim-support check."""
    sample_docs = [
        Document(
            page_content="Our return window is 30 days.",
            metadata={"id": "doc-ret-001", "intent": "cancel_order", "score": 0.92},
        ),
        Document(
            page_content="Refunds take 3 to 5 business days to post to your bank.",
            metadata={"id": "doc-ref-002", "intent": "check_refund_policy", "score": 0.88},
        ),
    ]
    state = {
        "messages": [HumanMessage(content="How long does a refund take?")],
        "kb_docs": sample_docs,
    }

    with (
        patch("agenticrag.nodes.get_llm") as mock_llm,
        patch("agenticrag.nodes.get_faithfulness_llm") as mock_faith,
    ):
        mock_llm.return_value.ainvoke = AsyncMock(
            return_value=AIMessage(content="Refunds are processed within 3 to 5 business days.")
        )
        mock_faith.return_value.ainvoke = AsyncMock(
            return_value=ClaimSupportCheck(supported=True, reason="Grounded")
        )

        result = await generate_from_kb(state)  # type: ignore[arg-type]

        assert result["source_used"] == "private_kb"
        assert len(result["messages"]) == 1
        assert "3 to 5 business days" in str(result["messages"][0].content)


@pytest.mark.asyncio
async def test_generate_from_kb_appends_note_on_unsupported_claim():
    """generate_from_kb adds guardrail disclaimer when claim support check fails."""
    sample_docs = [
        Document(
            page_content="Warranty covers defects for 12 months.",
            metadata={"id": "doc-war-001", "intent": "check_cancellation_fee", "score": 0.85},
        )
    ]
    state = {
        "messages": [HumanMessage(content="Does warranty cover water damage?")],
        "kb_docs": sample_docs,
    }

    with (
        patch("agenticrag.nodes.get_llm") as mock_llm,
        patch("agenticrag.nodes.get_faithfulness_llm") as mock_faith,
    ):
        mock_llm.return_value.ainvoke = AsyncMock(
            return_value=AIMessage(content="Yes, water damage is fully covered for 5 years.")
        )
        mock_faith.return_value.ainvoke = AsyncMock(
            return_value=ClaimSupportCheck(
                supported=False,
                reason="Water damage and 5 years not in context",
                unsupported_claims=["water damage covered", "5 years"],
            )
        )

        result = await generate_from_kb(state)  # type: ignore[arg-type]

        content = str(result["messages"][0].content)
        assert "Note: Please verify specific details" in content


def test_chat_response_source_attribution():
    """Validates that ChatResponse includes structured source citations."""
    from agenticrag.models import ChatResponse

    resp = ChatResponse(
        response="Your return window is 30 days.",
        thread_id="test-thread-citations",
        source_used="private_kb",
        sources=[
            {
                "doc_id": "doc-ret-001",
                "intent": "cancel_order",
                "score": 0.92,
                "content": "Our return window is 30 days.",
            }
        ],
    )
    assert resp.sources is not None
    assert len(resp.sources) == 1
    assert resp.sources[0]["doc_id"] == "doc-ret-001"
    assert resp.sources[0]["intent"] == "cancel_order"
    assert resp.sources[0]["score"] == 0.92
