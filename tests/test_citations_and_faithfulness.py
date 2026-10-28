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
@pytest.mark.asyncio
async def test_generate_from_kb_regenerates_on_unsupported_claim_and_succeeds():
    """generate_from_kb initiates 1-turn stricter regeneration when first attempt has unsupported claims."""
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
        # First generation produces hallucination; second generation (regen) is grounded
        mock_llm.return_value.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content="Yes, water damage is fully covered for 5 years."),
                AIMessage(
                    content="Our warranty covers manufacturer defects for 12 months. Water damage is not specified in our records."
                ),
            ]
        )
        # First faith check fails; second faith check passes
        mock_faith.return_value.ainvoke = AsyncMock(
            side_effect=[
                ClaimSupportCheck(
                    supported=False,
                    reason="Water damage and 5 years not in context",
                    unsupported_claims=["water damage covered", "5 years"],
                ),
                ClaimSupportCheck(supported=True, reason="Grounded in 12 month warranty context"),
            ]
        )

        result = await generate_from_kb(state)  # type: ignore[arg-type]

        assert result["source_used"] == "private_kb"
        content = str(result["messages"][0].content)
        assert "12 months" in content
        assert mock_llm.return_value.ainvoke.call_count == 2
        assert mock_faith.return_value.ainvoke.call_count == 2


@pytest.mark.asyncio
async def test_generate_from_kb_regeneration_still_unsupported_escalates():
    """When regenerated response is still unsupported, routes to answer_insufficient and escalates to human."""
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
            side_effect=[
                AIMessage(content="Water damage is covered for 5 years."),
                AIMessage(content="Water damage is still covered for 5 years."),
                AIMessage(
                    content="I apologize, but our records do not have sufficient information."
                ),
            ]
        )
        mock_faith.return_value.ainvoke = AsyncMock(
            return_value=ClaimSupportCheck(
                supported=False,
                reason="Water damage not in context",
                unsupported_claims=["water damage"],
            )
        )

        result = await generate_from_kb(state)  # type: ignore[arg-type]

        assert result["source_used"] == "insufficient_evidence"
        assert result.get("escalated") is True
        assert result.get("ticket_payload") is not None
        assert result["ticket_payload"]["reason"] == "faithfulness_violation"


def test_configurable_faithfulness_model():
    """Verifies that FAITHFULNESS_MODEL is configurable via Settings."""
    from agenticrag.clients import get_faithfulness_llm
    from agenticrag.config import Settings

    with patch("agenticrag.clients.get_settings") as mock_settings:
        mock_settings.return_value = Settings(
            faithfulness_model="groq/llama-3.3-70b-versatile",
            groq_api_key="test-key",
        )
        judge = get_faithfulness_llm()
        assert judge is not None


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


def test_faithfulness_benchmark_dataset_distribution_and_wilson_ci():
    """Verifies that faithfulness benchmark has >= 40 cases with 20 supported and 20 hallucinated."""
    from evals.faithfulness_eval import FAITHFULNESS_BENCHMARK_CASES, wilson_score_interval

    assert len(FAITHFULNESS_BENCHMARK_CASES) >= 40
    supported = [c for c in FAITHFULNESS_BENCHMARK_CASES if c["ground_truth_supported"]]
    unsupported = [c for c in FAITHFULNESS_BENCHMARK_CASES if not c["ground_truth_supported"]]
    assert len(supported) == 20
    assert len(unsupported) == 20

    # Wilson score interval calculation
    lower, upper = wilson_score_interval(0.90, 40, confidence=0.95)
    assert 0.70 < lower < 0.90
    assert 0.90 < upper <= 1.0
