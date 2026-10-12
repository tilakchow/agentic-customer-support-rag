"""Unit tests for pure routing, grading decision functions, and history formatting."""

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from agenticrag.nodes import (
    NodeName,
    route_after_router,
    decide_after_kb_grade,
    decide_after_web_grade,
    get_latest_question,
    get_conversation_history,
)


def test_route_after_router_kb():
    state = {"route": "kb"}
    assert route_after_router(state) == NodeName.RETRIEVE_KB.value


def test_route_after_router_direct():
    state = {"route": "direct"}
    assert route_after_router(state) == NodeName.DIRECT_ANSWER.value


def test_route_after_router_fallback():
    state = {}
    assert route_after_router(state) == NodeName.DIRECT_ANSWER.value


def test_decide_after_kb_grade_good():
    state = {"kb_grade": "good"}
    assert decide_after_kb_grade(state) == NodeName.GENERATE_FROM_KB.value


def test_decide_after_kb_grade_weak():
    state = {"kb_grade": "weak"}
    assert decide_after_kb_grade(state) == NodeName.SEARCH_WEB.value


def test_decide_after_kb_grade_weak_contact_number():
    # When user asks for a phone number and KB doesn't have it, do NOT search the open web!
    state = {
        "kb_grade": "weak",
        "messages": [HumanMessage(content="can i know the customer service number !")]
    }
    assert decide_after_kb_grade(state) == NodeName.ANSWER_INSUFFICIENT.value


def test_decide_after_web_grade_failure():
    # If search failed, immediately route to answer_insufficient without retrying
    state = {"web_search_failed": True, "web_grade": "weak", "retry_count": 0}
    assert decide_after_web_grade(state) == NodeName.ANSWER_INSUFFICIENT.value


def test_decide_after_web_grade_good():
    state = {"web_search_failed": False, "web_grade": "good", "retry_count": 0}
    assert decide_after_web_grade(state) == NodeName.GENERATE_FROM_WEB.value


def test_decide_after_web_grade_retry():
    state = {"web_search_failed": False, "web_grade": "weak", "retry_count": 0}
    assert decide_after_web_grade(state) == NodeName.REWRITE_QUERY.value


def test_decide_after_web_grade_max_retries():
    state = {"web_search_failed": False, "web_grade": "weak", "retry_count": 1}
    assert decide_after_web_grade(state) == NodeName.ANSWER_INSUFFICIENT.value


def test_get_latest_question():
    state = {
        "messages": [
            HumanMessage(content="First message"),
            AIMessage(content="First reply"),
            HumanMessage(content="Second message"),
        ]
    }
    assert get_latest_question(state) == "Second message"


def test_get_conversation_history_deduplication():
    # Verify that the current pending human message is excluded from prior history
    state = {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there! How can I help?"),
            HumanMessage(content="Where is my order?"),
        ]
    }
    history = get_conversation_history(state)
    assert "Where is my order?" not in history
    assert "Hello" in history
    assert "Hi there! How can I help?" in history
