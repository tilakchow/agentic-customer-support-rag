"""Unit tests for pure routing, grading decision functions, and history formatting."""

from langchain_core.messages import AIMessage, HumanMessage

from agenticrag.node_meta import NodeName
from agenticrag.nodes import (
    decide_after_kb_grade,
    decide_after_web_grade,
)
from agenticrag.routing import (
    get_conversation_history,
    get_latest_question,
    route_after_router,
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
        "messages": [HumanMessage(content="can i know the customer service number !")],
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


def test_validate_runtime_keys_smoke():
    from agenticrag.config import Settings

    settings = Settings(
        environment="development", groq_api_key="", tavily_api_key="", pinecone_api_key=""
    )
    # Non-strict validation safely completes without raising exception
    settings.validate_runtime_keys(strict=False)
    assert settings.environment == "development"
    assert settings.allow_in_memory_checkpoints is True
    assert settings.rate_limit == "20/minute"


def test_compiled_agent_smoke():
    from agenticrag.graph import create_agent

    app = create_agent()
    assert app is not None


def test_get_latest_question_edge_cases():
    """Empty state or non-human messages gracefully return empty string."""
    assert get_latest_question({}) == ""
    assert get_latest_question({"messages": []}) == ""
    assert get_latest_question({"messages": [AIMessage(content="Hello!")]}) == ""

    # Tuple representation ('human', 'message')
    assert get_latest_question({"messages": [("human", "Help me")]}) == "Help me"

    # Custom object with type="human"
    class CustomHumanMsg:
        type = "human"
        content = "My custom inquiry"

    assert get_latest_question({"messages": [CustomHumanMsg()]}) == "My custom inquiry"


def test_get_conversation_history_edge_cases():
    """Empty state, single message, and limit param handling in conversation history."""
    assert get_conversation_history({}) == "No previous conversation."
    assert get_conversation_history({"messages": []}) == "No previous conversation."
    assert (
        get_conversation_history({"messages": [HumanMessage(content="First")]})
        == "No previous conversation."
    )

    state = {
        "messages": [
            HumanMessage(content="1"),
            AIMessage(content="2"),
            HumanMessage(content="3"),
            AIMessage(content="4"),
            HumanMessage(content="5"),
        ]
    }
    # Limited to 2 prior turns
    limited_hist = get_conversation_history(state, limit=2)
    assert "User: 3" in limited_hist
    assert "Assistant: 4" in limited_hist
    assert "User: 1" not in limited_hist


def test_is_internal_contact_query():
    from agenticrag.routing import is_internal_contact_query

    assert is_internal_contact_query("What is your customer service number?") is True
    assert is_internal_contact_query("Can I have the toll free helpline?") is True
    assert is_internal_contact_query("What is the return window for laptops?") is False


def test_is_proprietary_policy_query():
    from agenticrag.routing import is_proprietary_policy_query

    assert is_proprietary_policy_query("What is your refund policy?") is True
    assert is_proprietary_policy_query("Is there a 2-year warranty?") is True
    assert is_proprietary_policy_query("How do I reset my password?") is False
