"""Unit tests for pure routing, grading decision functions, and history formatting."""

from langchain_core.messages import AIMessage, HumanMessage

from agenticrag.node_meta import NodeName
from agenticrag.routing import (
    decide_after_kb_grade,
    decide_after_web_grade,
    get_conversation_history,
    get_latest_question,
    route_after_router,
)


def test_route_after_router_kb():
    state = {"route": "kb"}
    assert route_after_router(state) == NodeName.CONDENSE_QUERY.value


def test_route_after_router_direct():
    state = {"route": "direct"}
    assert route_after_router(state) == NodeName.DIRECT_ANSWER.value


def test_route_after_router_fallback():
    state = {}
    assert route_after_router(state) == NodeName.DIRECT_ANSWER.value


def test_decide_after_kb_grade_good():
    state = {"kb_grade": "good"}
    assert decide_after_kb_grade(state) == NodeName.GENERATE_FROM_KB.value


def test_decide_after_kb_grade_default_disabled():
    # By default, WEB_FALLBACK_ENABLED=False -> always routes to answer_insufficient on weak KB evidence
    state = {"kb_grade": "weak", "messages": [HumanMessage(content="Where does FedEx ship?")]}
    assert decide_after_kb_grade(state) == NodeName.ANSWER_INSUFFICIENT.value


def test_decide_after_kb_grade_enabled_allowed_queries():
    from unittest.mock import MagicMock

    enabled_settings = MagicMock(web_fallback_enabled=True)
    allowed_queries = [
        "Does FedEx deliver on Saturdays in Germany?",
        "What are the physical dimensions of the XYZ bluetooth speaker?",
        "What is DHL Express standard international transit time?",
        "How do I install the battery in the remote control?",
    ]
    for q in allowed_queries:
        state = {"kb_grade": "weak", "messages": [HumanMessage(content=q)]}
        assert (
            decide_after_kb_grade(state, settings=enabled_settings) == NodeName.SEARCH_WEB.value
        ), f"Expected SEARCH_WEB for allowed query: {q}"


def test_decide_after_kb_grade_enabled_denied_categories_matrix():
    from unittest.mock import MagicMock

    enabled_settings = MagicMock(web_fallback_enabled=True)
    denied_matrix = {
        "policy": "What is your refund policy on opened merchandise?",
        "pricing": "What are your monthly subscription fees and discount promos?",
        "warranty": "Does your product warranty cover replacement screen defects?",
        "contact": "What is the toll-free customer support phone number?",
        "accounts": "How do I reset my account login credentials and 2FA?",
        "coding": "Write a python script to calculate fibonacci numbers",
    }
    for category, query in denied_matrix.items():
        state = {"kb_grade": "weak", "messages": [HumanMessage(content=query)]}
        assert (
            decide_after_kb_grade(state, settings=enabled_settings)
            == NodeName.ANSWER_INSUFFICIENT.value
        ), f"Expected ANSWER_INSUFFICIENT for denied category '{category}': {query}"


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


def test_is_allowed_web_fallback_query():
    from agenticrag.routing import is_allowed_web_fallback_query

    # Generic product / carrier / postal inquiries MUST be ALLOWED
    assert is_allowed_web_fallback_query("zip code lookup for shipment") is True
    assert is_allowed_web_fallback_query("FedEx international transit time") is True
    assert is_allowed_web_fallback_query("What are the product dimensions?") is True

    # Policy, pricing, warranty, contact, accounts, and coding MUST be BLOCKED
    assert is_allowed_web_fallback_query("promo code") is False
    assert is_allowed_web_fallback_query("discount code") is False
    assert is_allowed_web_fallback_query("return policy") is False
    assert is_allowed_web_fallback_query("warranty claim") is False
    assert is_allowed_web_fallback_query("support phone number") is False
    assert is_allowed_web_fallback_query("reset account password") is False
    assert is_allowed_web_fallback_query("write a python script") is False
    assert is_allowed_web_fallback_query("debug this code") is False


def test_human_escalation_intent_phrases():
    from agenticrag.routing import is_human_escalation_request

    # Intent phrases must escalate
    assert is_human_escalation_request("I want to speak to a manager") is True
    assert is_human_escalation_request("Can I talk to a supervisor?") is True
    assert is_human_escalation_request("I want a representative right now") is True
    assert is_human_escalation_request("Connect me to a live agent") is True

    # Negative test: standalone manager context must NOT escalate
    assert is_human_escalation_request("my manager asked me to track this order") is False
    assert is_human_escalation_request("our store operator received the package") is False


def test_policy_sensitive_intent_phrases():
    from agenticrag.routing import is_policy_sensitive_request

    assert is_policy_sensitive_request("I will take you to court") is True
    assert is_policy_sensitive_request("I am contacting my lawyer") is True

    # Standalone 'court' in innocent context must NOT trigger
    assert is_policy_sensitive_request("Deliver to 123 Court Street") is False


def test_get_truncated_conversation_history():
    """get_truncated_conversation_history truncates assistant messages to character limit and respects limit N."""
    from langchain_core.messages import AIMessage, HumanMessage

    from agenticrag.routing import get_truncated_conversation_history

    long_assistant_msg = "A" * 350
    state = {
        "messages": [
            HumanMessage(content="First turn"),
            AIMessage(content=long_assistant_msg),
            HumanMessage(content="Second turn"),
            AIMessage(content="Short assistant reply"),
            HumanMessage(content="Active turn"),
        ]
    }

    # Default settings (limit=4, cut_length=200)
    history = get_truncated_conversation_history(state, limit=4, assistant_char_limit=200)
    assert "First turn" in history
    assert "Second turn" in history
    assert "Short assistant reply" in history
    # Verify truncation
    assert "A" * 200 + "..." in history
    assert "A" * 205 not in history

    # Custom limit N=2
    history_short = get_truncated_conversation_history(state, limit=2, assistant_char_limit=200)
    assert "First turn" not in history_short
    assert "Second turn" in history_short
    assert "Short assistant reply" in history_short
