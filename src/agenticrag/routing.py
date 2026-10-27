"""Query inspection helpers and pure conditional routing edges for LangGraph."""

import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage

from agenticrag.config import get_settings
from agenticrag.node_meta import NodeName
from agenticrag.state import AgentState

logger = logging.getLogger(__name__)


def get_latest_question(state: AgentState) -> str:
    """Extracts the most recent user inquiry from the state messages."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
        if isinstance(message, tuple) and message[0] == "human":
            return str(message[1])
        if getattr(message, "type", None) == "human":
            return str(getattr(message, "content", ""))
    return ""


def get_conversation_history(state: AgentState, limit: int | None = None) -> str:
    """Extracts recent conversation turns prior to the active query."""
    if limit is None:
        limit = get_settings().max_thread_history

    messages = state.get("messages", [])
    if len(messages) <= 1:
        return "No previous conversation."

    prior_messages = messages[:-1]
    history = []
    for msg in prior_messages[-limit:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content if hasattr(msg, "content") else str(msg)
        content_snippet = str(content).strip()
        if content_snippet:
            history.append(f"{role}: {content_snippet}")

    return "\n".join(history) if history else "No previous conversation."


def is_internal_contact_query(query: str) -> bool:
    """Detects if query asks for internal company contact channels (phone, hotline, email)."""
    q = query.lower()
    contact_keywords = [
        "phone number",
        "customer service number",
        "customer support number",
        "toll free",
        "hotline",
        "telephone",
        "call you",
        "call support",
        "support phone",
        "helpdesk number",
        "contact number",
        "call center",
        "direct number",
        "helpline",
        "support number",
    ]
    return any(kw in q for kw in contact_keywords)


def is_proprietary_policy_query(query: str) -> bool:
    """Detects inquiries regarding proprietary company policies to prevent external web hallucination."""
    q = query.lower()
    policy_keywords = [
        "policy",
        "warranty",
        "terms of service",
        "terms and conditions",
    ]
    return any(kw in q for kw in policy_keywords)


def is_human_escalation_request(query: str) -> bool:
    """Detects explicit user requests to speak to a human representative, supervisor, or agent."""
    q = query.lower()
    escalation_keywords = [
        "talk to human",
        "speak to human",
        "talk to a human",
        "speak to a human",
        "speak with human",
        "speak with a human",
        "human agent",
        "human representative",
        "live agent",
        "real person",
        "talk to someone",
        "speak to someone",
        "customer service rep",
        "customer care rep",
        "representative",
        "supervisor",
        "manager",
        "support person",
        "operator",
        "escalate",
        "transfer to agent",
        "transfer me",
        "connect me to",
        "contact_human_agent",
        "dont even respond",
        "don't even respond",
        "no one is responding",
        "nobody responds",
        "speak with agent",
        "contact agent",
        "terrible customer service",
        "awful service",
    ]
    return any(kw in q for kw in escalation_keywords)


def is_policy_sensitive_request(query: str) -> bool:
    """Detects legal threats, lawsuits, regulatory filings, or severe escalation triggers."""
    q = query.lower()
    sensitive_keywords = [
        "lawyer",
        "attorney",
        "lawsuit",
        "sue you",
        "legal action",
        "litigation",
        "court",
        "consumer protection bureau",
        "ftc complaint",
        "better business bureau",
        "file a complaint with",
    ]
    return any(kw in q for kw in sensitive_keywords)


def is_allowed_web_fallback_query(query: str) -> bool:
    """Restricts open web fallback strictly from executing coding, programming, or scripting requests.

    Explicitly rejects programming/coding, algorithms, and developer script tasks.
    """
    q = query.lower()
    disallowed_keywords = [
        "python",
        "javascript",
        "typescript",
        "java ",
        "script",
        "code",
        "fibonacci",
        "algorithm",
        "sql query",
        "debug code",
        "write code",
        "write a function",
        "programming",
    ]
    return not any(kw in q for kw in disallowed_keywords)


def route_after_scope(
    state: AgentState,
) -> Literal["route_question", "refuse_out_of_scope", "escalate_to_human"]:
    """Conditional edge decision after scope classification."""
    scope = state.get("scope", "in_scope")
    if scope in ("needs_human", "policy_sensitive") or state.get("escalated", False):
        return NodeName.ESCALATE_TO_HUMAN.value
    if scope == "out_of_scope":
        return NodeName.REFUSE_OUT_OF_SCOPE.value
    return NodeName.ROUTE_QUESTION.value


def route_after_router(state: AgentState) -> Literal["condense_query", "direct_answer"]:
    """Conditional edge decision: routes to query condensation (for KB) or direct response."""
    if state.get("route") == "kb":
        return NodeName.CONDENSE_QUERY.value
    return NodeName.DIRECT_ANSWER.value


def decide_after_kb_grade(
    state: AgentState,
    settings: Any = None,
) -> Literal["generate_from_kb", "search_web", "answer_insufficient"]:
    """Conditional edge decision: proceeds to KB synthesis if grade is good, else falls back to web search or insufficient notice."""
    if state.get("kb_grade") == "good":
        return NodeName.GENERATE_FROM_KB.value

    if settings is None:
        settings = get_settings()
    if not settings.web_fallback_enabled:
        logger.info(
            "[Routing] Web fallback disabled by configuration. Routing to answer_insufficient."
        )
        return NodeName.ANSWER_INSUFFICIENT.value

    # Prevent searching the open web for proprietary contact numbers/channels
    question = get_latest_question(state)
    if is_internal_contact_query(question):
        logger.info(
            "[Routing] Contact inquiry without KB evidence. Skipping web search to prevent external contact hallucination."
        )
        return NodeName.ANSWER_INSUFFICIENT.value

    # Prevent searching the open web for proprietary company policies
    if is_proprietary_policy_query(question):
        logger.info(
            "[Routing] Proprietary policy inquiry without KB evidence. Skipping web search."
        )
        return NodeName.ANSWER_INSUFFICIENT.value

    # Prevent searching the open web for out-of-domain topics (e.g. coding, general trivia)
    if not is_allowed_web_fallback_query(question):
        logger.info(
            "[Routing] Query '%s' not eligible for web fallback. Routing to answer_insufficient.",
            question[:50],
        )
        return NodeName.ANSWER_INSUFFICIENT.value

    return NodeName.SEARCH_WEB.value


def decide_after_web_grade(
    state: AgentState,
    settings: Any = None,
) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
    """Conditional edge decision: routes to web synthesis, query rewrite retry, or insufficient notice."""
    # If search failed due to API/network exception, route immediately to answer_insufficient
    if state.get("web_search_failed", False):
        return NodeName.ANSWER_INSUFFICIENT.value

    if state.get("web_grade") == "good":
        return NodeName.GENERATE_FROM_WEB.value
    if settings is None:
        settings = get_settings()
    max_retries = settings.max_retries
    if state.get("retry_count", 0) < max_retries:
        return NodeName.REWRITE_QUERY.value
    return NodeName.ANSWER_INSUFFICIENT.value
