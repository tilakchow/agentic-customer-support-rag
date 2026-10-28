"""Query inspection helpers and pure conditional routing edges for LangGraph."""

import logging
import re
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


# Compiled word-boundary regexes for routing heuristics
_INTERNAL_CONTACT_REGEX = re.compile(
    r"\b(?:"
    r"support\s+phone|"
    r"phone\s+number|"
    r"customer\s+service\s+number|"
    r"customer\s+support\s+number|"
    r"toll\s+free|"
    r"hotline|"
    r"telephone|"
    r"call\s+you|"
    r"call\s+support|"
    r"helpdesk\s+number|"
    r"contact\s+number|"
    r"call\s+center|"
    r"direct\s+number|"
    r"helpline|"
    r"support\s+number"
    r")\b",
    re.IGNORECASE,
)

_PROPRIETARY_POLICY_REGEX = re.compile(
    r"\b(?:"
    r"policy|"
    r"warranty|"
    r"terms\s+of\s+service|"
    r"terms\s+and\s+conditions"
    r")\b",
    re.IGNORECASE,
)

_HUMAN_ESCALATION_REGEX = re.compile(
    r"\b(?:"
    r"(?:speak|talk|chat)\s+(?:to|with)\s+(?:a\s+)?(?:human|person|agent|representative|supervisor|manager)|"
    r"i\s+(?:need|want)\s+to\s+(?:speak|talk)\s+to\s+(?:a\s+|your\s+)?(?:human|person|agent|representative|supervisor|manager)|"
    r"i\s+want\s+(?:a\s+)?(?:human|person|agent|representative|supervisor|manager)|"
    r"human\s+(?:agent|representative|person|support)|"
    r"live\s+(?:agent|representative|person)|"
    r"real\s+person|"
    r"(?:transfer|connect)\s+(?:me\s+to|to)\s+(?:an?\s+)?(?:live\s+)?(?:agent|human|representative|supervisor|manager|operator)|"
    r"talk\s+to\s+someone\s+from\s+support|"
    r"escalate\s+(?:to\s+human|to\s+manager|to\s+supervisor|ticket)|"
    r"contact_human_agent|"
    r"(?:don'?t|dont|no\s+one\s+is|nobody)\s+(?:even\s+)?respond(?:s|ing)?|"
    r"terrible\s+customer\s+service|"
    r"awful\s+service"
    r")\b",
    re.IGNORECASE,
)

_POLICY_SENSITIVE_REGEX = re.compile(
    r"\b(?:"
    r"take\s+(?:\w+\s+)?to\s+court|"
    r"lawyer|"
    r"attorney|"
    r"lawsuit|"
    r"sue\s+you|"
    r"legal\s+action|"
    r"litigation|"
    r"consumer\s+protection\s+bureau|"
    r"ftc\s+complaint|"
    r"better\s+business\s+bureau|"
    r"file\s+a\s+complaint\s+with"
    r")\b",
    re.IGNORECASE,
)

_DISALLOWED_WEB_FALLBACK_REGEX = re.compile(
    r"\b(?:"
    r"write\s+(?:a\s+)?(?:python\s+|javascript\s+|bash\s+)?script|"
    r"write\s+(?:a\s+)?function|"
    r"write\s+(?:(?:a|some)\s+)?(?:javascript\s+|python\s+|typescript\s+)?code|"
    r"debug\s+(?:this\s+)?(?:recursive\s+)?(?:algorithm|code)|"
    r"(?:python|javascript|typescript|bash)\s+(?:script|code)|"
    r"fibonacci|"
    r"solve\s+(?:this\s+)?equation|"
    r"algorithm|"
    r"sql\s+query|"
    r"programming|"
    r"coding"
    r")\b",
    re.IGNORECASE,
)


def is_internal_contact_query(query: str) -> bool:
    """Detects if query asks for internal company contact channels (phone, hotline, email)."""
    return bool(_INTERNAL_CONTACT_REGEX.search(query))


def is_proprietary_policy_query(query: str) -> bool:
    """Detects inquiries regarding proprietary company policies to prevent external web hallucination."""
    return bool(_PROPRIETARY_POLICY_REGEX.search(query))


def is_human_escalation_request(query: str) -> bool:
    """Detects explicit user requests to speak to a human representative, supervisor, or agent."""
    return bool(_HUMAN_ESCALATION_REGEX.search(query))


def is_policy_sensitive_request(query: str) -> bool:
    """Detects legal threats, lawsuits, regulatory filings, or severe escalation triggers."""
    return bool(_POLICY_SENSITIVE_REGEX.search(query))


def is_allowed_web_fallback_query(query: str) -> bool:
    """Restricts open web fallback strictly from executing coding, programming, or scripting requests."""
    return not bool(_DISALLOWED_WEB_FALLBACK_REGEX.search(query))


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
