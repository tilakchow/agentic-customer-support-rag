"""Presentation metadata, streaming event descriptors, and UI helpers for LangGraph nodes."""

from enum import StrEnum
from typing import Any


class NodeName(StrEnum):
    """Enumeration of all node identifiers across the agent state graph."""
    ROUTE_QUESTION = "route_question"
    RETRIEVE_KB = "retrieve_kb"
    GRADE_KB_EVIDENCE = "grade_kb_evidence"
    SEARCH_WEB = "search_web"
    GRADE_WEB_EVIDENCE = "grade_web_evidence"
    REWRITE_QUERY = "rewrite_query"
    GENERATE_FROM_KB = "generate_from_kb"
    GENERATE_FROM_WEB = "generate_from_web"
    DIRECT_ANSWER = "direct_answer"
    ANSWER_INSUFFICIENT = "answer_insufficient"


NODE_METADATA: dict[str, dict[str, Any]] = {
    NodeName.ROUTE_QUESTION.value: {
        "status_message": "Analyzing question & routing...",
        "is_generation": False,
    },
    NodeName.RETRIEVE_KB.value: {
        "status_message": "Searching private knowledge base...",
        "is_generation": False,
    },
    NodeName.GRADE_KB_EVIDENCE.value: {
        "status_message": "Evaluating knowledge base evidence quality...",
        "is_generation": False,
    },
    NodeName.SEARCH_WEB.value: {
        "status_message": "Searching real-time web with Tavily...",
        "is_generation": False,
    },
    NodeName.GRADE_WEB_EVIDENCE.value: {
        "status_message": "Evaluating web search evidence...",
        "is_generation": False,
    },
    NodeName.REWRITE_QUERY.value: {
        "status_message": "Refining search query for better precision...",
        "is_generation": False,
    },
    NodeName.GENERATE_FROM_KB.value: {
        "status_message": "Synthesizing answer from knowledge base...",
        "is_generation": True,
    },
    NodeName.GENERATE_FROM_WEB.value: {
        "status_message": "Synthesizing answer from web results...",
        "is_generation": True,
    },
    NodeName.DIRECT_ANSWER.value: {
        "status_message": "Formulating direct response...",
        "is_generation": True,
    },
    NodeName.ANSWER_INSUFFICIENT.value: {
        "status_message": "Assessing insufficient context...",
        "is_generation": True,
    },
}


def get_node_status_message(node_name: str) -> str:
    """Derives a user-facing status message for a given graph node name."""
    meta = NODE_METADATA.get(node_name)
    return meta["status_message"] if meta else f"Executing {node_name}..."


def is_generation_node(node_name: str) -> bool:
    """Determines whether the given node streams final response tokens."""
    meta = NODE_METADATA.get(node_name)
    return meta["is_generation"] if meta else False
