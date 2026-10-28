"""Execution state schema passed across all LangGraph nodes."""

from typing import Annotated, Any, TypedDict, cast

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from agenticrag.config import get_settings


def trim_messages_reducer(
    existing: Any,
    incoming: Any,
) -> list[AnyMessage]:
    """Appends messages and retains at most MAX_THREAD_MESSAGES to prevent memory leaks."""
    updated: list[AnyMessage] = cast(list[AnyMessage], add_messages(existing, incoming))
    max_msgs = get_settings().max_thread_messages
    if max_msgs > 0 and len(updated) > max_msgs:
        return updated[-max_msgs:]
    return updated


class AgentState(TypedDict):
    """Execution state schema passed across all LangGraph nodes."""

    messages: Annotated[list[AnyMessage], trim_messages_reducer]
    current_query: str
    kb_docs: list[Document]
    kb_retrieval_failed: bool
    top_similarity_score: float
    web_results: str
    web_search_failed: bool
    kb_grade: str
    web_grade: str
    route: str  # Router intent: 'kb' or 'direct'
    source_used: str  # Origin of answer: 'private_kb', 'web_search', 'direct', 'insufficient_evidence', 'out_of_scope_refusal', 'human_escalation'
    retry_count: int
    scope: str  # 'in_scope', 'out_of_scope', 'needs_human', 'policy_sensitive'
    ticket_payload: dict[str, Any] | None
    escalated: bool
    tenant_id: str
    score_type: str  # 'rerank' or 'cosine'
