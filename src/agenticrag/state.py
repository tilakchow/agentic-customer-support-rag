"""Execution state schema passed across all LangGraph nodes."""

from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Execution state schema passed across all LangGraph nodes."""

    messages: Annotated[list[AnyMessage], add_messages]
    current_query: str
    kb_docs: list[Document]
    kb_retrieval_failed: bool
    top_similarity_score: float
    web_results: str
    web_search_failed: bool
    kb_grade: str
    web_grade: str
    route: str  # Router intent: 'kb' or 'direct'
    source_used: (
        str  # Origin of answer: 'private_kb', 'web_search', 'direct', 'insufficient_evidence'
    )
    retry_count: int
