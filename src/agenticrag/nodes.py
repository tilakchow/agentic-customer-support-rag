"""Core LangGraph nodes and decision-making logic for Agentic Customer Support RAG."""

import logging
from enum import StrEnum
from typing import Annotated, Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agenticrag.config import (
    MAX_RETRIES,
    get_kb_grader_llm,
    get_llm,
    get_retriever,
    get_router_llm,
    get_web_grader_llm,
    get_web_search,
)
from agenticrag.prompts import (
    CUSTOMER_SUPPORT_SYSTEM_PROMPT,
    DIRECT_ANSWER_PROMPT_TEMPLATE,
    INSUFFICIENT_EVIDENCE_PROMPT_TEMPLATE,
    KB_GENERATION_HUMAN_TEMPLATE,
    KB_GRADER_HUMAN_TEMPLATE,
    KB_GRADER_SYSTEM_PROMPT,
    REWRITE_QUERY_PROMPT_TEMPLATE,
    ROUTER_PROMPT_TEMPLATE,
    UNTRUSTED_CONTEXT_INSTRUCTION,
    WEB_GENERATION_HUMAN_TEMPLATE,
    WEB_GRADER_HUMAN_TEMPLATE,
    WEB_GRADER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# --- Shared Graph Node Enum & Metadata ---

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


# --- State Definition ---

class AgentState(TypedDict):
    """Execution state schema passed across all LangGraph nodes."""
    messages: Annotated[list[AnyMessage], add_messages]
    current_query: str
    kb_docs: list[Document]
    web_results: str
    web_search_failed: bool
    kb_grade: str
    web_grade: str
    route: str           # Router intent: 'kb' or 'direct'
    source_used: str     # Origin of answer: 'private_kb', 'web_search', 'direct', 'insufficient_evidence'
    retry_count: int


# --- Helper Functions ---

def get_latest_question(state: AgentState) -> str:
    """Extracts the most recent user inquiry from the state messages."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content).strip()
    return ""


def get_conversation_history(state: AgentState, limit: int = 6) -> str:
    """Formats a compact prior conversation history, excluding the current pending question."""
    messages = state.get("messages", [])
    if not messages:
        return "No previous conversation."

    # Exclude the very last human message to avoid duplicating the prompt inquiry
    prior_messages = messages[:-1]
    if not prior_messages:
        return "No previous conversation."

    history = []
    for msg in prior_messages[-limit:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content if hasattr(msg, "content") else str(msg)
        content_snippet = str(content).strip()
        if content_snippet:
            history.append(f"{role}: {content_snippet}")

    return "\n".join(history) if history else "No previous conversation."


# --- Graph Nodes ---

def route_question(state: AgentState) -> dict[str, Any]:
    """Analyzes user inquiry and classifies route to private knowledge base or direct conversation."""
    question = get_latest_question(state)
    history = get_conversation_history(state)

    prompt = ROUTER_PROMPT_TEMPLATE.format(history=history, question=question)
    decision = get_router_llm().invoke(prompt)

    logger.info("[Router] Decision: %s for query: %s", decision.route, question[:50])
    return {
        "current_query": question,
        "route": decision.route,
        "retry_count": 0,
        "web_search_failed": False,
    }


def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    """Conditional edge decision: routes to knowledge base retrieval or direct response."""
    if state.get("route") == "kb":
        return NodeName.RETRIEVE_KB.value
    return NodeName.DIRECT_ANSWER.value


def retrieve_kb(state: AgentState) -> dict[str, Any]:
    """Queries Pinecone vector store using the current search query."""
    query = state.get("current_query", "")
    retriever_client = get_retriever()
    if retriever_client:
        try:
            docs = retriever_client.invoke(query)
        except Exception as e:
            logger.error("[KB Retriever Error]: %s", e, exc_info=True)
            docs = []
    else:
        docs = []
    logger.info("[KB Retriever] Retrieved: %d chunks for query: '%s'", len(docs), query[:50])
    return {"kb_docs": docs}


def grade_kb_evidence(state: AgentState) -> dict[str, Any]:
    """Evaluates whether retrieved knowledge base documents are sufficient to answer the inquiry."""
    question = get_latest_question(state)
    docs = state.get("kb_docs", [])
    if not docs:
        logger.info("[KB Grader] No docs retrieved. Defaulting to weak grade.")
        return {"kb_grade": "weak"}

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )

    messages = [
        SystemMessage(content=KB_GRADER_SYSTEM_PROMPT),
        HumanMessage(content=KB_GRADER_HUMAN_TEMPLATE.format(question=question, context=context)),
    ]
    grade = get_kb_grader_llm().invoke(messages)
    logger.info("[KB Grader] Evidence grade: %s", grade.grade)
    return {"kb_grade": grade.grade}


def is_internal_contact_query(query: str) -> bool:
    """Detects if query asks for internal company contact channels (phone, hotline, email)."""
    q = query.lower()
    contact_keywords = [
        "phone number", "customer service number", "customer support number",
        "toll free", "hotline", "telephone", "call you", "call support",
        "support phone", "helpdesk number", "contact number", "call center",
        "direct number", "helpline", "support number"
    ]
    return any(kw in q for kw in contact_keywords)


def decide_after_kb_grade(state: AgentState) -> Literal["generate_from_kb", "search_web", "answer_insufficient"]:
    """Conditional edge decision: proceeds to KB synthesis if grade is good, else falls back to web search or insufficient notice."""
    if state.get("kb_grade") == "good":
        return NodeName.GENERATE_FROM_KB.value

    # Prevent searching the open web for proprietary contact numbers/channels
    question = get_latest_question(state)
    if is_internal_contact_query(question):
        logger.info("[Routing] Contact inquiry without KB evidence. Skipping web search to prevent external contact hallucination.")
        return NodeName.ANSWER_INSUFFICIENT.value

    return NodeName.SEARCH_WEB.value


def search_web(state: AgentState) -> dict[str, Any]:
    """Executes external web search via Tavily. Catches errors safely to prevent error leak."""
    query = state.get("current_query", "")
    logger.info("[Tavily Search] Query: %s", query)
    try:
        result = get_web_search().invoke({"query": query})
        if isinstance(result, dict):
            lines = []
            if result.get("answer"):
                lines.append(f"Tavily summary: {result.get('answer')}")
            for item in result.get("results", []):
                lines.append(f"Title: {item.get('title', '')}\nContent: {item.get('content', '')}")
            web_text = "\n\n".join(lines) if lines else ""
        else:
            web_text = str(result)
        return {"web_results": web_text, "web_search_failed": False}
    except Exception as e:
        logger.error("[Tavily Search Error]: %s", e, exc_info=True)
        # Never store raw exception text as evidence!
        return {"web_results": "", "web_search_failed": True}


def grade_web_evidence(state: AgentState) -> dict[str, Any]:
    """Evaluates whether web search results are relevant and sufficient to answer the question."""
    if state.get("web_search_failed", False):
        logger.warning("[Web Grader] Web search failed earlier. Bypassing LLM grading and setting weak grade.")
        return {"web_grade": "weak"}

    web_results = state.get("web_results", "").strip()
    if not web_results:
        return {"web_grade": "weak"}

    question = get_latest_question(state)
    messages = [
        SystemMessage(content=WEB_GRADER_SYSTEM_PROMPT),
        HumanMessage(content=WEB_GRADER_HUMAN_TEMPLATE.format(question=question, web_results=web_results)),
    ]
    grade = get_web_grader_llm().invoke(messages)
    logger.info("[Web Grader] Evidence grade: %s", grade.grade)
    return {"web_grade": grade.grade}


def decide_after_web_grade(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
    """Conditional edge decision: routes to web synthesis, query rewrite retry, or insufficient notice."""
    # If search failed due to API/network exception, route immediately to answer_insufficient
    if state.get("web_search_failed", False):
        return NodeName.ANSWER_INSUFFICIENT.value

    if state.get("web_grade") == "good":
        return NodeName.GENERATE_FROM_WEB.value
    if state.get("retry_count", 0) < MAX_RETRIES:
        return NodeName.REWRITE_QUERY.value
    return NodeName.ANSWER_INSUFFICIENT.value


def rewrite_query(state: AgentState) -> dict[str, Any]:
    """Rewrites inquiry to optimize retrieval precision for subsequent searches."""
    question = get_latest_question(state)
    retry_count = state.get("retry_count", 0) + 1
    prompt = REWRITE_QUERY_PROMPT_TEMPLATE.format(question=question)
    rewritten = get_llm().invoke(prompt).content.strip()
    logger.info("[Rewriter] Rewritten query: '%s' (retry: %d)", rewritten, retry_count)
    return {"current_query": rewritten, "retry_count": retry_count}


def generate_from_kb(state: AgentState) -> dict[str, Any]:
    """Synthesizes customer response grounded in retrieved internal knowledge base documents."""
    question = get_latest_question(state)
    history = get_conversation_history(state)
    context = "\n\n".join(
        f"[KB Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in state.get("kb_docs", [])
    )
    system_text = CUSTOMER_SUPPORT_SYSTEM_PROMPT + UNTRUSTED_CONTEXT_INSTRUCTION
    human_text = KB_GENERATION_HUMAN_TEMPLATE.format(
        context=context,
        history=history,
        question=question,
    )
    messages = [SystemMessage(content=system_text), HumanMessage(content=human_text)]
    answer = get_llm().invoke(messages).content
    return {"messages": [AIMessage(content=answer)], "source_used": "private_kb"}


def generate_from_web(state: AgentState) -> dict[str, Any]:
    """Synthesizes customer response grounded in verified external web search information."""
    question = get_latest_question(state)
    history = get_conversation_history(state)
    web_context = state.get("web_results", "")
    system_text = CUSTOMER_SUPPORT_SYSTEM_PROMPT + UNTRUSTED_CONTEXT_INSTRUCTION
    human_text = WEB_GENERATION_HUMAN_TEMPLATE.format(
        web_context=web_context,
        history=history,
        question=question,
    )
    messages = [SystemMessage(content=system_text), HumanMessage(content=human_text)]
    answer = get_llm().invoke(messages).content
    return {"messages": [AIMessage(content=answer)], "source_used": "web_search"}


def direct_answer(state: AgentState) -> dict[str, Any]:
    """Generates direct answer for greetings, conversational chit-chat, or meta questions."""
    question = get_latest_question(state)
    history = get_conversation_history(state)
    prompt = DIRECT_ANSWER_PROMPT_TEMPLATE.format(
        system_prompt=CUSTOMER_SUPPORT_SYSTEM_PROMPT,
        history=history,
        question=question
    )
    answer = get_llm().invoke(prompt).content
    return {"messages": [AIMessage(content=answer)], "source_used": "direct"}


def answer_insufficient(state: AgentState) -> dict[str, Any]:
    """Generates a polite notice when knowledge base and web searches cannot answer the customer inquiry."""
    question = get_latest_question(state)
    history = get_conversation_history(state)
    prompt = INSUFFICIENT_EVIDENCE_PROMPT_TEMPLATE.format(
        system_prompt=CUSTOMER_SUPPORT_SYSTEM_PROMPT,
        history=history,
        question=question
    )
    answer = get_llm().invoke(prompt).content
    return {"messages": [AIMessage(content=answer)], "source_used": "insufficient_evidence"}
