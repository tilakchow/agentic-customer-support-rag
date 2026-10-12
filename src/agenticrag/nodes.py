"""Core LangGraph nodes and decision-making logic for Agentic Customer Support RAG."""

import asyncio
import logging
from typing import Annotated, Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agenticrag.clients import (
    get_grader_llm,
    get_llm,
    get_retriever,
    get_router_llm,
    get_vectorstore,
    get_web_search,
)
from agenticrag.config import get_settings
from agenticrag.constants import RETRIEVER_K
from agenticrag.node_meta import (
    NODE_METADATA,
    NodeName,
    get_node_status_message,
    is_generation_node,
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

# Re-export node presentation metadata for backward compatibility
__all__ = [
    "AgentState",
    "NodeName",
    "NODE_METADATA",
    "get_node_status_message",
    "is_generation_node",
    "get_latest_question",
    "get_conversation_history",
    "route_question",
    "route_after_router",
    "retrieve_kb",
    "grade_kb_evidence",
    "is_internal_contact_query",
    "is_proprietary_policy_query",
    "decide_after_kb_grade",
    "search_web",
    "grade_web_evidence",
    "decide_after_web_grade",
    "rewrite_query",
    "generate_from_kb",
    "generate_from_web",
    "direct_answer",
    "answer_insufficient",
]


# --- State Definition ---


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


# --- Helper Functions (Pure & Sync) ---


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
        "return policy",
        "refund policy",
        "cancellation policy",
        "warranty policy",
        "exchange policy",
        "guarantee policy",
        "terms of service",
    ]
    return any(kw in q for kw in policy_keywords)


def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    """Conditional edge decision: routes to knowledge base retrieval or direct response."""
    if state.get("route") == "kb":
        return NodeName.RETRIEVE_KB.value
    return NodeName.DIRECT_ANSWER.value


def decide_after_kb_grade(
    state: AgentState,
) -> Literal["generate_from_kb", "search_web", "answer_insufficient"]:
    """Conditional edge decision: proceeds to KB synthesis if grade is good, else falls back to web search or insufficient notice."""
    if state.get("kb_grade") == "good":
        return NodeName.GENERATE_FROM_KB.value

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

    return NodeName.SEARCH_WEB.value


def decide_after_web_grade(
    state: AgentState,
) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
    """Conditional edge decision: routes to web synthesis, query rewrite retry, or insufficient notice."""
    # If search failed due to API/network exception, route immediately to answer_insufficient
    if state.get("web_search_failed", False):
        return NodeName.ANSWER_INSUFFICIENT.value

    if state.get("web_grade") == "good":
        return NodeName.GENERATE_FROM_WEB.value
    max_retries = get_settings().max_retries
    if state.get("retry_count", 0) < max_retries:
        return NodeName.REWRITE_QUERY.value
    return NodeName.ANSWER_INSUFFICIENT.value


# --- Async Graph Nodes ---


async def route_question(state: AgentState) -> dict[str, Any]:
    """Analyzes user inquiry and classifies route to private knowledge base or direct conversation."""
    question = get_latest_question(state)
    history = get_conversation_history(state)

    try:
        prompt = ROUTER_PROMPT_TEMPLATE.format(history=history, question=question)
        decision = await get_router_llm().ainvoke(prompt)
        route = decision.route
        logger.info("[Router] Decision: %s for query: %s", route, question[:50])
    except Exception as e:
        logger.warning(
            "[Router Error] Failed to route query (%s). Falling back to 'kb': %s", question[:50], e
        )
        route = "kb"

    return {
        "current_query": question,
        "route": route,
        "retry_count": 0,
        "web_search_failed": False,
        "kb_retrieval_failed": False,
        "top_similarity_score": 0.0,
    }


async def retrieve_kb(state: AgentState) -> dict[str, Any]:
    """Queries Pinecone vector store using the current search query with similarity scores."""
    query = state.get("current_query", "")
    vectorstore = get_vectorstore()
    retriever_client = get_retriever()
    kb_retrieval_failed = False
    docs: list[Document] = []
    top_score = 0.0

    if vectorstore:
        try:
            settings = get_settings()
            # Run blocking Pinecone network calls on a worker thread
            scored_results = await asyncio.to_thread(
                vectorstore.similarity_search_with_score,
                query,
                k=RETRIEVER_K,
                namespace=settings.namespace,
            )
            for doc, score in scored_results:
                doc.metadata["score"] = float(score)
                docs.append(doc)
            if scored_results:
                top_score = max(float(score) for _, score in scored_results)
        except Exception as e:
            logger.error("[KB Retriever Error]: %s", e, exc_info=True)
            kb_retrieval_failed = True
            docs = []
    elif retriever_client:
        try:
            docs = await asyncio.to_thread(retriever_client.invoke, query)
        except Exception as e:
            logger.error("[KB Retriever Error]: %s", e, exc_info=True)
            kb_retrieval_failed = True
            docs = []
    else:
        logger.error("[KB Retriever Error]: Pinecone retriever client is unavailable.")
        kb_retrieval_failed = True
        docs = []

    logger.info(
        "[KB Retriever] Retrieved: %d chunks for query: '%s' (top_score=%.4f, failed=%s)",
        len(docs),
        query[:50],
        top_score,
        kb_retrieval_failed,
    )
    return {
        "kb_docs": docs,
        "kb_retrieval_failed": kb_retrieval_failed,
        "top_similarity_score": top_score,
    }


async def grade_kb_evidence(state: AgentState) -> dict[str, Any]:
    """Evaluates whether retrieved knowledge base documents are sufficient to answer the inquiry."""
    question = get_latest_question(state)
    docs = state.get("kb_docs", [])
    if not docs:
        logger.info("[KB Grader] No docs retrieved. Defaulting to weak grade.")
        return {"kb_grade": "weak"}

    settings = get_settings()
    # Skip the KB grader LLM call when top similarity score is above the configured threshold
    if settings.skip_grader_score is not None:
        top_score = state.get("top_similarity_score")
        if top_score is None:
            top_score = max((float(d.metadata.get("score", 0.0)) for d in docs), default=0.0)
        if top_score >= settings.skip_grader_score:
            logger.info(
                "[KB Grader] Top similarity score %.4f >= threshold %.4f. Skipping LLM call and marking good.",
                top_score,
                settings.skip_grader_score,
            )
            return {"kb_grade": "good"}

    try:
        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}" for doc in docs
        )

        messages = [
            SystemMessage(content=KB_GRADER_SYSTEM_PROMPT),
            HumanMessage(
                content=KB_GRADER_HUMAN_TEMPLATE.format(question=question, context=context)
            ),
        ]
        grade = await get_grader_llm().ainvoke(messages)
        logger.info("[KB Grader] Evidence grade: %s", grade.grade)
        return {"kb_grade": grade.grade}
    except Exception as e:
        logger.warning(
            "[KB Grader Error] Failed to grade KB evidence. Falling back to 'weak': %s", e
        )
        return {"kb_grade": "weak"}


async def search_web(state: AgentState) -> dict[str, Any]:
    """Executes external web search via Tavily asynchronously with timeout."""
    query = state.get("current_query", "")
    logger.info("[Tavily Search] Query: %s", query)
    settings = get_settings()
    try:
        search_client = get_web_search()
        result = await asyncio.wait_for(
            asyncio.to_thread(search_client.invoke, {"query": query}),
            timeout=settings.tavily_timeout,
        )
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


async def grade_web_evidence(state: AgentState) -> dict[str, Any]:
    """Evaluates whether web search results are relevant and sufficient to answer the question."""
    if state.get("web_search_failed", False):
        logger.warning(
            "[Web Grader] Web search failed earlier. Bypassing LLM grading and setting weak grade."
        )
        return {"web_grade": "weak"}

    web_results = state.get("web_results", "").strip()
    if not web_results:
        return {"web_grade": "weak"}

    try:
        question = get_latest_question(state)
        messages = [
            SystemMessage(content=WEB_GRADER_SYSTEM_PROMPT),
            HumanMessage(
                content=WEB_GRADER_HUMAN_TEMPLATE.format(question=question, web_results=web_results)
            ),
        ]
        grade = await get_grader_llm().ainvoke(messages)
        logger.info("[Web Grader] Evidence grade: %s", grade.grade)
        return {"web_grade": grade.grade}
    except Exception as e:
        logger.warning(
            "[Web Grader Error] Failed to grade web evidence. Falling back to 'weak': %s", e
        )
        return {"web_grade": "weak"}


async def rewrite_query(state: AgentState) -> dict[str, Any]:
    """Rewrites inquiry to optimize retrieval precision for subsequent searches."""
    question = get_latest_question(state)
    retry_count = state.get("retry_count", 0) + 1
    try:
        prompt = REWRITE_QUERY_PROMPT_TEMPLATE.format(question=question)
        response = await get_llm().ainvoke(prompt)
        rewritten = (
            response.content.strip() if hasattr(response, "content") else str(response).strip()
        )
        logger.info("[Rewriter] Rewritten query: '%s' (retry: %d)", rewritten, retry_count)
        return {"current_query": rewritten, "retry_count": retry_count}
    except Exception as e:
        logger.warning("[Rewriter Error] Failed to rewrite query. Keeping original query: %s", e)
        return {"current_query": question, "retry_count": retry_count}


async def generate_from_kb(state: AgentState) -> dict[str, Any]:
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
    response = await get_llm().ainvoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)
    return {"messages": [AIMessage(content=answer)], "source_used": "private_kb"}


async def generate_from_web(state: AgentState) -> dict[str, Any]:
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
    response = await get_llm().ainvoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)
    return {"messages": [AIMessage(content=answer)], "source_used": "web_search"}


async def direct_answer(state: AgentState) -> dict[str, Any]:
    """Generates direct answer for greetings, conversational chit-chat, or meta questions."""
    question = get_latest_question(state)
    history = get_conversation_history(state)
    prompt = DIRECT_ANSWER_PROMPT_TEMPLATE.format(
        system_prompt=CUSTOMER_SUPPORT_SYSTEM_PROMPT, history=history, question=question
    )
    response = await get_llm().ainvoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return {"messages": [AIMessage(content=answer)], "source_used": "direct"}


async def answer_insufficient(state: AgentState) -> dict[str, Any]:
    """Generates a polite notice when knowledge base and web searches cannot answer the customer inquiry."""
    question = get_latest_question(state)
    history = get_conversation_history(state)
    prompt = INSUFFICIENT_EVIDENCE_PROMPT_TEMPLATE.format(
        system_prompt=CUSTOMER_SUPPORT_SYSTEM_PROMPT, history=history, question=question
    )
    response = await get_llm().ainvoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return {"messages": [AIMessage(content=answer)], "source_used": "insufficient_evidence"}
