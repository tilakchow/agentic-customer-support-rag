"""Core LangGraph execution nodes and error-resilient state handlers for Agentic RAG."""

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agenticrag.clients import (
    get_classifier_llm,
    get_grader_llm,
    get_llm,
    get_pinecone_client,
    get_router_llm,
    get_vectorstore,
    get_web_search,
)
from agenticrag.config import get_settings
from agenticrag.constants import RETRIEVER_K
from agenticrag.metrics import record_provider_failure
from agenticrag.prompts import (
    CONDENSE_QUERY_PROMPT_TEMPLATE,
    CUSTOMER_SUPPORT_SYSTEM_PROMPT,
    DIRECT_ANSWER_PROMPT_TEMPLATE,
    HUMAN_ESCALATION_TEMPLATE,
    INSUFFICIENT_EVIDENCE_PROMPT_TEMPLATE,
    KB_GENERATION_HUMAN_TEMPLATE,
    KB_GRADER_HUMAN_TEMPLATE,
    KB_GRADER_SYSTEM_PROMPT,
    OUT_OF_SCOPE_REFUSAL_MESSAGE,
    REWRITE_QUERY_PROMPT_TEMPLATE,
    ROUTER_PROMPT_TEMPLATE,
    SCOPE_CLASSIFIER_PROMPT_TEMPLATE,
    UNTRUSTED_CONTEXT_INSTRUCTION,
    WEB_GENERATION_HUMAN_TEMPLATE,
    WEB_GRADER_HUMAN_TEMPLATE,
    WEB_GRADER_SYSTEM_PROMPT,
    sanitize_untrusted_context,
)
from agenticrag.retrieval import retrieve_kb as retrieve_kb_documents
from agenticrag.routing import (
    get_conversation_history,
    get_latest_question,
    is_human_escalation_request,
    is_policy_sensitive_request,
)
from agenticrag.schemas import ScopeDecision
from agenticrag.state import AgentState

logger = logging.getLogger(__name__)


# --- Async Graph Execution Nodes ---


async def classify_scope(state: AgentState) -> dict[str, Any]:
    """Evaluates question scope before routing.

    Categorizes input into:
    - 'in_scope': Standard customer support queries (orders, account, returns, shipping, billing, chit-chat)
    - 'out_of_scope': Programming/code, creative writing, general trivia, homework, non-support tasks
    - 'needs_human': Explicit human escalation requests or persistent complex complaints
    - 'policy_sensitive': Legal threats, litigation, regulatory filings
    """
    question = get_latest_question(state)
    history = get_conversation_history(state)

    # Fast heuristic checks for deterministic edge cases
    if is_human_escalation_request(question):
        logger.info(
            "[Scope] Fast-track heuristic: human escalation requested for: %s", question[:50]
        )
        return {
            "scope": "needs_human",
            "current_query": question,
            "ticket_payload": None,
            "escalated": False,
        }

    if is_policy_sensitive_request(question):
        logger.info(
            "[Scope] Fast-track heuristic: policy sensitive inquiry detected: %s", question[:50]
        )
        return {
            "scope": "policy_sensitive",
            "current_query": question,
            "ticket_payload": None,
            "escalated": False,
        }

    # Deterministic check for obvious non-support out-of-scope requests
    q_lower = question.lower()
    obvious_oos = [
        "fibonacci",
        "python script",
        "write code",
        "write a script",
        "javascript code",
        "solve equation",
        "poem about",
        "write a poem",
        "write a story",
        "tell me a joke",
        "who is the president",
        "capital of",
        "speed of light",
        "distance to the moon",
    ]
    if any(kw in q_lower for kw in obvious_oos):
        logger.info(
            "[Scope] Fast-track heuristic: obvious out_of_scope detected: %s", question[:50]
        )
        return {
            "scope": "out_of_scope",
            "current_query": question,
            "ticket_payload": None,
            "escalated": False,
        }

    try:
        prompt = SCOPE_CLASSIFIER_PROMPT_TEMPLATE.format(history=history, question=question)
        decision: ScopeDecision = await get_classifier_llm().ainvoke(prompt)
        scope = decision.scope
        logger.info(
            "[Scope] Classification: %s (reason: %s) for query: %s",
            scope,
            decision.reason,
            question[:50],
        )
    except Exception as e:
        record_provider_failure(e)
        logger.warning("[Scope Error] Classifier failed (%s): %s", question[:50], e)
        # Fail-safe default: allow query to proceed to router
        scope = "in_scope"

    return {
        "scope": scope,
        "current_query": question,
        "ticket_payload": None,
        "escalated": False,
    }


async def refuse_out_of_scope(state: AgentState) -> dict[str, Any]:
    """Politely refuses out-of-scope queries (e.g. coding, general trivia, poems) without calling external retrievers or web search."""
    return {
        "messages": [AIMessage(content=OUT_OF_SCOPE_REFUSAL_MESSAGE)],
        "source_used": "out_of_scope_refusal",
        "scope": "out_of_scope",
    }


async def escalate_to_human(state: AgentState) -> dict[str, Any]:
    """Generates a structured customer support ticket payload and polite handoff response."""
    question = get_latest_question(state)
    scope = state.get("scope", "needs_human")
    ticket_id = f"TICK-{uuid.uuid4().hex[:8].upper()}"
    now_iso = datetime.now(UTC).isoformat()

    docs = state.get("kb_docs", [])
    doc_sources = [{"content": d.page_content[:200], "metadata": d.metadata} for d in docs]

    ticket_payload = {
        "ticket_id": ticket_id,
        "customer_query": question,
        "summary": f"Customer support escalation: {question[:120]}",
        "reason": scope,
        "timestamp": now_iso,
        "status": "escalated",
        "sources": doc_sources,
    }

    message_content = HUMAN_ESCALATION_TEMPLATE.format(ticket_id=ticket_id)

    return {
        "messages": [AIMessage(content=message_content)],
        "source_used": "human_escalation",
        "escalated": True,
        "ticket_payload": ticket_payload,
    }


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
        record_provider_failure(e)
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
    kb_retrieval_failed = False
    docs: list[Document] = []
    top_score = 0.0

    try:
        settings = get_settings()
        vectorstore = await asyncio.to_thread(get_vectorstore)
        if vectorstore is None:
            logger.error("[KB Retriever Error]: Pinecone vector store is unavailable.")
            return {
                "kb_docs": [],
                "kb_retrieval_failed": True,
                "top_similarity_score": 0.0,
            }

        pc = await asyncio.to_thread(get_pinecone_client) if settings.reranker_enabled else None
        res = await asyncio.to_thread(
            retrieve_kb_documents,
            query=query,
            k=RETRIEVER_K,
            vectorstore=vectorstore,
            settings=settings,
            pinecone_client=pc,
        )
        docs = res.docs
        top_score = res.top_score
    except Exception as e:
        logger.error("[KB Retriever Error]: %s", e, exc_info=True)
        record_provider_failure(e)
        kb_retrieval_failed = True
        docs = []
        top_score = 0.0

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
        raw_context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}" for doc in docs
        )
        context = sanitize_untrusted_context(raw_context)

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
        record_provider_failure(e)
        logger.warning(
            "[KB Grader Error] Failed to grade KB evidence. Falling back to 'weak': %s", e
        )
        return {"kb_grade": "weak"}


try:
    from tavily.errors import TavilyClientError  # type: ignore[import-not-found]
except ImportError:

    class TavilyClientError(Exception):  # type: ignore[no-redef]
        pass


__all__ = [
    "TavilyClientError",
    "answer_insufficient",
    "condense_query",
    "direct_answer",
    "generate_from_kb",
    "generate_from_web",
    "grade_kb_evidence",
    "grade_web_evidence",
    "retrieve_kb",
    "rewrite_query",
    "route_question",
    "search_web",
]


_tavily_semaphore: asyncio.Semaphore | None = None


def get_tavily_semaphore() -> asyncio.Semaphore:
    global _tavily_semaphore
    if _tavily_semaphore is None:
        settings = get_settings()
        _tavily_semaphore = asyncio.Semaphore(settings.tavily_max_concurrency)
    return _tavily_semaphore


async def search_web(state: AgentState) -> dict[str, Any]:
    """Executes external web search via Tavily asynchronously with timeout and concurrency throttling."""
    query = state.get("current_query", "")
    logger.info("[Tavily Search] Query: %s", query)
    settings = get_settings()
    timeout = settings.tavily_timeout
    sem = get_tavily_semaphore()

    try:
        async with sem:
            search_client = get_web_search()
            result = await asyncio.wait_for(
                asyncio.to_thread(search_client.invoke, {"query": query}),
                timeout=timeout,
            )
            if isinstance(result, dict):
                lines = []
                if result.get("answer"):
                    lines.append(f"Tavily summary: {result.get('answer')}")
                for item in result.get("results", []):
                    lines.append(
                        f"Title: {item.get('title', '')}\nContent: {item.get('content', '')}"
                    )
                web_text = "\n\n".join(lines) if lines else ""
            else:
                web_text = str(result)
            return {"web_results": web_text, "web_search_failed": False}
    except TimeoutError:
        logger.warning(
            "[Tavily Search Timeout]: Web search query '%s' timed out after %.1fs",
            query[:50],
            timeout,
        )
        return {"web_results": "", "web_search_failed": True}
    except TavilyClientError as e:
        record_provider_failure(e)
        logger.warning(
            "[Tavily Search Error]: Search failed for query '%s': %s",
            query[:50],
            e,
        )
        return {"web_results": "", "web_search_failed": True}
    except Exception as e:
        record_provider_failure(e)
        logger.warning(
            "[Tavily Search Error]: Unexpected failure for query '%s': %s",
            query[:50],
            e,
        )
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
        sanitized_web_results = sanitize_untrusted_context(web_results)
        messages = [
            SystemMessage(content=WEB_GRADER_SYSTEM_PROMPT),
            HumanMessage(
                content=WEB_GRADER_HUMAN_TEMPLATE.format(
                    question=question, web_results=sanitized_web_results
                )
            ),
        ]
        grade = await get_grader_llm().ainvoke(messages)
        logger.info("[Web Grader] Evidence grade: %s", grade.grade)
        return {"web_grade": grade.grade}
    except Exception as e:
        record_provider_failure(e)
        logger.warning(
            "[Web Grader Error] Failed to grade web evidence. Falling back to 'weak': %s", e
        )
        return {"web_grade": "weak"}


def _message_text(response: Any) -> str:
    """Extracts plain text from an LLM response regardless of its type."""
    return response.content if hasattr(response, "content") else str(response)


async def condense_query(state: AgentState) -> dict[str, Any]:
    """Condenses user query with conversation history into a standalone query.

    Skipped on the first user turn (when no prior assistant messages exist).
    """
    messages = state.get("messages", [])
    question = get_latest_question(state)

    # Count prior turns: if 0 or 1 user message and no prior assistant replies, skip LLM call
    has_history = any(
        isinstance(m, AIMessage) or (hasattr(m, "type") and m.type == "ai") for m in messages[:-1]
    )
    if not has_history:
        logger.info("[Condense Query] First turn detected; skipping query condensation.")
        return {"current_query": question}

    history = get_conversation_history(state)
    try:
        prompt = CONDENSE_QUERY_PROMPT_TEMPLATE.format(history=history, question=question)
        response = await get_llm().ainvoke(prompt)
        condensed = _message_text(response).strip()
        logger.info(
            "[Condense Query] Original: '%s' -> Condensed: '%s'", question[:50], condensed[:50]
        )
        return {"current_query": condensed}
    except Exception as e:
        record_provider_failure(e)
        logger.warning("[Condense Query Error] Failed to condense query. Keeping original: %s", e)
        return {"current_query": question}


async def rewrite_query(state: AgentState) -> dict[str, Any]:
    """Rewrites current search query to optimize retrieval precision for subsequent searches."""
    current_query = state.get("current_query") or get_latest_question(state)
    history = get_conversation_history(state)
    retry_count = state.get("retry_count", 0) + 1
    try:
        prompt = REWRITE_QUERY_PROMPT_TEMPLATE.format(history=history, current_query=current_query)
        response = await get_llm().ainvoke(prompt)
        rewritten = _message_text(response).strip()
        logger.info("[Rewriter] Rewritten query: '%s' (retry: %d)", rewritten, retry_count)
        return {"current_query": rewritten, "retry_count": retry_count}
    except Exception as e:
        record_provider_failure(e)
        logger.warning("[Rewriter Error] Failed to rewrite query. Keeping current query: %s", e)
        return {"current_query": current_query, "retry_count": retry_count}


DEFAULT_UNAVAILABLE_APOLOGY = "I apologize, but I am currently unable to complete your request due to a temporary service issue. Please try again shortly."


async def _generate_support_message(
    prompt_or_messages: Any, fallback_text: str, source_used: str
) -> dict[str, Any]:
    """Executes LLM synthesis with error containment and graceful customer fallback."""
    try:
        response = await get_llm().ainvoke(prompt_or_messages)
        content = _message_text(response)
        meta = getattr(response, "response_metadata", {}) or {}
        usage = getattr(response, "usage_metadata", None)
        ai_msg = AIMessage(content=str(content), response_metadata=meta, usage_metadata=usage)
        return {"messages": [ai_msg], "source_used": source_used}
    except Exception as e:
        record_provider_failure(e)
        logger.error("[%s Error] LLM generation failed: %s", source_used, e, exc_info=True)
        target_source = (
            "insufficient_evidence"
            if source_used in ("private_kb", "web_search", "insufficient_evidence")
            else source_used
        )
        return {"messages": [AIMessage(content=fallback_text)], "source_used": target_source}


async def generate_from_kb(state: AgentState) -> dict[str, Any]:
    """Synthesizes customer response grounded in retrieved internal knowledge base documents."""
    question = get_latest_question(state)
    history = get_conversation_history(state)
    raw_context = "\n\n".join(
        f"[KB Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in state.get("kb_docs", [])
    )
    context = sanitize_untrusted_context(raw_context)
    system_text = CUSTOMER_SUPPORT_SYSTEM_PROMPT + UNTRUSTED_CONTEXT_INSTRUCTION
    human_text = KB_GENERATION_HUMAN_TEMPLATE.format(
        context=context,
        history=history,
        question=question,
    )
    messages = [SystemMessage(content=system_text), HumanMessage(content=human_text)]
    return await _generate_support_message(messages, DEFAULT_UNAVAILABLE_APOLOGY, "private_kb")


async def generate_from_web(state: AgentState) -> dict[str, Any]:
    """Synthesizes customer response grounded in verified external web search information."""
    question = get_latest_question(state)
    history = get_conversation_history(state)
    web_context = sanitize_untrusted_context(state.get("web_results", ""))
    system_text = CUSTOMER_SUPPORT_SYSTEM_PROMPT + UNTRUSTED_CONTEXT_INSTRUCTION
    human_text = WEB_GENERATION_HUMAN_TEMPLATE.format(
        web_context=web_context,
        history=history,
        question=question,
    )
    messages = [SystemMessage(content=system_text), HumanMessage(content=human_text)]
    return await _generate_support_message(messages, DEFAULT_UNAVAILABLE_APOLOGY, "web_search")


async def direct_answer(state: AgentState) -> dict[str, Any]:
    """Generates direct answer for greetings, conversational chit-chat, or meta questions."""
    question = get_latest_question(state)
    history = get_conversation_history(state)
    prompt = DIRECT_ANSWER_PROMPT_TEMPLATE.format(
        system_prompt=CUSTOMER_SUPPORT_SYSTEM_PROMPT, history=history, question=question
    )
    return await _generate_support_message(
        prompt, "Hello! How can I assist you with your customer support inquiry?", "direct"
    )


async def answer_insufficient(state: AgentState) -> dict[str, Any]:
    """Generates a polite notice when knowledge base and web searches cannot answer the customer inquiry."""
    question = get_latest_question(state)
    history = get_conversation_history(state)
    prompt = INSUFFICIENT_EVIDENCE_PROMPT_TEMPLATE.format(
        system_prompt=CUSTOMER_SUPPORT_SYSTEM_PROMPT, history=history, question=question
    )
    fallback = "I apologize, but our records do not have sufficient information to answer your inquiry. Please contact our support team via Live Chat or submit a support ticket for further assistance."
    return await _generate_support_message(prompt, fallback, "insufficient_evidence")
