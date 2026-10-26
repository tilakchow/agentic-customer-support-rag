"""Core LangGraph execution nodes and error-resilient state handlers for Agentic RAG."""

import asyncio
import logging
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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
from agenticrag.metrics import record_provider_failure
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
    sanitize_untrusted_context,
)
from agenticrag.routing import (
    decide_after_kb_grade as _routing_decide_after_kb_grade,
)
from agenticrag.routing import (
    decide_after_web_grade as _routing_decide_after_web_grade,
)
from agenticrag.routing import (
    get_conversation_history,
    get_latest_question,
)
from agenticrag.state import AgentState

logger = logging.getLogger(__name__)


def decide_after_kb_grade(
    state: AgentState,
) -> Literal["generate_from_kb", "search_web", "answer_insufficient"]:
    """Conditional edge decision: delegates to routing logic using local settings."""
    return _routing_decide_after_kb_grade(state, settings=get_settings())


def decide_after_web_grade(
    state: AgentState,
) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
    """Conditional edge decision: delegates to routing logic using local settings."""
    return _routing_decide_after_web_grade(state, settings=get_settings())


# --- Async Graph Execution Nodes ---


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
    vectorstore = await asyncio.to_thread(get_vectorstore)
    retriever_client = await asyncio.to_thread(get_retriever)
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
            record_provider_failure(e)
            kb_retrieval_failed = True
            docs = []
    elif retriever_client:
        try:
            docs = await asyncio.to_thread(retriever_client.invoke, query)
        except Exception as e:
            logger.error("[KB Retriever Error]: %s", e, exc_info=True)
            record_provider_failure(e)
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
    "decide_after_kb_grade",
    "decide_after_web_grade",
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


async def rewrite_query(state: AgentState) -> dict[str, Any]:
    """Rewrites inquiry to optimize retrieval precision for subsequent searches."""
    question = get_latest_question(state)
    retry_count = state.get("retry_count", 0) + 1
    try:
        prompt = REWRITE_QUERY_PROMPT_TEMPLATE.format(question=question)
        response = await get_llm().ainvoke(prompt)
        rewritten = _message_text(response).strip()
        logger.info("[Rewriter] Rewritten query: '%s' (retry: %d)", rewritten, retry_count)
        return {"current_query": rewritten, "retry_count": retry_count}
    except Exception as e:
        record_provider_failure(e)
        logger.warning("[Rewriter Error] Failed to rewrite query. Keeping original query: %s", e)
        return {"current_query": question, "retry_count": retry_count}


DEFAULT_UNAVAILABLE_APOLOGY = "I apologize, but I am currently unable to complete your request due to a temporary service issue. Please try again shortly."


async def _generate_support_message(
    prompt_or_messages: Any, fallback_text: str, source_used: str
) -> dict[str, Any]:
    """Executes LLM synthesis with error containment and graceful customer fallback."""
    try:
        response = await get_llm().ainvoke(prompt_or_messages)
        content = _message_text(response)
        return {"messages": [AIMessage(content=str(content))], "source_used": source_used}
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
