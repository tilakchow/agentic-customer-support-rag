"""Core LangGraph execution nodes and error-resilient state handlers for Agentic RAG."""

import asyncio
import logging
import re
import uuid
from datetime import UTC, datetime
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agenticrag.auth import get_tenant_namespace
from agenticrag.clients import (
    get_classifier_llm,
    get_faithfulness_llm,
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
    CLAIM_SUPPORT_PROMPT_TEMPLATE,
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
    STRICT_REGEN_HUMAN_TEMPLATE,
    STRICT_REGEN_SYSTEM_PROMPT,
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
    get_truncated_conversation_history,
    is_human_escalation_request,
    is_policy_sensitive_request,
    is_proprietary_policy_query,
)
from agenticrag.schemas import ClaimSupportCheck, ScopeDecision, ToolCallRequest, ToolPlan
from agenticrag.security import (
    check_llm_injection,
    filter_canary_and_leaks,
    is_prompt_injection,
    is_suspicious_for_injection,
    redact_pii,
)
from agenticrag.state import AgentState
from agenticrag.telemetry import get_telemetry_collector, traced_node
from agenticrag.tools import create_support_ticket, get_order_status

logger = logging.getLogger(__name__)

_OBVIOUS_OOS_REGEX = re.compile(
    r"\b(?:"
    r"fibonacci|"
    r"python\s+script|"
    r"write\s+code|"
    r"write\s+(?:a\s+)?script|"
    r"javascript\s+code|"
    r"solve\s+(?:the\s+)?equation|"
    r"poem\s+about|"
    r"write\s+(?:a\s+)?poem|"
    r"write\s+(?:a\s+)?story|"
    r"tell\s+me\s+a\s+joke|"
    r"who\s+is\s+the\s+president|"
    r"capital\s+of|"
    r"speed\s+of\s+light|"
    r"distance\s+to\s+the\s+moon"
    r")\b",
    re.IGNORECASE,
)


# --- Async Graph Execution Nodes ---


@traced_node("classify_scope")
async def classify_scope(state: AgentState) -> dict[str, Any]:
    """Evaluates question scope before routing.

    Categorizes input into:
    - 'in_scope': Standard customer support queries (orders, account, returns, shipping, billing, chit-chat)
    - 'out_of_scope': Programming/code, creative writing, general trivia, homework, non-support tasks, prompt injections
    - 'needs_human': Explicit human escalation requests or persistent complex complaints
    - 'policy_sensitive': Legal threats, litigation, regulatory filings
    """
    question = get_latest_question(state)
    history = get_truncated_conversation_history(state)

    # Injection and adversarial prompt exfiltration check (fast-path regex + second-layer LLM)
    is_inj, inj_reason = is_prompt_injection(question)
    if not is_inj and is_suspicious_for_injection(question):
        is_inj, inj_reason = await check_llm_injection(question)

    if is_inj:
        logger.warning("[Security] Prompt injection blocked: %s", inj_reason)
        return {
            "scope": "out_of_scope",
            "current_query": question,
            "ticket_payload": None,
            "escalated": False,
        }

    # Fast heuristic checks for deterministic edge cases
    if is_human_escalation_request(question):
        logger.info(
            "[Scope] Fast-track heuristic: human escalation requested (len=%d)", len(question)
        )
        logger.debug("[Scope] Query: %s", redact_pii(question))
        return {
            "scope": "needs_human",
            "current_query": question,
            "ticket_payload": None,
            "escalated": False,
        }

    if is_policy_sensitive_request(question):
        logger.info(
            "[Scope] Fast-track heuristic: policy sensitive inquiry detected (len=%d)",
            len(question),
        )
        logger.debug("[Scope] Query: %s", redact_pii(question))
        return {
            "scope": "policy_sensitive",
            "current_query": question,
            "ticket_payload": None,
            "escalated": False,
        }

    # Deterministic check for obvious non-support out-of-scope requests (word-boundary regex)
    if _OBVIOUS_OOS_REGEX.search(question):
        logger.info(
            "[Scope] Fast-track heuristic: obvious out_of_scope detected (len=%d)", len(question)
        )
        logger.debug("[Scope] Query: %s", redact_pii(question))
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
            "[Scope] Classification: %s (reason: %s, len=%d)",
            scope,
            decision.reason,
            len(question),
        )
    except Exception as e:
        record_provider_failure(e)
        logger.warning("[Scope Error] Classifier failed (query_len=%d): %s", len(question), e)
        settings = get_settings()
        fail_mode = settings.fail_mode_scope_classifier
        get_telemetry_collector().record_fail_decision(
            node_name="scope_classifier",
            error_type=type(e).__name__,
            mode=fail_mode,
            details=str(e),
        )
        if fail_mode == "closed":
            # On error in closed mode, route to answer_insufficient with polite try-again + escalation hint
            return {
                "scope": "error_insufficient",
                "current_query": question,
                "ticket_payload": None,
                "escalated": False,
                "messages": [
                    AIMessage(
                        content="I apologize, but our system is currently experiencing a temporary issue processing your request. Please try again in a moment, or let us know if you would like to speak with a human support agent."
                    )
                ],
                "source_used": "insufficient_evidence",
            }
        scope = "in_scope"

    return {
        "scope": scope,
        "current_query": question,
        "ticket_payload": None,
        "escalated": False,
    }


@traced_node("refuse_out_of_scope")
async def refuse_out_of_scope(state: AgentState) -> dict[str, Any]:
    """Politely refuses out-of-scope queries (e.g. coding, general trivia, poems, injections) without calling external retrievers or web search."""
    question = get_latest_question(state)
    is_inj, _ = is_prompt_injection(question)
    refusal = (
        "I am an automated customer support assistant. I cannot follow instructions to override system rules or disclose internal instructions."
        if is_inj
        else OUT_OF_SCOPE_REFUSAL_MESSAGE
    )
    source = "prompt_injection_refusal" if is_inj else "out_of_scope_refusal"
    return {
        "messages": [AIMessage(content=refusal)],
        "source_used": source,
        "scope": "out_of_scope",
    }


@traced_node("escalate_to_human")
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
        "customer_query": redact_pii(question),
        "summary": f"Customer support escalation: {redact_pii(question[:120])}",
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


@traced_node("route_question")
async def route_question(state: AgentState) -> dict[str, Any]:
    """Analyzes user inquiry and classifies route to private knowledge base or direct conversation."""
    question = get_latest_question(state)
    history = get_truncated_conversation_history(state)

    try:
        prompt = ROUTER_PROMPT_TEMPLATE.format(history=history, question=question)
        decision = await get_router_llm().ainvoke(prompt)
        route = decision.route
        logger.info("[Router] Decision: %s (query_len=%d)", route, len(question))
        logger.debug("[Router] Redacted query: %s", redact_pii(question))
    except Exception as e:
        record_provider_failure(e)
        logger.warning(
            "[Router Error] Failed to route query (query_len=%d): %s",
            len(question),
            e,
        )
        settings = get_settings()
        fail_mode = settings.fail_mode_router
        get_telemetry_collector().record_fail_decision(
            node_name="router",
            error_type=type(e).__name__,
            mode=fail_mode,
            details=str(e),
        )
        route = "kb" if fail_mode == "open" else "direct"

    return {
        "current_query": question,
        "route": route,
        "retry_count": 0,
        "web_search_failed": False,
        "kb_retrieval_failed": False,
        "top_similarity_score": 0.0,
    }


@traced_node("plan_tools")
async def plan_tools(state: AgentState) -> dict[str, Any]:
    """Plans and executes customer support operational tools (order tracking, ticket creation).

    Enforces:
    - Structured output using ToolPlan schema
    - Capped at maximum 2 tool calls per turn
    - Pydantic validation of tool arguments
    - Multi-tenant isolation for order status lookups
    """
    question = get_latest_question(state)
    tenant_id = state.get("tenant_id", "default")

    # Fast heuristic checks for order IDs (e.g. ORD-1001) or ticket creation
    order_match = re.search(r"\b(ORD-\d+)\b", question, re.IGNORECASE)
    is_order_query = bool(
        order_match
        or re.search(r"\b(?:order|tracking|package|shipment|status)\b", question, re.IGNORECASE)
    )
    is_ticket_query = bool(
        re.search(
            r"\b(?:ticket|support\s+request|file\s+a\s+claim|complaint)\b",
            question,
            re.IGNORECASE,
        )
    )

    if not is_order_query and not is_ticket_query:
        return {"tool_calls": [], "tool_results": []}

    plan = ToolPlan(needs_tools=False, calls=[])

    # Try deterministic rule extraction first if clear order ID or ticket intent
    extracted_calls: list[ToolCallRequest] = []
    if order_match:
        extracted_calls.append(
            ToolCallRequest(tool_name="get_order_status", order_id=order_match.group(1).upper())
        )
    if is_ticket_query and any(
        kw in question.lower() for kw in ("create", "open", "submit", "file")
    ):
        extracted_calls.append(
            ToolCallRequest(
                tool_name="create_support_ticket",
                summary=redact_pii(question[:100]),
                order_id=order_match.group(1).upper() if order_match else None,
            )
        )

    if extracted_calls:
        plan = ToolPlan(needs_tools=True, calls=extracted_calls)
    else:
        # LLM structured output for fuzzy planning
        try:
            history = get_truncated_conversation_history(state)
            prompt = (
                "You are an operational tool planner for customer support.\n"
                "Available tools:\n"
                "1. get_order_status(order_id): Look up status, carrier, and tracking for an order ID (e.g. ORD-1001).\n"
                "2. create_support_ticket(summary, order_id): Create a support ticket for customer issues.\n\n"
                f"Recent Conversation:\n{history}\n\n"
                f"Customer Query: {question}\n\n"
                "Determine if any tools are needed. Max 2 calls. Return JSON matching the ToolPlan schema."
            )
            llm = get_llm().with_structured_output(ToolPlan, method="json_mode")
            plan_res = await llm.ainvoke(prompt)
            if isinstance(plan_res, ToolPlan):
                plan = plan_res
        except Exception as e:
            logger.debug("[Tool Planner LLM fallback]: %s", e)
            plan = ToolPlan(needs_tools=False, calls=[])

    # Enforce maximum 2 calls per turn constraint
    valid_calls = plan.calls[:2] if plan.needs_tools else []
    tool_results: list[dict[str, Any]] = []

    for call in valid_calls:
        if call.tool_name == "get_order_status" and call.order_id:
            res = get_order_status(order_id=call.order_id, tenant_id=tenant_id)
            tool_results.append({"tool": "get_order_status", "result": res})
        elif call.tool_name == "create_support_ticket":
            res = create_support_ticket(
                summary=call.summary or redact_pii(question[:120]),
                order_id=call.order_id,
                tenant_id=tenant_id,
            )
            tool_results.append({"tool": "create_support_ticket", "result": res})

    serialized_calls = [c.model_dump() for c in valid_calls]
    return {
        "tool_calls": serialized_calls,
        "tool_results": tool_results,
    }


@traced_node("retrieve_kb")
async def retrieve_kb(state: AgentState) -> dict[str, Any]:
    """Queries Pinecone vector store using the current search query with similarity scores."""
    query = state.get("current_query", "")
    kb_retrieval_failed = False
    docs: list[Document] = []
    top_score = 0.0
    score_type = "cosine"

    try:
        settings = get_settings()
        vectorstore = await asyncio.to_thread(get_vectorstore)
        if vectorstore is None:
            logger.error("[KB Retriever Error]: Pinecone vector store is unavailable.")
            return {
                "kb_docs": [],
                "kb_retrieval_failed": True,
                "top_similarity_score": 0.0,
                "score_type": "cosine",
            }

        pc = await asyncio.to_thread(get_pinecone_client) if settings.reranker_enabled else None
        tenant_id = state.get("tenant_id", "default")
        namespace = get_tenant_namespace(tenant_id)
        res = await asyncio.to_thread(
            retrieve_kb_documents,
            query=query,
            k=RETRIEVER_K,
            namespace=namespace,
            vectorstore=vectorstore,
            settings=settings,
            pinecone_client=pc,
        )
        docs = res.docs
        top_score = res.top_score
        score_type = res.score_type
    except Exception as e:
        logger.error("[KB Retriever Error]: %s", e, exc_info=True)
        record_provider_failure(e)
        kb_retrieval_failed = True
        docs = []
        top_score = 0.0
        score_type = "cosine"

    logger.info(
        "[KB Retriever] Retrieved: %d chunks (top_score=%.4f, score_type=%s, failed=%s, query_len=%d)",
        len(docs),
        top_score,
        score_type,
        kb_retrieval_failed,
        len(query),
    )
    logger.debug("[KB Retriever] Redacted query: %s", redact_pii(query))

    return {
        "kb_docs": docs,
        "kb_retrieval_failed": kb_retrieval_failed,
        "top_similarity_score": top_score,
        "score_type": score_type,
    }


@traced_node("grade_kb_evidence")
async def grade_kb_evidence(state: AgentState) -> dict[str, Any]:
    """Evaluates whether retrieved knowledge base documents are sufficient to answer the inquiry."""
    question = get_latest_question(state)
    docs = state.get("kb_docs", [])
    if not docs:
        logger.info("[KB Grader] No docs retrieved. Defaulting to weak grade.")
        return {"kb_grade": "weak"}

    settings = get_settings()
    score_type = state.get("score_type", "cosine")
    # Bypass the KB grader LLM call ONLY when score_type == "rerank" and top score is above threshold
    if score_type == "rerank" and settings.skip_grader_score is not None:
        top_score = state.get("top_similarity_score")
        if top_score is None:
            top_score = max((float(d.metadata.get("rerank_score", 0.0)) for d in docs), default=0.0)
        if top_score >= settings.skip_grader_score:
            logger.info(
                "[KB Grader] Top rerank score %.4f >= threshold %.4f. Skipping LLM call and marking good.",
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
        logger.warning("[KB Grader Error] Failed to grade KB evidence: %s", e)
        settings = get_settings()
        fail_mode = settings.fail_mode_grader
        get_telemetry_collector().record_fail_decision(
            node_name="grader",
            error_type=type(e).__name__,
            mode=fail_mode,
            details=str(e),
        )
        grade_val = "weak" if fail_mode == "closed" else "good"
        return {"kb_grade": grade_val}


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


@traced_node("search_web")
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
                    url = item.get("url", "")
                    lines.append(
                        f"Title: {item.get('title', '')}\nURL: {url}\nContent: {item.get('content', '')}"
                    )
                web_text = "\n\n".join(lines) if lines else ""
            else:
                web_text = str(result)
            return {"web_results": web_text, "web_search_failed": False}
    except TimeoutError:
        logger.warning(
            "[Tavily Search Timeout]: Web search query timed out after %.1fs (query_len=%d)",
            timeout,
            len(query),
        )
        return {"web_results": "", "web_search_failed": True}
    except TavilyClientError as e:
        record_provider_failure(e)
        logger.warning(
            "[Tavily Search Error]: Search failed (query_len=%d): %s",
            len(query),
            e,
        )
        return {"web_results": "", "web_search_failed": True}
    except Exception as e:
        record_provider_failure(e)
        logger.warning(
            "[Tavily Search Error]: Unexpected failure (query_len=%d): %s",
            len(query),
            e,
        )
        return {"web_results": "", "web_search_failed": True}


@traced_node("grade_web_evidence")
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
        logger.warning("[Web Grader Error] Failed to grade web evidence: %s", e)
        settings = get_settings()
        fail_mode = settings.fail_mode_grader
        get_telemetry_collector().record_fail_decision(
            node_name="grader",
            error_type=type(e).__name__,
            mode=fail_mode,
            details=str(e),
        )
        grade_val = "weak" if fail_mode == "closed" else "good"
        return {"web_grade": grade_val}


def _message_text(response: Any) -> str:
    """Extracts plain text from an LLM response regardless of its type."""
    return response.content if hasattr(response, "content") else str(response)


@traced_node("condense_query")
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

    history = get_truncated_conversation_history(state)
    try:
        prompt = CONDENSE_QUERY_PROMPT_TEMPLATE.format(history=history, question=question)
        response = await get_llm().ainvoke(prompt)
        condensed = _message_text(response).strip()
        logger.info(
            "[Condense Query] Original len=%d -> Condensed len=%d",
            len(question),
            len(condensed),
        )
        logger.debug("[Condense Query] Condensed: %s", redact_pii(condensed))
        return {"current_query": condensed}
    except Exception as e:
        record_provider_failure(e)
        logger.warning("[Condense Query Error] Failed to condense query. Keeping original: %s", e)
        return {"current_query": question}


@traced_node("rewrite_query")
async def rewrite_query(state: AgentState) -> dict[str, Any]:
    """Rewrites current search query to optimize retrieval precision for subsequent searches."""
    current_query = state.get("current_query") or get_latest_question(state)
    history = get_conversation_history(state)
    retry_count = state.get("retry_count", 0) + 1
    try:
        prompt = REWRITE_QUERY_PROMPT_TEMPLATE.format(history=history, current_query=current_query)
        response = await get_llm().ainvoke(prompt)
        rewritten = _message_text(response).strip()
        logger.info("[Rewriter] Rewritten query (retry: %d, len=%d)", retry_count, len(rewritten))
        logger.debug("[Rewriter] Rewritten: %s", redact_pii(rewritten))
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
        # Apply canary & leak filtering and PII redaction (M3)
        content = filter_canary_and_leaks(str(content))
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


async def check_claim_support(question: str, context: str, response: str) -> ClaimSupportCheck:
    """Evaluates whether all factual statements in the response are grounded in the reference context."""
    if not context or not response:
        return ClaimSupportCheck(supported=True, reason="Empty context or response.")

    settings = get_settings()
    is_policy_or_billing = is_proprietary_policy_query(question) or any(
        k in question.lower()
        for k in ("bill", "invoice", "refund", "charge", "payment", "policy", "warranty")
    )
    if settings.fail_mode_faithfulness == "adaptive":
        effective_mode = "closed" if is_policy_or_billing else "open"
    else:
        effective_mode = settings.fail_mode_faithfulness

    try:
        prompt = CLAIM_SUPPORT_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            response=response,
        )
        decision: ClaimSupportCheck = await get_faithfulness_llm().ainvoke(prompt)
        return decision
    except Exception as e:
        record_provider_failure(e)
        get_telemetry_collector().record_fail_decision(
            node_name="faithfulness",
            error_type=type(e).__name__,
            mode=effective_mode,
            details=str(e),
        )
        if effective_mode == "closed":
            logger.warning("[Faithfulness Fail-Closed] Evaluator error for policy/billing: %s", e)
            return ClaimSupportCheck(supported=False, reason=f"Evaluator error (fail-closed): {e}")
        logger.debug("[Faithfulness Fail-Open] Evaluator error: %s", e)
        return ClaimSupportCheck(supported=True, reason=f"Evaluator error (fail-open): {e}")


@traced_node("generate_from_kb")
async def generate_from_kb(state: AgentState) -> dict[str, Any]:
    """Synthesizes customer response grounded in retrieved internal knowledge base documents with citations."""
    question = get_latest_question(state)
    history = get_conversation_history(state)
    raw_context = "\n\n".join(
        f"[Source {idx + 1}: {doc.metadata.get('intent', doc.metadata.get('source', 'unknown'))} (id: {doc.metadata.get('id', f'doc-{idx + 1}')})]\n{doc.page_content}"
        for idx, doc in enumerate(state.get("kb_docs", []))
    )
    tool_results = state.get("tool_results")
    if tool_results:
        tool_ctx = "\n".join(str(r.get("result", r)) for r in tool_results)
        raw_context = f"{raw_context}\n\n[Operational Tool Records]\n{tool_ctx}".strip()

    context = sanitize_untrusted_context(raw_context)
    system_text = CUSTOMER_SUPPORT_SYSTEM_PROMPT + UNTRUSTED_CONTEXT_INSTRUCTION
    human_text = KB_GENERATION_HUMAN_TEMPLATE.format(
        context=context,
        history=history,
        question=question,
    )
    messages = [SystemMessage(content=system_text), HumanMessage(content=human_text)]
    result = await _generate_support_message(messages, DEFAULT_UNAVAILABLE_APOLOGY, "private_kb")

    # Post-generation faithfulness guardrail (M2 enforcement)
    ai_msgs = result.get("messages", [])
    if ai_msgs and context.strip():
        ans_text = _message_text(ai_msgs[-1])
        try:
            check = await check_claim_support(question=question, context=context, response=ans_text)
            if not check.supported:
                unsupported_str = (
                    "\n".join(f"- {c}" for c in check.unsupported_claims) or check.reason
                )
                logger.warning(
                    "[Faithfulness Guardrail] Unsupported claims detected on first attempt: %s. Initiating 1-turn grounded regeneration.",
                    unsupported_str,
                )
                regen_messages = [
                    SystemMessage(content=STRICT_REGEN_SYSTEM_PROMPT),
                    HumanMessage(
                        content=STRICT_REGEN_HUMAN_TEMPLATE.format(
                            unsupported_claims=unsupported_str,
                            context=context,
                            question=question,
                        )
                    ),
                ]
                regen_result = await _generate_support_message(
                    regen_messages, DEFAULT_UNAVAILABLE_APOLOGY, "private_kb"
                )
                regen_msgs = regen_result.get("messages", [])
                regen_ans_text = _message_text(regen_msgs[-1]) if regen_msgs else ""

                # Re-verify claim support on the regenerated output
                retry_check = await check_claim_support(
                    question=question, context=context, response=regen_ans_text
                )
                if retry_check.supported:
                    logger.info(
                        "[Faithfulness Guardrail] Regeneration succeeded with verified claims."
                    )
                    return regen_result
                else:
                    logger.warning(
                        "[Faithfulness Guardrail] Regeneration still failed verification: %s. Escalating to human and returning answer_insufficient.",
                        retry_check.unsupported_claims,
                    )
                    insufficient_res = await answer_insufficient(state)
                    insufficient_res["escalated"] = True
                    insufficient_res["ticket_payload"] = {
                        "reason": "faithfulness_violation",
                        "question": question,
                        "unsupported_claims": retry_check.unsupported_claims,
                        "details": retry_check.reason,
                    }
                    return insufficient_res
        except Exception as e:
            logger.debug("[Faithfulness Post-Check Non-fatal]: %s", e)

    return result


@traced_node("generate_from_web")
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


@traced_node("direct_answer")
async def direct_answer(state: AgentState) -> dict[str, Any]:
    """Generates direct answer for greetings, conversational chit-chat, or meta questions."""
    question = get_latest_question(state)
    history = get_conversation_history(state)
    tool_results = state.get("tool_results")
    if tool_results:
        tool_ctx = "\n".join(str(r.get("result", r)) for r in tool_results)
        history = f"{history}\n\n[Operational Tool Records]\n{tool_ctx}".strip()

    prompt = DIRECT_ANSWER_PROMPT_TEMPLATE.format(
        system_prompt=CUSTOMER_SUPPORT_SYSTEM_PROMPT, history=history, question=question
    )
    return await _generate_support_message(
        prompt, "Hello! How can I assist you with your customer support inquiry?", "direct"
    )


@traced_node("answer_insufficient")
async def answer_insufficient(state: AgentState) -> dict[str, Any]:
    """Generates a polite notice when knowledge base and web searches cannot answer the customer inquiry."""
    existing_messages = state.get("messages", [])
    if existing_messages and state.get("scope") == "error_insufficient":
        last_msg = existing_messages[-1]
        if isinstance(last_msg, AIMessage) or (hasattr(last_msg, "type") and last_msg.type == "ai"):
            return {"messages": [last_msg], "source_used": "insufficient_evidence"}

    question = get_latest_question(state)
    history = get_conversation_history(state)
    prompt = INSUFFICIENT_EVIDENCE_PROMPT_TEMPLATE.format(
        system_prompt=CUSTOMER_SUPPORT_SYSTEM_PROMPT, history=history, question=question
    )
    fallback = "I apologize, but our records do not have sufficient information to answer your inquiry. Please contact our support team via Live Chat or submit a support ticket for further assistance."
    return await _generate_support_message(prompt, fallback, "insufficient_evidence")
