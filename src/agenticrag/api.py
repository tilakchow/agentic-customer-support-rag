"""FastAPI production-grade API for Agentic Customer Support RAG."""

import os
import re
import uuid
import json
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator, Dict, Tuple
from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import HumanMessage
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from agenticrag.config import get_settings, get_retriever
from agenticrag.graph import get_agent_app
from agenticrag.nodes import get_node_status_message, is_generation_node

logger = logging.getLogger(__name__)

# --- Rate Limiter Setup ---
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# --- Optional API Key Authentication ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """Validates the API key if configured in settings; allows pass-through in development."""
    configured_key = get_settings().api_key
    if configured_key:
        if not api_key or api_key != configured_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key in 'X-API-Key' header."
            )
    return api_key


# --- In-Memory Short-TTL Answer Cache for Repeats ---
# Cache key: (clean_question, thread_id) -> (answer, source_used, expiry_timestamp)
_RESPONSE_CACHE: Dict[Tuple[str, str], Tuple[str, str, float]] = {}


def get_cached_response(query: str, thread_id: str) -> Optional[Tuple[str, str]]:
    key = (query.strip().lower(), thread_id)
    if key in _RESPONSE_CACHE:
        answer, source, expiry = _RESPONSE_CACHE[key]
        if time.time() < expiry:
            logger.info("[Cache Hit] Returning cached answer for query: '%s'", query[:40])
            return answer, source
        else:
            del _RESPONSE_CACHE[key]
    return None


def set_cached_response(query: str, thread_id: str, answer: str, source: str) -> None:
    ttl = get_settings().cache_ttl_seconds
    if ttl > 0:
        key = (query.strip().lower(), thread_id)
        _RESPONSE_CACHE[key] = (answer, source, time.time() + ttl)


# --- Application Lifespan Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes, validates dependencies, and pre-compiles graph at startup."""
    logger.info("Initializing Agentic-RAG backend lifespan...")
    settings = get_settings()
    settings.validate_runtime_keys(strict=False)

    # Pre-compile the agent graph on startup
    app.state.agent_app = get_agent_app()
    logger.info("Agentic-RAG backend startup complete.")
    yield
    _RESPONSE_CACHE.clear()
    logger.info("Agentic-RAG backend shutdown complete.")


# Initialize FastAPI application with lifespan handler
app = FastAPI(
    title="Agentic-RAG API",
    description="Hardened, production-ready FastAPI backend with rate limiting, async concurrency, and streaming",
    version="1.0.0",
    lifespan=lifespan
)

# Attach rate limiter to application state and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS with strict explicit origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def get_current_agent():
    """Retrieve compiled agent graph from application state or lazy singleton."""
    return getattr(app.state, "agent_app", None) or get_agent_app()


def clean_thread_id(raw_thread_id: Optional[str]) -> str:
    """Sanitizes thread_id and resolves whitespace bugs to avoid session collision."""
    cleaned = (raw_thread_id or "").strip()
    if not cleaned:
        return str(uuid.uuid4())
    # Restrict to safe slug characters (alphanumeric, hyphens, underscores)
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "", cleaned)[:128]
    return safe or str(uuid.uuid4())


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1500,
        description="User question or statement (max 1500 characters)"
    )
    thread_id: Optional[str] = Field(
        None,
        description="Session/Thread identifier for conversation memory"
    )
    stream: Optional[bool] = Field(
        False,
        description="Whether to stream the response as Server-Sent Events (SSE)"
    )

    @field_validator("message")
    @classmethod
    def validate_message_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message cannot be empty or pure whitespace.")
        return stripped


class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant's response text")
    thread_id: str = Field(..., description="Active thread ID for conversational continuity")
    source_used: Optional[str] = Field(None, description="Data source used (private_kb, web_search, direct, etc.)")


@app.get("/health")
def health_check():
    """Health status probe for monitoring and load balancers."""
    return {"status": "ok", "service": "Agentic-RAG Backend"}


@app.get("/kb/info")
def kb_info_endpoint():
    """Returns knowledge base vector store metadata and index status."""
    settings = get_settings()
    retriever_instance = get_retriever()
    return {
        "index_name": settings.index_name,
        "namespace": settings.namespace,
        "embedding_model": settings.embedding_model,
        "dimension": 384,
        "dataset": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "sample_size": 500,
        "primary_intent": "cancel_order & support policies",
        "retriever_k": 4,
        "status": "connected" if retriever_instance is not None else "disconnected"
    }


class KBSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)


@app.post("/kb/search")
@limiter.limit("30/minute")
def kb_search_endpoint(
    request: Request,
    body: KBSearchRequest,
    _auth: Optional[str] = Depends(verify_api_key)
):
    """Direct vector search endpoint with rate limiting and authentication."""
    retriever_instance = get_retriever()
    if not retriever_instance:
        raise HTTPException(status_code=503, detail="Knowledge base retriever is temporarily unavailable.")
    try:
        docs = retriever_instance.invoke(body.query)
        return {
            "query": body.query,
            "results": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        }
    except Exception as e:
        logger.error("[KBSearch Error]: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to execute search in knowledge base.")


async def stream_agent_generator(message: str, thread_id: str) -> AsyncGenerator[str, None]:
    """Generates Server-Sent Events asynchronously without blocking the Uvicorn event loop."""
    # Check short-TTL cache for duplicate query
    cached = get_cached_response(message, thread_id)
    if cached:
        cached_answer, cached_source = cached
        yield f"data: {json.dumps({'type': 'status', 'node': 'cache', 'message': 'Retrieved cached response...'})}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'token': cached_answer})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id, 'source_used': cached_source, 'full_content': cached_answer})}\n\n"
        return

    agent_app = get_current_agent()
    config = {"configurable": {"thread_id": thread_id}}
    input_data = {"messages": [HumanMessage(content=message)]}
    active_node = ""

    try:
        async for event in agent_app.astream_events(input_data, config=config, version="v2"):
            event_type = event.get("event")

            # Check if a new node started executing
            if event_type == "on_chain_start":
                node_name = event.get("metadata", {}).get("langgraph_node")
                if node_name and node_name != active_node:
                    active_node = node_name
                    status_text = get_node_status_message(node_name)
                    payload = json.dumps({"type": "status", "node": node_name, "message": status_text})
                    yield f"data: {payload}\n\n"

            # Stream LLM tokens when generating final answer
            elif event_type == "on_chat_model_stream":
                node_name = event.get("metadata", {}).get("langgraph_node")
                if node_name and is_generation_node(node_name):
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        payload = json.dumps({"type": "token", "token": chunk.content})
                        yield f"data: {payload}\n\n"

        # Graph execution completed; extract final state asynchronously
        state_snapshot = await agent_app.aget_state(config)
        final_state = state_snapshot.values
        source = final_state.get("source_used", "direct")

        messages = final_state.get("messages", [])
        last_content = messages[-1].content if messages else ""

        # Cache valid response
        if last_content:
            set_cached_response(message, thread_id, str(last_content), source)

        payload = json.dumps({
            "type": "done",
            "thread_id": thread_id,
            "source_used": source,
            "full_content": str(last_content)
        })
        yield f"data: {payload}\n\n"

    except Exception as e:
        logger.error("[Streaming Error]: %s", e, exc_info=True)
        # Never leak raw internal stack traces/provider errors to SSE clients
        error_payload = json.dumps({
            "type": "error",
            "error": "An unexpected error occurred while generating your response. Please try again."
        })
        yield f"data: {error_payload}\n\n"


@app.post("/chat")
@limiter.limit("20/minute")
async def chat_endpoint(
    request: Request,
    body: ChatRequest,
    _auth: Optional[str] = Depends(verify_api_key)
):
    """Chat endpoint supporting both JSON response and SSE streaming."""
    thread_id = clean_thread_id(body.thread_id)

    # If client requested streaming
    if body.stream:
        return StreamingResponse(
            stream_agent_generator(body.message, thread_id),
            media_type="text/event-stream"
        )

    # Check cache for repeats
    cached = get_cached_response(body.message, thread_id)
    if cached:
        cached_answer, cached_source = cached
        return ChatResponse(
            response=cached_answer,
            thread_id=thread_id,
            source_used=cached_source
        )

    # Non-blocking async invocation (ainvoke)
    agent_app = get_current_agent()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = await agent_app.ainvoke(
            {"messages": [HumanMessage(content=body.message)]},
            config=config
        )

        messages = result.get("messages", [])
        if not messages:
            raise HTTPException(status_code=500, detail="No response could be generated.")

        last_message = messages[-1]
        answer_content = last_message.content if hasattr(last_message, "content") else str(last_message)
        source = result.get("source_used", "unknown")

        # Cache response
        set_cached_response(body.message, thread_id, str(answer_content), source)

        return ChatResponse(
            response=str(answer_content),
            thread_id=thread_id,
            source_used=source
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[API Error] /chat failed: %s", e, exc_info=True)
        # Sanitize internal errors for client security
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again later."
        )


@app.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream_endpoint(
    request: Request,
    body: ChatRequest,
    _auth: Optional[str] = Depends(verify_api_key)
):
    """Direct Server-Sent Events streaming endpoint with rate limits."""
    thread_id = clean_thread_id(body.thread_id)
    return StreamingResponse(
        stream_agent_generator(body.message, thread_id),
        media_type="text/event-stream"
    )
