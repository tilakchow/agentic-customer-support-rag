import asyncio
import hmac
import logging
import re
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.cors import CORSMiddleware

from agenticrag.auth import verify_api_key
from agenticrag.clients import get_retriever, warm_up_clients
from agenticrag.config import get_settings
from agenticrag.constants import DATASET_NAME, EMBEDDING_DIM, KB_SAMPLE_SIZE, RETRIEVER_K
from agenticrag.graph import create_agent
from agenticrag.models import ChatRequest, ChatResponse, KBSearchRequest
from agenticrag.sse import stream_agent_generator

logger = logging.getLogger(__name__)


def get_client_rate_limit_key(request: Request) -> str:
    """Extracts rate limit identifier: verified API key if valid, otherwise Uvicorn-managed client IP."""
    api_key = request.headers.get("X-API-Key")
    configured_key = get_settings().api_key
    # Only assign a dedicated API key rate bucket if the key is genuine and matches configured secret
    if (
        api_key
        and configured_key
        and hmac.compare_digest(api_key.encode("utf-8"), configured_key.encode("utf-8"))
    ):
        return f"key:{api_key.strip()}"

    # Use Uvicorn-managed client IP (which securely processes proxy headers based on forwarded-allow-ips)
    return get_remote_address(request)


# Initialize rate limiter with proxy-aware key tracking
limiter = Limiter(key_func=get_client_rate_limit_key)


# --- Application Lifespan Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and cleans up state during application lifecycle."""
    settings = get_settings()
    if settings.environment.lower() == "production" and not settings.api_key:
        raise RuntimeError(
            "Production mode (ENVIRONMENT=production) requires API_KEY to be configured for security."
        )
    if settings.require_api_key and not settings.api_key:
        raise RuntimeError("REQUIRE_API_KEY is true, but API_KEY is empty or not configured.")
    if not settings.api_key:
        logger.warning(
            "[SECURITY NOTICE] Running without API_KEY. All /chat endpoints are publicly accessible."
        )
    settings.validate_runtime_keys(strict=False)

    if not settings.checkpoint_db and not settings.allow_in_memory_checkpoints:
        raise RuntimeError(
            "CHECKPOINT_DB must be set in production, or set ALLOW_IN_MEMORY_CHECKPOINTS=true for ephemeral testing."
        )

    # Warm up AI and embedding models in a worker thread to avoid cold starts
    await asyncio.to_thread(warm_up_clients)

    # Build the correct checkpointer and compile the agent graph ONCE at startup.
    # The AsyncSqliteSaver connection is managed here and closed cleanly on shutdown.
    if settings.checkpoint_db:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with aiosqlite.connect(settings.checkpoint_db) as conn:
            checkpointer = AsyncSqliteSaver(conn)
            app.state.agent_app = create_agent(checkpointer=checkpointer)

            logger.info(
                "Agentic-RAG backend startup complete (SQLite checkpointer: %s).",
                settings.checkpoint_db,
            )
            yield
            logger.info("Agentic-RAG backend shutdown complete (SQLite connection closed).")
    else:
        app.state.agent_app = create_agent()
        logger.info("Agentic-RAG backend startup complete (in-memory checkpointer).")
        yield
        logger.info("Agentic-RAG backend shutdown complete.")


# Initialize FastAPI application with lifespan handler
app = FastAPI(
    title="Agentic-RAG API",
    description="Hardened, production-ready FastAPI backend with rate limiting, async concurrency, and streaming",
    version="1.0.0",
    lifespan=lifespan,
)

# Attach rate limiter to application state and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


# Lazy CORS Middleware to avoid importing/reading settings on module load
class LazyCORSMiddleware(CORSMiddleware):
    """CORS middleware that dynamically resolves allowed origins from settings per request."""

    def __init__(self, inner_app, **kwargs):
        super().__init__(
            inner_app,
            allow_origins=[],
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            **kwargs,
        )
        self.preflight_explicit_allow_origin = True

    def is_allowed_origin(self, origin: str) -> bool:
        return origin in get_settings().allowed_origins


app.add_middleware(LazyCORSMiddleware)  # type: ignore[arg-type]


def get_current_agent():
    """Retrieve compiled agent graph from application state.

    Raises RuntimeError if called before the lifespan has initialised the app
    (e.g. in a test that bypasses the lifespan).  Tests that need the agent
    should supply their own via ``monkeypatch``.
    """
    agent = getattr(app.state, "agent_app", None)
    if agent is None:
        raise RuntimeError(
            "Agent application is not initialised. "
            "Ensure the FastAPI lifespan has started before calling this function."
        )
    return agent


def clean_thread_id(raw_thread_id: str | None) -> str:
    """Sanitizes thread_id and resolves whitespace bugs to avoid session collision."""
    cleaned = (raw_thread_id or "").strip()
    if not cleaned:
        return str(uuid.uuid4())
    # Restrict to safe slug characters (alphanumeric, hyphens, underscores)
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "", cleaned)[:128]
    return safe or str(uuid.uuid4())


@app.get("/")
def root():
    """Root endpoint for status check and service discovery."""
    return {
        "status": "online",
        "service": "Agentic-RAG Customer Support Backend",
        "docs": "/docs",
        "health": "/health",
        "kb_info": "/kb/info",
    }


@app.get("/health")
def health_check():
    """Health status probe for monitoring and load balancers."""
    return {"status": "ok", "service": "Agentic-RAG Backend"}


@app.get("/kb/info")
@limiter.limit("30/minute")
def kb_info_endpoint(request: Request, _auth: str | None = Depends(verify_api_key)):
    """Returns knowledge base vector store metadata and index status."""
    settings = get_settings()
    retriever_instance = get_retriever()
    return {
        "index_name": settings.index_name,
        "namespace": settings.namespace,
        "embedding_model": settings.embedding_model,
        "dimension": EMBEDDING_DIM,
        "sample_size": KB_SAMPLE_SIZE,
        "retriever_k": RETRIEVER_K,
        "dataset": DATASET_NAME,
        "retriever_connected": retriever_instance is not None,
    }


@app.post("/kb/search")
@limiter.limit("30/minute")
def kb_search_endpoint(
    request: Request, body: KBSearchRequest, _auth: str | None = Depends(verify_api_key)
):
    """Direct vector search endpoint with rate limiting and authentication."""
    retriever_instance = get_retriever()
    if not retriever_instance:
        raise HTTPException(
            status_code=503, detail="Knowledge base retriever is temporarily unavailable."
        )
    try:
        docs = retriever_instance.invoke(body.query)
        return {
            "query": body.query,
            "results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs],
        }
    except Exception as e:
        logger.error("[KBSearch Error]: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to execute search in knowledge base."
        ) from e


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(lambda: get_settings().rate_limit)
async def chat_endpoint(
    request: Request, body: ChatRequest, _auth: str | None = Depends(verify_api_key)
):
    """Chat endpoint returning direct JSON response."""
    thread_id = clean_thread_id(body.thread_id)

    agent_app = get_current_agent()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = await agent_app.ainvoke(
            {"messages": [HumanMessage(content=body.message)]}, config=config
        )

        messages = result.get("messages", [])
        if not messages:
            raise HTTPException(status_code=500, detail="No response could be generated.")

        last_message = messages[-1]
        answer_content = (
            last_message.content if hasattr(last_message, "content") else str(last_message)
        )
        source = result.get("source_used", "unknown")

        return ChatResponse(response=str(answer_content), thread_id=thread_id, source_used=source)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[API Error] /chat failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again later.",
        ) from e


@app.post("/chat/stream")
@limiter.limit(lambda: get_settings().rate_limit)
async def chat_stream_endpoint(
    request: Request, body: ChatRequest, _auth: str | None = Depends(verify_api_key)
):
    """Direct Server-Sent Events streaming endpoint with rate limits."""
    thread_id = clean_thread_id(body.thread_id)
    return StreamingResponse(
        stream_agent_generator(
            body.message, thread_id, agent_app=get_current_agent(), request=request
        ),
        media_type="text/event-stream",
    )
