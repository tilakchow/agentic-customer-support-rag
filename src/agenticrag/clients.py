"""Lazy factory functions for external AI, vector store, and search service clients."""

import logging
import time

from agenticrag.config import get_settings
from agenticrag.constants import RETRIEVER_K
from agenticrag.schemas import EvidenceGrade, RouteDecision

logger = logging.getLogger(__name__)

_llm_client = None
_web_search_client = None
_embeddings_client = None
_retriever_client = None
_last_retriever_failure_time: float = 0.0
RETRIEVER_BACKOFF_SECONDS: float = 30.0


def get_llm():
    """Lazily initializes ChatGroq LLM client with timeout and max_retries configured."""
    global _llm_client
    if _llm_client is None:
        from langchain_groq import ChatGroq
        settings = get_settings()
        _llm_client = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=settings.groq_api_key or None,
            request_timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
    return _llm_client


def get_router_llm():
    """Provides ChatGroq configured with RouteDecision structured output."""
    return get_llm().with_structured_output(RouteDecision, method="json_mode")


def get_grader_llm():
    """Provides ChatGroq configured with EvidenceGrade structured output for evidence grading."""
    return get_llm().with_structured_output(EvidenceGrade, method="json_mode")


# For backwards compatibility with any existing imports
get_kb_grader_llm = get_grader_llm
get_web_grader_llm = get_grader_llm


def get_web_search():
    """Lazily initializes Tavily search client."""
    global _web_search_client
    if _web_search_client is None:
        from langchain_tavily import TavilySearch
        settings = get_settings()
        _web_search_client = TavilySearch(
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
            tavily_api_key=settings.tavily_api_key or None
        )
    return _web_search_client


def get_embeddings():
    """Lazily initializes FastEmbed embeddings client."""
    global _embeddings_client
    if _embeddings_client is None:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        settings = get_settings()
        _embeddings_client = FastEmbedEmbeddings(model_name=settings.embedding_model)
    return _embeddings_client


def get_retriever():
    """Lazily connects to Pinecone vector store with connection failure backoff."""
    global _retriever_client, _last_retriever_failure_time
    if _retriever_client is not None:
        return _retriever_client
    settings = get_settings()
    if not settings.pinecone_api_key:
        logger.warning("PINECONE_API_KEY is not set. Pinecone retriever is disabled.")
        return None
    # Short backoff so Pinecone outage doesn't trigger reconnect on every request
    if time.time() - _last_retriever_failure_time < RETRIEVER_BACKOFF_SECONDS:
        return None
    try:
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone

        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.index_name)
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.namespace
        )
        _retriever_client = vectorstore.as_retriever(
            search_kwargs={"k": RETRIEVER_K, "namespace": settings.namespace}
        )
        _last_retriever_failure_time = 0.0
    except Exception as e:
        logger.warning("Pinecone index connection not ready or accessible: %s", e)
        _last_retriever_failure_time = time.time()
        _retriever_client = None
    return _retriever_client
