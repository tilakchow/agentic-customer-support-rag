"""Lazy factory functions for external AI, vector store, and search service clients."""

import logging
import time
from functools import lru_cache

from agenticrag.config import get_settings
from agenticrag.constants import RETRIEVER_K
from agenticrag.schemas import EvidenceGrade, RouteDecision

logger = logging.getLogger(__name__)

_vectorstore_client = None
_retriever_client = None
_last_retriever_failure_time: float = 0.0
RETRIEVER_BACKOFF_SECONDS: float = 30.0


@lru_cache(maxsize=1)
def get_llm():
    """Lazily initializes ChatGroq LLM client with timeout and max_retries configured."""
    from langchain_groq import ChatGroq

    settings = get_settings()
    return ChatGroq(
        model=settings.groq_model,
        temperature=0,
        api_key=settings.groq_api_key or None,
        request_timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )


def get_router_llm():
    """Provides ChatGroq configured with RouteDecision structured output."""
    return get_llm().with_structured_output(RouteDecision, method="json_mode")


def get_grader_llm():
    """Provides ChatGroq configured with EvidenceGrade structured output for evidence grading."""
    return get_llm().with_structured_output(EvidenceGrade, method="json_mode")


@lru_cache(maxsize=1)
def get_web_search():
    """Lazily initializes Tavily search client."""
    from langchain_tavily import TavilySearch

    settings = get_settings()
    return TavilySearch(
        max_results=5,
        topic="general",
        include_answer=True,
        include_raw_content=False,
        tavily_api_key=settings.tavily_api_key or None,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """Lazily initializes FastEmbed embeddings client."""
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

    settings = get_settings()
    kwargs = {"model_name": settings.embedding_model}
    if settings.fastembed_cache_dir:
        kwargs["cache_dir"] = settings.fastembed_cache_dir
    return FastEmbedEmbeddings(**kwargs)


def get_vectorstore():
    """Lazily connects to Pinecone vector store with 30s failure backoff."""
    global _vectorstore_client, _last_retriever_failure_time
    if _vectorstore_client is not None:
        return _vectorstore_client
    settings = get_settings()
    if not settings.pinecone_api_key:
        logger.warning("PINECONE_API_KEY is not set. Pinecone vectorstore is disabled.")
        return None
    if time.time() - _last_retriever_failure_time < RETRIEVER_BACKOFF_SECONDS:
        return None
    try:
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone

        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.index_name)
        _vectorstore_client = PineconeVectorStore(
            index=index, embedding=get_embeddings(), namespace=settings.namespace
        )
        _last_retriever_failure_time = 0.0
        return _vectorstore_client
    except Exception as e:
        logger.warning("Pinecone index connection not ready or accessible: %s", e)
        _last_retriever_failure_time = time.time()
        _vectorstore_client = None
        return None


def get_retriever():
    """Lazily connects to Pinecone vector store retriever."""
    global _retriever_client
    if _retriever_client is not None:
        return _retriever_client
    vs = get_vectorstore()
    if vs is None:
        return None
    settings = get_settings()
    _retriever_client = vs.as_retriever(
        search_kwargs={"k": RETRIEVER_K, "namespace": settings.namespace}
    )
    return _retriever_client


def reset_clients() -> None:
    """Resets client singletons and clears factory caches for clean testing."""
    global _vectorstore_client, _retriever_client, _last_retriever_failure_time
    _vectorstore_client = None
    _retriever_client = None
    _last_retriever_failure_time = 0.0
    get_llm.cache_clear()
    get_web_search.cache_clear()
    get_embeddings.cache_clear()


def warm_up_clients() -> None:
    """Pre-initializes LLM, embeddings, and vector store to prevent cold-start event loop blocking."""
    settings = get_settings()
    if settings.groq_api_key:
        try:
            get_llm()
            logger.info("LLM client warmed up successfully.")
        except Exception as e:
            logger.debug("LLM warmup skipped or non-fatal error: %s", e)
    try:
        get_embeddings()
        logger.info("Embeddings model warmed up successfully.")
    except Exception as e:
        logger.debug("Embeddings warmup skipped or non-fatal error: %s", e)
    if settings.pinecone_api_key:
        try:
            get_vectorstore()
            get_retriever()
            logger.info("Pinecone vector store warmed up successfully.")
        except Exception as e:
            logger.debug("Vectorstore warmup skipped or non-fatal error: %s", e)
