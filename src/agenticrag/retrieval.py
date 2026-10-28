"""Unified KB retrieval and neural reranking logic for graph nodes and evaluation scripts."""

import logging
from typing import Any, NamedTuple

from langchain_core.documents import Document

from agenticrag.clients import get_pinecone_client, get_vectorstore
from agenticrag.config import Settings, get_settings
from agenticrag.constants import RETRIEVER_K

logger = logging.getLogger(__name__)


class RetrievalResult(NamedTuple):
    """Result of retrieving documents from the knowledge base."""

    docs: list[Document]
    top_score: float


def retrieve_kb(
    query: str,
    *,
    k: int = RETRIEVER_K,
    namespace: str | None = None,
    vectorstore: Any = None,
    settings: Settings | None = None,
    pinecone_client: Any = None,
    reranker_enabled: bool | None = None,
) -> RetrievalResult:
    """Retrieves relevant knowledge base documents with vector search and optional neural reranking.

    Args:
        query: User search query.
        k: Maximum number of final documents to return.
        namespace: Target namespace for tenant isolation (defaults to settings.namespace).
        vectorstore: Pinecone vectorstore instance (defaults to get_vectorstore()).
        settings: Application settings (defaults to get_settings()).
        pinecone_client: Pinecone client instance for reranking (defaults to get_pinecone_client()).
        reranker_enabled: Override for settings.reranker_enabled.

    Returns:
        RetrievalResult containing the top-k documents and top similarity score.
    """
    settings = settings or get_settings()
    vs = vectorstore if vectorstore is not None else get_vectorstore()
    if vs is None:
        raise RuntimeError("Pinecone vectorstore is unavailable.")

    use_rerank = settings.reranker_enabled if reranker_enabled is None else reranker_enabled
    fetch_k = settings.rerank_fetch_k if use_rerank else k
    target_namespace = namespace if namespace is not None else settings.namespace

    scored_results = vs.similarity_search_with_score(
        query,
        k=fetch_k,
        namespace=target_namespace,
    )

    candidate_docs: list[Document] = []
    for doc, score in scored_results:
        doc.metadata["score"] = float(score)
        candidate_docs.append(doc)

    docs: list[Document] = []
    top_score = 0.0

    if use_rerank and len(candidate_docs) > k:
        try:
            pc = pinecone_client if pinecone_client is not None else get_pinecone_client()
            if pc and hasattr(pc, "inference"):
                doc_texts = [d.page_content for d in candidate_docs]
                rerank_res = None
                for attempt in range(4):
                    try:
                        rerank_res = pc.inference.rerank(
                            model=settings.rerank_model,
                            query=query,
                            documents=doc_texts,
                            top_n=k,
                        )
                        break
                    except Exception as err:
                        err_str = str(err)
                        if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < 3:
                            import time

                            time.sleep(1.5 * (attempt + 1))
                            continue
                        raise

                reranked_docs: list[Document] = []
                for item in getattr(rerank_res, "data", []):
                    idx = getattr(item, "index", 0)
                    r_score = float(getattr(item, "score", 0.0))
                    doc_match = candidate_docs[idx]
                    doc_match.metadata["rerank_score"] = r_score
                    reranked_docs.append(doc_match)
                if reranked_docs:
                    docs = reranked_docs
                    top_score = max(float(d.metadata.get("rerank_score", 0.0)) for d in docs)
            else:
                docs = candidate_docs[:k]
                if docs:
                    top_score = max(float(d.metadata.get("score", 0.0)) for d in docs)
        except Exception as rerank_err:
            logger.warning(
                "[KB Retriever Rerank Error]: %s. Falling back to vector similarity ranking.",
                rerank_err,
            )
            docs = candidate_docs[:k]
            if docs:
                top_score = max(float(d.metadata.get("score", 0.0)) for d in docs)
    else:
        docs = candidate_docs[:k]
        if docs:
            top_score = max(float(d.metadata.get("score", 0.0)) for d in docs)

    return RetrievalResult(docs=docs, top_score=top_score)
