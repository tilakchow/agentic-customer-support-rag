"""Unified KB retrieval and neural reranking logic for graph nodes and evaluation scripts."""

import logging
from typing import Any, Literal, NamedTuple

from langchain_core.documents import Document
from tenacity import retry, retry_if_exception, stop_after_delay, wait_random_exponential

from agenticrag.clients import get_pinecone_client, get_vectorstore
from agenticrag.config import Settings, get_settings
from agenticrag.constants import RETRIEVER_K
from agenticrag.telemetry import get_telemetry_collector

logger = logging.getLogger(__name__)


class RetrievalResult(NamedTuple):
    """Result of retrieving documents from the knowledge base."""

    docs: list[Document]
    top_score: float
    score_type: Literal["rerank", "cosine"] = "cosine"


def _is_retryable_rerank_error(exc: BaseException) -> bool:
    """Returns True if exception is a rate limit or transient service error."""
    err_str = str(exc)
    return (
        "429" in err_str
        or "RESOURCE_EXHAUSTED" in err_str
        or "Rate limit" in err_str
        or "timeout" in err_str.lower()
    )


@retry(
    reraise=True,
    stop=stop_after_delay(3.0),
    wait=wait_random_exponential(multiplier=0.1, max=1.0),
    retry=retry_if_exception(_is_retryable_rerank_error),
)
def _rerank_with_retry(
    pc: Any,
    model: str,
    query: str,
    documents: list[str],
    top_n: int,
) -> Any:
    return pc.inference.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=top_n,
    )


_bm25_index: Any = None
_bm25_corpus: list[Document] = []


def get_bm25_retriever(settings: Settings | None = None) -> tuple[Any, list[Document]]:
    """Lazily initializes and caches in-memory BM25 index over the Bitext KB documents."""
    global _bm25_index, _bm25_corpus
    if _bm25_index is not None and _bm25_corpus:
        return _bm25_index, _bm25_corpus

    settings = settings or get_settings()
    try:
        from datasets import load_dataset
        from rank_bm25 import BM25Okapi

        from agenticrag.constants import DATASET_NAME

        dataset = load_dataset(DATASET_NAME, split="train")
        sample_size = min(len(dataset), settings.kb_sample_size)
        sample_data = dataset.shuffle(seed=settings.kb_seed).select(range(sample_size))

        corpus: list[Document] = []
        for idx, row in enumerate(sample_data):
            content = f"User Request: {row['instruction']}\nRecommended Response: {row['response']}"
            meta = {
                "source": "bitext_customer_support",
                "category": row.get("category", "support"),
                "intent": row.get("intent", "general"),
                "id": f"chunk-{idx}",
            }
            corpus.append(Document(page_content=content, metadata=meta))

        tokenized_corpus = [doc.page_content.lower().split() for doc in corpus]
        _bm25_index = BM25Okapi(tokenized_corpus)
        _bm25_corpus = corpus
    except Exception as e:
        logger.warning("[BM25 Init Warning]: Failed to build BM25 index (%s). BM25 disabled.", e)
        _bm25_index = None
        _bm25_corpus = []

    return _bm25_index, _bm25_corpus


def reciprocal_rank_fusion(
    dense_docs: list[Document],
    bm25_docs: list[Document],
    rrf_k: int = 60,
    top_n: int = 20,
) -> list[Document]:
    """Fuses dense vector candidates and BM25 candidates using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    def doc_key(doc: Document, fallback_idx: int) -> str:
        return str(
            doc.metadata.get("id")
            or doc.metadata.get("doc_id")
            or f"doc_{hash(doc.page_content[:80])}_{fallback_idx}"
        )

    for rank, doc in enumerate(dense_docs, 1):
        key = doc_key(doc, rank)
        doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

    for rank, doc in enumerate(bm25_docs, 1):
        key = doc_key(doc, rank + 1000)
        if key not in doc_map:
            doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_n]
    fused: list[Document] = []
    for k_key in sorted_keys:
        d = doc_map[k_key]
        d.metadata["rrf_score"] = scores[k_key]
        fused.append(d)
    return fused


def retrieve_kb(
    query: str,
    *,
    k: int = RETRIEVER_K,
    namespace: str | None = None,
    vectorstore: Any = None,
    settings: Settings | None = None,
    pinecone_client: Any = None,
    reranker_enabled: bool | None = None,
    hybrid_enabled: bool | None = None,
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
        RetrievalResult containing the top-k documents, top similarity score, and score_type.
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

    use_hybrid = settings.hybrid_retrieval_enabled if hybrid_enabled is None else hybrid_enabled
    if use_hybrid:
        try:
            bm25, corpus = get_bm25_retriever(settings)
            if bm25 and corpus:
                tokenized_query = query.lower().split()
                bm25_scores = bm25.get_scores(tokenized_query)
                top_indices = sorted(
                    range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
                )[:fetch_k]
                bm25_candidates = [corpus[idx] for idx in top_indices]
                candidate_docs = reciprocal_rank_fusion(
                    dense_docs=candidate_docs,
                    bm25_docs=bm25_candidates,
                    rrf_k=settings.rrf_k,
                    top_n=fetch_k,
                )
        except Exception as bm25_err:
            logger.warning(
                "[Hybrid Retrieval Warning]: %s. Proceeding with dense candidates.", bm25_err
            )

    docs: list[Document] = []
    top_score = 0.0
    score_type: Literal["rerank", "cosine"] = "cosine"

    if use_rerank and len(candidate_docs) > k:
        try:
            pc = pinecone_client if pinecone_client is not None else get_pinecone_client()
            if pc and hasattr(pc, "inference"):
                doc_texts = [d.page_content for d in candidate_docs]
                rerank_res = _rerank_with_retry(
                    pc=pc,
                    model=settings.rerank_model,
                    query=query,
                    documents=doc_texts,
                    top_n=k,
                )

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
                    score_type = "rerank"
            else:
                docs = candidate_docs[:k]
                if docs:
                    top_score = max(float(d.metadata.get("score", 0.0)) for d in docs)
                score_type = "cosine"
        except Exception as rerank_err:
            logger.warning(
                "[KB Retriever Rerank Error]: %s. Falling back to vector similarity ranking.",
                rerank_err,
            )
            try:
                get_telemetry_collector().record_fail_decision(
                    node_name="retrieve_kb_reranker",
                    error_type=type(rerank_err).__name__,
                    mode="fallback_to_cosine",
                    details=str(rerank_err),
                )
            except Exception as tel_err:
                logger.debug("[Telemetry Non-fatal]: %s", tel_err)

            docs = candidate_docs[:k]
            if docs:
                top_score = max(float(d.metadata.get("score", 0.0)) for d in docs)
            score_type = "cosine"
    else:
        docs = candidate_docs[:k]
        if docs:
            top_score = max(float(d.metadata.get("score", 0.0)) for d in docs)
        score_type = "cosine"

    return RetrievalResult(docs=docs, top_score=top_score, score_type=score_type)
