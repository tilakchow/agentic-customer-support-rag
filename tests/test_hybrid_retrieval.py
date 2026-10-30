"""Unit tests for hybrid retrieval (BM25 + Dense Reciprocal Rank Fusion)."""

from langchain_core.documents import Document

from agenticrag.config import get_settings
from agenticrag.retrieval import reciprocal_rank_fusion


def test_hybrid_retrieval_setting_default_false():
    """Validates HYBRID_RETRIEVAL_ENABLED defaults to False."""
    settings = get_settings()
    assert settings.hybrid_retrieval_enabled is False
    assert settings.rrf_k == 60


def test_reciprocal_rank_fusion_logic():
    """Validates reciprocal rank fusion scoring and ranking."""
    doc1 = Document(
        page_content="Cancel order policy", metadata={"id": "doc-1", "intent": "cancel_order"}
    )
    doc2 = Document(
        page_content="Refund policy terms",
        metadata={"id": "doc-2", "intent": "check_refund_policy"},
    )
    doc3 = Document(
        page_content="Shipping delay details", metadata={"id": "doc-3", "intent": "delivery_period"}
    )

    dense_docs = [doc1, doc2]  # rank 1, rank 2
    bm25_docs = [doc2, doc3]  # rank 1, rank 2

    # doc2 appears in both (rank 2 in dense, rank 1 in bm25) -> highest score: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = ~0.0325
    # doc1 appears only in dense (rank 1) -> 1/(60+1) = 0.0164
    # doc3 appears only in bm25 (rank 2) -> 1/(60+2) = 0.0161
    fused = reciprocal_rank_fusion(dense_docs, bm25_docs, rrf_k=60, top_n=3)

    assert len(fused) == 3
    assert fused[0].metadata["id"] == "doc-2"
    assert fused[1].metadata["id"] == "doc-1"
    assert fused[2].metadata["id"] == "doc-3"
    assert fused[0].metadata["rrf_score"] > fused[1].metadata["rrf_score"]
