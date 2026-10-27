"""Idempotent ingestion pipeline for Bitext customer support dataset into Pinecone.

Includes content hash-diffing, effective date and KB version metadata enrichment,
and semantic cache synchronization.
"""

import hashlib
import json
import logging
import os
from datetime import UTC, datetime

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from agenticrag.cache import get_response_cache
from agenticrag.clients import get_embeddings
from agenticrag.config import get_settings
from agenticrag.constants import DATASET_NAME, EMBEDDING_DIM

logger = logging.getLogger(__name__)


def compute_content_hash(text: str) -> str:
    """Computes SHA-256 digest of document content for change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def enrich_chunks_with_lifecycle_metadata(
    chunks: list[Document],
    kb_version: str | None = None,
    effective_date: str | None = None,
) -> tuple[list[Document], list[str]]:
    """Enriches chunk metadata with version, effective_date, and content_hash.

    Generates deterministic, content-addressable chunk IDs.
    """
    if kb_version is None:
        kb_version = getattr(get_settings(), "kb_version", "v2.1.0")
    if effective_date is None:
        effective_date = datetime.now(UTC).strftime("%Y-%m-%d")

    enriched_chunks = []
    chunk_ids = []

    for idx, chunk in enumerate(chunks):
        content_hash = compute_content_hash(chunk.page_content)
        intent = chunk.metadata.get("intent", "general")
        unique_string = f"{intent}_{content_hash}"
        chunk_id = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()[:32]

        meta = dict(chunk.metadata)
        meta.update(
            {
                "version": kb_version,
                "effective_date": effective_date,
                "content_hash": content_hash,
                "chunk_index": idx,
                "id": chunk_id,
            }
        )
        enriched_chunks.append(Document(page_content=chunk.page_content, metadata=meta))
        chunk_ids.append(chunk_id)

    return enriched_chunks, chunk_ids


def diff_chunks_against_manifest(
    chunks: list[Document],
    chunk_ids: list[str],
    manifest_path: str,
) -> tuple[list[Document], list[str], dict[str, str]]:
    """Compares current chunks against a local manifest using content hashes.

    Returns only new or modified chunks that require re-embedding and upserting.
    """
    previous_manifest: dict[str, str] = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                previous_manifest = json.load(f)
        except Exception as e:
            logger.warning("[Ingest Manifest] Could not read previous manifest: %s", e)

    chunks_to_upsert: list[Document] = []
    ids_to_upsert: list[str] = []
    updated_manifest = dict(previous_manifest)

    for chunk, cid in zip(chunks, chunk_ids, strict=False):
        current_hash = chunk.metadata.get("content_hash", "")
        if cid not in previous_manifest or previous_manifest.get(cid) != current_hash:
            chunks_to_upsert.append(chunk)
            ids_to_upsert.append(cid)
            updated_manifest[cid] = current_hash

    return chunks_to_upsert, ids_to_upsert, updated_manifest


def run_ingestion() -> None:
    """Runs idempotent dataset ingestion into Pinecone vector store with hash-diffing."""
    settings = get_settings()

    if not settings.pinecone_api_key:
        error_msg = (
            "PINECONE_API_KEY environment variable is not set. "
            "Please set PINECONE_API_KEY in your .env file or environment before running ingestion."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        from datasets import load_dataset
    except ImportError as e:
        error_msg = (
            "The 'datasets' library is required to run the ingestion pipeline. "
            "Please install it with: uv pip install 'agenticrag[ingest]'"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    logger.info("Fetching dataset '%s' from Hugging Face...", DATASET_NAME)
    dataset = load_dataset(DATASET_NAME, split="train")

    sample_size = min(len(dataset), settings.kb_sample_size)
    sample_data = dataset.shuffle(seed=settings.kb_seed).select(range(sample_size))

    raw_docs = []
    for row in sample_data:
        content = f"User Request: {row['instruction']}\nRecommended Response: {row['response']}"
        meta = {
            "source": "bitext_customer_support",
            "category": row.get("category", "support"),
            "intent": row.get("intent", "general"),
        }
        doc = Document(page_content=content, metadata=meta)
        raw_docs.append(doc)

    logger.info("Loaded %d documents. Splitting into chunks...", len(raw_docs))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    base_chunks = splitter.split_documents(raw_docs)

    chunks, chunk_ids = enrich_chunks_with_lifecycle_metadata(
        base_chunks,
        kb_version=settings.kb_version,
    )

    manifest_dir = settings.fastembed_cache_dir or "scratch"
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, f"ingest_manifest_{settings.namespace}.json")

    chunks_to_upsert, ids_to_upsert, updated_manifest = diff_chunks_against_manifest(
        chunks, chunk_ids, manifest_path
    )

    index_name = settings.index_name
    namespace = settings.namespace

    logger.info(
        "Hash-diff check: %d/%d chunks need upsert (%d unchanged).",
        len(chunks_to_upsert),
        len(chunks),
        len(chunks) - len(chunks_to_upsert),
    )

    if not chunks_to_upsert:
        logger.info("All chunks are identical to previous ingestion. Skipping re-embedding.")
        return

    logger.info("Connecting to Pinecone index: %s...", index_name)
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        logger.info("Creating Pinecone index '%s' (dim=%d, cosine)...", index_name, EMBEDDING_DIM)
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    embeddings = get_embeddings()

    logger.info(
        "Upserting %d changed chunks with metadata (version=%s) to namespace '%s'...",
        len(chunks_to_upsert),
        settings.kb_version,
        namespace,
    )
    PineconeVectorStore.from_documents(
        documents=chunks_to_upsert,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace,
        ids=ids_to_upsert,
    )

    # Save manifest after successful upsert
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(updated_manifest, f, indent=2)
    except Exception as e:
        logger.warning("[Ingest Manifest] Could not write updated manifest: %s", e)

    # Invalidate cached responses for new KB version
    get_response_cache().invalidate_version(settings.kb_version)

    logger.info("Ingestion complete! Knowledge base is idempotent and ready.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ingestion()
