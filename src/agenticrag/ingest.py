"""Idempotent ingestion pipeline for Bitext customer support dataset into Pinecone."""

import hashlib
import logging

from datasets import load_dataset
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from agenticrag.clients import get_embeddings
from agenticrag.config import get_settings
from agenticrag.constants import DATASET_NAME, EMBEDDING_DIM

logger = logging.getLogger(__name__)


def run_ingestion() -> None:
    """Runs idempotent dataset ingestion into Pinecone vector store."""
    settings = get_settings()

    # Fail fast with clear message if Pinecone API key is missing
    if not settings.pinecone_api_key:
        error_msg = (
            "PINECONE_API_KEY environment variable is not set. "
            "Please set PINECONE_API_KEY in your .env file or environment before running ingestion."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info("Fetching dataset '%s' from Hugging Face...", DATASET_NAME)
    dataset = load_dataset(DATASET_NAME, split="train")

    # Sample data with configurable size and deterministic seed to avoid extreme indexing times
    sample_size = min(len(dataset), settings.kb_sample_size)
    sample_data = dataset.shuffle(seed=settings.kb_seed).select(range(sample_size))

    raw_docs = []
    for row in sample_data:
        content = f"User Request: {row['instruction']}\nRecommended Response: {row['response']}"
        meta = {
            "source": "bitext_customer_support",
            "category": row.get("category", "support"),
            "intent": row.get("intent", "general")
        }
        doc = Document(page_content=content, metadata=meta)
        raw_docs.append(doc)

    logger.info("Loaded %d documents. Splitting into chunks...", len(raw_docs))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(raw_docs)

    logger.info("Created %d chunks. Generating deterministic IDs...", len(chunks))

    # Generate deterministic chunk IDs (hash of content + intent) for idempotency
    chunk_ids = []
    for chunk in chunks:
        intent = chunk.metadata.get("intent", "")
        unique_string = f"{intent}_{chunk.page_content}"
        chunk_id = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()[:32]
        chunk_ids.append(chunk_id)

    index_name = settings.index_name
    namespace = settings.namespace

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

    # Share the exact same embedding model configuration used at runtime
    embeddings = get_embeddings()

    logger.info("Upserting %d chunks with deterministic IDs to namespace '%s'...", len(chunks), namespace)
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace,
        ids=chunk_ids
    )

    logger.info("Ingestion complete! Knowledge base is idempotent and ready.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ingestion()
