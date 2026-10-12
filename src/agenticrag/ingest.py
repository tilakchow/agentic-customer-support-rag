"""Idempotent ingestion pipeline for Bitext customer support dataset into Pinecone."""

import os
import hashlib
import logging
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from agenticrag.config import get_settings, get_embeddings, INDEX_NAME, NAMESPACE

load_dotenv()
logger = logging.getLogger(__name__)

DIMENSION = 384


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

    logger.info("Fetching dataset automatically from Hugging Face...")
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")

    # Sample data to avoid extreme upload times for initial indexing
    sample_data = dataset.select(range(500))

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

    logger.info("Connecting to Pinecone index: %s...", INDEX_NAME)
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        logger.info("Creating Pinecone index '%s' (dim=%d, cosine)...", INDEX_NAME, DIMENSION)
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    # Share the exact same embedding model configuration used at runtime
    embeddings = get_embeddings()

    logger.info("Upserting %d chunks with deterministic IDs to namespace '%s'...", len(chunks), NAMESPACE)
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
        ids=chunk_ids
    )

    logger.info("Ingestion complete! Knowledge base is idempotent and ready.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ingestion()
