import logging
import time
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# --- Application Settings with Pydantic Settings ---
class Settings(BaseSettings):
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    pinecone_api_key: str = Field(default="", validation_alias="PINECONE_API_KEY")
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")
    api_key: str = Field(default="", validation_alias="API_KEY")  # Optional secret for client auth; unset means public
    require_api_key: bool = Field(default=False, validation_alias="REQUIRE_API_KEY")

    index_name: str = Field(default="industry-agentic-rag-kb", validation_alias="INDEX_NAME")
    namespace: str = Field(default="bitext-support", validation_alias="NAMESPACE")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL")
    groq_model: str = Field(default="openai/gpt-oss-20b", validation_alias="GROQ_MODEL")
    max_retries: int = Field(default=1, validation_alias="MAX_RETRIES")
    llm_timeout: float = Field(default=30.0, validation_alias="LLM_TIMEOUT")
    llm_max_retries: int = Field(default=2, validation_alias="LLM_MAX_RETRIES")
    tavily_timeout: float = Field(default=15.0, validation_alias="TAVILY_TIMEOUT")

    # Security & Policy Settings
    allowed_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://agentic-customer-support-rag.vercel.app",
    ]
    rate_limit: str = "20/minute"
    max_message_length: int = 1500
    max_thread_history: int = 10

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def validate_runtime_keys(self, strict: bool = False) -> None:
        """Validates that necessary API keys are present for execution."""
        missing = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")

        if missing:
            msg = f"Missing required environment variables: {', '.join(missing)}."
            if strict:
                raise RuntimeError(msg)
            logger.warning(
                "%s Some agent capabilities will be limited or require mocking.",
                msg
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Convenience constants backed by settings
INDEX_NAME = get_settings().index_name
NAMESPACE = get_settings().namespace
MAX_RETRIES = get_settings().max_retries


# --- Pydantic Models for Structured LLM Output ---
class RouteDecision(BaseModel):
    route: Literal["kb", "direct"] = Field(
        default="kb",
        description="Use kb for questions needing Agentic RAG docs; direct for greetings/simple chat."
    )

    @model_validator(mode="before")
    @classmethod
    def handle_route_aliases(cls, data):
        if isinstance(data, dict) and "route" not in data:
            for key in ["decision", "destination", "choice", "answer"]:
                if key in data and data[key] in ("kb", "direct"):
                    data["route"] = data[key]
                    break
        return data


class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(
        default="weak",
        description="good means evidence can answer the question; weak means not enough evidence."
    )

    @model_validator(mode="before")
    @classmethod
    def handle_grade_aliases(cls, data):
        if isinstance(data, dict) and "grade" not in data:
            for key in ["answer", "score", "decision", "result"]:
                if key in data and data[key] in ("good", "weak"):
                    data["grade"] = data[key]
                    break
        return data


# --- Lazy Client Factories ---

_llm_client = None
_web_search_client = None
_embeddings_client = None
_retriever_client = None


def get_llm():
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
    return get_llm().with_structured_output(RouteDecision, method="json_mode")


def get_kb_grader_llm():
    return get_llm().with_structured_output(EvidenceGrade, method="json_mode")


def get_web_grader_llm():
    return get_llm().with_structured_output(EvidenceGrade, method="json_mode")


def get_web_search():
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
    global _embeddings_client
    if _embeddings_client is None:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        settings = get_settings()
        _embeddings_client = FastEmbedEmbeddings(model_name=settings.embedding_model)
    return _embeddings_client


_last_retriever_failure_time: float = 0.0
RETRIEVER_BACKOFF_SECONDS: float = 30.0


def get_retriever():
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
            search_kwargs={"k": 4, "namespace": settings.namespace}
        )
        _last_retriever_failure_time = 0.0
    except Exception as e:
        logger.warning("Pinecone index connection not ready or accessible: %s", e)
        _last_retriever_failure_time = time.time()
        _retriever_client = None
    return _retriever_client


# Module-level property proxies for backwards compatibility
class _RetrieverProxy:
    def invoke(self, *args, **kwargs):
        r = get_retriever()
        if r is None:
            return []
        return r.invoke(*args, **kwargs)

    def __bool__(self):
        return get_retriever() is not None

retriever = _RetrieverProxy()
