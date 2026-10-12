"""Configuration and environment settings management for Agentic Customer Support RAG."""

import logging
from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from agenticrag.constants import KB_SAMPLE_SIZE

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""

    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    pinecone_api_key: str = Field(default="", validation_alias="PINECONE_API_KEY")
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")
    api_key: str = Field(
        default="", validation_alias="API_KEY"
    )  # Optional secret for client auth; unset means public
    require_api_key: bool = Field(default=False, validation_alias="REQUIRE_API_KEY")
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")

    index_name: str = Field(default="industry-agentic-rag-kb", validation_alias="INDEX_NAME")
    namespace: str = Field(default="bitext-support", validation_alias="NAMESPACE")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL"
    )
    groq_model: str = Field(default="openai/gpt-oss-20b", validation_alias="GROQ_MODEL")
    max_retries: int = Field(default=1, validation_alias="MAX_RETRIES")
    llm_timeout: float = Field(default=30.0, validation_alias="LLM_TIMEOUT")
    llm_max_retries: int = Field(default=2, validation_alias="LLM_MAX_RETRIES")
    tavily_timeout: float = Field(default=15.0, validation_alias="TAVILY_TIMEOUT")
    tavily_timeout_seconds: float = Field(default=10.0, validation_alias="TAVILY_TIMEOUT_SECONDS")

    # Security & Policy Settings
    allowed_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://agentic-customer-support-rag.vercel.app",
    ]
    rate_limit: str = "20/minute"
    max_message_length: int = 1500
    max_thread_history: int = 10
    web_fallback_enabled: bool = Field(default=True, validation_alias="WEB_FALLBACK_ENABLED")
    skip_grader_score: float | None = Field(default=None, validation_alias="SKIP_GRADER_SCORE")
    checkpoint_db: str | None = Field(default=None, validation_alias="CHECKPOINT_DB")
    allow_in_memory_checkpoints: bool | None = Field(
        default=None, validation_alias="ALLOW_IN_MEMORY_CHECKPOINTS"
    )
    fastembed_cache_dir: str | None = Field(default=None, validation_alias="FASTEMBED_CACHE_PATH")
    kb_sample_size: int = Field(default=KB_SAMPLE_SIZE, validation_alias="KB_SAMPLE_SIZE")
    kb_seed: int = Field(default=42, validation_alias="KB_SEED")
    tavily_max_concurrency: int = Field(default=5, validation_alias="TAVILY_MAX_CONCURRENCY")
    max_thread_messages: int = Field(default=40, validation_alias="MAX_THREAD_MESSAGES")

    trusted_proxies: Annotated[list[str], NoDecode] = Field(
        default_factory=list, validation_alias="TRUSTED_PROXIES"
    )

    @model_validator(mode="after")
    def _set_checkpoint_defaults(self) -> "Settings":
        if self.allow_in_memory_checkpoints is None:
            self.allow_in_memory_checkpoints = self.environment.lower() != "production"
        return self

    @field_validator("allowed_origins", "trusted_proxies", mode="before")
    @classmethod
    def parse_comma_separated_list(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

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
            logger.warning("%s Some agent capabilities will be limited or require mocking.", msg)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns singleton cached instance of Application Settings."""
    return Settings()
