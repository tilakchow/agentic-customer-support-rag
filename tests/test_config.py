"""Tests for Settings loading from real environment variables (monkeypatch.setenv)."""

import pytest

from agenticrag.config import Settings


def test_settings_defaults():
    """Settings loads sensible defaults without any env vars."""
    s = Settings()
    assert s.environment == "development"
    assert s.rate_limit == "20/minute"
    assert s.max_thread_history == 10
    assert s.max_message_length == 1500
    assert isinstance(s.allowed_origins, list)
    assert len(s.allowed_origins) > 0


def test_settings_allowed_origins_comma_separated(monkeypatch):
    """Comma-separated ALLOWED_ORIGINS env var parses to list without NoDecode errors."""
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://app.example.com, https://portal.example.com")
    s = Settings()
    assert s.allowed_origins == ["http://app.example.com", "https://portal.example.com"]


def test_settings_allowed_origins_single(monkeypatch):
    """Single ALLOWED_ORIGINS string parses to one-element list."""
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://only.example.com")
    s = Settings()
    assert s.allowed_origins == ["https://only.example.com"]


def test_settings_allowed_origins_empty_string(monkeypatch):
    """Empty ALLOWED_ORIGINS yields empty list."""
    monkeypatch.setenv("ALLOWED_ORIGINS", "")
    s = Settings()
    assert s.allowed_origins == []


def test_settings_environment_from_env(monkeypatch):
    """ENVIRONMENT env var is correctly picked up."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    s = Settings()
    assert s.environment == "production"


def test_settings_rate_limit_from_env(monkeypatch):
    """RATE_LIMIT env var overrides the default."""
    monkeypatch.setenv("RATE_LIMIT", "50/minute")
    s = Settings()
    assert s.rate_limit == "50/minute"


def test_settings_max_thread_history_from_env(monkeypatch):
    """MAX_THREAD_HISTORY env var parses as int."""
    monkeypatch.setenv("MAX_THREAD_HISTORY", "20")
    s = Settings()
    assert s.max_thread_history == 20


def test_settings_invalid_max_thread_history(monkeypatch):
    """Non-numeric MAX_THREAD_HISTORY raises ValidationError."""
    from pydantic import ValidationError

    monkeypatch.setenv("MAX_THREAD_HISTORY", "not-a-number")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_require_api_key_from_env(monkeypatch):
    """REQUIRE_API_KEY=true env var sets require_api_key to True."""
    monkeypatch.setenv("REQUIRE_API_KEY", "true")
    s = Settings()
    assert s.require_api_key is True


def test_settings_checkpoint_db_default_is_none():
    """CHECKPOINT_DB defaults to None when not set."""
    s = Settings()
    assert s.checkpoint_db is None


def test_settings_checkpoint_db_from_env(monkeypatch, tmp_path):
    """CHECKPOINT_DB env var captures file path."""
    db_path = str(tmp_path / "test.sqlite")
    monkeypatch.setenv("CHECKPOINT_DB", db_path)
    s = Settings()
    assert s.checkpoint_db == db_path


def test_settings_skip_grader_score_from_env(monkeypatch):
    """SKIP_GRADER_SCORE env var parses as float."""
    monkeypatch.setenv("SKIP_GRADER_SCORE", "0.85")
    s = Settings()
    assert s.skip_grader_score == pytest.approx(0.85)


def test_settings_api_key_from_env(monkeypatch):
    """API_KEY env var is captured."""
    monkeypatch.setenv("API_KEY", "super-secret-key")
    s = Settings()
    assert s.api_key == "super-secret-key"


def test_settings_validate_runtime_keys_warns_when_missing(caplog):
    """validate_runtime_keys logs a warning when API keys are missing in non-strict mode."""
    import logging

    s = Settings(groq_api_key="", tavily_api_key="", pinecone_api_key="")
    with caplog.at_level(logging.WARNING, logger="agenticrag.config"):
        s.validate_runtime_keys(strict=False)
    assert any(
        "Missing required environment variables" in record.message for record in caplog.records
    )


def test_settings_validate_runtime_keys_defaults_to_strict_in_production():
    """validate_runtime_keys defaults strict=True when environment is production."""
    s = Settings(environment="production", groq_api_key="", tavily_api_key="", pinecone_api_key="")
    with pytest.raises(RuntimeError, match="Missing required environment variables"):
        s.validate_runtime_keys()


def test_settings_validate_runtime_keys_strict_raises():
    """validate_runtime_keys raises RuntimeError in strict mode when keys missing."""
    s = Settings(groq_api_key="", tavily_api_key="", pinecone_api_key="")
    with pytest.raises(RuntimeError, match="Missing required environment variables"):
        s.validate_runtime_keys(strict=True)


def test_settings_allow_in_memory_checkpoints_defaults(monkeypatch):
    """allow_in_memory_checkpoints defaults to True in dev and False in prod."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    s_dev = Settings()
    assert s_dev.allow_in_memory_checkpoints is True

    monkeypatch.setenv("ENVIRONMENT", "production")
    s_prod = Settings()
    assert s_prod.allow_in_memory_checkpoints is False


def test_settings_allow_in_memory_checkpoints_override(monkeypatch):
    """Explicit ALLOW_IN_MEMORY_CHECKPOINTS overrides environment default."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ALLOW_IN_MEMORY_CHECKPOINTS", "true")
    s = Settings()
    assert s.allow_in_memory_checkpoints is True


def test_settings_max_thread_messages_from_env(monkeypatch):
    """MAX_THREAD_MESSAGES env var parses correctly."""
    monkeypatch.setenv("MAX_THREAD_MESSAGES", "50")
    s = Settings()
    assert s.max_thread_messages == 50


def test_settings_tavily_config_from_env(monkeypatch):
    """TAVILY_TIMEOUT and TAVILY_MAX_CONCURRENCY parse correctly."""
    monkeypatch.setenv("TAVILY_TIMEOUT", "25.0")
    monkeypatch.setenv("TAVILY_MAX_CONCURRENCY", "8")
    s = Settings()
    assert s.tavily_timeout == 25.0
    assert s.tavily_max_concurrency == 8
