"""Configuration loader for the Streamlit frontend."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class FrontendConfig:
    """Frontend runtime configuration resolved from environment variables."""

    backend_url: str
    api_key: str
    request_timeout: float = 15.0
    stream_timeout: float = 60.0

    @classmethod
    def from_env(cls) -> FrontendConfig:
        """Loads configuration from environment variables with safe defaults.

        Returns:
            FrontendConfig instance.
        """
        raw_backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000").strip()
        backend_url = raw_backend_url.rstrip("/") if raw_backend_url else "http://localhost:8000"
        api_key = os.environ.get("API_KEY", "").strip()

        return cls(
            backend_url=backend_url,
            api_key=api_key,
        )


def get_config() -> FrontendConfig:
    """Returns the frontend configuration singleton."""
    return FrontendConfig.from_env()
