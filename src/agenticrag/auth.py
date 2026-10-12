"""Authentication dependencies and constant-time API key validation for API routes."""

import hmac

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from agenticrag.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    """Verifies incoming X-API-Key header against server configured key using constant-time comparison."""
    settings = get_settings()
    configured_key = settings.api_key

    # When API_KEY is unset/empty in settings, the API is considered public
    if not configured_key:
        return api_key

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required API key in X-API-Key header."
        )

    if not hmac.compare_digest(api_key.encode("utf-8"), configured_key.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key."
        )

    return api_key
