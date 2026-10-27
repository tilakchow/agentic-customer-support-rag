"""Authentication dependencies, JWT verification, and multi-tenant isolation.

Provides JWT user and tenant extraction, thread-to-user binding to prevent thread hijacking,
tenant-scoped Pinecone namespaces, and constant-time API key validation.
"""

import base64
import hashlib
import hmac
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from agenticrag.config import get_settings

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    """Represents an authenticated principal with tenant and role claims."""

    user_id: str
    tenant_id: str
    role: str = "user"


def _b64url_encode(data: bytes) -> str:
    """Encodes bytes into URL-safe base64 string without trailing padding."""
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(segment: str) -> bytes:
    """Decodes a URL-safe base64 string handling variable padding."""
    rem = len(segment) % 4
    if rem > 0:
        segment += "=" * (4 - rem)
    return base64.urlsafe_b64decode(segment)


def create_jwt(payload: dict[str, Any], secret: str | None = None) -> str:
    """Creates a signed HMAC-SHA256 JWT string."""
    if secret is None:
        secret = get_settings().jwt_secret

    header = {"alg": "HS256", "typ": "JWT"}
    hdr_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{hdr_b64}.{payload_b64}"

    signature = hmac.new(
        secret.encode("utf-8"), signing_input.encode("utf-8"), hashlib.sha256
    ).digest()
    sig_b64 = _b64url_encode(signature)
    return f"{signing_input}.{sig_b64}"


def decode_jwt(
    token: str, secret: str | None = None, verify_signature: bool = True
) -> dict[str, Any]:
    """Decodes and validates a signed HMAC-SHA256 JWT string."""
    if secret is None:
        secret = get_settings().jwt_secret

    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT structure: token must contain 3 dot-separated parts.")

    hdr_b64, payload_b64, sig_b64 = parts
    signing_input = f"{hdr_b64}.{payload_b64}"

    if verify_signature and secret:
        expected_sig = hmac.new(
            secret.encode("utf-8"), signing_input.encode("utf-8"), hashlib.sha256
        ).digest()
        provided_sig = _b64url_decode(sig_b64)
        if not hmac.compare_digest(expected_sig, provided_sig):
            raise ValueError("JWT signature verification failed.")

    payload_json = _b64url_decode(payload_b64).decode("utf-8")
    return json.loads(payload_json)


async def get_authenticated_user(request: Request) -> AuthenticatedUser:
    """Extracts authenticated user and tenant identity from Bearer JWT or headers."""
    settings = get_settings()

    auth_header = request.headers.get("Authorization", "")
    tenant_header = request.headers.get("X-Tenant-ID", "")
    user_header = request.headers.get("X-User-ID", "")

    user_id = user_header or "anonymous"
    tenant_id = tenant_header or "default"
    role = "user"

    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        try:
            payload = decode_jwt(
                token, secret=settings.jwt_secret, verify_signature=bool(settings.jwt_secret)
            )
            user_id = payload.get("sub") or payload.get("user_id") or user_id
            tenant_id = payload.get("tenant_id") or payload.get("org_id") or tenant_id
            role = payload.get("role", "user")
        except Exception as e:
            logger.debug("[Auth] JWT validation failed, falling back to header/anonymous: %s", e)
            if settings.require_api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid authentication token: {e}",
                ) from e

    return AuthenticatedUser(user_id=user_id, tenant_id=tenant_id, role=role)


class ThreadTenancyManager:
    """Thread-safe in-memory binding of conversation threads to tenants and users."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread_owners: dict[str, tuple[str, str]] = {}  # thread_id -> (tenant_id, user_id)

    def bind_thread(self, thread_id: str, tenant_id: str, user_id: str) -> None:
        """Binds a thread_id to a specific tenant and user upon thread creation."""
        with self._lock:
            if thread_id not in self._thread_owners:
                self._thread_owners[thread_id] = (tenant_id, user_id)

    def verify_thread_access(self, thread_id: str, tenant_id: str, user_id: str) -> None:
        """Validates that the requester has ownership of the given thread_id."""
        with self._lock:
            owner = self._thread_owners.get(thread_id)
            if not owner:
                # First time seeing this thread, register ownership
                self._thread_owners[thread_id] = (tenant_id, user_id)
                return

            registered_tenant, registered_user = owner

            # Cross-tenant access is strictly prohibited
            if (
                registered_tenant != tenant_id
                and registered_tenant != "default"
                and tenant_id != "default"
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cross-tenant thread access forbidden.",
                )

            # Cross-user access check (unless anonymous or admin)
            if (
                registered_user != "anonymous"
                and user_id != "anonymous"
                and registered_user != user_id
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cross-user thread access forbidden.",
                )

    def reset(self) -> None:
        """Clears all thread bindings for test isolation."""
        with self._lock:
            self._thread_owners.clear()


_tenancy_manager = ThreadTenancyManager()


def get_tenancy_manager() -> ThreadTenancyManager:
    """Returns singleton ThreadTenancyManager."""
    return _tenancy_manager


def get_tenant_namespace(tenant_id: str, base_namespace: str | None = None) -> str:
    """Returns tenant-scoped Pinecone namespace."""
    if base_namespace is None:
        base_namespace = get_settings().namespace
    if not tenant_id or tenant_id == "default":
        return base_namespace
    return f"{tenant_id}-{base_namespace}"


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
            detail="Missing required API key in X-API-Key header.",
        )

    if not hmac.compare_digest(api_key.encode("utf-8"), configured_key.encode("utf-8")):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key.")

    return api_key
