"""Authentication dependencies, JWT verification, and multi-tenant isolation.

Provides JWT user and tenant extraction, thread-to-user binding to prevent thread hijacking,
tenant-scoped Pinecone namespaces, and constant-time API key validation.
"""

import hmac
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
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


def create_jwt(payload: dict[str, Any], secret: str | None = None) -> str:
    """Creates a signed HMAC-SHA256 JWT string using PyJWT."""
    settings = get_settings()
    if secret is None:
        secret = settings.jwt_secret

    claims = dict(payload)
    if "exp" not in claims:
        # Default 1 hour expiry if not specified
        claims["exp"] = datetime.now(UTC) + timedelta(hours=1)
    if "iat" not in claims:
        claims["iat"] = datetime.now(UTC)
    if settings.jwt_issuer and "iss" not in claims:
        claims["iss"] = settings.jwt_issuer
    if settings.jwt_audience and "aud" not in claims:
        claims["aud"] = settings.jwt_audience

    return jwt.encode(claims, secret, algorithm="HS256")


def decode_jwt(
    token: str, secret: str | None = None, verify_signature: bool = True
) -> dict[str, Any]:
    """Decodes and validates a signed HMAC-SHA256 JWT string using PyJWT."""
    settings = get_settings()
    if secret is None:
        secret = settings.jwt_secret

    options = {
        "verify_signature": verify_signature,
        "require": ["exp"],
        "verify_exp": True,
    }

    kwargs: dict[str, Any] = {
        "algorithms": ["HS256"],  # Require HS256 only (rejects 'none' and other algs)
        "options": options,
        "leeway": 10,  # small leeway for clock skew
    }
    if settings.jwt_issuer:
        kwargs["issuer"] = settings.jwt_issuer
    if settings.jwt_audience:
        kwargs["audience"] = settings.jwt_audience

    try:
        return jwt.decode(token, secret, **kwargs)
    except jwt.PyJWTError as e:
        raise ValueError(f"JWT validation error: {e}") from e


async def get_authenticated_user(request: Request) -> AuthenticatedUser:
    """Extracts authenticated user and tenant identity from Bearer JWT only.

    Header fallback (X-Tenant-ID / X-User-ID) is removed. If ALLOW_ANONYMOUS is
    enabled and no token is supplied, tenant 'default' and user 'anonymous' are returned.
    Invalid or expired tokens always return 401.
    """
    settings = get_settings()
    auth_header = request.headers.get("Authorization", "").strip()

    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        try:
            payload = decode_jwt(token, secret=settings.jwt_secret, verify_signature=True)
            user_id = payload.get("sub") or payload.get("user_id")
            tenant_id = payload.get("tenant_id") or payload.get("org_id")
            role = payload.get("role", "user")

            if not user_id or not tenant_id:
                raise ValueError("JWT must contain user identifier ('sub') and 'tenant_id'.")

            return AuthenticatedUser(user_id=str(user_id), tenant_id=str(tenant_id), role=str(role))
        except Exception as e:
            logger.warning("[Auth] JWT verification failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid or expired authentication token: {e}",
            ) from e

    # No Bearer token supplied: check if anonymous access is explicitly allowed
    if settings.allow_anonymous:
        # Per-session anonymous thread ownership is generated from client IP / session
        client_host = request.client.host if request.client else "local"
        anon_user_id = f"anon-{client_host}"
        return AuthenticatedUser(user_id=anon_user_id, tenant_id="default", role="anonymous")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing required Bearer authentication token.",
    )


class BaseThreadTenancyStorage(ABC):
    """Abstract interface for thread ownership storage."""

    @abstractmethod
    def bind_thread(self, thread_id: str, tenant_id: str, user_id: str) -> None:
        """Binds a thread_id to tenant_id and user_id."""
        pass

    @abstractmethod
    def get_thread_owner(self, thread_id: str) -> tuple[str, str] | None:
        """Returns (tenant_id, user_id) tuple or None."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets storage."""
        pass


class InMemoryThreadTenancyStorage(BaseThreadTenancyStorage):
    """Thread-safe in-memory thread tenancy storage."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread_owners: dict[str, tuple[str, str]] = {}

    def bind_thread(self, thread_id: str, tenant_id: str, user_id: str) -> None:
        with self._lock:
            if thread_id not in self._thread_owners:
                self._thread_owners[thread_id] = (tenant_id, user_id)

    def get_thread_owner(self, thread_id: str) -> tuple[str, str] | None:
        with self._lock:
            return self._thread_owners.get(thread_id)

    def reset(self) -> None:
        with self._lock:
            self._thread_owners.clear()


class RedisThreadTenancyStorage(BaseThreadTenancyStorage):
    """Redis-backed thread tenancy storage used when REDIS_URL is configured."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._client: Any = None
        self._fallback = InMemoryThreadTenancyStorage()

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import redis

                self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            except Exception as e:
                logger.warning(
                    "[Redis Tenancy] Failed to connect to Redis (%s), falling back to in-memory: %s",
                    self.redis_url,
                    e,
                )
                return None
        return self._client

    def bind_thread(self, thread_id: str, tenant_id: str, user_id: str) -> None:
        client = self._get_client()
        if client:
            try:
                client.hsetnx(f"thread_owner:{thread_id}", "tenant_id", tenant_id)
                client.hsetnx(f"thread_owner:{thread_id}", "user_id", user_id)
                return
            except Exception as e:
                logger.warning("[Redis Tenancy] Error binding thread: %s", e)
        self._fallback.bind_thread(thread_id, tenant_id, user_id)

    def get_thread_owner(self, thread_id: str) -> tuple[str, str] | None:
        client = self._get_client()
        if client:
            try:
                data = client.hgetall(f"thread_owner:{thread_id}")
                if data and "tenant_id" in data and "user_id" in data:
                    return (data["tenant_id"], data["user_id"])
            except Exception as e:
                logger.warning("[Redis Tenancy] Error reading thread: %s", e)
        return self._fallback.get_thread_owner(thread_id)

    def reset(self) -> None:
        self._fallback.reset()
        client = self._get_client()
        if client:
            try:
                keys = client.keys("thread_owner:*")
                if keys:
                    client.delete(*keys)
            except Exception as e:
                logger.warning("[Redis Tenancy] Error resetting: %s", e)


class ThreadTenancyManager:
    """Manages thread ownership across tenants and users backed by modular storage."""

    def __init__(self, storage: BaseThreadTenancyStorage | None = None):
        self.storage = storage or InMemoryThreadTenancyStorage()

    def bind_thread(self, thread_id: str, tenant_id: str, user_id: str) -> None:
        self.storage.bind_thread(thread_id, tenant_id, user_id)

    def verify_thread_access(self, thread_id: str, tenant_id: str, user_id: str) -> None:
        """Validates that the requester has ownership of the given thread_id."""
        owner = self.storage.get_thread_owner(thread_id)
        if not owner:
            # First time seeing this thread: register ownership
            self.storage.bind_thread(thread_id, tenant_id, user_id)
            return

        registered_tenant, registered_user = owner

        # Cross-tenant access is strictly forbidden
        if (
            registered_tenant != tenant_id
            and registered_tenant != "default"
            and tenant_id != "default"
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cross-tenant thread access forbidden.",
            )

        # Cross-user access check: anonymous users are bound per session user_id
        if registered_user != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cross-user thread access forbidden.",
            )

    def reset(self) -> None:
        self.storage.reset()


_tenancy_manager: ThreadTenancyManager | None = None


def get_tenancy_manager() -> ThreadTenancyManager:
    """Returns singleton ThreadTenancyManager with Redis or in-memory backend."""
    global _tenancy_manager
    if _tenancy_manager is None:
        settings = get_settings()
        if settings.redis_url:
            storage: BaseThreadTenancyStorage = RedisThreadTenancyStorage(settings.redis_url)
        else:
            storage = InMemoryThreadTenancyStorage()
        _tenancy_manager = ThreadTenancyManager(storage)
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
