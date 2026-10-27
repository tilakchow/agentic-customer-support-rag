"""Semantic and exact response cache partitioned by KB version and tenant identity.

Provides fast sub-millisecond responses for repeated queries, with automatic
invalidation whenever the knowledge base version advances.
"""

import hashlib
import logging
import threading
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from typing import Any

from agenticrag.config import get_settings

logger = logging.getLogger(__name__)


def compute_query_hash(query: str) -> str:
    """Produces normalized SHA-256 hash for query caching."""
    normalized = " ".join(query.lower().strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


class ResponseCache:
    """Thread-safe LRU response cache keyed by (kb_version, tenant_id, query_hash)."""

    def __init__(self, maxsize: int = 1000, default_ttl_seconds: int = 3600):
        self._maxsize = maxsize
        self._default_ttl = timedelta(seconds=default_ttl_seconds)
        self._cache: OrderedDict[str, tuple[dict[str, Any], datetime]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0

    def _build_key(self, query: str, tenant_id: str, kb_version: str) -> str:
        q_hash = compute_query_hash(query)
        return f"{kb_version}:{tenant_id}:{q_hash}"

    def get(
        self,
        query: str,
        tenant_id: str = "default",
        kb_version: str | None = None,
    ) -> dict[str, Any] | None:
        """Retrieves cached response if present, unexpired, and matches KB version."""
        if kb_version is None:
            kb_version = getattr(get_settings(), "kb_version", "v2.1.0")

        key = self._build_key(query, tenant_id, kb_version)
        now = datetime.now(UTC)

        with self._lock:
            item = self._cache.get(key)
            if not item:
                self._misses += 1
                return None

            data, expiry = item
            if now > expiry:
                # Expired item
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (LRU hit)
            self._cache.move_to_end(key)
            self._hits += 1
            return dict(data)

    def set(
        self,
        query: str,
        response_data: dict[str, Any],
        tenant_id: str = "default",
        kb_version: str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        """Stores response in cache with TTL and active KB version key."""
        if kb_version is None:
            kb_version = getattr(get_settings(), "kb_version", "v2.1.0")

        key = self._build_key(query, tenant_id, kb_version)
        ttl = timedelta(seconds=ttl_seconds) if ttl_seconds is not None else self._default_ttl
        expiry = datetime.now(UTC) + ttl

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self._maxsize:
                # Evict oldest entry
                self._cache.popitem(last=False)
            self._cache[key] = (dict(response_data), expiry)

    def invalidate_version(self, kb_version: str) -> int:
        """Evicts all cache entries belonging to a retired KB version."""
        prefix = f"{kb_version}:"
        evicted = 0
        with self._lock:
            keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
            for k in keys_to_delete:
                del self._cache[k]
                evicted += 1
        return evicted

    def clear(self) -> None:
        """Clears entire cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, Any]:
        """Returns cache telemetry stats."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests) if total_requests > 0 else 0.0
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
            }


_global_cache = ResponseCache()


def get_response_cache() -> ResponseCache:
    """Returns singleton ResponseCache instance."""
    return _global_cache
