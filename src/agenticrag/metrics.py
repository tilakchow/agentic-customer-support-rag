"""Provider health telemetry and operational failure tracking.

Replaces basic integer counter with structured, thread-safe provider health tracking,
rolling 5-minute failure rates, per-provider error isolation, and operational metrics.
"""

import threading
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any


class ProviderHealthTracker:
    """Thread-safe operational telemetry tracker for external services."""

    def __init__(self, rolling_window_minutes: int = 5):
        self._lock = threading.Lock()
        self._rolling_window = timedelta(minutes=rolling_window_minutes)
        self._total_failures: int = 0
        self._last_error: str | None = None
        self._provider_counts: dict[str, int] = {
            "groq": 0,
            "pinecone": 0,
            "tavily": 0,
            "database": 0,
            "general": 0,
        }
        self._failure_timestamps: deque[datetime] = deque()

    def record_failure(self, error: Exception | str, provider: str | None = None) -> None:
        """Records a provider failure with timestamp and provider classification."""
        err_str = str(error)[:250]
        now = datetime.now(UTC)

        # Infer provider from error string or class if not explicitly given
        if not provider:
            err_lower = err_str.lower()
            if "pinecone" in err_lower:
                provider = "pinecone"
            elif "tavily" in err_lower:
                provider = "tavily"
            elif "groq" in err_lower or "rate limit" in err_lower:
                provider = "groq"
            elif "sqlite" in err_lower or "postgres" in err_lower or "db" in err_lower:
                provider = "database"
            else:
                provider = "general"

        with self._lock:
            self._total_failures += 1
            self._last_error = err_str
            self._provider_counts[provider] = self._provider_counts.get(provider, 0) + 1
            self._failure_timestamps.append(now)
            self._evict_expired(now)

    def _evict_expired(self, now: datetime) -> None:
        """Removes timestamps outside the rolling window."""
        cutoff = now - self._rolling_window
        while self._failure_timestamps and self._failure_timestamps[0] < cutoff:
            self._failure_timestamps.popleft()

    def get_metrics(self) -> dict[str, Any]:
        """Returns structured health snapshot including rolling window failure count."""
        now = datetime.now(UTC)
        with self._lock:
            self._evict_expired(now)
            return {
                "provider_failures": self._total_failures,
                "rolling_5m_failures": len(self._failure_timestamps),
                "providers": dict(self._provider_counts),
                "last_error": self._last_error,
                "status": "degraded" if len(self._failure_timestamps) >= 5 else "healthy",
            }

    def reset(self) -> None:
        """Resets all metrics (for test isolation)."""
        with self._lock:
            self._total_failures = 0
            self._last_error = None
            for p in self._provider_counts:
                self._provider_counts[p] = 0
            self._failure_timestamps.clear()


# Singleton health tracker instance
_tracker = ProviderHealthTracker()


def record_provider_failure(error: Exception | str, provider: str | None = None) -> None:
    """Records an external provider failure event."""
    _tracker.record_failure(error, provider=provider)


def get_provider_metrics() -> dict[str, Any]:
    """Returns current external provider health metrics."""
    return _tracker.get_metrics()


def reset_metrics() -> None:
    """Resets provider failure metrics (useful in unit testing)."""
    _tracker.reset()
