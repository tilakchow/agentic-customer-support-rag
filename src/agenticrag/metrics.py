"""Provider health telemetry and operational failure tracking."""

import threading
from typing import Any

_lock = threading.Lock()
_provider_failures: int = 0
_last_provider_error: str | None = None


def record_provider_failure(error: Exception | str) -> None:
    """Records an external provider failure event."""
    global _provider_failures, _last_provider_error
    with _lock:
        _provider_failures += 1
        _last_provider_error = str(error)[:250]


def get_provider_metrics() -> dict[str, Any]:
    """Returns current external provider health metrics."""
    with _lock:
        return {
            "provider_failures": _provider_failures,
            "last_error": _last_provider_error,
        }


def reset_metrics() -> None:
    """Resets provider failure metrics (useful in unit testing)."""
    global _provider_failures, _last_provider_error
    with _lock:
        _provider_failures = 0
        _last_provider_error = None
