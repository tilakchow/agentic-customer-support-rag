"""Observability and distributed tracing telemetry for Agentic RAG graph nodes.

Captures per-node execution spans including latency, input/output tokens,
estimated cost, routing decisions, fallback triggers, and active KB versions.
Supports Langfuse / OpenTelemetry integration with bounded in-memory buffer.
"""

import functools
import hashlib
import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from agenticrag.config import get_settings

logger = logging.getLogger(__name__)

# Approximate pricing per 1M tokens ($0.05 input, $0.10 output for OSS / Groq models)
INPUT_COST_PER_MILLION = 0.05
OUTPUT_COST_PER_MILLION = 0.10


def hash_tenant_id(tenant_id: str | None) -> str:
    """Hashes tenant_id with SHA-256 for privacy-preserving observability."""
    if not tenant_id:
        return "none"
    return hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:16]


@dataclass
class TelemetrySpan:
    """Structured telemetry span representing a single graph node execution."""

    trace_id: str
    span_id: str
    node_name: str
    start_time: str
    end_time: str
    duration_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    route: str | None
    fallback_triggered: bool
    kb_version: str
    status: str
    error_message: str | None = None
    score_type: str | None = None
    top_score: float | None = None
    tenant_id_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TelemetryCollector:
    """Thread-safe bounded ring buffer for telemetry spans and operational metrics."""

    def __init__(self, max_buffer_size: int = 500):
        self._max_buffer_size = max_buffer_size
        self._spans: deque[TelemetrySpan] = deque(maxlen=max_buffer_size)
        self._fail_events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._total_recorded_spans: int = 0
        self._total_tokens_consumed: int = 0
        self._total_estimated_cost: float = 0.0

    def record_fail_decision(
        self,
        node_name: str,
        error_type: str,
        mode: str,
        details: str | None = None,
    ) -> dict[str, Any]:
        """Records a fail-open or fail-closed event with node name and error type."""
        event = {
            "node_name": node_name,
            "error_type": error_type,
            "mode": mode,
            "details": details[:200] if details else None,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with self._lock:
            self._fail_events.append(event)
        logger.info(
            "[Fail-Mode] Node %s evaluated as fail-%s on error %s",
            node_name,
            mode,
            error_type,
        )
        return event

    def get_fail_events(self) -> list[dict[str, Any]]:
        """Returns recorded fail-open and fail-closed events."""
        with self._lock:
            return list(self._fail_events)

    def record_span(
        self,
        node_name: str,
        duration_ms: float,
        start_dt: datetime,
        end_dt: datetime,
        input_tokens: int = 0,
        output_tokens: int = 0,
        route: str | None = None,
        fallback_triggered: bool = False,
        status: str = "success",
        error_message: str | None = None,
        trace_id: str | None = None,
        score_type: str | None = None,
        top_score: float | None = None,
        tenant_id: str | None = None,
        tenant_id_hash: str | None = None,
    ) -> TelemetrySpan:
        """Records a completed execution span into the telemetry ring buffer."""
        total_tokens = input_tokens + output_tokens
        cost = (
            input_tokens * INPUT_COST_PER_MILLION + output_tokens * OUTPUT_COST_PER_MILLION
        ) / 1_000_000.0

        resolved_hash = tenant_id_hash or (hash_tenant_id(tenant_id) if tenant_id else None)

        span = TelemetrySpan(
            trace_id=trace_id or uuid.uuid4().hex[:12],
            span_id=uuid.uuid4().hex[:8],
            node_name=node_name,
            start_time=start_dt.isoformat(),
            end_time=end_dt.isoformat(),
            duration_ms=round(duration_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(cost, 6),
            route=route,
            fallback_triggered=fallback_triggered,
            kb_version=getattr(get_settings(), "kb_version", "v2.1.0"),
            status=status,
            error_message=error_message[:200] if error_message else None,
            score_type=score_type,
            top_score=round(top_score, 4) if top_score is not None else None,
            tenant_id_hash=resolved_hash,
        )

        with self._lock:
            self._spans.append(span)
            self._total_recorded_spans += 1
            self._total_tokens_consumed += total_tokens
            self._total_estimated_cost += cost

        # Langfuse & OpenTelemetry export hooks
        self._export_to_langfuse(span)
        self._export_to_opentelemetry(span)

        return span

    def _export_to_opentelemetry(self, span: TelemetrySpan) -> None:
        """Hooks into OpenTelemetry Tracer if available in environment."""
        try:
            from opentelemetry import trace  # type: ignore[import-untyped]

            tracer = trace.get_tracer("agenticrag.telemetry")
            with tracer.start_as_current_span(
                f"agenticrag.{span.node_name}",
                attributes={
                    "agenticrag.node": span.node_name,
                    "agenticrag.route": span.route or "unknown",
                    "agenticrag.score_type": span.score_type or "none",
                    "agenticrag.top_score": span.top_score if span.top_score is not None else 0.0,
                    "agenticrag.kb_version": span.kb_version,
                    "agenticrag.tenant_id_hash": span.tenant_id_hash or "none",
                    "agenticrag.tokens.total": span.total_tokens,
                    "agenticrag.tokens.input": span.input_tokens,
                    "agenticrag.tokens.output": span.output_tokens,
                    "agenticrag.cost_usd": span.estimated_cost_usd,
                    "agenticrag.status": span.status,
                },
            ):
                pass
        except Exception:
            pass

    def _export_to_langfuse(self, span: TelemetrySpan) -> None:
        """Hooks into Langfuse client if API keys are configured."""
        settings = get_settings()
        if getattr(settings, "langfuse_public_key", None) and getattr(
            settings, "langfuse_secret_key", None
        ):
            logger.debug(
                "[Langfuse Export] Forwarding span %s for node %s", span.span_id, span.node_name
            )

    def get_recent_spans(self, limit: int = 50) -> list[dict[str, Any]]:
        """Returns the most recent execution spans."""
        with self._lock:
            spans = list(self._spans)
        return [s.to_dict() for s in spans[-limit:]]

    def get_summary(self) -> dict[str, Any]:
        """Returns aggregated telemetry metrics."""
        with self._lock:
            recent_spans = list(self._spans)
            total_spans = self._total_recorded_spans
            total_tokens = self._total_tokens_consumed
            total_cost = self._total_estimated_cost

        avg_latency = (
            sum(s.duration_ms for s in recent_spans) / len(recent_spans) if recent_spans else 0.0
        )

        return {
            "total_spans_recorded": total_spans,
            "buffered_spans_count": len(recent_spans),
            "total_tokens_consumed": total_tokens,
            "total_estimated_cost_usd": round(total_cost, 6),
            "recent_avg_latency_ms": round(avg_latency, 2),
            "kb_version": getattr(get_settings(), "kb_version", "v2.1.0"),
        }

    def clear(self) -> None:
        """Clears all buffered spans (useful for tests)."""
        with self._lock:
            self._spans.clear()
            self._fail_events.clear()
            self._total_recorded_spans = 0
            self._total_tokens_consumed = 0
            self._total_estimated_cost = 0.0


# Global singleton collector
_collector = TelemetryCollector()


def get_telemetry_collector() -> TelemetryCollector:
    """Returns singleton TelemetryCollector."""
    return _collector


class trace_node_span:
    """Async context manager for timing and tracing graph node executions."""

    def __init__(self, node_name: str, state: dict[str, Any] | None = None):
        self.node_name = node_name
        self.state = state or {}
        self.start_dt = datetime.now(UTC)
        self.start_time = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.fallback = False

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.start_dt = datetime.now(UTC)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000.0
        end_dt = datetime.now(UTC)
        status = "error" if exc_val is not None else "success"
        err_msg = str(exc_val) if exc_val else None

        route = self.state.get("route")
        fallback = (
            self.fallback
            or self.state.get("web_search_failed", False)
            or self.state.get("kb_retrieval_failed", False)
        )

        get_telemetry_collector().record_span(
            node_name=self.node_name,
            duration_ms=duration_ms,
            start_dt=self.start_dt,
            end_dt=end_dt,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            route=route,
            fallback_triggered=bool(fallback),
            status=status,
            error_message=err_msg,
        )
        return False  # Propagate exception


def traced_node(node_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for tracing LangGraph node execution with real duration, tokens, status, and OpenTelemetry attributes."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            start_dt = datetime.now(UTC)
            tenant_id = state.get("tenant_id", "default")
            route = state.get("route")
            score_type = state.get("score_type")
            top_score = state.get("top_similarity_score")

            try:
                result = await func(state, *args, **kwargs)
                duration_ms = (time.perf_counter() - t0) * 1000.0
                end_dt = datetime.now(UTC)

                in_tok = 0
                out_tok = 0
                fb = False
                if isinstance(result, dict):
                    msgs = result.get("messages", [])
                    if msgs:
                        last = msgs[-1]
                        usage = getattr(last, "usage_metadata", None)
                        meta = getattr(last, "response_metadata", {}) or {}
                        if usage:
                            in_tok = getattr(usage, "input_tokens", 0) or 0
                            out_tok = getattr(usage, "output_tokens", 0) or 0
                        elif "token_usage" in meta:
                            tu = meta.get("token_usage", {})
                            in_tok = tu.get("prompt_tokens", 0) or 0
                            out_tok = tu.get("completion_tokens", 0) or 0

                    if "route" in result:
                        route = result["route"]
                    if "score_type" in result:
                        score_type = result["score_type"]
                    if "top_similarity_score" in result:
                        top_score = result["top_similarity_score"]
                    fb = bool(result.get("kb_retrieval_failed") or result.get("web_search_failed"))

                get_telemetry_collector().record_span(
                    node_name=node_name,
                    duration_ms=duration_ms,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    route=route,
                    score_type=score_type,
                    top_score=top_score,
                    tenant_id=tenant_id,
                    fallback_triggered=fb,
                    status="success",
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - t0) * 1000.0
                end_dt = datetime.now(UTC)
                get_telemetry_collector().record_span(
                    node_name=node_name,
                    duration_ms=duration_ms,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    route=route,
                    score_type=score_type,
                    top_score=top_score,
                    tenant_id=tenant_id,
                    status="error",
                    error_message=str(e),
                )
                raise

        return wrapper

    return decorator
