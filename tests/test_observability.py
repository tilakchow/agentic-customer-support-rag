"""Unit and integration tests for M4 Observability: spans, provider telemetry, and hardened health."""

from datetime import UTC, datetime

import pytest
from starlette.testclient import TestClient

from agenticrag.api import app
from agenticrag.metrics import get_provider_metrics, record_provider_failure, reset_metrics
from agenticrag.telemetry import TelemetryCollector, get_telemetry_collector


@pytest.fixture(autouse=True)
def clean_metrics_and_telemetry():
    """Ensures each test starts with a clean telemetry buffer and reset metrics."""
    reset_metrics()
    get_telemetry_collector().clear()
    yield
    reset_metrics()
    get_telemetry_collector().clear()


def test_telemetry_collector_span_recording():
    """Records spans with latency, tokens, cost, route, and active KB version."""
    collector = TelemetryCollector(max_buffer_size=10)
    start = datetime.now(UTC)
    end = datetime.now(UTC)

    span = collector.record_span(
        node_name="generate_from_kb",
        duration_ms=250.5,
        start_dt=start,
        end_dt=end,
        input_tokens=500,
        output_tokens=150,
        route="kb",
        fallback_triggered=False,
        status="success",
    )

    assert span.node_name == "generate_from_kb"
    assert span.total_tokens == 650
    assert span.estimated_cost_usd > 0
    assert span.route == "kb"
    assert span.fallback_triggered is False
    assert span.kb_version == "v2.1.0"

    recent = collector.get_recent_spans()
    assert len(recent) == 1
    assert recent[0]["span_id"] == span.span_id

    summary = collector.get_summary()
    assert summary["total_spans_recorded"] == 1
    assert summary["total_tokens_consumed"] == 650


def test_provider_health_tracker_rolling_window_and_providers():
    """Verifies per-provider classification and rolling 5m failure rate."""
    reset_metrics()

    record_provider_failure("Groq API rate limit exceeded 429")
    record_provider_failure("Pinecone vector query timed out")
    record_provider_failure("Tavily search connection error")

    metrics = get_provider_metrics()
    assert metrics["provider_failures"] == 3
    assert metrics["rolling_5m_failures"] == 3
    assert metrics["providers"]["groq"] == 1
    assert metrics["providers"]["pinecone"] == 1
    assert metrics["providers"]["tavily"] == 1
    assert metrics["status"] == "healthy"


def test_public_health_check_endpoint():
    """Public /health endpoint returns standard liveness without exposing internal stack traces."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Agentic-RAG Backend"
    assert "retriever_connected" in data
    assert "provider_failures" in data


def test_detailed_health_check_endpoint():
    """Protected /health/detailed endpoint exposes granular provider and telemetry data."""
    client = TestClient(app)
    response = client.get("/health/detailed")
    assert response.status_code == 200
    data = response.json()
    assert "provider_metrics" in data
    assert "telemetry_summary" in data
    assert "status" in data
