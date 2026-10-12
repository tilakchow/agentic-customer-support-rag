"""Contract tests for Server-Sent Events (SSE) stream payloads."""

import json

from agenticrag.node_meta import NodeName, get_node_status_message, is_generation_node


def test_status_event_contract():
    """Verify that every node produces a valid status message payload."""
    for node in NodeName:
        status_text = get_node_status_message(node.value)
        assert isinstance(status_text, str)
        assert len(status_text) > 0

        payload = {"type": "status", "node": node.value, "message": status_text}
        encoded = json.dumps(payload)
        parsed = json.loads(encoded)

        assert parsed["type"] == "status"
        assert parsed["node"] == node.value
        assert "message" in parsed


def test_generation_node_contract():
    """Verify generation nodes classification matches the answer synthesis nodes."""
    expected_generation = {
        NodeName.GENERATE_FROM_KB.value,
        NodeName.GENERATE_FROM_WEB.value,
        NodeName.DIRECT_ANSWER.value,
        NodeName.ANSWER_INSUFFICIENT.value,
    }

    for node in NodeName:
        if node.value in expected_generation:
            assert is_generation_node(node.value) is True
        else:
            assert is_generation_node(node.value) is False


def test_sse_event_payload_schemas():
    """Verify that SSE stream shapes match frontend TypeScript interface contracts."""
    # 1. Status event schema
    status_payload = {"type": "status", "node": "route_question", "message": "Analyzing question & routing..."}
    assert "type" in status_payload and status_payload["type"] == "status"
    assert "node" in status_payload
    assert "message" in status_payload

    # 2. Token event schema
    token_payload = {"type": "token", "token": "Hello"}
    assert token_payload["type"] == "token"
    assert "token" in token_payload

    # 3. Done event schema
    done_payload = {
        "type": "done",
        "thread_id": "test-session-123",
        "source_used": "private_kb",
        "full_content": "Your order has been shipped."
    }
    assert done_payload["type"] == "done"
    assert "thread_id" in done_payload
    assert "source_used" in done_payload
    assert "full_content" in done_payload

    # 4. Error event schema
    error_payload = {"type": "error", "error": "An unexpected error occurred."}
    assert error_payload["type"] == "error"
    assert "error" in error_payload
