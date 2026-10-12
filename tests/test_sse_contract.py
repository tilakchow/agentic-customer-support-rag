"""Contract and streaming tests for Server-Sent Events (SSE) stream payloads."""

import json

import pytest

from agenticrag.node_meta import NodeName, get_node_status_message, is_generation_node
from agenticrag.sse import stream_agent_generator


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


class FakeToken:
    def __init__(self, content: str):
        self.content = content


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeStateSnapshot:
    def __init__(self, values: dict):
        self.values = values


class FakeCompiledGraph:
    def __init__(self, should_raise: bool = False):
        self.should_raise = should_raise

    async def astream_events(self, input_data, config=None, version="v2"):
        if self.should_raise:
            raise RuntimeError("CRITICAL_INTERNAL_LEAKED_KEY_9999")
        # 1. status frame
        yield {
            "event": "on_chain_start",
            "metadata": {"langgraph_node": "route_question"},
            "data": {},
        }
        # 2. retrieval failure status frame
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "retrieve_kb"},
            "data": {"output": {"kb_retrieval_failed": True}},
        }
        # 3. token frame
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "generate_from_kb"},
            "data": {"chunk": FakeToken("Hello! How can I assist you?")},
        }

    async def aget_state(self, config):
        return FakeStateSnapshot(
            {
                "messages": [FakeMessage("Hello! How can I assist you?")],
                "source_used": "private_kb",
            }
        )


@pytest.mark.asyncio
async def test_sse_stream_exact_frames_success():
    """Asserts exact SSE data frames for status, token, and done."""
    fake_graph = FakeCompiledGraph(should_raise=False)
    events = []
    async for frame in stream_agent_generator("Need help", "session-42", agent_app=fake_graph):
        events.append(frame)

    assert len(events) == 4

    # Frame 1: status for route_question
    assert events[0] == (
        'data: {"type": "status", "node": "route_question", "message": "Analyzing question & routing..."}\n\n'
    )

    # Frame 2: status for retrieve_kb failure
    assert events[1] == (
        'data: {"type": "status", "node": "retrieve_kb", "message": "Knowledge base retrieval unavailable or failed..."}\n\n'
    )

    # Frame 3: token
    assert events[2] == ('data: {"type": "token", "token": "Hello! How can I assist you?"}\n\n')

    # Frame 4: done
    parsed_done = json.loads(events[3].replace("data: ", "").strip())
    assert parsed_done["type"] == "done"
    assert parsed_done["thread_id"] == "session-42"
    assert parsed_done["source_used"] == "private_kb"
    assert parsed_done["full_content"] == "Hello! How can I assist you?"


@pytest.mark.asyncio
async def test_sse_stream_sanitized_error_frame_no_leak():
    """Asserts graph failure produces a sanitized error frame with no leaked text."""
    fake_graph = FakeCompiledGraph(should_raise=True)
    events = []
    async for frame in stream_agent_generator("Need help", "session-42", agent_app=fake_graph):
        events.append(frame)

    assert len(events) == 1
    raw_frame = events[0]
    assert "CRITICAL_INTERNAL_LEAKED_KEY_9999" not in raw_frame
    parsed_error = json.loads(raw_frame.replace("data: ", "").strip())
    assert parsed_error["type"] == "error"
    assert "error" in parsed_error
    assert "unexpected error occurred" in parsed_error["error"]
