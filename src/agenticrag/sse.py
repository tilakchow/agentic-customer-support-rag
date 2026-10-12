import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import HumanMessage
from starlette.requests import Request

from agenticrag.graph import get_agent_app
from agenticrag.node_meta import get_node_status_message, is_generation_node

logger = logging.getLogger(__name__)


def sse(event: dict) -> str:
    """Formats an event payload dictionary as a Server-Sent Events data frame."""
    return f"data: {json.dumps(event)}\n\n"


async def stream_agent_generator(
    message: str,
    thread_id: str,
    agent_app: Any = None,
    request: Request | None = None,
) -> AsyncGenerator[str, None]:
    """Generates Server-Sent Events asynchronously with client disconnect detection."""
    if agent_app is None:
        agent_app = get_agent_app()

    config = {"configurable": {"thread_id": thread_id}}
    input_data = {"messages": [HumanMessage(content=message)]}
    active_node = ""

    try:
        async for event in agent_app.astream_events(input_data, config=config, version="v2"):
            if request is not None and await request.is_disconnected():
                logger.info("[Streaming] Client disconnected prematurely. Aborting stream.")
                return

            event_type = event.get("event")

            # Check if a new node started executing
            if event_type == "on_chain_start":
                node_name = event.get("metadata", {}).get("langgraph_node")
                if node_name and node_name != active_node:
                    active_node = node_name
                    status_text = get_node_status_message(node_name)
                    yield sse({"type": "status", "node": node_name, "message": status_text})

            # Check for node completion updates (e.g. retrieval failure notice)
            elif event_type == "on_chain_end":
                node_name = event.get("metadata", {}).get("langgraph_node")
                if node_name == "retrieve_kb":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and output.get("kb_retrieval_failed"):
                        yield sse(
                            {
                                "type": "status",
                                "node": "retrieve_kb",
                                "message": "Knowledge base retrieval unavailable or failed...",
                            }
                        )

            # Stream LLM tokens when generating final answer
            elif event_type == "on_chat_model_stream":
                node_name = event.get("metadata", {}).get("langgraph_node")
                if node_name and is_generation_node(node_name):
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield sse({"type": "token", "token": str(chunk.content)})

        # Graph execution completed; extract final state asynchronously
        state_snapshot = await agent_app.aget_state(config)
        final_state = state_snapshot.values
        source = final_state.get("source_used", "direct")

        messages = final_state.get("messages", [])
        last_content = messages[-1].content if messages else ""

        yield sse(
            {
                "type": "done",
                "thread_id": thread_id,
                "source_used": source,
                "full_content": str(last_content),
            }
        )

    except asyncio.CancelledError:
        logger.info("[Streaming] Stream cancelled due to client disconnect.")
        raise
    except Exception as e:
        logger.error("[Streaming Error]: %s", e, exc_info=True)
        # Never leak raw internal stack traces/provider errors to SSE clients
        yield sse(
            {
                "type": "error",
                "error": "An unexpected error occurred while generating your response. Please try again.",
            }
        )
