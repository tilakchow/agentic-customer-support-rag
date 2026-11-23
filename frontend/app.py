"""Streamlit Application Entrypoint for Agentic Customer Support RAG."""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from frontend.api_client import ApiClient, ApiClientError
from frontend.config import get_config


@st.cache_resource
def get_api_client() -> ApiClient:
    """Creates a cached ApiClient singleton."""
    config = get_config()
    return ApiClient(
        base_url=config.backend_url,
        api_key=config.api_key,
        request_timeout=config.request_timeout,
        stream_timeout=config.stream_timeout,
    )


def init_session_state() -> None:
    """Initializes Streamlit session state keys if not already set."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


def render_sidebar(client: ApiClient) -> None:
    """Renders the application sidebar with health status and KB info."""
    with st.sidebar:
        st.title("⚙️ System Status")

        # Backend health probe
        try:
            health = client.check_health()
            status_text = health.get("status", "ok")
            st.success(f"🟢 Backend: {status_text.capitalize()}")
        except ApiClientError:
            st.error("🔴 Backend: Offline")
            st.caption(f"Target: `{client.base_url}`")

        st.divider()

        # Knowledge base metadata
        with st.expander("📚 Knowledge Base Info", expanded=False):
            try:
                kb_info = client.get_kb_info()
                st.write(f"**Index:** `{kb_info.get('index_name', 'N/A')}`")
                st.write(f"**Namespace:** `{kb_info.get('namespace', 'N/A')}`")
                st.write(f"**Dimension:** `{kb_info.get('dimension', 'N/A')}`")
                st.write(f"**Sample Size:** `{kb_info.get('sample_size', 'N/A')}`")
            except ApiClientError:
                st.caption("Unable to fetch knowledge base info.")

        st.divider()

        # Session reset action
        st.caption(f"Thread: `{st.session_state.thread_id[:8]}...`")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Renders retrieved knowledge base sources in an expandable section."""
    with st.expander(f"📚 Sources & Citations ({len(sources)})", expanded=False):
        for idx, src in enumerate(sources, start=1):
            doc_id = src.get("doc_id", f"Doc {idx}")
            score = src.get("score", 0.0)
            intent = src.get("intent", "general")
            content = src.get("content", "")

            st.markdown(f"**[{idx}] {doc_id}** `intent: {intent}` | `score: {score:.3f}`")
            if content:
                st.markdown(f"> {content}")
            if idx < len(sources):
                st.divider()


def render_ticket(ticket: dict[str, Any]) -> None:
    """Renders an escalated support ticket card."""
    with st.expander("🎫 Support Ticket Escalation", expanded=True):
        st.info("A support ticket has been prepared for human agent handoff.")
        st.json(ticket)


def render_message(msg: dict[str, Any]) -> None:
    """Renders a single message from history."""
    role = msg.get("role", "user")
    content = msg.get("content", "")

    with st.chat_message(role):
        steps = msg.get("steps")
        if steps:
            with st.expander("🔍 Agent Reasoning Steps", expanded=False):
                for step in steps:
                    st.write(f"- {step}")

        st.markdown(content)

        sources = msg.get("sources")
        if sources:
            render_sources(sources)

        ticket = msg.get("ticket_payload")
        if ticket:
            render_ticket(ticket)


def handle_user_input(client: ApiClient, user_query: str) -> None:
    """Processes user query and streams the assistant response."""
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Stream assistant response
    with st.chat_message("assistant"):
        status_box = st.status("Agent Reasoning...", expanded=True)
        steps: list[str] = []
        token_placeholder = st.empty()
        full_tokens = ""
        sources: list[dict[str, Any]] | None = None
        ticket_payload: dict[str, Any] | None = None
        source_used: str | None = None
        error_message: str | None = None

        stream_gen = client.chat_stream(
            message=user_query,
            thread_id=st.session_state.thread_id,
            session_id=st.session_state.session_id,
        )

        for event in stream_gen:
            event_type = event.get("type")

            if event_type == "status":
                msg = event.get("message", "Processing...")
                steps.append(msg)
                status_box.write(f"• {msg}")

            elif event_type == "token":
                token = event.get("token", "")
                full_tokens += token
                token_placeholder.markdown(full_tokens + "▌")

            elif event_type == "done":
                thread_id = event.get("thread_id")
                if thread_id:
                    st.session_state.thread_id = thread_id
                source_used = event.get("source_used")
                sources = event.get("sources")
                ticket_payload = event.get("ticket_payload")
                full_content = event.get("full_content")
                if full_content:
                    full_tokens = full_content

            elif event_type == "error":
                error_message = event.get("error", "An unknown error occurred.")
                status_box.update(label="Error Encountered", state="error", expanded=False)
                st.error(error_message)

        if not error_message:
            status_box.update(label="Reasoning Complete", state="complete", expanded=False)
            token_placeholder.markdown(full_tokens)

            if sources:
                render_sources(sources)
            if ticket_payload:
                render_ticket(ticket_payload)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_tokens,
                    "steps": steps,
                    "sources": sources,
                    "source_used": source_used,
                    "ticket_payload": ticket_payload,
                }
            )


def main() -> None:
    """Main application layout and execution flow."""
    st.set_page_config(
        page_title="Agentic Customer Support",
        page_icon="💬",
        layout="wide",
    )

    init_session_state()
    client = get_api_client()
    render_sidebar(client)

    st.title("💬 Agentic Customer Support")
    st.caption("Production Customer Support RAG with LangGraph routing and verification")

    # Render conversation history
    for msg in st.session_state.messages:
        render_message(msg)

    # Chat input
    if user_query := st.chat_input("Ask a customer support question..."):
        handle_user_input(client, user_query)


if __name__ == "__main__":
    main()
