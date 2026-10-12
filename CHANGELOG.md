# Changelog

All notable changes to the Agentic Customer Support RAG project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-13

### Added
- **Checkpointer Lifecycle Management**: Async SQLite checkpointer opening and closing managed strictly in FastAPI `lifespan` context manager, eliminating unclosed connections and silent fallback bugs.
- **Production Persistence Guard**: Startup check refusing in-memory checkpointing when `ENVIRONMENT=production` unless `ALLOW_IN_MEMORY_CHECKPOINTS=true` is set.
- **Thread State Bounding**: Added `trim_messages_reducer` capping retained messages per thread at `MAX_THREAD_MESSAGES` (default 40) to prevent unbounded memory growth.
- **Tavily Timeout & Concurrency Controls**: Configured `TAVILY_TIMEOUT_SECONDS` (default 10s) with structured error fallback and `asyncio.Semaphore(TAVILY_MAX_CONCURRENCY)` to avoid connection hangs and API rate limit exhaustion.
- **Anti-Spoofing Rate Limiting**: Added `TRUSTED_PROXIES` configuration with rightmost untrusted IP traversal in `X-Forwarded-For` chain.
- **Startup Warmup & Thread Guards**: Added client warmup during application lifespan to eliminate cold-start event loop freezes; guarded client initializers with `threading.Lock`.
- **Vercel Edge Proxy Routes**: Added `frontend/api/chat.ts` and `frontend/api/chat-stream.ts` supporting full chunked SSE streaming for split frontend/backend deployments.
- **Frontend State Modernization**: Extracted `useThreads` (with safe localStorage handling) and `useChatStream` (with thread switching cancellation); added `ErrorBoundary` recovery component.
- **Comprehensive Test Suite & CI Hardening**: Expanded test suite to 123+ unit and integration tests with >93% coverage; added CLI, ingestion, and schema test suites; integrated Vitest, pip-audit, npm audit, and Dependabot.

### Changed
- Refactored `src/agenticrag/nodes.py` to remove legacy backward-compatibility re-exports and deduplicated LLM content extraction with `_message_text`.
- Updated `pyproject.toml` mypy target `python_version` to 3.12 matching `requires-python`.
- Enhanced `.github/workflows/ci.yml` with coverage threshold `85%`, Docker image validation, and security auditing.
