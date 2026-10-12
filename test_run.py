#!/usr/bin/env python
"""Smoke test script for Agentic-RAG pipeline verification."""

import sys
import logging
from agenticrag.config import get_settings
from agenticrag.graph import get_agent_app
from agenticrag.nodes import route_after_router, decide_after_kb_grade, decide_after_web_grade

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_run")


def run_smoke_test():
    logger.info("1. Validating configuration settings...")
    settings = get_settings()
    logger.info("   Index: %s, Namespace: %s", settings.index_name, settings.namespace)
    settings.validate_runtime_keys(strict=False)

    logger.info("2. Validating pure decision functions...")
    assert route_after_router({"route": "kb"}) == "retrieve_kb"
    assert route_after_router({"route": "direct"}) == "direct_answer"
    assert decide_after_kb_grade({"kb_grade": "good"}) == "generate_from_kb"
    assert decide_after_kb_grade({"kb_grade": "weak"}) == "search_web"
    assert decide_after_web_grade({"web_search_failed": True}) == "answer_insufficient"
    logger.info("   Decision functions verified successfully.")

    logger.info("3. Compiling LangGraph agent state graph...")
    app = get_agent_app()
    assert app is not None
    logger.info("   Agent state graph compiled successfully.")

    logger.info("All smoke tests passed! System is ready.")


if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception as e:
        logger.error("Smoke test failed: %s", e, exc_info=True)
        sys.exit(1)
