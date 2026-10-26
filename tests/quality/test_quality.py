"""Quality evaluation test suite using DeepEval Faithfulness and AnswerRelevancy.

Marked with @pytest.mark.live. Skipped if no JUDGE_API_KEY / OPENAI_API_KEY is configured.
The judge model is configurable via JUDGE_MODEL environment variable and must not be
the application's default model.
"""

import json
import os
from pathlib import Path

import pytest
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from agenticrag.graph import create_agent

GOLDEN_PATH = Path("evals/golden.jsonl")


def load_in_kb_dev_cases() -> list[dict]:
    if not GOLDEN_PATH.exists():
        return []
    with open(GOLDEN_PATH, encoding="utf-8-sig") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
            and json.loads(line).get("split") == "dev"
            and json.loads(line).get("type") == "in_kb"
        ][:5]  # Sample first 5 for quality test gate


QUALITY_CASES = load_in_kb_dev_cases()


@pytest.mark.live
@pytest.mark.parametrize("case", QUALITY_CASES, ids=[c["id"] for c in QUALITY_CASES])
def test_rag_quality_deepeval(case: dict) -> None:
    """Evaluates response faithfulness and relevancy via DeepEval metrics."""
    judge_model = os.getenv("JUDGE_MODEL", "gpt-4o")
    judge_api_key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not judge_api_key:
        pytest.skip(
            "JUDGE_API_KEY / OPENAI_API_KEY not configured. Skipping DeepEval quality test."
        )

    graph = create_agent()
    state_input = {
        "messages": [{"role": "user", "content": case["query"]}],
        "thread_id": f"quality-test-{case['id']}",
    }
    result = graph.invoke(state_input)

    output = result.get("generation", "")
    context_docs = [
        d.page_content if hasattr(d, "page_content") else str(d)
        for d in result.get("documents", [])
    ]

    test_case = LLMTestCase(
        input=case["query"],
        actual_output=output,
        retrieval_context=context_docs,
    )

    faithfulness = FaithfulnessMetric(threshold=0.7, model=judge_model)
    relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge_model)

    faithfulness.measure(test_case)
    relevancy.measure(test_case)

    assert faithfulness.is_successful(), f"Faithfulness check failed: {faithfulness.reason}"
    assert relevancy.is_successful(), f"Relevancy check failed: {relevancy.reason}"
