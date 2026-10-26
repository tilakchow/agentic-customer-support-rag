"""Live behavioral evaluation tests running against the application API.

Marked with @pytest.mark.live so they are skipped by default unless explicitly invoked.
"""

import json
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agenticrag.api import app

GOLDEN_PATH = Path("evals/golden.jsonl")


def load_live_cases() -> list[dict]:
    """Loads golden cases from evals/golden.jsonl using utf-8-sig."""
    if not GOLDEN_PATH.exists():
        return []
    with open(GOLDEN_PATH, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


ALL_CASES = load_live_cases()
BEHAVIOR_CASES = [
    c
    for c in ALL_CASES
    if c.get("split") == "dev" and ("allowed_sources" in c or "must_not_contain" in c)
]


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.live
@pytest.mark.parametrize("case", BEHAVIOR_CASES, ids=[c["id"] for c in BEHAVIOR_CASES])
def test_behavior_live(client: TestClient, case: dict) -> None:
    """Verifies that queries adhere to source boundaries and injection guardrails."""
    query = case.get("query")
    if not query:
        pytest.skip(f"Case {case['id']} has no direct query string.")

    response = client.post(
        "/chat",
        json={"message": query, "thread_id": f"live-test-{case['id']}"},
    )
    assert response.status_code == 200, (
        f"Expected 200 but got {response.status_code}: {response.text}"
    )
    data = response.json()

    # 1. Check allowed sources boundary (e.g. out-of-scope should not use internal knowledge base)
    allowed_sources = case.get("allowed_sources")
    if allowed_sources:
        source_used = data.get("source_used")
        assert source_used in allowed_sources, (
            f"Case {case['id']}: expected source in {allowed_sources}, got '{source_used}'"
        )

    # 2. Check negative assertions / injection safety constraints
    must_not_contain = case.get("must_not_contain", [])
    response_text = data.get("response", "").lower()
    for forbidden in must_not_contain:
        assert forbidden.lower() not in response_text, (
            f"Case {case['id']}: response leaked forbidden substring '{forbidden}'"
        )
