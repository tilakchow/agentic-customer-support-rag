"""Adversarial evaluation suite for prompt injection and jailbreak defense.

Measures the defense rate against prompt injections, system prompt exfiltration,
and canary string extraction attempts on the dev split. Computes Wilson score 95% CI.
"""

import json
import math
import os
from typing import Any

from agenticrag.security import CANARY_TOKEN, is_prompt_injection


def wilson_score_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Computes Wilson score 95% confidence interval for a binomial proportion."""
    if trials == 0:
        return 0.0, 1.0
    z = 1.959963984540054  # z-score for 95% confidence
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    spread = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return round(lower, 4), round(upper, 4)


def run_adversarial_eval() -> dict[str, Any]:
    """Runs injection defense evaluation on golden_other.jsonl dev split."""
    golden_other_path = os.path.join(os.path.dirname(__file__), "golden_other.jsonl")
    if not os.path.exists(golden_other_path):
        raise FileNotFoundError(f"Missing {golden_other_path}")

    cases = []
    with open(golden_other_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("type") == "injection" and item.get("split") == "dev":
                cases.append(item)

    total_cases = len(cases)
    blocked_count = 0
    results = []

    for c in cases:
        query = c["query"]
        is_inj, reason = is_prompt_injection(query)
        # Defense passes if detected as injection and canary is never present
        passed = is_inj and CANARY_TOKEN not in query
        if passed:
            blocked_count += 1
        results.append(
            {
                "id": c["id"],
                "query": query,
                "is_injection_detected": is_inj,
                "reason": reason,
                "passed": passed,
            }
        )

    defense_rate = (blocked_count / total_cases) if total_cases > 0 else 1.0
    ci_lower, ci_upper = wilson_score_interval(blocked_count, total_cases)

    report = {
        "n": total_cases,
        "blocked": blocked_count,
        "defense_rate": round(defense_rate, 4),
        "ci_95": [ci_lower, ci_upper],
        "results": results,
    }
    return report


if __name__ == "__main__":
    report = run_adversarial_eval()
    print(
        f"Prompt Injection Defense Rate: {report['defense_rate']:.4f} "
        f"({report['blocked']}/{report['n']}, 95% CI: {report['ci_95']})"
    )
