"""Answer quality evaluation script using an LLM judge on realistic customer queries.

Evaluates 3 core dimensions on 1-5 scales:
- Correctness (1-5): Factual alignment with expected answer or domain policy.
- Groundedness (1-5): Extent to which claims are substantiated by evidence.
- Tone (1-5): Professionalism, empathy, and conciseness.

Outputs bootstrap 95% confidence intervals for all metrics.
"""

import argparse
import asyncio
import json
import logging
import random
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agenticrag.clients import get_llm
from agenticrag.graph import create_agent

logger = logging.getLogger(__name__)


class QualityJudgement(BaseModel):
    correctness: int = Field(
        ge=1, le=5, description="Correctness score from 1 (completely wrong) to 5 (accurate)."
    )
    groundedness: int = Field(
        ge=1, le=5, description="Groundedness score from 1 (hallucinated) to 5 (substantiated)."
    )
    tone: int = Field(
        ge=1,
        le=5,
        description="Tone score from 1 (unprofessional/rude) to 5 (empathetic, concise, professional).",
    )
    rationale: str = Field(description="Brief explanation of the judgment.")


JUDGE_PROMPT_TEMPLATE = """You are an expert customer support evaluator.
Evaluate the following assistant response to a customer inquiry.

Customer Question:
{question}

Assistant Response:
{response}

Reference Intent / Domain Context:
{context}

Score each dimension strictly from 1 to 5:
- Correctness (1-5): Does the response correctly address the user query within customer support scope?
- Groundedness (1-5): Are the claims verifiable, grounded, and free of baseless promises or hallucinations?
- Tone (1-5): Is the tone empathetic, polite, helpful, and concise?

Return your evaluation as a JSON object matching the requested schema.
"""


def boot_ci(vals: list[float], n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n_boot))
    return round(means[int(0.025 * n_boot)], 3), round(means[int(0.975 * n_boot)], 3)


async def evaluate_single(
    agent: Any,
    judge_llm: Any,
    case: dict[str, Any],
) -> dict[str, Any] | None:
    query = case.get("query", "")
    if not query:
        return None

    try:
        # Run agent
        res = await agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": f"quality-{case.get('id', 'temp')}"}},
        )
        answer = res.get("messages", [])[-1].content if res.get("messages") else ""
        context = case.get("expected_intent") or case.get("type", "customer_support")

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=query,
            response=answer,
            context=context,
        )
        judgement: QualityJudgement = await judge_llm.ainvoke(prompt)
        return {
            "id": case.get("id"),
            "correctness": float(judgement.correctness),
            "groundedness": float(judgement.groundedness),
            "tone": float(judgement.tone),
            "rationale": judgement.rationale,
            "answer": answer,
        }
    except Exception as e:
        logger.warning("Failed to evaluate case %s: %s", case.get("id"), e)
        return None


async def run_quality_eval(
    dataset_path: str = "evals/golden_realistic.jsonl",
    sample_size: int | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    with open(dataset_path, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    if sample_size and sample_size < len(cases):
        rng = random.Random(42)
        cases = rng.sample(cases, sample_size)

    agent = create_agent()
    try:
        judge_llm = get_llm(model=judge_model).with_structured_output(
            QualityJudgement, method="json_mode"
        )
    except Exception as e:
        logger.error("Could not initialize judge LLM: %s", e)
        return {"status": "UNMEASURED", "reason": str(e)}

    results = []
    for c in cases:
        r = await evaluate_single(agent, judge_llm, c)
        if r:
            results.append(r)

    if not results:
        return {"status": "UNMEASURED", "reason": "No successful evaluations completed."}

    correctness_scores = [r["correctness"] for r in results]
    groundedness_scores = [r["groundedness"] for r in results]
    tone_scores = [r["tone"] for r in results]

    def summarize(vals: list[float]) -> dict[str, Any]:
        mean_v = round(sum(vals) / len(vals), 3)
        ci = boot_ci(vals)
        return {"mean": mean_v, "ci95": ci, "n": len(vals)}

    report = {
        "dataset": dataset_path,
        "sample_size": len(results),
        "correctness": summarize(correctness_scores),
        "groundedness": summarize(groundedness_scores),
        "tone": summarize(tone_scores),
        "overall_quality": round(
            (sum(correctness_scores) + sum(groundedness_scores) + sum(tone_scores))
            / (3 * len(results)),
            3,
        ),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate answer quality with LLM judge.")
    parser.add_argument(
        "--dataset", default="evals/golden_realistic.jsonl", help="Path to evaluation dataset"
    )
    parser.add_argument("--sample", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--judge-model", default=None, help="LLM judge model override")
    args = parser.parse_args()

    report = asyncio.run(run_quality_eval(args.dataset, args.sample, args.judge_model))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
