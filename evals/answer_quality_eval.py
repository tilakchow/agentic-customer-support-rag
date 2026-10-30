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
from pydantic import AliasChoices, BaseModel, Field

from agenticrag.clients import get_llm
from agenticrag.graph import create_agent

logger = logging.getLogger(__name__)


class QualityJudgement(BaseModel):
    correctness: int = Field(
        ge=1,
        le=5,
        default=5,
        description="Correctness score from 1 (completely wrong) to 5 (accurate).",
        validation_alias=AliasChoices("correctness", "Correctness"),
    )
    groundedness: int = Field(
        ge=1,
        le=5,
        default=5,
        description="Groundedness score from 1 (hallucinated) to 5 (substantiated).",
        validation_alias=AliasChoices("groundedness", "Groundedness"),
    )
    tone: int = Field(
        ge=1,
        le=5,
        default=5,
        description="Tone score from 1 (unprofessional/rude) to 5 (empathetic, concise, professional).",
        validation_alias=AliasChoices("tone", "Tone"),
    )
    rationale: str = Field(
        default="",
        description="Brief explanation of the judgment.",
        validation_alias=AliasChoices("rationale", "Rationale", "reason", "Reason"),
    )


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

Return valid JSON with exact lowercase keys:
{{"correctness": <int 1-5>, "groundedness": <int 1-5>, "tone": <int 1-5>, "rationale": "<brief explanation>"}}
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
    max_retries: int = 3,
) -> dict[str, Any] | None:
    query = case.get("query", "")
    if not query:
        return None

    for attempt in range(max_retries):
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
            err_msg = str(e)
            if "429" in err_msg or "rate_limit" in err_msg.lower():
                backoff = 4.0 * (2**attempt)
                logger.warning(
                    "Rate limit on case %s (attempt %d/%d): waiting %.1fs",
                    case.get("id"),
                    attempt + 1,
                    max_retries,
                    backoff,
                )
                await asyncio.sleep(backoff)
            else:
                logger.warning("Failed to evaluate case %s: %s", case.get("id"), e)
                return None
    return None


async def run_quality_eval(
    dataset_path: str = "evals/golden_realistic.jsonl",
    sample_size: int | None = None,
    judge_model: str | None = None,
    delay: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    with open(dataset_path, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    if sample_size and sample_size < len(cases):
        rng = random.Random(seed)
        cases = rng.sample(cases, sample_size)

    agent = create_agent()
    judge_raw = get_llm(model=judge_model)
    try:
        judge_llm = judge_raw.with_structured_output(QualityJudgement, method="json_mode")
    except Exception as e:
        logger.error("Failed to initialize structured judge LLM: %s", e)
        return {"status": "UNMEASURED", "reason": str(e)}

    results = []
    for idx, c in enumerate(cases, 1):
        r = await evaluate_single(agent, judge_llm, c)
        if r:
            results.append(r)
            print(
                f"[{idx:02d}/{len(cases):02d}] {c.get('id')}: Correctness={r['correctness']} Groundedness={r['groundedness']} Tone={r['tone']}"
            )
        else:
            print(f"[{idx:02d}/{len(cases):02d}] {c.get('id')}: FAILED / SKIPPED")
        if delay > 0:
            await asyncio.sleep(delay)

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
    parser.add_argument(
        "--sample",
        "--limit",
        dest="sample",
        type=int,
        default=None,
        help="Optional sample limit",
    )
    parser.add_argument("--judge-model", default=None, help="LLM judge model override")
    parser.add_argument("--delay", type=float, default=1.5, help="Pacing delay between calls")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    report = asyncio.run(
        run_quality_eval(
            args.dataset,
            sample_size=args.sample,
            judge_model=args.judge_model,
            delay=args.delay,
            seed=args.seed,
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
