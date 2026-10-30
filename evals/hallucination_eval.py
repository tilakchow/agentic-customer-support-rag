"""Evaluation script for Milestone T3 cheap faithfulness pre-check vs LLM check.

Runs cases from evals/hallucination_cases.jsonl, evaluating cheap entity pre-check
against LLM check_claim_support. Reports precision, recall, and token savings.
"""

import argparse
import asyncio
import json
import logging

from agenticrag.nodes import cheap_faithfulness_precheck, check_claim_support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_evaluation(dataset_path: str, delay: float = 1.0, limit: int | None = None) -> dict:
    cases = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    if limit:
        cases = cases[:limit]

    print(f"\nLoaded {len(cases)} hallucination test cases from {dataset_path}\n")

    precheck_correct = 0
    llm_correct = 0
    agreement_count = 0

    # Confusion matrix: Ground truth vs Pre-check
    # Positive class = Grounded (True), Negative class = Hallucinated (False)
    # TP: GT grounded & precheck passed
    # FP: GT hallucinated & precheck passed
    # FN: GT grounded & precheck flagged
    # TN: GT hallucinated & precheck flagged
    tp, fp, fn, tn = 0, 0, 0, 0

    results = []

    for i, c in enumerate(cases, 1):
        cid = c["id"]
        ctype = c.get("type", "unknown")
        q = c["question"]
        ctx = c["context"]
        ans = c["response"]
        gt = c["ground_truth_grounded"]

        # Run cheap pre-check
        pre_passed, missing = cheap_faithfulness_precheck(ctx, ans)

        # Run LLM check with retry backoff
        llm_passed = None
        for attempt in range(5):
            try:
                check_result = await check_claim_support(q, ctx, ans)
                llm_passed = check_result.supported
                break
            except Exception as e:
                wait_time = 2.0 * (attempt + 1)
                logger.warning(
                    "[Case %s] LLM call failed: %s; retrying in %.1fs", cid, e, wait_time
                )
                await asyncio.sleep(wait_time)

        if llm_passed is None:
            llm_passed = False

        if pre_passed == gt:
            precheck_correct += 1
        if llm_passed == gt:
            llm_correct += 1
        if pre_passed == llm_passed:
            agreement_count += 1

        if gt and pre_passed:
            tp += 1
        elif not gt and pre_passed:
            fp += 1
        elif gt and not pre_passed:
            fn += 1
        else:
            tn += 1

        results.append(
            {
                "id": cid,
                "type": ctype,
                "gt": gt,
                "precheck": pre_passed,
                "missing": missing,
                "llm": llm_passed,
            }
        )

        status_str = "OK" if pre_passed == gt else "MISMATCH"
        print(
            f"[{i:02d}/{len(cases):02d}] {cid:14} type={ctype:18} GT={str(gt):5} PreCheck={str(pre_passed):5} LLM={str(llm_passed):5} -> {status_str}"
        )
        if missing:
            print(f"     Flagged missing entities: {missing}")

        if delay > 0:
            await asyncio.sleep(delay)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    accuracy = (tp + tn) / len(cases) if cases else 0.0
    agreement_rate = agreement_count / len(cases) if cases else 0.0

    # Hallucination detection rate (recall on negative class): TN / (TN + FP)
    detection_rate = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    report = {
        "n_cases": len(cases),
        "precheck_accuracy": accuracy,
        "precheck_precision": precision,
        "precheck_recall": recall,
        "hallucination_detection_rate": detection_rate,
        "agreement_with_llm": agreement_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "details": results,
    }

    print("\n" + "=" * 70)
    print("FAITHFULNESS PRE-CHECK EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Cases:                 {len(cases)}")
    print(f"Pre-Check Accuracy:          {accuracy:.3f} ({precheck_correct}/{len(cases)})")
    print(f"Pre-Check Precision:         {precision:.3f} (Grounded Precision)")
    print(f"Pre-Check Recall:            {recall:.3f} (Grounded Recall)")
    print(f"Hallucination Catch Rate:    {detection_rate:.3f} ({tn}/{tn + fp})")
    print(f"Agreement with LLM Check:    {agreement_rate:.3f} ({agreement_count}/{len(cases)})")
    print("=" * 70)

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evals/hallucination_cases.jsonl", help="Dataset path")
    parser.add_argument("--delay", type=float, default=1.2, help="Pacing delay between LLM calls")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.dataset, delay=args.delay, limit=args.limit))


if __name__ == "__main__":
    main()
