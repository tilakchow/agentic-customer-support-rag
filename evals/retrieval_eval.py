"""Retrieval evaluation script reporting recall@4, precision@4, MRR, score distributions,

and evidence grader bypass metrics for reranker_enabled=false vs true.
"""

import argparse
import json
import random
from typing import Any

from agenticrag.config import get_settings
from agenticrag.constants import RETRIEVER_K
from agenticrag.retrieval import retrieve_kb


def boot_ci(vals: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n_boot))
    return round(means[int(0.025 * n_boot)], 3), round(means[int(0.975 * n_boot)], 3)


def pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * p))]


def evaluate_retrieval(
    rows: list[dict[str, Any]],
    *,
    split: str,
    k: int = RETRIEVER_K,
    reranker_enabled: bool,
    skip_grader_score: float | None = None,
) -> dict[str, Any]:
    hit: list[float] = []
    prec: list[float] = []
    rr: list[float] = []
    in_scores: list[float] = []
    in_kb_scores: list[float] = []
    oos_scores: list[float] = []
    adv_scores: list[float] = []

    for row in rows:
        if "query" not in row or row.get("split") != split:
            continue
        query = row["query"]
        res = retrieve_kb(query, k=k, reranker_enabled=reranker_enabled)
        top = res.top_score

        row_type = row.get("type", "")
        expected_intent = row.get("expected_intent")

        if expected_intent is None:
            if row_type == "injection":
                adv_scores.append(top)
            else:
                oos_scores.append(top)
            continue

        rel = [d.metadata.get("intent") == expected_intent for d in res.docs]
        hit.append(float(any(rel)))
        prec.append(sum(rel) / k)
        rr.append(next((1.0 / (i + 1) for i, r in enumerate(rel) if r), 0.0))
        in_scores.append(top)
        if row_type == "in_kb":
            in_kb_scores.append(top)

    def summ(v: list[float]) -> dict[str, Any]:
        mean_val = round(sum(v) / len(v), 3) if v else 0.0
        return {"mean": mean_val, "ci95": boot_ci(v), "n": len(v)}

    all_oos_adv = oos_scores + adv_scores
    score_p10_in = round(pct(sorted(in_scores), 0.10), 4) if in_scores else 0.0
    score_p10_in_kb = round(pct(sorted(in_kb_scores), 0.10), 4) if in_kb_scores else 0.0
    score_p90_oos = round(pct(sorted(all_oos_adv), 0.90), 4) if all_oos_adv else 0.0

    eval_out: dict[str, Any] = {
        "split": split,
        "reranker_enabled": reranker_enabled,
        "k": k,
        "recall@4": summ(hit),
        "precision@4": summ(prec),
        "mrr": summ(rr),
        "top_score_in_kb_p10": score_p10_in,
        "top_score_in_kb_only_p10": score_p10_in_kb,
        "top_score_oos_p90": score_p90_oos,
        "top_score_oos_max": round(max(all_oos_adv), 4) if all_oos_adv else 0.0,
        "n_in_scope": len(in_scores),
        "n_in_kb_only": len(in_kb_scores),
        "n_oos": len(oos_scores),
        "n_adversarial": len(adv_scores),
        "n_oos_total": len(all_oos_adv),
    }

    if skip_grader_score is not None:
        in_bypassed = sum(s >= skip_grader_score for s in in_kb_scores)
        in_bypass_rate = round(in_bypassed / len(in_kb_scores), 3) if in_kb_scores else 0.0

        oos_false_bypassed = sum(s >= skip_grader_score for s in all_oos_adv)
        oos_false_bypass_rate = (
            round(oos_false_bypassed / len(all_oos_adv), 3) if all_oos_adv else 0.0
        )

        eval_out["skip_grader_score"] = skip_grader_score
        eval_out["in_kb_bypass_rate"] = {
            "rate": in_bypass_rate,
            "bypassed": in_bypassed,
            "total": len(in_kb_scores),
        }
        eval_out["false_bypass_rate"] = {
            "rate": oos_false_bypass_rate,
            "bypassed": oos_false_bypassed,
            "total": len(all_oos_adv),
        }

    return eval_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KB retrieval and evidence scores.")
    parser.add_argument("path", nargs="?", default="evals/golden.jsonl", help="Dataset path")
    parser.add_argument("split", nargs="?", default="dev", help="Dataset split (dev or test)")
    parser.add_argument(
        "thresh", nargs="?", type=float, default=0.0, help="Recall@4 threshold gate"
    )
    parser.add_argument(
        "--reranker",
        choices=["true", "false", "both"],
        default="both",
        help="Whether to test with neural reranker, without, or compare both",
    )
    parser.add_argument(
        "--skip-grader-score",
        type=float,
        default=None,
        help="Optional override for skip_grader_score threshold calibration",
    )

    args = parser.parse_args()
    settings = get_settings()
    skip_score = (
        args.skip_grader_score if args.skip_grader_score is not None else settings.skip_grader_score
    )

    with open(args.path, encoding="utf-8-sig") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    output: dict[str, Any] = {}
    if args.reranker == "both":
        res_false = evaluate_retrieval(
            rows,
            split=args.split,
            reranker_enabled=False,
            skip_grader_score=0.45,
        )
        res_true = evaluate_retrieval(
            rows,
            split=args.split,
            reranker_enabled=True,
            skip_grader_score=skip_score,
        )
        output = {
            "split": args.split,
            "reranker_disabled": res_false,
            "reranker_enabled": res_true,
        }
        # Backward compatibility for top-level keys
        output["recall@4"] = res_true["recall@4"]
        output["precision@4"] = res_true["precision@4"]
        output["mrr"] = res_true["mrr"]
        output["top_score_in_kb_p10"] = res_true["top_score_in_kb_p10"]
        output["top_score_oos_p90"] = res_true["top_score_oos_p90"]
        output["n_oos"] = res_true["n_oos_total"]
    else:
        use_rerank = args.reranker == "true"
        res = evaluate_retrieval(
            rows,
            split=args.split,
            reranker_enabled=use_rerank,
            skip_grader_score=skip_score,
        )
        output = res

    print(json.dumps(output, indent=2))

    current_recall = (
        output["reranker_enabled"]["recall@4"]["mean"]
        if "reranker_enabled" in output
        else output["recall@4"]["mean"]
    )
    assert current_recall >= args.thresh, (
        f"recall@4 ({current_recall}) below threshold {args.thresh}"
    )


if __name__ == "__main__":
    main()
