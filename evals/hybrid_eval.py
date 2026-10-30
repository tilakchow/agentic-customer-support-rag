"""Benchmark script for Milestone T7 hybrid retrieval experiment.

Evaluates dense retrieval vs dense + BM25 reciprocal rank fusion (RRF k=60) on evals/golden_hybrid.jsonl.
Reports Recall@4, Precision@4, MRR, bootstrap 95% CIs, and p50/p95 latency.
"""

import argparse
import json
import random
import time
from typing import Any

from agenticrag.retrieval import retrieve_kb


def boot_ci(vals: list[float], n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n_boot))
    return round(means[int(0.025 * n_boot)], 3), round(means[int(0.975 * n_boot)], 3)


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    sorted_v = sorted(vals)
    return round(sorted_v[int(len(sorted_v) * p)], 3)


def evaluate_split(
    cases: list[dict[str, Any]],
    *,
    hybrid_enabled: bool,
    k: int = 4,
) -> dict[str, Any]:
    hits: list[float] = []
    precisions: list[float] = []
    mrrs: list[float] = []
    latencies: list[float] = []

    for c in cases:
        query = c["query"]
        expected_intent = c.get("expected_intent")

        t0 = time.perf_counter()
        res = retrieve_kb(query, k=k, hybrid_enabled=hybrid_enabled, reranker_enabled=False)
        dur = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(dur)

        retrieved_intents = [doc.metadata.get("intent") for doc in res.docs]

        # Recall@k (1.0 if expected_intent in top-k else 0.0)
        is_hit = 1.0 if expected_intent in retrieved_intents else 0.0
        hits.append(is_hit)

        # Precision@k (fraction of top-k docs matching intent)
        matching_count = sum(1.0 for intent in retrieved_intents if intent == expected_intent)
        prec = matching_count / len(retrieved_intents) if retrieved_intents else 0.0
        precisions.append(prec)

        # MRR (reciprocal rank of first matching doc)
        rr = 0.0
        for rank, intent in enumerate(retrieved_intents, 1):
            if intent == expected_intent:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

    n = len(cases)
    recall_mean = round(sum(hits) / n, 3) if n else 0.0
    prec_mean = round(sum(precisions) / n, 3) if n else 0.0
    mrr_mean = round(sum(mrrs) / n, 3) if n else 0.0

    return {
        "hybrid_enabled": hybrid_enabled,
        "n": n,
        "recall@4": {"mean": recall_mean, "ci95": boot_ci(hits), "n": n},
        "precision@4": {"mean": prec_mean, "ci95": boot_ci(precisions), "n": n},
        "mrr": {"mean": mrr_mean, "ci95": boot_ci(mrrs), "n": n},
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_mean_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid retrieval performance.")
    parser.add_argument(
        "--dataset", default="evals/golden_hybrid.jsonl", help="Hard cases dataset path"
    )
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    print("\n=======================================================")
    print(f"HYBRID RETRIEVAL BENCHMARK (Dataset: {args.dataset}, n={len(cases)})")
    print("=======================================================\n")

    print("Evaluating Dense Retrieval (Hybrid OFF)...")
    off_res = evaluate_split(cases, hybrid_enabled=False, k=4)

    print("Evaluating Hybrid Retrieval (Dense + BM25 RRF k=60, Hybrid ON)...")
    on_res = evaluate_split(cases, hybrid_enabled=True, k=4)

    print("\n" + "=" * 78)
    print(f"{'Metric':<22} | {'Hybrid OFF (Dense Only)':<24} | {'Hybrid ON (Dense + BM25)':<24}")
    print("-" * 78)
    print(
        f"{'Recall@4':<22} | {off_res['recall@4']['mean']:<5} (CI: {str(off_res['recall@4']['ci95']):<14}) | {on_res['recall@4']['mean']:<5} (CI: {str(on_res['recall@4']['ci95']):<14})"
    )
    print(
        f"{'Precision@4':<22} | {off_res['precision@4']['mean']:<5} (CI: {str(off_res['precision@4']['ci95']):<14}) | {on_res['precision@4']['mean']:<5} (CI: {str(on_res['precision@4']['ci95']):<14})"
    )
    print(
        f"{'MRR':<22} | {off_res['mrr']['mean']:<5} (CI: {str(off_res['mrr']['ci95']):<14}) | {on_res['mrr']['mean']:<5} (CI: {str(on_res['mrr']['ci95']):<14})"
    )
    print(
        f"{'Latency p50':<22} | {off_res['latency_p50_ms']} ms{'':<18} | {on_res['latency_p50_ms']} ms"
    )
    print(
        f"{'Latency p95':<22} | {off_res['latency_p95_ms']} ms{'':<18} | {on_res['latency_p95_ms']} ms"
    )
    print("=" * 78 + "\n")

    # Gate decision
    off_res["recall@4"]["mean"]
    recall_on = on_res["recall@4"]["mean"]
    ci_off_upper = off_res["recall@4"]["ci95"][1]

    if recall_on > ci_off_upper:
        print(
            f"DECISION: Hybrid ON significantly improves recall ({recall_on} > {ci_off_upper}). Enable by default."
        )
    else:
        print(
            f"DECISION: Hybrid ON recall ({recall_on}) does not improve outside CI ({off_res['recall@4']['ci95']})."
        )
        print("RESULT: tried, no measured gain. Default remains HYBRID_RETRIEVAL_ENABLED=False.")

    return {"off": off_res, "on": on_res}


if __name__ == "__main__":
    main()
