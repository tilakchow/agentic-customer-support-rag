"""Measures end-to-end graph latency (p50, p95) and token counts across execution paths (kb, web, direct).

Writes measured results to evals/latency.json.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from typing import Any

from langchain_core.messages import HumanMessage

from agenticrag.config import get_settings
from agenticrag.graph import create_agent

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("evals.latency_run")


def get_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * p))]


def extract_tokens(messages: list[Any]) -> dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for m in messages:
        if hasattr(m, "response_metadata") and isinstance(m.response_metadata, dict):
            token_usage = m.response_metadata.get("token_usage")
            if token_usage and isinstance(token_usage, dict):
                prompt_tokens += int(token_usage.get("prompt_tokens", 0))
                completion_tokens += int(token_usage.get("completion_tokens", 0))
                total_tokens += int(token_usage.get("total_tokens", 0))
                continue
        if hasattr(m, "usage_metadata") and isinstance(m.usage_metadata, dict):
            prompt_tokens += int(m.usage_metadata.get("input_tokens", 0))
            completion_tokens += int(m.usage_metadata.get("output_tokens", 0))
            total_tokens += int(m.usage_metadata.get("total_tokens", 0))

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


async def run_latency_benchmark(
    n_per_path: int = 5, output_file: str = "evals/latency.json"
) -> dict[str, Any]:
    settings = get_settings()

    # Pre-flight check: required keys must exist
    missing_keys = []
    if not settings.groq_api_key:
        missing_keys.append("GROQ_API_KEY")
    if not settings.pinecone_api_key:
        missing_keys.append("PINECONE_API_KEY")

    if missing_keys:
        sys.exit(
            f"Error: Required API keys ({', '.join(missing_keys)}) are missing. "
            "Cannot run latency benchmark without real provider access. Do not fabricate numbers."
        )

    # Standard representative queries for each path
    path_queries: dict[str, list[str]] = {
        "kb": [
            "can I cancel my order? #12345",
            "how can i get my bill from Company?",
            "what do I have to do to unsubscribe to your newsletter?",
            "I entered a wrong delivery address, can you correct it?",
            "check how long reimbursements usually take",
            "how do I track the status of my order #99231?",
            "can I delete an article from my purchase?",
        ],
        "direct": [
            "Hello! Can you help me today?",
            "Good morning! Who are you?",
            "Thank you so much, that answers my question!",
            "Hi there! I just wanted to say hello.",
            "What kind of customer support services can you provide?",
            "Hey bot, how are you doing today?",
        ],
        "web": [
            "What is the capital city of Australia?",
            "Who won the FIFA Men's World Cup in 2022?",
            "What is the distance between the Earth and the Moon in kilometers?",
            "Who wrote the play Romeo and Juliet?",
            "What is the official currency of Japan?",
            "When did the Apollo 11 mission land on the moon?",
        ],
    }

    agent = create_agent()
    results_by_path: dict[str, list[dict[str, Any]]] = {"kb": [], "web": [], "direct": []}

    print(f"Running latency benchmark ({n_per_path} queries per path)...")
    for expected_path, queries in path_queries.items():
        sample = queries[:n_per_path]
        for q in sample:
            thread_id = f"latency-bench-{uuid.uuid4().hex[:8]}"
            t0 = time.perf_counter()
            try:
                res = await agent.ainvoke(
                    {"messages": [HumanMessage(content=q)]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                elapsed = time.perf_counter() - t0
                source_used = res.get("source_used", "")

                # Map source_used to path name
                actual_path = (
                    "kb"
                    if source_used in ("private_kb", "knowledge_base")
                    else "web"
                    if source_used in ("web_search",)
                    else "direct"
                    if source_used in ("direct", "direct_answer")
                    else expected_path
                )

                tokens = extract_tokens(res.get("messages", []))
                record = {
                    "query": q,
                    "target_path": expected_path,
                    "actual_path": actual_path,
                    "source_used": source_used,
                    "latency_s": round(elapsed, 3),
                    "tokens": tokens,
                }
                results_by_path[actual_path].append(record)
                print(
                    f"  [{actual_path.upper()}] latency={elapsed:.3f}s | tokens={tokens['total_tokens']} | q='{q[:40]}'"
                )
            except Exception as e:
                print(f"  [ERROR] Failed query '{q[:40]}': {e}")

    # Compute statistics per path
    path_stats: dict[str, Any] = {}
    all_latencies: list[float] = []
    all_tokens: list[int] = []

    for path_name, records in results_by_path.items():
        if not records:
            path_stats[path_name] = {"n": 0, "status": "no_samples"}
            continue
        lats = sorted(r["latency_s"] for r in records)
        all_latencies.extend(lats)

        totals = [r["tokens"]["total_tokens"] for r in records]
        prompts = [r["tokens"]["prompt_tokens"] for r in records]
        completions = [r["tokens"]["completion_tokens"] for r in records]
        all_tokens.extend(totals)

        path_stats[path_name] = {
            "n": len(records),
            "latency_p50_s": round(pct(lats, 0.50), 3),
            "latency_p95_s": round(pct(lats, 0.95), 3),
            "latency_mean_s": round(sum(lats) / len(lats), 3),
            "prompt_tokens_p50": int(pct(sorted(prompts), 0.50)),
            "completion_tokens_p50": int(pct(sorted(completions), 0.50)),
            "total_tokens_p50": int(pct(sorted(totals), 0.50)),
            "total_tokens_mean": round(sum(totals) / len(totals), 1),
        }

    overall_lats = sorted(all_latencies)
    overall_stats = {
        "n": len(all_latencies),
        "latency_p50_s": round(pct(overall_lats, 0.50), 3) if overall_lats else 0.0,
        "latency_p95_s": round(pct(overall_lats, 0.95), 3) if overall_lats else 0.0,
        "total_tokens_mean": round(sum(all_tokens) / len(all_tokens), 1) if all_tokens else 0.0,
    }

    final_report = {
        "git_sha": get_git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_total": len(all_latencies),
        "paths": path_stats,
        "overall": overall_stats,
        "records": results_by_path,
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nLatency benchmark complete. Written to {output_file}.")
    print(json.dumps({"overall": overall_stats, "paths": path_stats}, indent=2))
    return final_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real graph latency benchmarks.")
    parser.add_argument("--n-per-path", type=int, default=5, help="Number of queries per path")
    parser.add_argument("--output", default="evals/latency.json", help="Output JSON path")
    args = parser.parse_args()

    asyncio.run(run_latency_benchmark(n_per_path=args.n_per_path, output_file=args.output))


if __name__ == "__main__":
    main()
