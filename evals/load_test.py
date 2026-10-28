"""Load testing script for Agentic Customer Support RAG.

Executes concurrent requests measuring latency distributions (p50, p95, p99),
throughput (req/sec), and success/error rates under load.
"""

import argparse
import asyncio
import json
import logging
import statistics
import time
from typing import Any

import httpx

from agenticrag.api import app

logger = logging.getLogger(__name__)

BENCHMARK_PROMPTS = [
    "How can I track my package ORD-1001?",
    "What is your return policy for international orders?",
    "Can I change my delivery address after placing an order?",
    "How long do refunds take to reflect in my bank account?",
    "I received a damaged item, what should I do?",
    "Can you reset my password please?",
    "I want to cancel my recent subscription order.",
    "Do you ship to Canada and Australia?",
    "Can I speak with a human agent right now?",
    "Where is order ORD-1002 currently located?",
    "What payment methods do you accept on your store?",
    "My order arrived missing an accessory, please assist.",
    "How do I apply a discount promo code at checkout?",
    "Can you look up my order status for ORD-1003?",
    "Is standard shipping free on orders above fifty dollars?",
]


def pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
    return sorted_vals[idx]


async def run_worker(
    queue: asyncio.Queue,
    client: httpx.AsyncClient,
    results: list[dict[str, Any]],
) -> None:
    while not queue.empty():
        try:
            item_id, prompt = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        start_time = time.perf_counter()
        status = 0
        error_msg = None
        try:
            resp = await client.post(
                "/chat",
                json={"query": prompt},
                timeout=60.0,
            )
            status = resp.status_code
        except Exception as e:
            error_msg = str(e)
            status = 500

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        results.append(
            {
                "id": item_id,
                "status": status,
                "latency_ms": latency_ms,
                "error": error_msg,
            }
        )
        queue.task_done()


async def execute_load_test(
    total_queries: int = 200,
    concurrency: int = 5,
    base_url: str | None = None,
) -> dict[str, Any]:
    queue: asyncio.Queue = asyncio.Queue()
    for i in range(total_queries):
        prompt = BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]
        queue.put_nowait((i, prompt))

    results: list[dict[str, Any]] = []

    start_total = time.perf_counter()
    if base_url:
        async with httpx.AsyncClient(base_url=base_url) as client:
            workers = [
                asyncio.create_task(run_worker(queue, client, results)) for _ in range(concurrency)
            ]
            await asyncio.gather(*workers)
    else:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            workers = [
                asyncio.create_task(run_worker(queue, client, results)) for _ in range(concurrency)
            ]
            await asyncio.gather(*workers)

    total_duration_sec = time.perf_counter() - start_total

    latencies = sorted(r["latency_ms"] for r in results)
    successful = sum(1 for r in results if r["status"] == 200)
    failed = len(results) - successful

    p50 = round(pct(latencies, 0.50), 2)
    p95 = round(pct(latencies, 0.95), 2)
    p99 = round(pct(latencies, 0.99), 2)
    mean_lat = round(statistics.mean(latencies), 2) if latencies else 0.0
    throughput = round(len(results) / total_duration_sec, 2) if total_duration_sec > 0 else 0.0

    return {
        "total_queries": len(results),
        "concurrency": concurrency,
        "total_duration_sec": round(total_duration_sec, 2),
        "throughput_rps": throughput,
        "successful_requests": successful,
        "failed_requests": failed,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "mean_latency_ms": mean_lat,
    }


def main():
    parser = argparse.ArgumentParser(description="Run load tests on Agentic RAG /chat endpoint.")
    parser.add_argument(
        "--queries", type=int, default=200, help="Total queries to execute (default 200)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=5, help="Concurrency level (e.g. 5 or 20)"
    )
    parser.add_argument(
        "--url", default=None, help="Optional external base URL (e.g. http://localhost:8000)"
    )
    args = parser.parse_args()

    print(f"Starting load test: {args.queries} queries at concurrency {args.concurrency}...")
    metrics = asyncio.run(execute_load_test(args.queries, args.concurrency, args.url))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
