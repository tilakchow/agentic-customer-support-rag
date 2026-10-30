"""Token and LLM Call Benchmark for Agentic Customer Support RAG.

Measures input, output, and reasoning tokens across graph nodes and execution paths:
(kb, direct, tool, refuse, escalate, insufficient).

Writes measured results to evals/tokens.json and prints a formatted summary table.
"""

import argparse
import asyncio
import json
import logging
import os
import random
import subprocess
import sys
import time
import uuid
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from agenticrag.config import get_settings
from agenticrag.graph import create_agent

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("evals.token_report")


def get_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def pct(sorted_vals: list[float] | list[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
    return float(sorted_vals[idx])


class NodeTokenCallbackHandler(BaseCallbackHandler):
    """Tracks token consumption and reasoning tokens per LangGraph node and query."""

    def __init__(self) -> None:
        self.run_to_node: dict[str, str] = {}
        self.node_stats: dict[str, dict[str, int]] = {}
        self.total_llm_calls: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_reasoning_tokens: int = 0

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        node = (metadata or {}).get("langgraph_node", "unknown")
        self.run_to_node[str(run_id)] = node

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        self.total_llm_calls += 1
        node = self.run_to_node.get(str(run_id), "unknown")
        if node not in self.node_stats:
            self.node_stats[node] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
            }

        inp = 0
        out = 0
        reas = 0

        # 1. Check llm_output
        llm_out = getattr(response, "llm_output", {}) or {}
        tu = llm_out.get("token_usage", {})
        if tu:
            inp = int(tu.get("prompt_tokens", 0) or 0)
            out = int(tu.get("completion_tokens", 0) or 0)
            details = tu.get("completion_tokens_details") or {}
            reas = int(details.get("reasoning_tokens", 0) or 0)

        # 2. Check generation usage_metadata (takes precedence / fallback)
        generations = getattr(response, "generations", []) or []
        for gen_list in generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg:
                    um = getattr(msg, "usage_metadata", {}) or {}
                    if um:
                        inp = int(um.get("input_tokens", inp) or 0)
                        out = int(um.get("output_tokens", out) or 0)
                        otd = um.get("output_token_details") or {}
                        reas = int(otd.get("reasoning", reas) or 0)

        tot = inp + out
        self.node_stats[node]["calls"] += 1
        self.node_stats[node]["input_tokens"] += inp
        self.node_stats[node]["output_tokens"] += out
        self.node_stats[node]["reasoning_tokens"] += reas
        self.node_stats[node]["total_tokens"] += tot

        self.total_input_tokens += inp
        self.total_output_tokens += out
        self.total_reasoning_tokens += reas


def determine_actual_path(res: dict[str, Any]) -> str:
    """Categorizes graph outcome into one of: kb, direct, tool, refuse, escalate, insufficient."""
    source_used = str(res.get("source_used", ""))
    tool_calls = res.get("tool_calls") or []
    tool_results = res.get("tool_results") or []
    scope = str(res.get("scope", ""))
    escalated = bool(res.get("escalated", False))

    if len(tool_calls) > 0 or len(tool_results) > 0:
        return "tool"
    if source_used in ("out_of_scope_refusal", "refuse") or scope == "out_of_scope":
        return "refuse"
    if (
        source_used in ("human_escalation", "escalate")
        or scope in ("needs_human", "policy_sensitive")
        or escalated
    ):
        return "escalate"
    if source_used in ("insufficient_evidence", "insufficient"):
        return "insufficient"
    if source_used in ("direct", "direct_answer"):
        return "direct"
    if source_used in ("private_kb", "knowledge_base"):
        return "kb"

    return "kb"


def build_stratified_dataset(limit: int = 40, seed: int = 42) -> list[dict[str, Any]]:
    """Builds a stratified evaluation dataset representing all 6 paths from golden cases."""
    rng = random.Random(seed)

    # 1. KB Cases (from in-KB dev set)
    kb_pool: list[dict[str, Any]] = []
    golden_path = os.path.join(os.path.dirname(__file__), "golden.jsonl")
    if os.path.exists(golden_path):
        with open(golden_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") == "dev" and row.get("type") == "in_kb":
                    q = row.get("query", "")
                    # Clean out placeholder tokens for natural evaluation
                    q_clean = (
                        q.replace("{{Order Number}}", "#10492")
                        .replace("{{Person Name}}", "ACME Support")
                        .replace("{{Account Category}}", "personal")
                        .replace("{{Delivery Country}}", "Canada")
                    )
                    kb_pool.append(
                        {
                            "id": row.get("id"),
                            "query": q_clean,
                            "target_path": "kb",
                            "expected_intent": row.get("expected_intent"),
                        }
                    )

    # 2. Refusal Cases (out-of-scope & prompt injections)
    refuse_pool: list[dict[str, Any]] = []
    if os.path.exists(golden_path):
        with open(golden_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") == "dev" and row.get("type") in ("out_of_scope", "injection"):
                    refuse_pool.append(
                        {
                            "id": row.get("id"),
                            "query": row.get("query"),
                            "target_path": "refuse",
                            "expected_intent": row.get("type"),
                        }
                    )

    # 3. Tool Cases (order lookup & ticket requests)
    tool_pool: list[dict[str, Any]] = [
        {
            "id": "TOOL-01",
            "query": "Where is my order ORD-1001 right now?",
            "target_path": "tool",
            "expected_intent": "get_order_status",
        },
        {
            "id": "TOOL-02",
            "query": "Can you check tracking for ORD-1002?",
            "target_path": "tool",
            "expected_intent": "get_order_status",
        },
        {
            "id": "TOOL-03",
            "query": "What is the delivery status of ORD-1003?",
            "target_path": "tool",
            "expected_intent": "get_order_status",
        },
        {
            "id": "TOOL-04",
            "query": "Has order ORD-1004 been returned or delivered?",
            "target_path": "tool",
            "expected_intent": "get_order_status",
        },
        {
            "id": "TOOL-05",
            "query": "Please open a support ticket for my broken shipment item ORD-1001",
            "target_path": "tool",
            "expected_intent": "create_support_ticket",
        },
        {
            "id": "TOOL-06",
            "query": "Submit a support ticket regarding billing discrepancy on my order",
            "target_path": "tool",
            "expected_intent": "create_support_ticket",
        },
        {
            "id": "TOOL-07",
            "query": "Check status of ord-1001 package tracking",
            "target_path": "tool",
            "expected_intent": "get_order_status",
        },
        {
            "id": "TOOL-08",
            "query": "Please create a ticket for account recovery assistance",
            "target_path": "tool",
            "expected_intent": "create_support_ticket",
        },
    ]

    # 4. Direct Answer Cases (chit-chat, capabilities, greetings)
    direct_pool: list[dict[str, Any]] = [
        {
            "id": "DIR-01",
            "query": "Hello! How can you help me today?",
            "target_path": "direct",
            "expected_intent": "greeting",
        },
        {
            "id": "DIR-02",
            "query": "Good morning! Who are you and what do you do?",
            "target_path": "direct",
            "expected_intent": "identity",
        },
        {
            "id": "DIR-03",
            "query": "Thank you so much, that completely answers my question!",
            "target_path": "direct",
            "expected_intent": "gratitude",
        },
        {
            "id": "DIR-04",
            "query": "Hi there! Just wanted to say hello to customer support.",
            "target_path": "direct",
            "expected_intent": "greeting",
        },
        {
            "id": "DIR-05",
            "query": "What kind of customer support services can you provide?",
            "target_path": "direct",
            "expected_intent": "capabilities",
        },
        {
            "id": "DIR-06",
            "query": "Hey bot, hope you are having a wonderful day!",
            "target_path": "direct",
            "expected_intent": "chit_chat",
        },
    ]

    # 5. Escalation Cases (demands for supervisor, human agent, litigation threats)
    escalate_pool: list[dict[str, Any]] = [
        {
            "id": "ESC-01",
            "query": "I want to speak with a human manager immediately",
            "target_path": "escalate",
            "expected_intent": "human_escalation",
        },
        {
            "id": "ESC-02",
            "query": "Transfer me to a live supervisor right now, your service is terrible",
            "target_path": "escalate",
            "expected_intent": "human_escalation",
        },
        {
            "id": "ESC-03",
            "query": "I will file a lawsuit against your company with my legal attorney",
            "target_path": "escalate",
            "expected_intent": "policy_sensitive",
        },
        {
            "id": "ESC-04",
            "query": "I demand to talk to a real human person right now!!",
            "target_path": "escalate",
            "expected_intent": "human_escalation",
        },
        {
            "id": "ESC-05",
            "query": "I am submitting a formal regulatory fraud report against this store",
            "target_path": "escalate",
            "expected_intent": "policy_sensitive",
        },
        {
            "id": "ESC-06",
            "query": "Connect me to a representative immediately, do not give me bot answers",
            "target_path": "escalate",
            "expected_intent": "human_escalation",
        },
    ]

    # 6. Insufficient Evidence Cases (specific in-scope support questions absent from KB)
    insufficient_pool: list[dict[str, Any]] = [
        {
            "id": "INS-01",
            "query": "What is your warranty for items damaged during orbital spaceflight?",
            "target_path": "insufficient",
            "expected_intent": "insufficient_evidence",
        },
        {
            "id": "INS-02",
            "query": "Can I pay for my order using Martian cryptocurrency tokens?",
            "target_path": "insufficient",
            "expected_intent": "insufficient_evidence",
        },
        {
            "id": "INS-03",
            "query": "What is the return policy for hazardous industrial nuclear materials?",
            "target_path": "insufficient",
            "expected_intent": "insufficient_evidence",
        },
        {
            "id": "INS-04",
            "query": "Do you provide drone submarine delivery to international underwater habitats?",
            "target_path": "insufficient",
            "expected_intent": "insufficient_evidence",
        },
    ]

    # Stratified quotas for N = 40
    # Ratio: kb: 10, direct: 6, tool: 8, refuse: 6, escalate: 6, insufficient: 4
    if limit == 40:
        quotas = {
            "kb": 10,
            "direct": 6,
            "tool": 8,
            "refuse": 6,
            "escalate": 6,
            "insufficient": 4,
        }
    else:
        # Scale proportionally
        ratio = limit / 40.0
        quotas = {
            "kb": max(1, int(round(10 * ratio))),
            "direct": max(1, int(round(6 * ratio))),
            "tool": max(1, int(round(8 * ratio))),
            "refuse": max(1, int(round(6 * ratio))),
            "escalate": max(1, int(round(6 * ratio))),
            "insufficient": max(1, int(round(4 * ratio))),
        }

    dataset: list[dict[str, Any]] = []

    rng.shuffle(kb_pool)
    dataset.extend(kb_pool[: quotas["kb"]])

    rng.shuffle(direct_pool)
    dataset.extend(direct_pool[: quotas["direct"]])

    rng.shuffle(tool_pool)
    dataset.extend(tool_pool[: quotas["tool"]])

    rng.shuffle(refuse_pool)
    dataset.extend(refuse_pool[: quotas["refuse"]])

    rng.shuffle(escalate_pool)
    dataset.extend(escalate_pool[: quotas["escalate"]])

    rng.shuffle(insufficient_pool)
    dataset.extend(insufficient_pool[: quotas["insufficient"]])

    # Cap to exact limit
    dataset = dataset[:limit]
    return dataset


async def run_token_benchmark(
    limit: int = 40,
    seed: int = 42,
    output_file: str = "evals/tokens.json",
    query_delay: float = 1.0,
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
            "Cannot run token report without real provider access. Do not fabricate numbers."
        )

    dataset = build_stratified_dataset(limit=limit, seed=seed)
    print(f"\nLoaded stratified sample of {len(dataset)} cases across 6 execution paths.")
    print(f"Provider: Groq ({settings.groq_model}) | Output: {output_file}\n")

    agent = create_agent()
    records: list[dict[str, Any]] = []
    all_node_events: dict[str, list[dict[str, int]]] = {}

    total_tokens_consumed_overall = 0
    total_calls_overall = 0

    for idx, item in enumerate(dataset, start=1):
        q = item["query"]
        target = item["target_path"]
        cid = item.get("id", f"case-{idx}")

        thread_id = f"token-bench-{uuid.uuid4().hex[:8]}"
        tracker = NodeTokenCallbackHandler()

        # Exponential backoff on HTTP 429
        max_retries = 5
        res = None
        for attempt in range(max_retries):
            try:
                res = await agent.ainvoke(
                    {"messages": [HumanMessage(content=q)]},
                    config={"callbacks": [tracker], "configurable": {"thread_id": thread_id}},
                )
                break
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate_limit" in err_str or "tpm" in err_str:
                    wait_s = (2**attempt) * 2.0 + random.uniform(0.5, 1.5)
                    print(
                        f"  [429 Rate Limit] Backing off for {wait_s:.1f}s on query {idx}/{len(dataset)}..."
                    )
                    await asyncio.sleep(wait_s)
                else:
                    print(f"  [Error] Query failed: {e}")
                    break

        if res is None:
            continue

        actual_path = determine_actual_path(res)
        q_inp = tracker.total_input_tokens
        q_out = tracker.total_output_tokens
        q_reas = tracker.total_reasoning_tokens
        q_tot = q_inp + q_out
        q_calls = tracker.total_llm_calls

        total_tokens_consumed_overall += q_tot
        total_calls_overall += q_calls

        # Record node events
        for n_name, n_data in tracker.node_stats.items():
            if n_name not in all_node_events:
                all_node_events[n_name] = []
            all_node_events[n_name].append(n_data)

        rec = {
            "id": cid,
            "query": q,
            "target_path": target,
            "actual_path": actual_path,
            "llm_calls": q_calls,
            "input_tokens": q_inp,
            "output_tokens": q_out,
            "reasoning_tokens": q_reas,
            "total_tokens": q_tot,
            "nodes": tracker.node_stats,
        }
        records.append(rec)

        print(
            f"[{idx:02d}/{len(dataset)}] [{actual_path.upper():12s}] calls={q_calls} | "
            f"tokens={q_tot:4d} (in={q_inp:4d}, out={q_out:3d}, reas={q_reas:3d}) | {q[:45]}"
        )

        if query_delay > 0:
            await asyncio.sleep(query_delay)

    # Compute Path Statistics
    path_names = ["kb", "direct", "tool", "refuse", "escalate", "insufficient"]
    path_stats: dict[str, Any] = {}

    for p in path_names:
        p_recs = [r for r in records if r["actual_path"] == p]
        if not p_recs:
            path_stats[p] = {"n": 0, "status": "no_samples"}
            continue

        n = len(p_recs)
        calls = sorted(r["llm_calls"] for r in p_recs)
        inps = sorted(r["input_tokens"] for r in p_recs)
        outs = sorted(r["output_tokens"] for r in p_recs)
        reass = sorted(r["reasoning_tokens"] for r in p_recs)
        tots = sorted(r["total_tokens"] for r in p_recs)

        path_stats[p] = {
            "n": n,
            "llm_calls_mean": round(sum(calls) / n, 2),
            "llm_calls_p95": int(pct(calls, 0.95)),
            "input_tokens_mean": round(sum(inps) / n, 1),
            "input_tokens_p95": int(pct(inps, 0.95)),
            "output_tokens_mean": round(sum(outs) / n, 1),
            "output_tokens_p95": int(pct(outs, 0.95)),
            "reasoning_tokens_mean": round(sum(reass) / n, 1),
            "reasoning_tokens_p95": int(pct(reass, 0.95)),
            "total_tokens_mean": round(sum(tots) / n, 1),
            "total_tokens_p95": int(pct(tots, 0.95)),
        }

    # Compute Node Statistics
    node_stats: dict[str, Any] = {}
    total_queries = len(records)
    for n_name, events in all_node_events.items():
        if not events:
            continue
        calls = [e["calls"] for e in events]
        inps = sorted(e["input_tokens"] for e in events)
        outs = sorted(e["output_tokens"] for e in events)
        reass = sorted(e["reasoning_tokens"] for e in events)
        tots = sorted(e["total_tokens"] for e in events)
        n = len(events)

        node_stats[n_name] = {
            "total_invocations": sum(calls),
            "calls_per_query_mean": round(sum(calls) / max(1, total_queries), 2),
            "input_tokens_mean": round(sum(inps) / n, 1),
            "input_tokens_p95": int(pct(inps, 0.95)),
            "output_tokens_mean": round(sum(outs) / n, 1),
            "output_tokens_p95": int(pct(outs, 0.95)),
            "reasoning_tokens_mean": round(sum(reass) / n, 1),
            "reasoning_tokens_p95": int(pct(reass, 0.95)),
            "total_tokens_mean": round(sum(tots) / n, 1),
            "total_tokens_p95": int(pct(tots, 0.95)),
        }

    # Overall Summary
    all_tots = sorted(r["total_tokens"] for r in records)
    all_calls = sorted(r["llm_calls"] for r in records)
    overall_summary = {
        "n_total": len(records),
        "total_tokens_consumed": total_tokens_consumed_overall,
        "total_llm_calls": total_calls_overall,
        "llm_calls_per_query_mean": round(sum(all_calls) / max(1, len(records)), 2),
        "tokens_per_query_mean": round(sum(all_tots) / max(1, len(records)), 1),
        "tokens_per_query_p95": int(pct(all_tots, 0.95)),
    }

    report = {
        "git_sha": get_git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_type": f"stratified_dev_{limit}",
        "overall": overall_summary,
        "paths": path_stats,
        "nodes": node_stats,
        "records": records,
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print Formatted Markdown Table
    print("\n" + "=" * 90)
    print("TOKEN & CALL BENCHMARK TABLE ACROSS PATHS (evals/tokens.json)")
    print("=" * 90)
    print(
        f"{'Path':<14} | {'n':<4} | {'LLM Calls (mean/p95)':<20} | {'Input Tok (mean/p95)':<20} | {'Output Tok (mean/p95)':<21} | {'Reasoning (mean/p95)'}"
    )
    print("-" * 105)
    for p in path_names:
        st = path_stats.get(p, {})
        if not st or st.get("n", 0) == 0:
            print(f"{p:<14} | 0    | {'-':<20} | {'-':<20} | {'-':<21} | -")
            continue
        c_str = f"{st['llm_calls_mean']:.2f} / {st['llm_calls_p95']}"
        i_str = f"{st['input_tokens_mean']:.1f} / {st['input_tokens_p95']}"
        o_str = f"{st['output_tokens_mean']:.1f} / {st['output_tokens_p95']}"
        r_str = f"{st['reasoning_tokens_mean']:.1f} / {st['reasoning_tokens_p95']}"
        print(f"{p:<14} | {st['n']:<4} | {c_str:<20} | {i_str:<20} | {o_str:<21} | {r_str}")
    print("-" * 105)
    tot_str = f"{overall_summary['tokens_per_query_mean']:.1f} / {overall_summary['tokens_per_query_p95']}"
    call_str = f"{overall_summary['llm_calls_per_query_mean']:.2f}"
    print(
        f"{'OVERALL':<14} | {overall_summary['n_total']:<4} | {call_str:<20} | {'Mean/p95 Total: ' + tot_str:<43} |"
    )
    print("=" * 90)

    print("\n" + "=" * 90)
    print("TOKEN BENCHMARK TABLE PER NODE")
    print("=" * 90)
    print(
        f"{'Node Name':<20} | {'Calls/Q':<8} | {'Input Tok (mean/p95)':<20} | {'Output Tok (mean/p95)':<21} | {'Reasoning (mean/p95)'}"
    )
    print("-" * 105)
    for n_name, st in sorted(node_stats.items()):
        c_str = f"{st['calls_per_query_mean']:.2f}"
        i_str = f"{st['input_tokens_mean']:.1f} / {st['input_tokens_p95']}"
        o_str = f"{st['output_tokens_mean']:.1f} / {st['output_tokens_p95']}"
        r_str = f"{st['reasoning_tokens_mean']:.1f} / {st['reasoning_tokens_p95']}"
        print(f"{n_name:<20} | {c_str:<8} | {i_str:<20} | {o_str:<21} | {r_str}")
    print("=" * 90)

    print(f"\nTotal tokens used: {total_tokens_consumed_overall}")
    print(f"Results written to: {output_file}\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run token & LLM call benchmark on stratified sample."
    )
    parser.add_argument("--limit", type=int, default=40, help="Number of queries to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output", default="evals/tokens.json", help="Output JSON path")
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between queries to prevent TPM rate limits",
    )
    args = parser.parse_args()

    asyncio.run(
        run_token_benchmark(
            limit=args.limit, seed=args.seed, output_file=args.output, query_delay=args.delay
        )
    )


if __name__ == "__main__":
    main()
