"""Faithfulness and groundedness evaluation runner using claim verification and LLM-as-a-judge.

Measures the proportion of factual statements in generated answers that are strictly supported
by the retrieved knowledge base chunks. Gracefully handles absence of JUDGE_API_KEY.
"""

import asyncio
import json
import os
from typing import Any

from langchain_core.messages import HumanMessage

from agenticrag.graph import create_agent
from agenticrag.nodes import check_claim_support


async def run_faithfulness_eval(sample_limit: int = 15) -> dict[str, Any]:
    """Runs end-to-end faithfulness and claim-support verification across in-KB dev queries."""
    golden_path = os.path.join(os.path.dirname(__file__), "golden.jsonl")
    if not os.path.exists(golden_path):
        print(f"Golden dataset not found at {golden_path}")
        return {"status": "skipped", "reason": "golden.jsonl missing"}

    cases = []
    with open(golden_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("split") == "dev":
                cases.append(item)

    if sample_limit and len(cases) > sample_limit:
        cases = cases[:sample_limit]

    print(f"[Faithfulness Eval] Running evaluation on {len(cases)} dev cases...")

    app = create_agent()
    results = []
    supported_count = 0
    total_evaluated = 0

    for idx, c in enumerate(cases):
        query = c["query"]
        thread_id = f"faith-eval-{idx}"

        try:
            res = await app.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": thread_id}},
            )

            messages = res.get("messages", [])
            last_msg = messages[-1].content if messages else ""
            source = res.get("source_used", "")

            # If the response came from private KB, evaluate faithfulness against retrieved context
            if source == "private_kb":
                docs = res.get("kb_docs", [])
                context = "\n\n".join(d.page_content for d in docs)
                check = await check_claim_support(
                    question=query, context=context, response=str(last_msg)
                )

                total_evaluated += 1
                if check.supported:
                    supported_count += 1

                results.append(
                    {
                        "query": query,
                        "source": source,
                        "supported": check.supported,
                        "reason": check.reason,
                        "unsupported_claims": check.unsupported_claims,
                    }
                )
                status_str = "PASS" if check.supported else "FAIL"
                print(f"[{status_str}] Query {idx + 1}/{len(cases)}: {query[:45]}...")
            else:
                # Direct answers or refusals without KB context
                results.append(
                    {
                        "query": query,
                        "source": source,
                        "supported": True,
                        "reason": "Non-KB or direct response",
                    }
                )
        except Exception as e:
            print(f"[ERROR] Query {idx + 1} failed: {e}")

    faithfulness_score = (supported_count / total_evaluated) if total_evaluated > 0 else 1.0
    print(
        f"\n[Faithfulness Summary] Grounded: {supported_count}/{total_evaluated} ({faithfulness_score:.4f})"
    )

    report = {
        "total_cases": len(cases),
        "evaluated_kb_cases": total_evaluated,
        "grounded_count": supported_count,
        "faithfulness_score": faithfulness_score,
        "results": results,
    }
    return report


if __name__ == "__main__":
    report = asyncio.run(run_faithfulness_eval(sample_limit=10))
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
