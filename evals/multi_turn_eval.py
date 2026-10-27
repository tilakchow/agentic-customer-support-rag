"""Multi-turn retrieval evaluation measuring recall@4 with and without query condensation."""

import asyncio
import json
import random
import sys
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agenticrag.nodes import condense_query
from agenticrag.retrieval import retrieve_kb

PATH = sys.argv[1] if len(sys.argv) > 1 else "evals/golden.jsonl"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "dev"


def boot_ci(vals: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n_boot))
    return round(means[int(0.025 * n_boot)], 3), round(means[int(0.975 * n_boot)], 3)


async def run_multi_turn_eval(path: str = PATH, split: str = SPLIT) -> dict[str, Any]:
    with open(path, encoding="utf-8-sig") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    mt_rows = [r for r in rows if r.get("split") == split and "turns" in r]
    hits_without: list[float] = []
    hits_with: list[float] = []
    details: list[dict[str, Any]] = []

    for r in mt_rows:
        turns: list[str] = r["turns"]
        expected = r["expected_intent"]
        last_turn = turns[-1]

        # 1. Without condensing: retrieve using only the raw final turn
        res_without = retrieve_kb(last_turn, k=4)
        intents_without = [d.metadata.get("intent") for d in res_without.docs]
        hit_without = float(expected in intents_without)
        hits_without.append(hit_without)

        # 2. With condensing: build message history and condense query
        msgs: list[HumanMessage | AIMessage] = []
        for turn in turns[:-1]:
            msgs.append(HumanMessage(content=turn))
            msgs.append(AIMessage(content="I can assist you with that."))
        msgs.append(HumanMessage(content=last_turn))

        state: dict[str, Any] = {"messages": msgs, "current_query": last_turn}
        condensed_res = await condense_query(state)  # type: ignore[arg-type]
        condensed_q = condensed_res.get("current_query", last_turn)

        # Retrieve using condensed query
        res_with = retrieve_kb(condensed_q, k=4)
        intents_with = [d.metadata.get("intent") for d in res_with.docs]
        hit_with = float(expected in intents_with)
        hits_with.append(hit_with)

        details.append(
            {
                "id": r.get("id"),
                "turns": turns,
                "expected_intent": expected,
                "last_turn": last_turn,
                "condensed_query": condensed_q,
                "hit_without": bool(hit_without),
                "hit_with": bool(hit_with),
            }
        )

    def summ(v: list[float]) -> dict[str, Any]:
        mean_val = round(sum(v) / len(v), 3) if v else 0.0
        return {"mean": mean_val, "ci95": boot_ci(v), "n": len(v)}

    return {
        "split": split,
        "n": len(mt_rows),
        "recall@4_without_condensing": summ(hits_without),
        "recall@4_with_condensing": summ(hits_with),
        "details": details,
    }


def main() -> None:
    results = asyncio.run(run_multi_turn_eval(PATH, SPLIT))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
