import json
import random
import sys

from agenticrag.clients import get_vectorstore
from agenticrag.config import get_settings

K = 4
PATH = sys.argv[1] if len(sys.argv) > 1 else "evals/golden.jsonl"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "dev"
THRESH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0  # gate off until baseline exists


def boot_ci(vals, n_boot=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n_boot))
    return round(means[int(0.025 * n_boot)], 3), round(means[int(0.975 * n_boot)], 3)


def pct(sorted_vals, p):
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * p))]


def main():
    vs, ns = get_vectorstore(), get_settings().namespace
    hit, rr, prec, in_scores, oos_scores = [], [], [], [], []
    with open(PATH, encoding="utf-8-sig") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for row in rows:
        if "query" not in row or row.get("split") != SPLIT:
            continue
        res = vs.similarity_search_with_score(row["query"], k=K, namespace=ns)
        top = max((s for _, s in res), default=0.0)
        if row.get("expected_intent") is None:
            oos_scores.append(top)
            continue
        rel = [d.metadata.get("intent") == row["expected_intent"] for d, _ in res]
        hit.append(float(any(rel)))
        prec.append(sum(rel) / K)
        rr.append(next((1 / (i + 1) for i, r in enumerate(rel) if r), 0.0))
        in_scores.append(top)

    def summ(v):
        return {"mean": round(sum(v) / len(v), 3), "ci95": boot_ci(v), "n": len(v)}

    out = {
        "split": SPLIT,
        "recall@4": summ(hit),
        "precision@4": summ(prec),
        "mrr": summ(rr),
        "top_score_in_kb_p10": round(pct(sorted(in_scores), 0.10), 3),
        "top_score_oos_p90": round(pct(sorted(oos_scores), 0.90), 3) if oos_scores else None,
        "n_oos": len(oos_scores),
    }
    print(json.dumps(out, indent=2))
    assert out["recall@4"]["mean"] >= THRESH, "recall@4 below threshold"


if __name__ == "__main__":
    main()
