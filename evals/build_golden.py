"""Build the in-KB portion of the golden set from rows NOT in the index."""

import collections
import json
import random

from datasets import load_dataset

from agenticrag.config import get_settings
from agenticrag.constants import DATASET_NAME

N_TOTAL = 100  # in-KB cases (about 60% of a 170-case set)
random.seed(42)

s = get_settings()
ds = load_dataset(DATASET_NAME, split="train").shuffle(seed=s.kb_seed)
n = min(len(ds), s.kb_sample_size)
indexed, pool = ds.select(range(n)), ds.select(range(n, len(ds)))
kb_intents = {r["intent"] for r in indexed}

by_intent = collections.defaultdict(list)
for r in pool:
    if r["intent"] in kb_intents:
        by_intent[r["intent"]].append(r["instruction"])

# Round-robin across intents so every intent is represented
intents = sorted(by_intent)
rows, i = [], 0
while len(rows) < N_TOTAL:
    intent = intents[i % len(intents)]
    q = random.choice(by_intent[intent])
    rows.append(
        {"id": f"hp-{len(rows) + 1:03d}", "query": q, "expected_intent": intent, "type": "in_kb"}
    )
    i += 1

# 70/30 dev/test split, fixed once
random.shuffle(rows)
for j, r in enumerate(rows):
    r["split"] = "test" if j % 10 >= 7 else "dev"

with open("evals/golden_in_kb.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"{len(rows)} cases, {len(kb_intents)} intents in KB")
