import json

from agenticrag.clients import get_vectorstore
from agenticrag.config import get_settings

s = get_settings()
vs = get_vectorstore()
intents = sorted(
    {
        json.loads(line)["expected_intent"]
        for line in open("evals/golden.jsonl", encoding="utf-8-sig")
        if line.strip() and ('"expected_intent": "' in line or '"expected_intent":"' in line)
    }
)
present = []
for i in intents:
    r = vs.similarity_search("help", k=1, namespace=s.namespace, filter={"intent": i})
    print(("PRESENT " if r else "MISSING ") + i)
    if r:
        present.append(i)
print(f"\n{len(present)} of {len(intents)} intents in the index")
