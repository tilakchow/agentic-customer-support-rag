from agenticrag.clients import get_embeddings, get_vectorstore
from agenticrag.config import get_settings

s = get_settings()
print(
    "index:",
    s.index_name,
    "| namespace:",
    repr(s.namespace),
    "| embedding_model:",
    s.embedding_model,
)

vs = get_vectorstore()
stats = vs._index.describe_index_stats()
print("index dim:", stats.get("dimension"))
print("namespaces:", {k: v.get("vector_count") for k, v in stats.get("namespaces", {}).items()})
print("embedding dim:", len(get_embeddings().embed_query("test")))

for q in [
    "How do I track my order?",
    "I want to cancel my order",
    "how can I change my shipping address",
]:
    print("\nQ:", q)
    for d, sc in vs.similarity_search_with_score(q, k=4, namespace=s.namespace):
        print(f"  {sc:.3f} | {d.metadata.get('intent')} | {d.page_content[:70]!r}")
