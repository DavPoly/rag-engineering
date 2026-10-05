import json
import time

from policy_answer_service.config import get_settings
from policy_answer_service.retrieval import (
    BM25Retriever,
    HybridRetriever,
    VectorRetriever,
    evaluate,
    load_chunks,
)

settings = get_settings()
chunks = load_chunks(settings.data_dir / "chunks.jsonl")
golden_set = [json.loads(line) for line in open(settings.golden_set_path)]

bm25 = BM25Retriever(chunks)
vector = VectorRetriever(chunks, settings.embedding_model_api)
hybrid = HybridRetriever(bm25, vector)

# One batched request for all golden-set questions; Vector and Hybrid reuse it.
vector.embed_queries([item["question"] for item in golden_set])

for name, retriever in [("BM25", bm25), ("Vector", vector), ("Hybrid", hybrid)]:
    start = time.perf_counter()
    results = evaluate(retriever, golden_set)
    elapsed = (time.perf_counter() - start) / len(golden_set)
    print(
        f"{name}: recall@5={results['recall@5']:.2f}  MRR={results['mrr']:.2f}  "
        f"avg_latency={elapsed * 1000:.0f}ms"
    )
