import json, time
from policy_answer_service.config import get_settings
from policy_answer_service.retrieval import (
    load_chunks, BM25Retriever, VectorRetriever, HybridRetriever,
    RerankedRetriever, evaluate,
)

settings = get_settings()
chunks = load_chunks(settings.data_dir / "chunks.jsonl")
golden_set = [json.loads(l) for l in open(settings.golden_set_path)]

bm25 = BM25Retriever(chunks)
vector = VectorRetriever(chunks, settings.embedding_model_name)
hybrid = HybridRetriever(bm25, vector)
vector_reranked = RerankedRetriever(vector)
reranked = RerankedRetriever(hybrid)

for name, retriever in [("BM25", bm25), ("Vector", vector), ("Hybrid", hybrid), ("Vector+Reranker", vector_reranked), ("Hybrid+Reranker", reranked)]:
    start = time.perf_counter()
    results = evaluate(retriever, golden_set)
    elapsed = (time.perf_counter() - start) / len(golden_set)
    print(f"{name}: recall@5={results['recall@5']:.2f}  MRR={results['mrr']:.2f}  avg_latency={elapsed*1000:.0f}ms")

