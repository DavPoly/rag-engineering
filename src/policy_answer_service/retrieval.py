"""
Retrieval ladder: keyword (BM25) -> dense vector -> hybrid (RRF) -> + reranker.

Implement and measure each rung in order on the golden set (recall@5, MRR)
before climbing to the next. Stop when the gain stops justifying the
complexity, and record that call in DECISIONS.md.

TODO:
- class BM25Retriever
- class VectorRetriever (embedding model swappable via config)
- def reciprocal_rank_fusion(ranked_lists, k=60) -> list
- class RerankedRetriever (cross-encoder, added last)
- def evaluate(retriever, golden_set) -> dict[str, float]  # recall@5, MRR
"""