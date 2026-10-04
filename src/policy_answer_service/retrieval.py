"""
Retrieval ladder: keyword (BM25) -> dense vector -> hybrid (RRF) -> + reranker.

Implement and measure each rung in order on the golden set (recall@5, MRR)
before climbing to the next. Stop when the gain stops justifying the
complexity, and record that call in DECISIONS.md.
"""

import json
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


def load_chunks(chunks_path: Path) -> list[dict]:
    with open(chunks_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([c["text"].lower().split() for c in chunks]) if chunks else None

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self.chunks, scores, strict=True), key=lambda x: -x[1])
        return [c for c, _ in ranked[:top_k]]


class VectorRetriever:
    def __init__(self, chunks: list[dict], model_name: str):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode([c["text"] for c in chunks], normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_emb = self.model.encode(query, normalize_embeddings=True)
        scores = self.embeddings @ query_emb  # cosine sim, since normalized
        ranked = sorted(zip(self.chunks, scores, strict=True), key=lambda x: -x[1])
        return [c for c, _ in ranked[:top_k]]


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
            chunk_lookup[cid] = chunk
    ranked_ids = sorted(scores, key=lambda cid: -scores[cid])
    return [chunk_lookup[cid] for cid in ranked_ids]


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, vector: VectorRetriever):
        self.bm25 = bm25
        self.vector = vector

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        bm25_results = self.bm25.retrieve(query, top_k=20)
        vector_results = self.vector.retrieve(query, top_k=20)
        fused = reciprocal_rank_fusion([bm25_results, vector_results])
        return fused[:top_k]


class RerankedRetriever:
    def __init__(
        self,
        hybrid: HybridRetriever,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.hybrid = hybrid
        self.reranker = CrossEncoder(reranker_model)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        candidates = self.hybrid.retrieve(query, top_k=15)
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: -x[1])
        return [c for c, _ in ranked[:top_k]]


def evaluate(retriever, golden_set: list[dict], top_k: int = 5) -> dict:
    recalls, reciprocal_ranks = [], []
    for item in golden_set:
        if item["category"] == "unanswerable":
            continue  # recall@k/MRR don't apply; handle refusal separately
        expected = set(item["expected_chunks"])
        retrieved = retriever.retrieve(item["question"], top_k=top_k)
        retrieved_ids = [c["chunk_id"] for c in retrieved]
        hit = bool(expected & set(retrieved_ids))
        recalls.append(hit)
        rank = next((i + 1 for i, cid in enumerate(retrieved_ids) if cid in expected), None)
        reciprocal_ranks.append(1 / rank if rank else 0)
    return {
        "recall@5": sum(recalls) / len(recalls),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }
