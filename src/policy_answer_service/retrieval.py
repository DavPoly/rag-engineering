import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

from policy_answer_service.config import get_settings


def load_chunks(chunks_path: Path) -> list[dict]:
    with open(chunks_path) as f:
        return [json.loads(line) for line in f]


# --------------------------------------------------------------------------
# Rung 1 — BM25 (keyword)
# --------------------------------------------------------------------------


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        # rank_bm25 divides by the corpus size, so an empty corpus needs its own path.
        self.bm25 = None
        if chunks:
            self.tokenized = [c["text"].lower().split() for c in chunks]
            self.bm25 = BM25Okapi(self.tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self.chunks, scores, strict=True), key=lambda x: -x[1])
        return [c for c, _ in ranked[:top_k]]


# --------------------------------------------------------------------------
# Embedding cache — avoid re-embedding the same chunks twice
# --------------------------------------------------------------------------


def _cache_key(chunks: list[dict], model_name: str) -> str:
    # Version tag: bump when the windowing scheme changes so stale entries are ignored.
    content = "".join(c["text"] for c in chunks) + model_name + f"|windows={MAX_WORDS_PER_WINDOW}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _load_cached_embeddings(
    chunks: list[dict], model_name: str, cache_dir: Path
) -> tuple[np.ndarray, np.ndarray] | None:
    cache_path = cache_dir / f"emb_{_cache_key(chunks, model_name)}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None


def _save_cached_embeddings(
    chunks: list[dict],
    model_name: str,
    cache_dir: Path,
    embeddings: tuple[np.ndarray, np.ndarray],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"emb_{_cache_key(chunks, model_name)}.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings, f)


# --------------------------------------------------------------------------
# Rung 2 — Dense vector search via the OpenRouter embeddings endpoint
# --------------------------------------------------------------------------


MAX_EMBEDDING_BATCH = 128  # the OpenRouter embedding endpoint rejects larger inputs with a 400
# The endpoint caps each input at 512 tokens. Measured on these policies: 531 words -> 922 tokens,
# so 200 words stays comfortably under the cap (~350 tokens).
MAX_WORDS_PER_WINDOW = 200


def _embed_texts(client: OpenAI, model_name: str, texts: list[str]) -> np.ndarray:
    """Embed texts in API-sized batches, returning unnormalized vectors in input order."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), MAX_EMBEDDING_BATCH):
        batch = texts[start : start + MAX_EMBEDDING_BATCH]
        response = client.embeddings.create(model=model_name, input=batch)
        vectors.extend(d.embedding for d in response.data)
    return np.array(vectors)


def _split_into_windows(text: str, max_words: int = MAX_WORDS_PER_WINDOW) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


class VectorRetriever:
    def __init__(self, chunks: list[dict], model_name: str):
        self.chunks = chunks
        self.model_name = model_name
        settings = get_settings()
        self.client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
        cache_dir = settings.data_dir / "embedding_cache"

        cached = _load_cached_embeddings(chunks, model_name, cache_dir)
        if cached is not None:
            self.embeddings, self._owners = cached
        else:
            windows: list[str] = []
            owners: list[int] = []
            for chunk_index, chunk in enumerate(chunks):
                for window in _split_into_windows(chunk["text"]):
                    windows.append(window)
                    owners.append(chunk_index)
            raw = _embed_texts(self.client, model_name, windows)
            self.embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)
            self._owners = np.array(owners)
            _save_cached_embeddings(chunks, model_name, cache_dir, (self.embeddings, self._owners))
        self._query_embeddings: dict[str, np.ndarray] = {}

    def embed_queries(self, queries: list[str]) -> None:
        """Embed every not-yet-seen query, one API request per MAX_EMBEDDING_BATCH questions."""
        missing = [q for q in dict.fromkeys(queries) if q not in self._query_embeddings]
        if not missing:
            return
        raw = _embed_texts(self.client, self.model_name, missing)
        for query, vector in zip(missing, raw, strict=True):
            self._query_embeddings[query] = vector / np.linalg.norm(vector)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        self.embed_queries([query])
        window_scores = self.embeddings @ self._query_embeddings[query]
        # A chunk scores as its best-matching window.
        chunk_scores = np.full(len(self.chunks), -np.inf)
        np.maximum.at(chunk_scores, self._owners, window_scores)
        ranked = sorted(zip(self.chunks, chunk_scores, strict=True), key=lambda x: -x[1])
        return [c for c, _ in ranked[:top_k]]


# --------------------------------------------------------------------------
# Rung 3 — Hybrid via Reciprocal Rank Fusion (weighted)
# --------------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]], weights: list[float] | None = None, k: int = 60
) -> list[dict]:
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}
    for w, ranked_list in zip(weights, ranked_lists, strict=True):
        for rank, chunk in enumerate(ranked_list):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0) + w / (k + rank + 1)
            chunk_lookup[cid] = chunk
    ranked_ids = sorted(scores, key=lambda cid: -scores[cid])
    return [chunk_lookup[cid] for cid in ranked_ids]


class HybridRetriever:
    def __init__(
        self, bm25: BM25Retriever, vector: VectorRetriever, weights: list[float] | None = None
    ):
        self.bm25 = bm25
        self.vector = vector
        self.weights = weights or [1.0, 1.0]

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        bm25_results = self.bm25.retrieve(query, top_k=20)
        vector_results = self.vector.retrieve(query, top_k=20)
        fused = reciprocal_rank_fusion([bm25_results, vector_results], weights=self.weights, k=20)
        return fused[:top_k]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def evaluate(retriever, golden_set: list[dict], top_k: int = 5) -> dict:
    recalls, reciprocal_ranks = [], []
    for item in golden_set:
        if item["category"] == "unanswerable":
            continue  # recall@k/MRR don't apply; handled by the refusal path
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
