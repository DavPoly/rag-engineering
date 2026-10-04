import numpy as np

from policy_answer_service.config import get_settings
from policy_answer_service.ingestion import chunk_document, load_documents
from policy_answer_service.retrieval import (
    BM25Retriever,
    VectorRetriever,
    evaluate,
    reciprocal_rank_fusion,
)


def _policy_chunks() -> list[dict]:
    settings = get_settings()
    docs = load_documents(settings.policy_docs_dir)
    return [c.model_dump() for d in docs for c in chunk_document(d)]


def test_bm25_finds_exact_product_code():
    retriever = BM25Retriever(_policy_chunks())
    top = retriever.retrieve("WR-1187-C", top_k=1)[0]
    assert top["chunk_id"] == "warranty-coverage-by-product-code#warranty-reference-table"


def test_bm25_empty_corpus_returns_nothing():
    retriever = BM25Retriever([])
    assert retriever.retrieve("refund", top_k=5) == []


def test_vector_retriever_returns_top_k(mocker):
    def fake_encode(texts, normalize_embeddings=True):
        if isinstance(texts, str):
            return np.array([1.0, 0.0])
        return np.array([[1.0, 0.0] if "refund" in t else [0.0, 1.0] for t in texts])

    model = mocker.patch("policy_answer_service.retrieval.SentenceTransformer").return_value
    model.encode.side_effect = fake_encode

    chunks = [
        {"chunk_id": "a", "text": "refund policy"},
        {"chunk_id": "b", "text": "shipping times"},
        {"chunk_id": "c", "text": "refund timing"},
    ]
    retriever = VectorRetriever(chunks, model_name="unused")
    results = retriever.retrieve("refund", top_k=2)

    assert len(results) == 2
    assert {r["chunk_id"] for r in results} == {"a", "c"}


def test_reciprocal_rank_fusion_combines_lists():
    list_a = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    list_b = [{"chunk_id": "b"}, {"chunk_id": "c"}]
    fused = reciprocal_rank_fusion([list_a, list_b])

    assert {c["chunk_id"] for c in fused} == {"a", "b", "c"}
    assert fused[0]["chunk_id"] == "b"


class _StubRetriever:
    def __init__(self, results: dict[str, list[str]]):
        self.results = results

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return [{"chunk_id": cid} for cid in self.results[query]]


def test_evaluate_computes_recall_and_mrr():
    golden_set = [
        {"question": "q1", "category": "answerable", "expected_chunks": ["a"]},
        {"question": "q2", "category": "answerable", "expected_chunks": ["b"]},
        {"question": "q3", "category": "unanswerable", "expected_chunks": []},
    ]
    retriever = _StubRetriever({"q1": ["x", "a"], "q2": ["c"]})

    metrics = evaluate(retriever, golden_set, top_k=5)

    assert metrics["recall@5"] == 0.5
    assert metrics["mrr"] == 0.25
