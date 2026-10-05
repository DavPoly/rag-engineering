from types import SimpleNamespace

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


def _fake_embeddings_create(model, input):
    def vec(text: str) -> list[float]:
        return [1.0, 0.0] if "refund" in text else [0.0, 1.0]

    return SimpleNamespace(data=[SimpleNamespace(embedding=vec(t)) for t in input])


def test_vector_retriever_returns_top_k(mocker):
    client = mocker.patch("policy_answer_service.retrieval.OpenAI").return_value
    client.embeddings.create.side_effect = _fake_embeddings_create

    chunks = [
        {"chunk_id": "a", "text": "refund policy"},
        {"chunk_id": "b", "text": "shipping times"},
        {"chunk_id": "c", "text": "refund timing"},
    ]
    retriever = VectorRetriever(chunks, model_name="unused")
    results = retriever.retrieve("refund", top_k=2)

    assert len(results) == 2
    assert {r["chunk_id"] for r in results} == {"a", "c"}


def test_embed_queries_batches_into_one_request_and_reuses_it(mocker):
    client = mocker.patch("policy_answer_service.retrieval.OpenAI").return_value
    client.embeddings.create.side_effect = _fake_embeddings_create

    chunks = [
        {"chunk_id": "a", "text": "refund policy"},
        {"chunk_id": "b", "text": "shipping times"},
    ]
    retriever = VectorRetriever(chunks, model_name="unused")
    calls_before = client.embeddings.create.call_count

    questions = ["refund rules", "shipping times", "refund timing"]
    retriever.embed_queries(questions)
    for question in questions:
        retriever.retrieve(question, top_k=1)
    retriever.retrieve("refund rules", top_k=1)  # repeat query, e.g. Hybrid reusing Vector

    assert client.embeddings.create.call_count - calls_before == 1


def test_vector_retriever_splits_large_batches_under_api_limit(mocker):
    batch_sizes: list[int] = []

    def capped_create(model, input):
        batch_sizes.append(len(input))
        if len(input) > 128:
            raise ValueError("Too big: expected array to have <=128 items")
        return _fake_embeddings_create(model, input)

    client = mocker.patch("policy_answer_service.retrieval.OpenAI").return_value
    client.embeddings.create.side_effect = capped_create
    mocker.patch("policy_answer_service.retrieval._load_cached_embeddings", return_value=None)
    mocker.patch("policy_answer_service.retrieval._save_cached_embeddings")

    chunks = [{"chunk_id": f"c{i}", "text": f"refund note {i}"} for i in range(200)]
    retriever = VectorRetriever(chunks, model_name="unused-large-batch")

    assert max(batch_sizes) <= 128
    assert retriever.embeddings.shape[0] == 200


def test_vector_retriever_scores_long_chunks_by_best_window(mocker):
    def word_limited_create(model, input):
        if any(len(text.split()) > 200 for text in input):
            raise ValueError("Embedding input exceeds the model maximum")
        return _fake_embeddings_create(model, input)

    client = mocker.patch("policy_answer_service.retrieval.OpenAI").return_value
    client.embeddings.create.side_effect = word_limited_create
    mocker.patch("policy_answer_service.retrieval._load_cached_embeddings", return_value=None)
    mocker.patch("policy_answer_service.retrieval._save_cached_embeddings")

    # "refund" only appears in the second window of the long chunk.
    long_text = " ".join(["filler"] * 250 + ["refund"] * 10)
    chunks = [
        {"chunk_id": "long", "text": long_text},
        {"chunk_id": "short", "text": "shipping times"},
    ]
    retriever = VectorRetriever(chunks, model_name="unused-windows")
    results = retriever.retrieve("refund", top_k=1)

    assert results[0]["chunk_id"] == "long"


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
