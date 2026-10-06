"""
Tests for the /ask endpoint.

The pipeline dependency is replaced with a fake, so these tests make no
embedding or LLM calls.
"""

from fastapi.testclient import TestClient

from policy_answer_service.api import AnswerPipeline, app, get_pipeline
from policy_answer_service.generation import ModelReplyError
from policy_answer_service.models import Citation, ModelResponse

client = TestClient(app)


class FakePipeline:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.questions: list[str] = []

    def answer(self, question: str) -> ModelResponse:
        self.questions.append(question)
        return self.response


class FakeRetriever:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        self.calls.append((query, top_k))
        return self.chunks[:top_k]


def _override(fake: FakePipeline) -> None:
    app.dependency_overrides[get_pipeline] = lambda: fake


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_ask_returns_200_and_valid_schema():
    response_model = ModelResponse(
        answer="You have 45 days.",
        citations=[
            Citation(chunk_id="returns-policy-v2#return-window", document_title="Returns v2")
        ],
        confidence=0.9,
        answerable=True,
    )
    fake = FakePipeline(response_model)
    _override(fake)

    response = client.post("/ask", json={"question": "how many days do I have to return"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "You have 45 days."
    assert body["answerable"] is True
    assert body["confidence"] == 0.9
    assert body["citations"] == [
        {"chunk_id": "returns-policy-v2#return-window", "document_title": "Returns v2"}
    ]
    assert fake.questions == ["how many days do I have to return"]


def test_ask_returns_refusal_for_unanswerable_question():
    refusal = ModelResponse(
        answer="I couldn't find this in Northwind's policy documents.",
        citations=[],
        confidence=0.2,
        answerable=False,
    )
    _override(FakePipeline(refusal))

    response = client.post("/ask", json={"question": "do you price match amazon"})

    assert response.status_code == 200
    assert response.json()["answerable"] is False
    assert response.json()["citations"] == []


def test_ask_with_empty_question_returns_422():
    _override(
        FakePipeline(ModelResponse(answer="", citations=[], confidence=0.0, answerable=False))
    )

    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422


def test_pipeline_retrieves_top_k_then_generates(mocker):
    chunks = [
        {"chunk_id": "a#x", "document_title": "A", "effective_date": "2025-01-01", "text": "t"}
    ]
    retriever = FakeRetriever(chunks)
    generate = mocker.patch(
        "policy_answer_service.api.generate_answer",
        return_value=ModelResponse(answer="ok", citations=[], confidence=0.5, answerable=False),
    )
    pipeline = AnswerPipeline(retriever=retriever, client="client", model="m", top_k=3)

    result = pipeline.answer("what is the policy")

    assert retriever.calls == [("what is the policy", 3)]
    generate.assert_called_once_with("client", "m", "what is the policy", chunks)
    assert result.answer == "ok"


def test_ask_returns_502_when_model_reply_is_unusable():
    class BrokenPipeline:
        def answer(self, question: str) -> ModelResponse:
            raise ModelReplyError("Model reply was not JSON (finish_reason=length): ''")

    app.dependency_overrides[get_pipeline] = lambda: BrokenPipeline()

    response = client.post("/ask", json={"question": "how many days"})

    assert response.status_code == 502
    assert "unusable reply" in response.json()["detail"]
