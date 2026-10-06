import json
from types import SimpleNamespace

import pytest

from policy_answer_service.generation import (
    NOT_IN_POLICIES,
    filter_superseded,
    generate_answer,
    score_correctness,
    score_faithfulness,
)

V1 = {
    "chunk_id": "returns-policy-v1#return-window",
    "document_title": "Returns Policy v1",
    "section": "Return window",
    "text": "You have 30 days to return items.",
    "effective_date": "2023-02-01",
    "version": "1",
}
V2 = {
    "chunk_id": "returns-policy-v2#return-window",
    "document_title": "Returns Policy v2",
    "section": "Return window",
    "text": "You have 45 days to return items.",
    "effective_date": "2025-06-01",
    "version": "2",
}
SHIPPING = {
    "chunk_id": "standard-and-express-shipping-policy#shipping-options",
    "document_title": "Standard and Express Shipping",
    "section": "Shipping options",
    "text": "Standard shipping costs $4.95.",
    "effective_date": "2024-01-01",
    "version": "1",
}


class _Completions:
    def __init__(self, owner: "FakeClient") -> None:
        self.owner = owner

    def create(self, **kwargs):
        self.owner.calls.append(kwargs)
        message = SimpleNamespace(content=self.owner.content)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice])


class FakeClient:
    """Stands in for the OpenAI client; returns a fixed JSON string and records calls."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=_Completions(self))


def _llm_reply(**fields) -> str:
    return json.dumps(fields)


def test_filter_superseded_keeps_latest_version_per_document():
    kept = filter_superseded([V1, V2, SHIPPING])

    assert [c["chunk_id"] for c in kept] == [V2["chunk_id"], SHIPPING["chunk_id"]]


def test_generate_answer_cites_retrieved_chunks():
    client = FakeClient(
        _llm_reply(
            answer="You have 45 days.",
            citations=[V2["chunk_id"]],
            confidence=0.9,
            answerable=True,
        )
    )

    response = generate_answer(client, "model-x", "how many days do I have", [V2])

    assert response.answerable is True
    assert response.answer == "You have 45 days."
    assert [c.chunk_id for c in response.citations] == [V2["chunk_id"]]
    assert response.citations[0].document_title == "Returns Policy v2"


def test_generate_answer_uses_newest_version_only():
    client = FakeClient(
        _llm_reply(answer="45 days.", citations=[V2["chunk_id"]], confidence=0.8, answerable=True)
    )

    generate_answer(client, "model-x", "how many days", [V1, V2])

    prompt = client.calls[0]["messages"][1]["content"]
    assert V2["text"] in prompt
    assert V1["text"] not in prompt


def test_generate_answer_refuses_when_not_in_policies():
    client = FakeClient(_llm_reply(answer="Yes", citations=[], confidence=0.2, answerable=False))

    response = generate_answer(client, "model-x", "do you price match", [SHIPPING])

    assert response.answerable is False
    assert response.answer == NOT_IN_POLICIES
    assert response.citations == []


def test_empty_retrieval_refuses_without_calling_the_llm():
    client = FakeClient(_llm_reply(answer="x", citations=[], confidence=1.0, answerable=True))

    response = generate_answer(client, "model-x", "anything", [])

    assert response.answerable is False
    assert response.answer == NOT_IN_POLICIES
    assert client.calls == []


def test_answerable_without_valid_citation_is_refused():
    client = FakeClient(
        _llm_reply(
            answer="45 days.", citations=["made-up#section"], confidence=0.9, answerable=True
        )
    )

    response = generate_answer(client, "model-x", "how many days", [V2])

    assert response.answerable is False
    assert response.answer == NOT_IN_POLICIES


def test_citations_outside_the_retrieved_set_are_dropped():
    client = FakeClient(
        _llm_reply(
            answer="45 days.",
            citations=["made-up#section", V2["chunk_id"]],
            confidence=0.9,
            answerable=True,
        )
    )

    response = generate_answer(client, "model-x", "how many days", [V2])

    assert [c.chunk_id for c in response.citations] == [V2["chunk_id"]]


def test_confidence_is_clamped_to_unit_range():
    client = FakeClient(
        _llm_reply(answer="45 days.", citations=[V2["chunk_id"]], confidence=1.7, answerable=True)
    )

    response = generate_answer(client, "model-x", "how many days", [V2])

    assert response.confidence == 1.0


def test_generate_answer_accepts_json_wrapped_in_code_fence():
    reply = (
        "```json\n"
        + _llm_reply(answer="45 days.", citations=[V2["chunk_id"]], confidence=0.9, answerable=True)
        + "\n```"
    )
    client = FakeClient(reply)

    response = generate_answer(client, "model-x", "how many days", [V2])

    assert response.answerable is True


def test_non_json_reply_raises_with_the_raw_reply_and_finish_reason():
    client = FakeClient("")

    with pytest.raises(ValueError, match="finish_reason=stop"):
        generate_answer(client, "model-x", "how many days", [V2])


def test_score_faithfulness_returns_clamped_float():
    client = FakeClient(json.dumps({"score": 3}))

    score = score_faithfulness(client, "model-x", "how many days", "45 days.", [V2])

    assert score == 1.0


def test_score_correctness_compares_answer_to_reference_note():
    client = FakeClient(json.dumps({"score": 0.2}))

    score = score_correctness(
        client,
        "model-x",
        "as a gold member can i return something 50 days after it arrived",
        "No, you cannot.",
        "Yes. Gold's 60 days overrides v2's standard 45 days.",
    )

    assert score == 0.2
    prompt = client.calls[0]["messages"][1]["content"]
    assert "Gold's 60 days overrides" in prompt
    assert "No, you cannot." in prompt
