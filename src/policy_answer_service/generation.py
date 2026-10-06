"""
Answer generation: structured, cited answers with an honest refusal path.

Retrieved chunks go through filter_superseded (newest effective_date per
document), then the LLM answers from those excerpts only. Any citation that
was not in the excerpts is dropped, and an answer with no valid citation is
refused.
"""

import json
import re

from openai import OpenAI

from policy_answer_service.config import get_settings
from policy_answer_service.models import Citation, ModelResponse

NOT_IN_POLICIES = (
    "I couldn't find this in Northwind's policy documents, so I can't answer it. "
    "Please check with a supervisor."
)

_VERSION_SUFFIX = re.compile(r"-v\d+$")

SYSTEM_PROMPT = """You answer questions about Northwind Goods policies using only the excerpts.
Rules:
- Use only facts stated in the excerpts. Do not use outside knowledge or guess figures.
- Cite the chunk_id of every excerpt you rely on.
- If the excerpts do not answer the question, set "answerable" to false.
Respond with JSON only: {"answer": string, "citations": [chunk_id, ...],
"confidence": number from 0 to 1, "answerable": boolean}."""

JUDGE_PROMPT = """Score from 0 to 1 how fully the ANSWER is supported by the EXCERPTS.
1 means every claim in the answer is stated in the excerpts.
0 means the answer is contradicted or unsupported.
Respond with JSON only: {"score": number from 0 to 1}."""

CORRECTNESS_PROMPT = """Score from 0 to 1 how well the ANSWER agrees with the REFERENCE.
1 means the answer gives the same conclusion and key facts as the reference.
0 means the answer reaches the opposite conclusion or gets the key facts wrong.
Do not reward extra detail that is consistent with the reference, and do not penalise wording.
Respond with JSON only: {"score": number from 0 to 1}."""


class ModelReplyError(ValueError):
    """The model's reply could not be parsed as the JSON we asked for."""


def build_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)


def _document_family(chunk: dict) -> str:
    """'returns-policy-v2#return-window' -> 'returns-policy'."""
    document_slug = chunk["chunk_id"].split("#", 1)[0]
    return _VERSION_SUFFIX.sub("", document_slug)


def filter_superseded(chunks: list[dict]) -> list[dict]:
    """Keep only the chunks from the newest effective_date within each document family."""
    latest: dict[str, str] = {}
    for chunk in chunks:
        family = _document_family(chunk)
        latest[family] = max(latest.get(family, chunk["effective_date"]), chunk["effective_date"])
    return [c for c in chunks if c["effective_date"] == latest[_document_family(c)]]


def _format_excerpts(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c['chunk_id']}] {c['document_title']} (effective {c['effective_date']})\n{c['text']}"
        for c in chunks
    )


def _clamp_unit(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _refusal(confidence: float) -> ModelResponse:
    return ModelResponse(
        answer=NOT_IN_POLICIES,
        citations=[],
        confidence=confidence,
        answerable=False,
    )


def _parse_json_reply(completion) -> dict:
    """Parse the model's JSON reply, tolerating a code fence or prose around the object."""
    content = completion.choices[0].message.content or ""
    text = content.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    finish_reason = completion.choices[0].finish_reason
    raise ModelReplyError(
        f"Model reply was not JSON (finish_reason={finish_reason}): {content[:300]!r}"
    )


def generate_answer(client: OpenAI, model: str, question: str, chunks: list[dict]) -> ModelResponse:
    if not chunks:
        return _refusal(confidence=0.0)

    current = filter_superseded(chunks)
    allowed = {c["chunk_id"]: c for c in current}
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Excerpts:\n\n{_format_excerpts(current)}\n\nQuestion: {question}",
            },
        ],
    )
    data = _parse_json_reply(completion)
    confidence = _clamp_unit(data.get("confidence", 0.0))

    cited = [cid for cid in dict.fromkeys(data.get("citations", [])) if cid in allowed]
    if not data.get("answerable", False) or not cited:
        return _refusal(confidence)

    return ModelResponse(
        answer=data["answer"],
        citations=[
            Citation(chunk_id=cid, document_title=allowed[cid]["document_title"]) for cid in cited
        ],
        confidence=confidence,
        answerable=True,
    )


def score_faithfulness(
    client: OpenAI, model: str, question: str, answer: str, cited_chunks: list[dict]
) -> float:
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"EXCERPTS:\n\n{_format_excerpts(cited_chunks)}\n\n"
                    f"QUESTION: {question}\n\nANSWER: {answer}"
                ),
            },
        ],
    )
    data = _parse_json_reply(completion)
    return _clamp_unit(data["score"])


def score_correctness(
    client: OpenAI, model: str, question: str, answer: str, reference: str
) -> float:
    """Compare an answer to a reference note. Unlike faithfulness, this checks the conclusion."""
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": CORRECTNESS_PROMPT},
            {
                "role": "user",
                "content": (f"QUESTION: {question}\n\nREFERENCE: {reference}\n\nANSWER: {answer}"),
            },
        ],
    )
    data = _parse_json_reply(completion)
    return _clamp_unit(data["score"])
