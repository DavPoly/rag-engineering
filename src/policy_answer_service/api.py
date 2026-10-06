"""
FastAPI service layer. Answers questions at POST /ask by running retrieval over the
chunks produced by ingestion, then generation with citations and the refusal path.

Run ingestion ahead of time so data/chunks.jsonl exists.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from openai import OpenAI

from policy_answer_service.config import get_settings
from policy_answer_service.generation import ModelReplyError, build_client, generate_answer
from policy_answer_service.models import ModelResponse, UserRequest
from policy_answer_service.retrieval import VectorRetriever, load_chunks

logger = logging.getLogger(__name__)

app = FastAPI(title="Policy Answer Service")


@dataclass
class AnswerPipeline:
    retriever: VectorRetriever
    client: OpenAI
    model: str
    top_k: int

    def answer(self, question: str) -> ModelResponse:
        chunks = self.retriever.retrieve(question, top_k=self.top_k)
        return generate_answer(self.client, self.model, question, chunks)


@lru_cache
def get_pipeline() -> AnswerPipeline:
    """Built once per process. Loads the chunks and embeds them; the embeddings are
    cached on disk after the first run."""
    settings = get_settings()
    chunks = load_chunks(settings.data_dir / "chunks.jsonl")
    return AnswerPipeline(
        retriever=VectorRetriever(chunks, settings.embedding_model_api),
        client=build_client(),
        model=settings.llm_model,
        top_k=settings.retrieval_top_k,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=ModelResponse)
def ask(
    request: UserRequest,
    pipeline: Annotated[AnswerPipeline, Depends(get_pipeline)],
) -> ModelResponse:
    try:
        return pipeline.answer(request.question)
    except ModelReplyError as error:
        # The model answered, but not in the format we asked for. Log the detail, not the reply.
        logger.warning("unusable model reply: %s", error)
        raise HTTPException(
            status_code=502,
            detail="The answer model returned an unusable reply. Please try again.",
        ) from error
