"""
Pydantic request/response models for the /ask endpoint.
"""

from pydantic import BaseModel, Field


class UserRequest(BaseModel):
    question: str = Field(min_length=1)


class Citation(BaseModel):
    chunk_id: str
    document_title: str


class ModelResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    answerable: bool  # False for the "not in the policies" case


class Chunk(BaseModel):
    chunk_id: str  # e.g. "returns-policy-v2#eligibility"
    document_title: str
    section: str
    text: str
    effective_date: str
    version: str
