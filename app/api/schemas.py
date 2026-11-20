from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    status: str
    size_bytes: int
    uploaded_at: datetime
    duplicate: bool = False


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=36, pattern=r"^[A-Za-z0-9-]+$")
    doc_ids: list[str] | None = Field(default=None, max_length=50)
    mode: Literal["dense", "hybrid", "hybrid_rerank"] | None = None


class CitationOut(BaseModel):
    n: int
    filename: str
    page: int
    snippet: str
    doc_id: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[CitationOut]
    rewritten_question: str
    timings_ms: dict[str, float]


class MessageOut(BaseModel):
    role: str
    content: str
    rewritten_question: str | None
    citations: list[CitationOut] | None
    created_at: datetime


class SessionOut(BaseModel):
    session_id: str
    messages: list[MessageOut]


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    api: bool
    qdrant: bool
    llm: bool
    retrieval_mode: str
