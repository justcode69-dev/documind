"""Cross-encoder reranking: scores (query, chunk) pairs jointly — slower than bi-encoder
retrieval, so we only run it on the top-20 candidates."""

from typing import Protocol

from app.retrieval.qdrant_store import RetrievedChunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str | None = None, max_length: int = 512):
        from sentence_transformers import CrossEncoder  # heavy import, keep lazy

        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = self.model.predict([(query, c.text) for c in chunks], show_progress_bar=False)
        for c, s in zip(chunks, scores, strict=True):
            c.rerank_score = float(s)
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_n]
