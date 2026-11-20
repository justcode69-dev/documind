"""Retrieval pipeline: dense | hybrid (dense+BM25, RRF) | hybrid + cross-encoder rerank."""

from langsmith import traceable

from app.ingestion.embedder import Embedder
from app.retrieval.qdrant_store import QdrantStore, RetrievedChunk
from app.retrieval.reranker import Reranker

MODES = ("dense", "hybrid", "hybrid_rerank")


class Retriever:
    def __init__(self, store: QdrantStore, embedder: Embedder, reranker: Reranker | None,
                 top_k_retrieve: int = 20, top_k_final: int = 5):
        self.store, self.embedder, self.reranker = store, embedder, reranker
        self.top_k_retrieve, self.top_k_final = top_k_retrieve, top_k_final

    @traceable(run_type="retriever", name="retrieve")
    def retrieve(self, query: str, *, mode: str = "hybrid_rerank",
                 doc_ids: list[str] | None = None) -> list[RetrievedChunk]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        candidates = self.store.search(
            dense_query=self.embedder.embed_query(query), text_query=query,
            hybrid=mode != "dense", limit=self.top_k_retrieve, doc_ids=doc_ids,
        )
        if mode == "hybrid_rerank":
            if self.reranker is None:
                raise RuntimeError("hybrid_rerank requested but no reranker is loaded")
            return self._rerank(query, candidates)
        return candidates[: self.top_k_final]

    @traceable(name="rerank")
    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return self.reranker.rerank(query, candidates, self.top_k_final)
