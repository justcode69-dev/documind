"""Builds all heavy components once at startup. Tests build a container with fakes instead."""

import logging
from dataclasses import dataclass

from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db.models import make_session_factory
from app.ingestion.embedder import Embedder, SentenceTransformerEmbedder
from app.llm.chain import build_llm
from app.retrieval.hybrid import Retriever
from app.retrieval.qdrant_store import QdrantStore, make_client
from app.retrieval.reranker import CrossEncoderReranker, Reranker

log = logging.getLogger("documind")


@dataclass
class Services:
    settings: Settings
    db: sessionmaker
    embedder: Embedder
    store: QdrantStore
    reranker: Reranker | None
    retriever: Retriever
    llm: object | None


def build_services(settings: Settings, *, embedder: Embedder | None = None, reranker: Reranker | None = None,
                   llm: object | None = None, load_reranker: bool | None = None) -> Services:
    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model, device=settings.device)
    if reranker is None and (load_reranker if load_reranker is not None
                             else settings.retrieval_mode == "hybrid_rerank"):
        reranker = CrossEncoderReranker(settings.reranker_model, device=settings.device)
    client = make_client(settings.qdrant_url,
                         settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
                         settings.qdrant_path)
    store = QdrantStore(client, settings.collection_name, embedder.dim, server=bool(settings.qdrant_url))
    retriever = Retriever(store, embedder, reranker, settings.top_k_retrieve, settings.top_k_final)
    llm = llm if llm is not None else build_llm(settings)
    if llm is None:
        log.warning("No LLM API key configured; /chat will return 503 until GEMINI_API_KEY or GROQ_API_KEY is set.")
    return Services(settings, make_session_factory(settings.sqlite_path), embedder, store, reranker, retriever, llm)
