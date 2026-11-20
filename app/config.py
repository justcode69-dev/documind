"""Central settings, loaded from environment / .env (never hard-code keys)."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ChunkingStrategy = Literal["recursive", "fixed"]
RetrievalMode = Literal["dense", "hybrid", "hybrid_rerank"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore",
        env_ignore_empty=True,  # `API_KEY=` in .env means "not set", not "empty password"
    )

    # --- API ---
    api_key: SecretStr | None = Field(
        default=None, description="If set, every request must send header X-API-Key with this value."
    )
    max_upload_mb: int = 20
    max_pages: int = 500
    rate_limit_per_minute: int = 30  # per client IP, for POST /chat and POST /documents
    cors_origins: list[str] = ["http://localhost:8501"]

    # --- Storage ---
    qdrant_url: str | None = None  # e.g. http://localhost:6333 ; None -> embedded local mode
    qdrant_api_key: SecretStr | None = None
    qdrant_path: str = "data/qdrant"  # used only in embedded mode (":memory:" for tests)
    collection_name: str = "documind_chunks"
    sqlite_path: str = "data/documind.db"

    # --- Ingestion ---
    chunking_strategy: ChunkingStrategy = "recursive"
    chunk_size: int = 800
    chunk_overlap: int = 100

    # --- Models ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    device: str | None = None  # "cuda" / "cpu" / None (auto)

    # --- Retrieval ---
    retrieval_mode: RetrievalMode = "hybrid_rerank"
    top_k_retrieve: int = 20
    top_k_final: int = 5

    # --- LLM ---
    llm_provider: Literal["gemini", "groq"] = "groq"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 1  # keep low: with a fallback provider, failing over fast beats retrying
    history_turns: int = 4  # past user+assistant pairs used for query rewriting


@lru_cache
def get_settings() -> Settings:
    return Settings()
