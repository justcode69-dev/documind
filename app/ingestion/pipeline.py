"""PDF bytes -> parsed pages -> chunks -> embeddings -> Qdrant. Shared by the API and eval."""

from dataclasses import dataclass

from langsmith import traceable

from app.ingestion.chunker import Chunk, chunk_pages
from app.ingestion.embedder import Embedder
from app.ingestion.parser import parse_pdf
from app.retrieval.qdrant_store import QdrantStore


@dataclass
class IngestResult:
    pages: int
    chunks: int


def build_chunks(data: bytes, *, doc_id: str, filename: str, strategy: str, chunk_size: int,
                 chunk_overlap: int, max_pages: int = 500) -> tuple[int, list[Chunk]]:
    pages = parse_pdf(data, max_pages=max_pages)
    chunks = chunk_pages(pages, doc_id=doc_id, filename=filename, strategy=strategy,
                         chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return len(pages), chunks


@traceable(name="ingest_pdf")
def ingest_pdf(data: bytes, *, doc_id: str, filename: str, store: QdrantStore, embedder: Embedder,
               strategy: str = "recursive", chunk_size: int = 800, chunk_overlap: int = 100,
               max_pages: int = 500) -> IngestResult:
    n_pages, chunks = build_chunks(data, doc_id=doc_id, filename=filename, strategy=strategy,
                                   chunk_size=chunk_size, chunk_overlap=chunk_overlap, max_pages=max_pages)
    vectors = embedder.embed_documents([c.text for c in chunks])
    store.upsert(chunks, vectors)
    return IngestResult(pages=n_pages, chunks=len(chunks))
