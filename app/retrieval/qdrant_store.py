"""Qdrant wrapper: one collection, named dense + sparse vectors, payload filter on doc_id."""

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from app.ingestion.chunker import Chunk
from app.retrieval import sparse

DENSE, SPARSE = "dense", "sparse"


@dataclass
class RetrievedChunk:
    id: str
    text: str
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    score: float
    rerank_score: float | None = None


def make_client(url: str | None, api_key: str | None, path: str) -> QdrantClient:
    if url:
        return QdrantClient(url=url, api_key=api_key, timeout=10)
    if path == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(path=path)  # embedded mode: no Docker needed for quick local runs


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int, server: bool = False):
        self.client, self.collection, self.dim, self.server = client, collection, dim, server
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            existing = info.config.params.vectors[DENSE].size
            if existing != self.dim:
                raise RuntimeError(
                    f"Collection '{self.collection}' has dim {existing}, embedder has {self.dim}. "
                    "Use a new COLLECTION_NAME after changing the embedding model."
                )
            return
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=self.dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        # Payload index makes doc_id filtering fast on a real Qdrant server (no effect in local mode).
        if self.server:
            self.client.create_payload_index(self.collection, "doc_id", models.PayloadSchemaType.KEYWORD)

    @staticmethod
    def point_id(doc_id: str, chunk_index: int) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{chunk_index}"))

    def upsert(self, chunks: list[Chunk], dense_vectors: list[list[float]], batch: int = 128) -> None:
        points = [
            models.PointStruct(
                id=self.point_id(c.doc_id, c.chunk_index),
                vector={DENSE: v, SPARSE: sparse.doc_sparse_vector(c.text)},
                payload=c.payload(),
            )
            for c, v in zip(chunks, dense_vectors, strict=True)
        ]
        for i in range(0, len(points), batch):
            self.client.upsert(self.collection, points[i : i + batch], wait=True)

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=self._filter([doc_id])),
                           wait=True)

    def count(self, doc_id: str | None = None) -> int:
        f = self._filter([doc_id]) if doc_id else None
        return self.client.count(self.collection, count_filter=f, exact=True).count

    @staticmethod
    def _filter(doc_ids: list[str] | None) -> models.Filter | None:
        if not doc_ids:
            return None
        return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids))])

    def search(self, *, dense_query: list[float], text_query: str, hybrid: bool, limit: int,
               doc_ids: list[str] | None = None) -> list[RetrievedChunk]:
        flt = self._filter(doc_ids)
        if hybrid:
            # Reciprocal Rank Fusion of dense and BM25 candidate lists, done inside Qdrant.
            res = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(query=dense_query, using=DENSE, limit=limit, filter=flt),
                    models.Prefetch(query=sparse.query_sparse_vector(text_query), using=SPARSE,
                                    limit=limit, filter=flt),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        else:
            res = self.client.query_points(self.collection, query=dense_query, using=DENSE, limit=limit,
                                           query_filter=flt, with_payload=True)
        return [
            RetrievedChunk(
                id=str(p.id), text=p.payload["text"], doc_id=p.payload["doc_id"],
                filename=p.payload["filename"], page=p.payload["page"],
                chunk_index=p.payload["chunk_index"], score=float(p.score),
            )
            for p in res.points
        ]

    def healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
