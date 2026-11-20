import hashlib
import logging
import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import get_services
from app.api.schemas import DocumentOut
from app.api.security import rate_limited
from app.db import crud
from app.ingestion.parser import PDFError, looks_like_pdf
from app.ingestion.pipeline import ingest_pdf
from app.services.container import Services

router = APIRouter(prefix="/documents", tags=["documents"])
log = logging.getLogger("documind")

ALLOWED_TYPES = {"application/pdf", "application/x-pdf"}


def _out(doc, duplicate: bool = False) -> DocumentOut:
    return DocumentOut(doc_id=doc.id, filename=doc.filename, pages=doc.pages, chunks=doc.chunks,
                       status=doc.status, size_bytes=doc.size_bytes, uploaded_at=doc.uploaded_at,
                       duplicate=duplicate)


def _safe_name(name: str | None) -> str:
    base = os.path.basename((name or "document.pdf").replace("\\", "/"))
    base = "".join(ch for ch in base if ch.isprintable())[:200].strip()
    return base or "document.pdf"


# Plain `def`: parsing + embedding are CPU-bound, so FastAPI runs this in its threadpool.
@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(rate_limited)])
def upload_document(file: UploadFile = File(...), svc: Services = Depends(get_services)):
    s = svc.settings
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PDF files are accepted")
    limit = s.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)  # never read more than limit+1 bytes into memory
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"File exceeds {s.max_upload_mb} MB")
    if not looks_like_pdf(data):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "File content is not a PDF")

    filename = _safe_name(file.filename)
    sha = hashlib.sha256(data).hexdigest()
    with svc.db() as db:
        existing = crud.get_document_by_hash(db, sha)
        if existing and existing.status == "ready":
            return _out(existing, duplicate=True)
        if existing:  # a previous failed/partial attempt: clean up and retry
            svc.store.delete_document(existing.id)
            crud.delete_document(db, existing)

        doc = crud.create_document(db, doc_id=str(uuid.uuid4()), filename=filename, sha256=sha,
                                   size_bytes=len(data))
        try:
            res = ingest_pdf(data, doc_id=doc.id, filename=filename, store=svc.store, embedder=svc.embedder,
                             strategy=s.chunking_strategy, chunk_size=s.chunk_size,
                             chunk_overlap=s.chunk_overlap, max_pages=s.max_pages)
        except PDFError as e:
            crud.mark_document(db, doc, status="failed", error=str(e))
            svc.store.delete_document(doc.id)
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
        except Exception as e:
            log.exception("Ingestion failed for doc %s", doc.id)  # id only — never log document text
            crud.mark_document(db, doc, status="failed", error=type(e).__name__)
            svc.store.delete_document(doc.id)
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Ingestion failed") from e
        crud.mark_document(db, doc, status="ready", pages=res.pages, chunks=res.chunks)
        log.info("Ingested doc %s: %d pages, %d chunks", doc.id, res.pages, res.chunks)
        return _out(doc)


@router.get("", response_model=list[DocumentOut])
def list_documents(svc: Services = Depends(get_services)):
    with svc.db() as db:
        return [_out(d) for d in crud.list_documents(db)]


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, svc: Services = Depends(get_services)):
    with svc.db() as db:
        doc = crud.get_document(db, doc_id)
        if not doc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
        svc.store.delete_document(doc_id)  # vectors first, so a crash never leaves orphan vectors
        crud.delete_document(db, doc)
