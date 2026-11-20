"""Small data-access layer so API code never builds queries itself."""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ChatSession, Document, Message


# ---------- documents ----------
def get_document(db: Session, doc_id: str) -> Document | None:
    return db.get(Document, doc_id)


def get_document_by_hash(db: Session, sha256: str) -> Document | None:
    return db.scalar(select(Document).where(Document.sha256 == sha256))


def list_documents(db: Session) -> list[Document]:
    return list(db.scalars(select(Document).order_by(Document.uploaded_at.desc())))


def create_document(db: Session, *, doc_id: str, filename: str, sha256: str, size_bytes: int) -> Document:
    doc = Document(id=doc_id, filename=filename, sha256=sha256, size_bytes=size_bytes)
    db.add(doc)
    db.commit()
    return doc


def mark_document(db: Session, doc: Document, *, status: str, pages: int = 0, chunks: int = 0,
                  error: str | None = None) -> Document:
    doc.status, doc.pages, doc.chunks, doc.error = status, pages, chunks, error
    db.commit()
    return doc


def delete_document(db: Session, doc: Document) -> None:
    db.delete(doc)
    db.commit()


# ---------- chat ----------
def get_or_create_session(db: Session, session_id: str | None) -> ChatSession:
    if session_id:
        existing = db.get(ChatSession, session_id)
        if existing:
            return existing
    s = ChatSession(id=session_id or str(uuid.uuid4()))
    db.add(s)
    db.commit()
    return s


def get_session(db: Session, session_id: str) -> ChatSession | None:
    return db.get(ChatSession, session_id)


def recent_history(db: Session, session_id: str, turns: int) -> list[Message]:
    """Last `turns` user/assistant pairs, oldest first."""
    rows = db.scalars(
        select(Message).where(Message.session_id == session_id).order_by(Message.id.desc()).limit(turns * 2)
    )
    return list(reversed(list(rows)))


def add_message(db: Session, session_id: str, role: str, content: str, *,
                rewritten_question: str | None = None, citations: list | None = None) -> Message:
    m = Message(session_id=session_id, role=role, content=content,
                rewritten_question=rewritten_question, citations=citations)
    db.add(m)
    db.commit()
    return m
