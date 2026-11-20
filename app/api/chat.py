from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_services
from app.api.schemas import ChatRequest, ChatResponse, MessageOut, SessionOut
from app.api.security import rate_limited
from app.db import crud
from app.llm.chain import LLMUnavailable
from app.services.chat import chat_turn
from app.services.container import Services

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limited)])
def chat(req: ChatRequest, svc: Services = Depends(get_services)):
    if req.mode == "hybrid_rerank" and svc.reranker is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Reranker not loaded on this server")
    try:
        res = chat_turn(svc, question=req.question.strip(), session_id=req.session_id,
                        doc_ids=req.doc_ids or None, mode=req.mode)
    except LLMUnavailable as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e
    return ChatResponse(session_id=res.session_id, answer=res.answer,
                        citations=[asdict(c) for c in res.citations],
                        rewritten_question=res.rewritten_question, timings_ms=res.timings_ms)


@router.get("/sessions/{session_id}", response_model=SessionOut)
def get_session(session_id: str, svc: Services = Depends(get_services)):
    with svc.db() as db:
        s = crud.get_session(db, session_id)
        if not s:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
        return SessionOut(session_id=s.id, messages=[
            MessageOut(role=m.role, content=m.content, rewritten_question=m.rewritten_question,
                       citations=m.citations, created_at=m.created_at) for m in s.messages
        ])
