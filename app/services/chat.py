"""One chat turn: history -> rewrite -> retrieve (+rerank) -> grounded answer -> persist."""

import time
from dataclasses import asdict, dataclass

from langsmith import traceable

from app.db import crud
from app.llm.chain import Citation, generate_answer
from app.llm.rewriter import rewrite_question
from app.services.container import Services


@dataclass
class ChatResult:
    session_id: str
    answer: str
    citations: list[Citation]
    rewritten_question: str
    timings_ms: dict[str, float]


@traceable(name="chat_turn")
def chat_turn(svc: Services, *, question: str, session_id: str | None, doc_ids: list[str] | None,
              mode: str | None = None) -> ChatResult:
    mode = mode or svc.settings.retrieval_mode
    t: dict[str, float] = {}
    with svc.db() as db:
        session = crud.get_or_create_session(db, session_id)
        history = [(m.role, m.content) for m in crud.recent_history(db, session.id, svc.settings.history_turns)]

        t0 = time.perf_counter()
        standalone = rewrite_question(svc.llm, question, history)
        t["rewrite"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        chunks = svc.retriever.retrieve(standalone, mode=mode, doc_ids=doc_ids)
        t["retrieve"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        answer, citations = generate_answer(svc.llm, standalone, chunks)
        t["generate"] = (time.perf_counter() - t0) * 1000
        t["total"] = sum(t.values())

        crud.add_message(db, session.id, "user", question, rewritten_question=standalone)
        crud.add_message(db, session.id, "assistant", answer, citations=[asdict(c) for c in citations])
    return ChatResult(session.id, answer, citations, standalone, {k: round(v, 1) for k, v in t.items()})
