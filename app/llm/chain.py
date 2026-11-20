"""LLM factory (Gemini primary, Groq fallback) + grounded answer generation with citations."""

import logging
import re
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langsmith import traceable

from app.config import Settings
from app.llm.prompts import NOT_FOUND, answer_prompt, format_context
from app.retrieval.qdrant_store import RetrievedChunk

log = logging.getLogger("documind")


class LLMUnavailable(RuntimeError):
    pass


def _gemini(s: Settings) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=s.gemini_model, google_api_key=s.gemini_api_key.get_secret_value(),
                                  temperature=0, timeout=s.llm_timeout_s, max_retries=s.llm_max_retries)


def _groq(s: Settings) -> BaseChatModel:
    from langchain_groq import ChatGroq

    return ChatGroq(model=s.groq_model, api_key=s.groq_api_key.get_secret_value(), temperature=0,
                    timeout=s.llm_timeout_s, max_retries=s.llm_max_retries)


def build_llm(s: Settings):
    """Primary provider from LLM_PROVIDER; the other one (if its key is set) becomes a fallback,
    so a free-tier rate limit on one provider doesn't take the app down."""
    builders = {"gemini": (_gemini, s.gemini_api_key), "groq": (_groq, s.groq_api_key)}
    order = [s.llm_provider] + [p for p in builders if p != s.llm_provider]
    models = [builders[p][0](s) for p in order if builders[p][1]]
    if not models:
        return None
    return models[0].with_fallbacks(models[1:]) if len(models) > 1 else models[0]


def message_text(msg: BaseMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content.strip()
    # Some providers return a list of content blocks.
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content).strip()


def describe_llm_error(e: Exception, max_len: int = 200) -> str:
    """Short, user-safe reason (provider errors carry status + message, never the API key)."""
    msg = " ".join(str(e).split())
    if not msg:
        return type(e).__name__
    return f"{type(e).__name__}: {msg[:max_len]}{'…' if len(msg) > max_len else ''}"


@dataclass
class Citation:
    n: int
    filename: str
    page: int
    snippet: str
    doc_id: str


_CITE_RE = re.compile(r"\[(\d{1,2})\]")
# Some models (e.g. gpt-oss) write 【3】, ［3］ or 【3†L1-L4】 instead of [3]; normalise those first.
_ALT_CITE_RE = re.compile(r"[\[【［]\s*(\d{1,2})\s*(?:†[^\]】］]*)?[\]】］]")


def normalize_citations(answer: str) -> str:
    return _ALT_CITE_RE.sub(lambda m: f"[{m.group(1)}]", answer)


def extract_citations(answer: str, chunks: list[RetrievedChunk], snippet_len: int = 240) -> list[Citation]:
    seen, out = set(), []
    for m in _CITE_RE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= len(chunks) and n not in seen:
            seen.add(n)
            c = chunks[n - 1]
            snippet = c.text[:snippet_len] + ("…" if len(c.text) > snippet_len else "")
            out.append(Citation(n=n, filename=c.filename, page=c.page, snippet=snippet, doc_id=c.doc_id))
    return sorted(out, key=lambda c: c.n)


def strip_invalid_citations(answer: str, n_chunks: int) -> str:
    """Remove hallucinated markers like [7] when only 5 passages were given."""
    return _CITE_RE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_chunks else "", answer)


@traceable(name="generate_answer")
def generate_answer(llm, question: str, chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
    if not chunks:
        return NOT_FOUND, []
    if llm is None:
        raise LLMUnavailable("No LLM configured: set GEMINI_API_KEY or GROQ_API_KEY in .env")
    try:
        msg = (answer_prompt | llm).invoke({"context": format_context(chunks), "question": question})
    except Exception as e:
        reason = describe_llm_error(e)
        log.warning("LLM call failed: %s", reason)
        raise LLMUnavailable(f"LLM call failed: {reason}") from e
    answer = strip_invalid_citations(normalize_citations(message_text(msg)), len(chunks))
    return answer, extract_citations(answer, chunks)
