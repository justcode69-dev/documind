"""Follow-up question -> standalone question, using recent chat history."""

from langsmith import traceable

from app.llm.chain import message_text
from app.llm.prompts import rewrite_prompt


def format_history(history: list[tuple[str, str]], max_chars: int = 600) -> str:
    lines = []
    for role, content in history:
        text = content if len(content) <= max_chars else content[:max_chars] + "…"
        lines.append(f"{'User' if role == 'user' else 'Assistant'}: {text}")
    return "\n".join(lines)


@traceable(name="rewrite_query")
def rewrite_question(llm, question: str, history: list[tuple[str, str]]) -> str:
    # No history -> nothing to resolve; skip the LLM call (saves latency and free-tier quota).
    if not history or llm is None:
        return question
    try:
        msg = (rewrite_prompt | llm).invoke({"history": format_history(history), "question": question})
    except Exception:
        return question  # degrade gracefully: retrieval still works on the raw question
    rewritten = message_text(msg).strip().strip('"').splitlines()[0].strip() if message_text(msg) else ""
    return rewritten or question
