"""Speed up writing the test set: the LLM drafts one question per sampled page.

    python -m eval.draft_questions --pdf-dir eval/pdfs --per-doc 20

Writes eval/testset.draft.json. REVIEW IT BY HAND before renaming to testset.json:
fix wrong questions, add follow-up-style/Hinglish ones, and add extra expected_pages when the
answer appears on more than one page. Unreviewed LLM questions copy the page's wording and make
keyword search look better than it is.
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

from app.config import get_settings
from app.ingestion.parser import parse_pdf
from app.llm.chain import build_llm, message_text

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You write exam-style questions that a student would ask about their notes. The question "
               "must be answerable ONLY from the given page. Paraphrase — do not copy phrases from the page. "
               "Reply with JSON only: {{\"question\": \"...\"}}"),
    ("human", "PAGE TEXT:\n{page}"),
])


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", type=Path, default=Path("eval/pdfs"))
    ap.add_argument("--per-doc", type=int, default=20)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--sleep", type=float, default=4.0)
    ap.add_argument("--out", type=Path, default=Path("eval/testset.draft.json"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    llm = build_llm(get_settings())
    if llm is None:
        raise SystemExit("Set GEMINI_API_KEY or GROQ_API_KEY in .env")
    rng = random.Random(args.seed)
    items = []
    for pdf in sorted(args.pdf_dir.glob("*.pdf")):
        pages = [p for p in parse_pdf(pdf.read_bytes()) if len(p.text) >= args.min_chars]
        for page in rng.sample(pages, min(args.per_doc, len(pages))):
            try:
                raw = message_text((PROMPT | llm).invoke({"page": page.text[:3000]}))
                q = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))["question"].strip()
            except Exception as e:
                print(f"  skip {pdf.name} p{page.page_number}: {type(e).__name__}")
                continue
            items.append({"id": f"q{len(items) + 1}", "question": q, "filename": pdf.name,
                          "expected_pages": [page.page_number]})
            print(f"  {pdf.name} p{page.page_number}: {q}")
            time.sleep(args.sleep)
    args.out.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(items)} drafts -> {args.out}. Review, then save as eval/testset.json")


if __name__ == "__main__":
    main()
