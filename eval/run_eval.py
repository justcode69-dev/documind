"""Compare chunking strategies x retrieval modes on a labelled test set.

    python -m eval.run_eval --pdf-dir eval/pdfs --testset eval/testset.json
    python -m eval.run_eval ... --faithfulness          # + LLM-as-judge on the best config

Every strategy gets its own in-memory Qdrant collection, so the eval never touches app data.
Questions are searched across ALL eval PDFs (no doc filter) — the harder, more honest setting.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from app.config import get_settings
from app.ingestion.pipeline import build_chunks
from app.llm.chain import build_llm, generate_answer, message_text
from app.llm.prompts import NOT_FOUND, format_context, judge_prompt
from app.retrieval.hybrid import MODES, Retriever
from app.retrieval.qdrant_store import QdrantStore
from eval.metrics import first_relevant_rank, hit_at_k, mrr_at_k, percentile

RESULTS_DIR = Path(__file__).parent / "results"


def load_testset(path: Path, pdf_dir: Path) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    available = {p.name for p in pdf_dir.glob("*.pdf")}
    bad = [q.get("id", i) for i, q in enumerate(items)
           if not q.get("question") or q.get("filename") not in available or not q.get("expected_pages")]
    if bad:
        sys.exit(f"Invalid test items (missing question/expected_pages or unknown filename): {bad}\n"
                 f"PDFs found in {pdf_dir}: {sorted(available)}")
    return items


def build_index(strategy: str, pdfs: list[Path], embedder, settings) -> QdrantStore:
    store = QdrantStore(QdrantClient(":memory:"), f"eval_{strategy}", embedder.dim)
    total = 0
    for pdf in pdfs:
        _, chunks = build_chunks(pdf.read_bytes(), doc_id=pdf.name, filename=pdf.name, strategy=strategy,
                                 chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
                                 max_pages=settings.max_pages)
        store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
        total += len(chunks)
    print(f"  [{strategy}] indexed {len(pdfs)} PDFs -> {total} chunks")
    return store


def judge(llm, chunks, answer: str) -> bool:
    msg = (judge_prompt | llm).invoke({"context": format_context(chunks), "answer": answer})
    return message_text(msg).strip().upper().startswith("YES")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", type=Path, default=Path("eval/pdfs"))
    ap.add_argument("--testset", type=Path, default=Path("eval/testset.json"))
    ap.add_argument("--strategies", default="fixed,recursive")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--limit", type=int, default=0, help="only first N questions (quick check)")
    ap.add_argument("--faithfulness", action="store_true", help="also generate answers + LLM-as-judge")
    ap.add_argument("--faith-config", default="recursive:hybrid_rerank", help="strategy:mode for faithfulness")
    ap.add_argument("--sleep", type=float, default=4.0, help="seconds between LLM calls (free-tier limits)")
    args = ap.parse_args()

    settings = get_settings()
    strategies, modes = args.strategies.split(","), args.modes.split(",")
    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {args.pdf_dir}")
    tests = load_testset(args.testset, args.pdf_dir)
    if args.limit:
        tests = tests[: args.limit]
    print(f"{len(tests)} questions, {len(pdfs)} PDFs, strategies={strategies}, modes={modes}")

    from app.ingestion.embedder import SentenceTransformerEmbedder
    from app.retrieval.reranker import CrossEncoderReranker

    embedder = SentenceTransformerEmbedder(settings.embedding_model, device=settings.device)
    reranker = None
    if "hybrid_rerank" in modes:
        reranker = CrossEncoderReranker(settings.reranker_model, device=settings.device)

    rows, per_question, retrievers = [], [], {}
    for strategy in strategies:
        retriever = Retriever(build_index(strategy, pdfs, embedder, settings), embedder, reranker,
                              settings.top_k_retrieve, settings.top_k_final)
        retrievers[strategy] = retriever
        for mode in modes:
            ranks, lat = [], []
            for q in tests:
                t0 = time.perf_counter()
                got = retriever.retrieve(q["question"], mode=mode)
                lat.append((time.perf_counter() - t0) * 1000)
                rank = first_relevant_rank([(c.filename, c.page) for c in got], q["filename"], q["expected_pages"])
                ranks.append(rank)
                per_question.append({"id": q.get("id"), "strategy": strategy, "mode": mode, "rank": rank})
            row = {"strategy": strategy, "mode": mode, "hit@1": hit_at_k(ranks, 1), "hit@5": hit_at_k(ranks, 5),
                   "mrr@5": mrr_at_k(ranks, 5), "p50_ms": percentile(lat, 50), "p95_ms": percentile(lat, 95)}
            rows.append(row)
            print(f"  {strategy:9s} {mode:13s} Hit@1 {row['hit@1']:.0%}  Hit@5 {row['hit@5']:.0%}  "
                  f"MRR@5 {row['mrr@5']:.3f}  p50 {row['p50_ms']:.0f} ms")

    faith = None
    if args.faithfulness:
        strategy, mode = args.faith_config.split(":")
        llm = build_llm(settings)
        if llm is None:
            sys.exit("--faithfulness needs GEMINI_API_KEY or GROQ_API_KEY in .env")
        supported = not_found = cite_ok = cite_total = errors = 0
        for i, q in enumerate(tests, 1):
            chunks = retrievers[strategy].retrieve(q["question"], mode=mode)
            try:
                answer, cits = generate_answer(llm, q["question"], chunks)
                time.sleep(args.sleep)
                ok = judge(llm, chunks, answer)
                time.sleep(args.sleep)
            except Exception as e:  # keep going on rate-limit / network errors
                errors += 1
                print(f"    q{i}: LLM error {type(e).__name__}")
                continue
            supported += ok
            not_found += answer.strip() == NOT_FOUND
            cite_total += bool(cits)
            cite_ok += any(c.filename == q["filename"] and c.page in q["expected_pages"] for c in cits)
            print(f"    q{i}/{len(tests)} faithful={ok}")
        judged = len(tests) - errors
        faith = {"config": args.faith_config, "judged": judged, "errors": errors,
                 "faithfulness": supported / judged if judged else 0.0,
                 "not_found_rate": not_found / judged if judged else 0.0,
                 "citation_page_accuracy": cite_ok / cite_total if cite_total else 0.0}
        print(f"  Faithfulness {faith['faithfulness']:.0%} | citation page accuracy "
              f"{faith['citation_page_accuracy']:.0%} | not-found {faith['not_found_rate']:.0%}")

    write_report(rows, faith, per_question, tests, pdfs, settings)


def write_report(rows, faith, per_question, tests, pdfs, settings) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {"questions": len(tests), "pdfs": [p.name for p in pdfs], "embedding": settings.embedding_model,
            "reranker": settings.reranker_model, "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap, "top_k_retrieve": settings.top_k_retrieve,
            "top_k_final": settings.top_k_final}
    (RESULTS_DIR / f"results_{stamp}.json").write_text(
        json.dumps({"meta": meta, "rows": rows, "faithfulness": faith, "per_question": per_question}, indent=2))

    lines = [f"# DocuMind eval results ({stamp})", "",
             f"{len(tests)} questions over {len(pdfs)} PDFs · embeddings `{settings.embedding_model}` · "
             f"reranker `{settings.reranker_model}` · chunk {settings.chunk_size}/{settings.chunk_overlap} · "
             f"top-{settings.top_k_retrieve} → top-{settings.top_k_final}", "",
             "| Chunking | Retrieval | Hit@1 | Hit@5 | MRR@5 | p50 latency | p95 latency |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['strategy']} | {r['mode']} | {r['hit@1']:.0%} | {r['hit@5']:.0%} | {r['mrr@5']:.3f} | "
                     f"{r['p50_ms']:.0f} ms | {r['p95_ms']:.0f} ms |")
    if faith:
        lines += ["", f"**Answer quality** (`{faith['config']}`, {faith['judged']} judged): "
                  f"faithfulness {faith['faithfulness']:.0%} · citation page accuracy "
                  f"{faith['citation_page_accuracy']:.0%} · \"not found\" rate {faith['not_found_rate']:.0%}"]
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RESULTS_DIR / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
