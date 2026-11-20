import json
import sys

from eval import metrics, run_eval
from tests.conftest import PAGES, make_pdf
from tests.fakes import FakeEmbedder, FakeReranker


def test_metrics():
    ranks = [1, 3, None, 6]
    assert metrics.hit_at_k(ranks, 1) == 0.25
    assert metrics.hit_at_k(ranks, 5) == 0.5
    assert abs(metrics.mrr_at_k(ranks, 5) - (1 + 1 / 3) / 4) < 1e-9
    assert metrics.first_relevant_rank([("a", 1), ("b", 2), ("b", 3)], "b", [3, 9]) == 3


def test_run_eval_end_to_end(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "notes.pdf").write_bytes(make_pdf(PAGES))
    testset = tmp_path / "t.json"
    testset.write_text(json.dumps([
        {"id": "q1", "question": "Bernoulli equation pressure speed", "filename": "notes.pdf", "expected_pages": [1]},
        {"id": "q2", "question": "Reynolds number turbulent laminar", "filename": "notes.pdf", "expected_pages": [2]},
    ]))
    import app.ingestion.embedder as emb
    import app.retrieval.reranker as rr

    monkeypatch.setattr(emb, "SentenceTransformerEmbedder", lambda *a, **k: FakeEmbedder())
    monkeypatch.setattr(rr, "CrossEncoderReranker", lambda *a, **k: FakeReranker())
    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(sys, "argv", ["run_eval", "--pdf-dir", str(pdf_dir), "--testset", str(testset)])
    run_eval.main()

    report = (tmp_path / "results" / "RESULTS.md").read_text()
    assert report.count("| fixed |") == 3 and report.count("| recursive |") == 3
    data = json.loads(next((tmp_path / "results").glob("results_*.json")).read_text())
    assert all(r["hit@5"] == 1.0 for r in data["rows"])
