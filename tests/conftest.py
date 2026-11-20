import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.container import build_services
from tests.fakes import FakeEmbedder, FakeReranker, RecordingLLM

PAGES = [
    "Bernoulli's principle states that an increase in the speed of a fluid occurs simultaneously "
    "with a decrease in pressure. The Bernoulli equation is P + 1/2 rho v^2 + rho g h = constant.",
    "Reynolds number Re = rho v D / mu predicts laminar or turbulent flow. Flow is laminar when "
    "Re is below 2000 and turbulent above 4000.",
    "PM Kisan Samman Nidhi scheme gives eligible farmer families Rs 6000 per year in three "
    "instalments, transferred directly to bank accounts.",
]


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def pdf_bytes() -> bytes:
    return make_pdf(PAGES)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, qdrant_url=None, qdrant_path=":memory:", sqlite_path=":memory:",
                    max_upload_mb=1, rate_limit_per_minute=1000, gemini_api_key=None, groq_api_key=None)


@pytest.fixture
def llm():
    RecordingLLM.calls = []
    return RecordingLLM(responses=["Bernoulli equation hai P + 1/2 rho v^2 + rho g h = constant [1]."])


@pytest.fixture
def client(settings, llm):
    services = build_services(settings, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=llm)
    with TestClient(create_app(settings, services)) as c:
        c.services = services
        yield c
