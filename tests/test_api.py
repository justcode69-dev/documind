from tests.conftest import make_pdf


def upload(client, data, name="notes.pdf", ctype="application/pdf"):
    return client.post("/documents", files={"file": (name, data, ctype)})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["qdrant"] is True and body["llm"] is True and body["status"] == "ok"


def test_upload_list_and_duplicate(client, pdf_bytes):
    r = upload(client, pdf_bytes)
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["pages"] == 3 and doc["chunks"] >= 3 and doc["status"] == "ready"
    assert client.services.store.count(doc["doc_id"]) == doc["chunks"]

    again = upload(client, pdf_bytes, name="copy.pdf")
    assert again.json()["duplicate"] is True and again.json()["doc_id"] == doc["doc_id"]

    listed = client.get("/documents").json()
    assert [d["doc_id"] for d in listed] == [doc["doc_id"]]


def test_rejects_wrong_type_fake_pdf_and_oversize(client):
    assert upload(client, b"hello", name="a.txt", ctype="text/plain").status_code == 415
    assert upload(client, b"not really a pdf").status_code == 415  # content-type lies
    assert upload(client, b"%PDF-1.4" + b"0" * (1024 * 1024 + 10)).status_code == 413


def test_rejects_textless_pdf(client):
    assert upload(client, make_pdf([""])).status_code == 422


def test_filename_is_sanitised(client, pdf_bytes):
    r = upload(client, pdf_bytes, name="../../etc/evil.pdf")
    assert r.json()["filename"] == "evil.pdf"


def test_chat_returns_page_citation(client, pdf_bytes):
    upload(client, pdf_bytes)
    r = client.post("/chat", json={"question": "What is the Bernoulli equation?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["citations"] and body["citations"][0]["page"] == 1
    assert body["citations"][0]["filename"] == "notes.pdf"
    assert set(body["timings_ms"]) >= {"rewrite", "retrieve", "generate", "total"}


def test_followup_is_rewritten_with_history(client, pdf_bytes, llm):
    upload(client, pdf_bytes)
    first = client.post("/chat", json={"question": "What is the Bernoulli equation?"}).json()
    # Next LLM responses: rewrite output, then answer.
    llm.responses = ["What is the Reynolds number formula?", "Re = rho v D / mu [1]."]
    llm.i = 0
    r = client.post("/chat", json={"question": "aur Reynolds ka formula?", "session_id": first["session_id"]})
    body = r.json()
    assert body["rewritten_question"] == "What is the Reynolds number formula?"
    assert body["citations"][0]["page"] == 2

    hist = client.get(f"/sessions/{first['session_id']}").json()["messages"]
    assert [m["role"] for m in hist] == ["user", "assistant", "user", "assistant"]


def test_doc_filter_and_delete(client, pdf_bytes):
    a = upload(client, pdf_bytes, name="a.pdf").json()
    b = upload(client, make_pdf(["Ohm's law: V = I R. Voltage equals current times resistance."]),
               name="b.pdf").json()
    r = client.post("/chat", json={"question": "Bernoulli equation", "doc_ids": [b["doc_id"]]}).json()
    assert all(c["doc_id"] == b["doc_id"] for c in r["citations"])

    assert client.delete(f"/documents/{a['doc_id']}").status_code == 204
    assert client.services.store.count(a["doc_id"]) == 0
    assert client.delete(f"/documents/{a['doc_id']}").status_code == 404


def test_no_documents_says_not_found(client):
    r = client.post("/chat", json={"question": "anything?"}).json()
    assert r["answer"] == "Document mein nahi mila." and r["citations"] == []


def test_chat_without_llm_returns_503(settings, pdf_bytes):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.services.container import build_services
    from tests.fakes import FakeEmbedder, FakeReranker

    svc = build_services(settings, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=None)
    svc.llm = None
    with TestClient(create_app(settings, svc)) as c:
        upload(c, pdf_bytes)
        assert c.post("/chat", json={"question": "Bernoulli?"}).status_code == 503
        assert c.get("/health").json()["status"] == "degraded"


def test_api_key_and_rate_limit(settings, pdf_bytes):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.services.container import build_services
    from tests.fakes import FakeEmbedder, FakeReranker, RecordingLLM

    s = settings.model_copy(update={"api_key": "s3cret", "rate_limit_per_minute": 2})
    from pydantic import SecretStr

    s.api_key = SecretStr("s3cret")
    svc = build_services(s, embedder=FakeEmbedder(), reranker=FakeReranker(),
                         llm=RecordingLLM(responses=["x"]))
    with TestClient(create_app(s, svc)) as c:
        assert c.get("/documents").status_code == 401
        h = {"X-API-Key": "s3cret"}
        assert c.get("/documents", headers=h).status_code == 200
        codes = [c.post("/chat", json={"question": "q"}, headers=h).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
