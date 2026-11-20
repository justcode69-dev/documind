"""Streamlit UI:  streamlit run ui/streamlit_app.py   (API must be running)."""

import os

import requests
import streamlit as st

API_URL = os.getenv("DOCUMIND_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("DOCUMIND_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API_URL}{path}", headers=HEADERS, timeout=kw.pop("timeout", 60), **kw)
    except requests.RequestException:
        st.error(f"API se connect nahi ho paaya ({API_URL}). Kya uvicorn chal raha hai?")
        st.stop()
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        st.error(f"{r.status_code}: {detail}")
        return None
    return r.json() if r.content else {}


if "session_id" not in st.session_state:
    st.session_state.session_id = None
    st.session_state.messages = []

# ---------------- sidebar: documents ----------------
with st.sidebar:
    st.title("📄 DocuMind")
    health = api("GET", "/health", timeout=10) or {}
    st.caption(f"Qdrant {'✅' if health.get('qdrant') else '❌'} · LLM {'✅' if health.get('llm') else '❌'}"
               f" · mode `{health.get('retrieval_mode', '?')}`")

    uploads = st.file_uploader("PDF upload karo", type=["pdf"], accept_multiple_files=True)
    if uploads and st.button("Upload & index", type="primary"):
        for f in uploads:
            with st.spinner(f"Indexing {f.name}…"):
                res = api("POST", "/documents", files={"file": (f.name, f.getvalue(), "application/pdf")},
                          timeout=600)
            if res:
                note = " (already indexed)" if res.get("duplicate") else ""
                st.success(f"{res['filename']}: {res['pages']} pages, {res['chunks']} chunks{note}")

    docs = api("GET", "/documents") or []
    ready = [d for d in docs if d["status"] == "ready"]
    st.subheader("Documents")
    selected = []
    for d in ready:
        cols = st.columns([5, 1])
        if cols[0].checkbox(f"{d['filename']} · {d['pages']}p", value=True, key=f"sel_{d['doc_id']}"):
            selected.append(d["doc_id"])
        if cols[1].button("🗑", key=f"del_{d['doc_id']}", help="Delete"):
            api("DELETE", f"/documents/{d['doc_id']}")
            st.rerun()
    if not ready:
        st.info("Abhi koi document nahi hai.")

    mode = st.selectbox("Retrieval", ["(server default)", "hybrid_rerank", "hybrid", "dense"])
    if st.button("New chat"):
        st.session_state.session_id, st.session_state.messages = None, []
        st.rerun()


# ---------------- main: chat ----------------
def show_citations(cits, rewritten=None, timings=None):
    if not cits and not rewritten:
        return
    with st.expander(f"Sources ({len(cits)})"):
        for c in cits:
            st.markdown(f"**[{c['n']}] {c['filename']} — page {c['page']}**")
            st.caption(c["snippet"])
        if rewritten:
            st.caption(f"Searched for: _{rewritten}_")
        if timings:
            st.caption(" · ".join(f"{k} {v:.0f} ms" for k, v in timings.items()))


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_citations(m.get("citations", []), m.get("rewritten"), m.get("timings"))

if question := st.chat_input("Apne documents se kuch bhi pucho…"):
    if ready and not selected:
        st.warning("Kam se kam ek document select karo.")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    payload = {"question": question, "session_id": st.session_state.session_id, "doc_ids": selected or None}
    if mode != "(server default)":
        payload["mode"] = mode
    with st.chat_message("assistant"), st.spinner("Soch raha hoon…"):
        res = api("POST", "/chat", json=payload, timeout=120)
        if res:
            st.session_state.session_id = res["session_id"]
            st.markdown(res["answer"])
            show_citations(res["citations"], res["rewritten_question"], res["timings_ms"])
            st.session_state.messages.append({
                "role": "assistant", "content": res["answer"], "citations": res["citations"],
                "rewritten": res["rewritten_question"], "timings": res["timings_ms"],
            })
