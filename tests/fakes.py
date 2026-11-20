"""Offline fakes so the full pipeline runs in CI without model downloads or API keys."""

import hashlib
import math

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.retrieval.sparse import tokenize


class FakeEmbedder:
    """Hashed bag-of-words, L2-normalised. Similar texts -> similar vectors; deterministic."""

    dim = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeReranker:
    """Scores by query-token overlap."""

    def rerank(self, query, chunks, top_n):
        q = set(tokenize(query))
        for c in chunks:
            c.rerank_score = float(len(q & set(tokenize(c.text))))
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_n]


class RecordingLLM(FakeListChatModel):
    """FakeListChatModel that also records the prompts it received."""

    calls: list = []

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        return super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)
