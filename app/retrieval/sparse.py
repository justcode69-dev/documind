"""BM25 sparse vectors without downloading any model.

Document side: BM25 term-frequency saturation  tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl)).
Query side: weight 1 per unique term.
IDF is applied by Qdrant itself (sparse vector config uses Modifier.IDF), so the score
Qdrant returns is a proper BM25 score that stays correct as documents are added/removed.
"""

import re
import zlib
from collections import Counter

from qdrant_client import models

K1, B = 1.2, 0.75
AVG_DOC_LEN = 120.0  # rough average tokens per ~800-char chunk; only affects length normalisation

# Latin words/numbers, plus Devanagari runs (incl. matras), so Hindi notes tokenize sensibly.
_TOKEN_RE = re.compile(r"[ऀ-ॿ]+|[a-z0-9]+(?:'[a-z]+)?")

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i if in into is it its
    of on or our she so that the their them then there these they this to was we were what
    when where which who why will with you your hai hain ka ki ke ko se me mein aur ya kya""".split()
)


def tokenize(text: str) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    return [t for t in toks if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def _index(token: str) -> int:
    # Stable across processes (unlike hash()); collisions in 2^31 space are negligible here.
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF


def _to_sparse(weights: dict[int, float]) -> models.SparseVector:
    idx = sorted(weights)
    return models.SparseVector(indices=idx, values=[weights[i] for i in idx])


def doc_sparse_vector(text: str) -> models.SparseVector:
    toks = tokenize(text)
    dl = len(toks) or 1
    weights: dict[int, float] = {}
    for tok, tf in Counter(toks).items():
        w = tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / AVG_DOC_LEN))
        i = _index(tok)
        weights[i] = weights.get(i, 0.0) + w
    return _to_sparse(weights)


def query_sparse_vector(text: str) -> models.SparseVector:
    return _to_sparse({_index(t): 1.0 for t in set(tokenize(text))})
