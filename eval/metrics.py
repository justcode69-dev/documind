"""Retrieval metrics. `rank` = 1-based position of the first relevant chunk, or None if absent."""

from statistics import mean


def first_relevant_rank(retrieved: list[tuple[str, int]], filename: str, expected_pages: list[int]) -> int | None:
    """retrieved: [(filename, page), ...] in ranked order."""
    pages = set(expected_pages)
    for i, (fn, page) in enumerate(retrieved, 1):
        if fn == filename and page in pages:
            return i
    return None


def hit_at_k(ranks: list[int | None], k: int) -> float:
    return mean(1.0 if r is not None and r <= k else 0.0 for r in ranks) if ranks else 0.0


def mrr_at_k(ranks: list[int | None], k: int) -> float:
    return mean(1.0 / r if r is not None and r <= k else 0.0 for r in ranks) if ranks else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[idx]
