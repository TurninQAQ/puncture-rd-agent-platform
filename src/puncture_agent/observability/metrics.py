"""Dependency-free retrieval metrics used by the RAG evaluation contract."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def recall_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int
) -> float:
    """Fraction of unique relevant documents found in the first ``k`` results."""

    _validate_k(k)
    relevant = set(relevant_ids)
    if not relevant:
        return 1.0
    hits = relevant.intersection(retrieved_ids[:k])
    return len(hits) / len(relevant)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """Inverse rank of the first relevant result; zero when there is no hit."""

    relevant = set(relevant_ids)
    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevance_by_id: dict[str, float],
    k: int,
) -> float:
    """Normalized discounted cumulative gain with non-negative relevance."""

    _validate_k(k)
    if any(score < 0 for score in relevance_by_id.values()):
        raise ValueError("relevance scores must be non-negative")

    def dcg(scores: Sequence[float]) -> float:
        return sum(
            (2**score - 1) / math.log2(index + 2)
            for index, score in enumerate(scores)
        )

    actual = [relevance_by_id.get(document_id, 0.0) for document_id in retrieved_ids[:k]]
    ideal = sorted(relevance_by_id.values(), reverse=True)[:k]
    ideal_score = dcg(ideal)
    return dcg(actual) / ideal_score if ideal_score else 1.0
