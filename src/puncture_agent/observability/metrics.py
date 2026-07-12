"""Dependency-free retrieval and agent metrics used by the evaluation contract.

Metric edge-case policy (version ``metrics-v1``):

* ``k`` must be positive.
* Empty relevance for Recall@K scores ``1.0`` only when the caller intends an
  explicit no-answer / empty-relevance case (current scaffold behaviour).
* Reciprocal rank is ``0.0`` when there is no hit.
* NDCG@K ideal score of zero yields ``1.0`` (nothing to rank).
* Relevance scores must be non-negative.
* Duplicate retrieved IDs are preserved in rank order for MRR/NDCG; Recall uses
  set intersection of the first ``k`` ranks.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

METRICS_SCHEMA_VERSION = "metrics-v1"


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


def mean_reciprocal_rank(
    ranked_lists: Sequence[Sequence[str]],
    relevant_sets: Sequence[Iterable[str]],
) -> float:
    """Mean reciprocal rank across multiple queries."""

    if not ranked_lists:
        raise ValueError("ranked_lists must be non-empty")
    if len(ranked_lists) != len(relevant_sets):
        raise ValueError("ranked_lists and relevant_sets length mismatch")
    scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(ranked_lists, relevant_sets)
    ]
    return sum(scores) / len(scores)


def active_version_hit_rate(
    cases: Sequence[Mapping[str, Any]],
) -> float:
    """Fraction of cases that retrieved the required current document version.

    Each case mapping must provide:

    * ``required_version`` (str | None): when null/empty the case is skipped;
    * ``retrieved_versions`` (Sequence[str]): versions actually retrieved;
    * optional ``current_version_hit`` (bool) which takes precedence when set.
    """

    scored: list[float] = []
    for case in cases:
        required = case.get("required_version")
        if required is None or required == "":
            explicit = case.get("current_version_hit")
            if explicit is None:
                continue
            scored.append(1.0 if explicit else 0.0)
            continue
        explicit = case.get("current_version_hit")
        if explicit is not None:
            scored.append(1.0 if explicit else 0.0)
            continue
        retrieved = list(case.get("retrieved_versions") or [])
        scored.append(1.0 if required in retrieved else 0.0)
    if not scored:
        return 1.0
    return sum(scored) / len(scored)


def acl_violation_rate(
    unauthorized_chunk_count: int,
    total_retrieved_chunk_count: int,
) -> float:
    """Unauthorized retrieved chunks / all retrieved chunks.

    Empty retrieval yields ``0.0`` (no violation observed). Counts must be
    non-negative integers.
    """

    if unauthorized_chunk_count < 0 or total_retrieved_chunk_count < 0:
        raise ValueError("chunk counts must be non-negative")
    if unauthorized_chunk_count > total_retrieved_chunk_count:
        raise ValueError("unauthorized count cannot exceed total retrieved")
    if total_retrieved_chunk_count == 0:
        return 0.0
    return unauthorized_chunk_count / total_retrieved_chunk_count


def precision_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int
) -> float:
    """Unique relevant hits in the first ``k`` results / min(k, retrieved)."""

    _validate_k(k)
    window = list(retrieved_ids[:k])
    if not window:
        return 0.0 if set(relevant_ids) else 1.0
    relevant = set(relevant_ids)
    hits = len(relevant.intersection(window))
    return hits / len(window)


def tool_selection_precision(
    called_tools: Iterable[str], expected_tools: Iterable[str]
) -> float:
    called = {name for name in called_tools if name}
    expected = set(expected_tools)
    if not called:
        return 1.0 if not expected else 0.0
    if not expected:
        # No expectation means we do not punish extra tools at the precision
        # layer; forbidden-tool checks are separate.
        return 1.0
    return len(called.intersection(expected)) / len(called)


def tool_selection_recall(
    called_tools: Iterable[str], expected_tools: Iterable[str]
) -> float:
    called = {name for name in called_tools if name}
    expected = set(expected_tools)
    if not expected:
        return 1.0
    return len(called.intersection(expected)) / len(expected)


def tool_parameter_validity_rate(
    calls: Sequence[Mapping[str, Any]],
    predicates: Sequence[Mapping[str, Any]],
) -> float:
    """Evaluate exact field predicates against recorded tool requests.

    Each predicate mapping supports:

    * ``tool_name`` (required)
    * ``field`` (dotted path into ``request``)
    * ``operator``: ``eq`` | ``ne`` | ``gt`` | ``gte`` | ``lt`` | ``lte`` | ``in`` | ``exists``
    * ``value`` (optional depending on operator)
    * optional ``required`` (default True): when True a missing call fails
    """

    if not predicates:
        return 1.0

    by_tool: dict[str, list[Mapping[str, Any]]] = {}
    for call in calls:
        name = call.get("tool_name")
        if not isinstance(name, str) or not name:
            continue
        by_tool.setdefault(name, []).append(call)

    outcomes: list[float] = []
    for predicate in predicates:
        tool_name = predicate.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("predicate.tool_name is required")
        field = predicate.get("field")
        operator = predicate.get("operator", "eq")
        expected = predicate.get("value")
        required = bool(predicate.get("required", True))
        matches = by_tool.get(tool_name) or []
        if not matches:
            outcomes.append(0.0 if required else 1.0)
            continue
        # Evaluate against the first matching call for determinism.
        request = matches[0].get("request") or {}
        actual = _read_path(request, str(field) if field is not None else "")
        outcomes.append(1.0 if _compare(actual, operator, expected) else 0.0)
    return sum(outcomes) / len(outcomes)


def retry_recovery_rate(
    results: Sequence[Mapping[str, Any]],
) -> float:
    """Cases marked as recovery scenarios that ended in the expected status."""

    scored = [
        1.0 if item.get("recovery_correct") else 0.0
        for item in results
        if item.get("is_recovery_case")
    ]
    if not scored:
        return 1.0
    return sum(scored) / len(scored)


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile for non-empty numeric samples."""

    if not values:
        raise ValueError("values must be non-empty")
    if not 0.0 <= p <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(float(v) for v in values)
    if p == 0:
        return ordered[0]
    rank = math.ceil((p / 100.0) * len(ordered)) - 1
    rank = min(max(rank, 0), len(ordered) - 1)
    return ordered[rank]


def _read_path(payload: Any, path: str) -> Any:
    if not path:
        return payload
    current = payload
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "exists":
        return actual is not None
    if operator == "eq":
        return actual == expected
    if operator == "ne":
        return actual != expected
    if operator == "gt":
        return actual is not None and actual > expected
    if operator == "gte":
        return actual is not None and actual >= expected
    if operator == "lt":
        return actual is not None and actual < expected
    if operator == "lte":
        return actual is not None and actual <= expected
    if operator == "in":
        return actual in expected
    raise ValueError(f"unsupported predicate operator: {operator!r}")
