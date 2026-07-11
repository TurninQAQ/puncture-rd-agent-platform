"""Offline golden-set and ablation evaluation with no external dependencies."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .client import RagService
from .errors import RagServiceError
from .models import RetrievalRequest, RetrievalResponse


@dataclass(frozen=True)
class GoldenQuery:
    """One synthetic or internal labelled retrieval case."""

    name: str
    request: RetrievalRequest
    relevant_document_ids: tuple[str, ...] = ()
    relevant_chunk_ids: tuple[str, ...] = ()
    expected_version: str | None = None
    expect_no_answer: bool = False
    forbidden_document_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("golden query name must be non-empty")
        for field_name in ("relevant_document_ids", "relevant_chunk_ids", "forbidden_document_ids"):
            values = tuple(getattr(self, field_name))
            if len(set(values)) != len(values) or any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{field_name} must contain unique non-empty strings")
            object.__setattr__(self, field_name, values)
        if self.expect_no_answer and (self.relevant_document_ids or self.relevant_chunk_ids):
            raise ValueError("no-answer cases must not define relevant IDs")
        if not self.expect_no_answer and not (self.relevant_document_ids or self.relevant_chunk_ids):
            raise ValueError("answerable cases must define relevant document or chunk IDs")


@dataclass(frozen=True)
class CaseEvaluation:
    name: str
    returned_document_ids: tuple[str, ...]
    returned_chunk_ids: tuple[str, ...]
    latency_ms: float
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    ndcg_at_10: float
    version_correct: bool | None
    acl_leak_count: int
    no_answer_correct: bool
    error_code: str | None = None


@dataclass(frozen=True)
class EvaluationReport:
    profile: str
    query_count: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    correct_version_hit_rate: float
    version_case_count: int
    acl_leak_count: int
    no_answer_accuracy: float
    p50_latency_ms: float
    p95_latency_ms: float
    error_count: int
    cases: tuple[CaseEvaluation, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_service(
    service: RagService,
    cases: Sequence[GoldenQuery],
    *,
    profile: str = "hybrid",
) -> EvaluationReport:
    """Run the labelled set and calculate metrics from actual responses/latency."""

    if not cases:
        raise ValueError("evaluation requires at least one golden query")
    case_results: list[CaseEvaluation] = []
    for case in cases:
        started = time.perf_counter()
        response: RetrievalResponse | None = None
        error_code: str | None = None
        try:
            response = service.retrieve(case.request)
        except RagServiceError as exc:
            error_code = exc.code
        latency_ms = (time.perf_counter() - started) * 1000.0
        chunks = response.chunks if response is not None else ()
        document_ids = tuple(chunk.document_id for chunk in chunks)
        chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
        relevance = tuple(_is_relevant(case, document_id, chunk_id) for document_id, chunk_id in zip(document_ids, chunk_ids))
        relevant_total = len(case.relevant_chunk_ids or case.relevant_document_ids)
        recall_5 = _recall_at_k(case, document_ids[:5], chunk_ids[:5], relevant_total)
        recall_10 = _recall_at_k(case, document_ids[:10], chunk_ids[:10], relevant_total)
        first_relevant = next((index for index, relevant in enumerate(relevance, start=1) if relevant), None)
        reciprocal_rank = 0.0 if first_relevant is None else 1.0 / first_relevant
        ndcg = _ndcg_at_10(relevance, relevant_total)
        version_correct = None
        if case.expected_version is not None:
            version_correct = any(
                _is_relevant(case, chunk.document_id, chunk.chunk_id)
                and chunk.version == case.expected_version
                for chunk in chunks
            )
        acl_leaks = sum(document_id in set(case.forbidden_document_ids) for document_id in document_ids)
        predicted_no_answer = not chunks
        case_results.append(
            CaseEvaluation(
                name=case.name,
                returned_document_ids=document_ids,
                returned_chunk_ids=chunk_ids,
                latency_ms=round(latency_ms, 6),
                recall_at_5=recall_5,
                recall_at_10=recall_10,
                reciprocal_rank=reciprocal_rank,
                ndcg_at_10=ndcg,
                version_correct=version_correct,
                acl_leak_count=acl_leaks,
                no_answer_correct=error_code is None and predicted_no_answer == case.expect_no_answer,
                error_code=error_code,
            )
        )

    answerable = [result for result, case in zip(case_results, cases) if not case.expect_no_answer]
    version_cases = [result for result in case_results if result.version_correct is not None]
    latencies = [result.latency_ms for result in case_results]
    return EvaluationReport(
        profile=profile,
        query_count=len(case_results),
        recall_at_5=_mean([result.recall_at_5 for result in answerable]),
        recall_at_10=_mean([result.recall_at_10 for result in answerable]),
        mrr=_mean([result.reciprocal_rank for result in answerable]),
        ndcg_at_10=_mean([result.ndcg_at_10 for result in answerable]),
        correct_version_hit_rate=_mean([1.0 if result.version_correct else 0.0 for result in version_cases]),
        version_case_count=len(version_cases),
        acl_leak_count=sum(result.acl_leak_count for result in case_results),
        no_answer_accuracy=_mean([1.0 if result.no_answer_correct else 0.0 for result in case_results]),
        p50_latency_ms=_percentile(latencies, 50.0),
        p95_latency_ms=_percentile(latencies, 95.0),
        error_count=sum(result.error_code is not None for result in case_results),
        cases=tuple(case_results),
    )


def evaluate_ablations(
    services: Mapping[str, RagService],
    cases: Sequence[GoldenQuery],
) -> Mapping[str, EvaluationReport]:
    """Evaluate BM25/dense/hybrid variants built with identical data and filters."""

    if not services:
        raise ValueError("at least one ablation service is required")
    return {
        profile: evaluate_service(service, cases, profile=profile)
        for profile, service in sorted(services.items())
    }


def _is_relevant(case: GoldenQuery, document_id: str, chunk_id: str) -> bool:
    if case.relevant_chunk_ids:
        return chunk_id in set(case.relevant_chunk_ids)
    return document_id in set(case.relevant_document_ids)


def _recall_at_k(
    case: GoldenQuery,
    document_ids: Sequence[str],
    chunk_ids: Sequence[str],
    relevant_total: int,
) -> float:
    if relevant_total <= 0:
        return 0.0
    if case.relevant_chunk_ids:
        hits = len(set(chunk_ids).intersection(case.relevant_chunk_ids))
    else:
        hits = len(set(document_ids).intersection(case.relevant_document_ids))
    return min(1.0, hits / relevant_total)


def _ndcg_at_10(relevance: Sequence[bool], relevant_total: int) -> float:
    if relevant_total <= 0:
        return 0.0
    dcg = sum(
        (1.0 / math.log2(rank + 1.0)) if relevant else 0.0
        for rank, relevant in enumerate(relevance[:10], start=1)
    )
    ideal_count = min(relevant_total, 10)
    ideal = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


__all__ = [
    "CaseEvaluation",
    "EvaluationReport",
    "GoldenQuery",
    "evaluate_ablations",
    "evaluate_service",
]
