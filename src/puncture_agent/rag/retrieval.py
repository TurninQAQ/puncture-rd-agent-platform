"""Hybrid retrieval pipeline: filters, lexical+dense recall, RRF, and evidence output."""

from __future__ import annotations

import hashlib
import math
import re
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, TimeoutError as FutureTimeoutError, wait
from dataclasses import dataclass, field
from threading import Lock
from types import MappingProxyType
from typing import Any, Protocol

from .backends import (
    BackendProtocolError,
    BackendTimeout,
    BackendUnavailable,
    EmbeddingBackend,
    EmbeddingUnavailable,
    IndexIncompatible,
    IndexedChunk,
    ParentRecord,
    RerankerBackend,
    RerankerUnavailable,
    RetrievalFilters,
    SearchBackend,
    SearchHit,
    chunk_matches_filters,
    estimate_tokens,
    parent_matches_filters,
)
from .errors import RagServiceError
from .models import EnterpriseRagConfig, RetrievalRequest, RetrievalResponse, RetrievedChunk


_ALLOWED_METADATA_FILTERS = frozenset({"status", "source_type", "category", "language"})
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class RagRuntimeConfig:
    """Production-only controls kept outside the stable Agent-facing contracts."""

    rrf_k: int = 60
    context_budget_tokens: int = 4000
    minimum_relevance: float = 0.15
    reranker_failure_mode: str = "fail"
    max_query_characters: int = 8192
    max_filter_count: int = 16
    allowed_metadata_filters: frozenset[str] = _ALLOWED_METADATA_FILTERS
    trace_redaction_patterns: tuple[str, ...] = ()
    trace_query_mode: str = "hash"
    recall_mode: str = "hybrid"
    use_reranker: bool = True
    expand_parent_context: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.rrf_k, bool) or not isinstance(self.rrf_k, int) or self.rrf_k <= 0:
            raise ValueError("rrf_k must be a positive integer")
        if (
            isinstance(self.context_budget_tokens, bool)
            or not isinstance(self.context_budget_tokens, int)
            or self.context_budget_tokens <= 0
        ):
            raise ValueError("context_budget_tokens must be a positive integer")
        if not isinstance(self.minimum_relevance, (int, float)) or isinstance(self.minimum_relevance, bool):
            raise ValueError("minimum_relevance must be numeric")
        if not math.isfinite(float(self.minimum_relevance)) or not 0.0 <= self.minimum_relevance <= 1.0:
            raise ValueError("minimum_relevance must be in [0, 1]")
        if self.reranker_failure_mode not in {"degrade", "fail"}:
            raise ValueError("reranker_failure_mode must be degrade or fail")
        if self.trace_query_mode not in {"hash", "redacted"}:
            raise ValueError("trace_query_mode must be hash or redacted")
        if self.recall_mode not in {"hybrid", "lexical", "dense"}:
            raise ValueError("recall_mode must be hybrid, lexical, or dense")
        if not isinstance(self.use_reranker, bool) or not isinstance(self.expand_parent_context, bool):
            raise ValueError("ablation switches must be booleans")
        if self.max_query_characters <= 0 or self.max_filter_count <= 0:
            raise ValueError("query and filter limits must be positive")
        object.__setattr__(self, "allowed_metadata_filters", frozenset(self.allowed_metadata_filters))
        object.__setattr__(self, "trace_redaction_patterns", tuple(self.trace_redaction_patterns))
        for pattern in self.trace_redaction_patterns:
            re.compile(pattern)


class RagTraceSink(Protocol):
    def record(self, trace: Mapping[str, Any]) -> None: ...


class NullTraceSink:
    def record(self, trace: Mapping[str, Any]) -> None:
        return None


class InMemoryTraceSink:
    """Thread-safe trace collector for tests and local observability integration."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._records: list[Mapping[str, Any]] = []

    def record(self, trace: Mapping[str, Any]) -> None:
        with self._lock:
            self._records.append(MappingProxyType(dict(trace)))

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        with self._lock:
            return tuple(self._records)


@dataclass(frozen=True)
class FusionCandidate:
    chunk: IndexedChunk
    rrf_score: float
    lexical_rank: int | None = None
    dense_rank: int | None = None
    lexical_score: float | None = None
    dense_score: float | None = None
    pre_rerank_rank: int = 0


def normalize_query(query: str) -> str:
    """Normalize Unicode and expand only approved aliases without removing terms."""

    normalized = " ".join(unicodedata.normalize("NFC", query).strip().split())
    folded = normalized.casefold()
    additions: list[str] = []
    if "nii.gz" in folded and "nifti" not in folded:
        additions.append("NIfTI")
    elif "nifti" in folded and "nii.gz" not in folded:
        additions.append("nii.gz")
    if "mcs" in folded and "mimics" not in folded:
        additions.append("Mimics")
    return " ".join([normalized, *additions])


def reciprocal_rank_fusion(
    lexical_hits: Sequence[SearchHit],
    dense_hits: Sequence[SearchHit],
    *,
    k: int = 60,
) -> tuple[FusionCandidate, ...]:
    """Fuse two ranked branches using exact RRF and deterministic tie breaking."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("RRF k must be a positive integer")
    by_id: dict[str, dict[str, Any]] = {}
    for branch, hits in (("lexical", lexical_hits), ("dense", dense_hits)):
        seen: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            _validate_search_hit(hit)
            chunk_id = hit.chunk.chunk_id
            if chunk_id in seen:
                raise BackendProtocolError(f"{branch} branch returned a duplicate chunk ID")
            seen.add(chunk_id)
            entry = by_id.setdefault(chunk_id, {"chunk": hit.chunk, "rrf_score": 0.0})
            if entry["chunk"] != hit.chunk:
                raise BackendProtocolError("branches disagree on chunk payload for the same chunk ID")
            entry["rrf_score"] += 1.0 / (k + rank)
            entry[f"{branch}_rank"] = rank
            entry[f"{branch}_score"] = hit.score
    candidates = [
        FusionCandidate(
            chunk=entry["chunk"],
            rrf_score=entry["rrf_score"],
            lexical_rank=entry.get("lexical_rank"),
            dense_rank=entry.get("dense_rank"),
            lexical_score=entry.get("lexical_score"),
            dense_score=entry.get("dense_score"),
        )
        for entry in by_id.values()
    ]
    candidates.sort(key=lambda candidate: candidate.chunk.chunk_id)
    candidates.sort(key=lambda candidate: candidate.chunk.updated_at, reverse=True)
    candidates.sort(key=lambda candidate: candidate.rrf_score, reverse=True)
    return tuple(
        FusionCandidate(
            chunk=candidate.chunk,
            rrf_score=candidate.rrf_score,
            lexical_rank=candidate.lexical_rank,
            dense_rank=candidate.dense_rank,
            lexical_score=candidate.lexical_score,
            dense_score=candidate.dense_score,
            pre_rerank_rank=rank,
        )
        for rank, candidate in enumerate(candidates, start=1)
    )


class HybridRetrievalEngine:
    def __init__(
        self,
        config: EnterpriseRagConfig,
        search_backend: SearchBackend,
        embedding_backend: EmbeddingBackend,
        reranker_backend: RerankerBackend | None,
        *,
        runtime: RagRuntimeConfig | None = None,
        trace_sink: RagTraceSink | None = None,
    ) -> None:
        self.config = config
        self.search_backend = search_backend
        self.embedding_backend = embedding_backend
        self.reranker_backend = reranker_backend
        self.runtime = runtime or RagRuntimeConfig()
        self.trace_sink = trace_sink or NullTraceSink()

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        if not isinstance(request, RetrievalRequest):
            raise RagServiceError(
                "RAG_INVALID_REQUEST",
                "retrieve requires a RetrievalRequest",
                retryable=False,
            )
        started = time.perf_counter()
        deadline = started + self.config.timeout_seconds
        stage_latencies: dict[str, float] = {}
        warnings: list[str] = []
        rewritten_query = ""
        trace_id = _trace_id(request.request_id)
        try:
            request_started = time.perf_counter()
            filters = self._validate_and_build_filters(request)
            rewritten_query = normalize_query(request.query)
            stage_latencies["request_validation_ms"] = _elapsed_ms(request_started)

            compatibility_started = time.perf_counter()
            descriptor = _call_with_deadline(
                self.search_backend.descriptor,
                deadline,
                "index validation",
            )
            self._validate_index_compatibility(descriptor)
            stage_latencies["index_validation_ms"] = _elapsed_ms(compatibility_started)

            embedding_started = time.perf_counter()
            query_vector = (
                ()
                if self.runtime.recall_mode == "lexical"
                else self._embed_query(rewritten_query, deadline)
            )
            stage_latencies["embedding_ms"] = _elapsed_ms(embedding_started)

            recall_started = time.perf_counter()
            lexical_hits, dense_hits = self._recall(rewritten_query, query_vector, filters, deadline)
            stage_latencies["recall_ms"] = _elapsed_ms(recall_started)

            fusion_started = time.perf_counter()
            fused = reciprocal_rank_fusion(lexical_hits, dense_hits, k=self.runtime.rrf_k)
            # Mandatory post-filtering protects against a provider that accidentally
            # omitted a backend filter.  It also prevents ACL leaks from test fakes.
            stage_latencies["fusion_filter_ms"] = _elapsed_ms(fusion_started)

            rerank_started = time.perf_counter()
            rerank_limit = max(self.config.rerank_top_k, request.top_k)
            reranked, degraded = self._rerank(
                rewritten_query,
                fused[:rerank_limit],
                deadline,
            )
            if degraded:
                warnings.append("RERANKER_DEGRADED")
            stage_latencies["rerank_ms"] = _elapsed_ms(rerank_started)

            relevant = [item for item in reranked if item[1] >= self.runtime.minimum_relevance]
            if not relevant:
                warnings.append("NO_RELEVANT_KNOWLEDGE")
                response = RetrievalResponse(
                    request_id=request.request_id,
                    rewritten_query=rewritten_query,
                    chunks=(),
                    retrieval_mode=self._retrieval_mode(),
                    trace_id=trace_id,
                    latency_ms=_elapsed_ms(started),
                    warnings=tuple(warnings),
                )
                self._record_trace(
                    request,
                    response,
                    filters,
                    descriptor,
                    lexical_hits,
                    dense_hits,
                    fused,
                    relevant,
                    stage_latencies,
                    degraded,
                )
                return response

            expansion_started = time.perf_counter()
            chunks = self._expand_and_format(
                relevant,
                filters,
                top_k=request.top_k,
                descriptor_generation=descriptor.generation,
                deadline=deadline,
            )
            stage_latencies["parent_expansion_ms"] = _elapsed_ms(expansion_started)
            if not chunks:
                warnings.append("NO_RELEVANT_KNOWLEDGE")
            response = RetrievalResponse(
                request_id=request.request_id,
                rewritten_query=rewritten_query,
                chunks=chunks,
                retrieval_mode=self._retrieval_mode(),
                trace_id=trace_id,
                latency_ms=_elapsed_ms(started),
                warnings=tuple(warnings),
            )
            self._record_trace(
                request,
                response,
                filters,
                descriptor,
                lexical_hits,
                dense_hits,
                fused,
                relevant,
                stage_latencies,
                degraded,
            )
            return response
        except RagServiceError as exc:
            self._record_error_trace(request, trace_id, rewritten_query, exc, started, stage_latencies)
            raise
        except IndexIncompatible as exc:
            error = RagServiceError("RAG_INDEX_INCOMPATIBLE", str(exc), retryable=False)
            self._record_error_trace(request, trace_id, rewritten_query, error, started, stage_latencies)
            raise error from exc
        except BackendProtocolError as exc:
            error = RagServiceError("RAG_PROTOCOL_ERROR", str(exc), retryable=False)
            self._record_error_trace(request, trace_id, rewritten_query, error, started, stage_latencies)
            raise error from exc
        except BackendTimeout as exc:
            error = RagServiceError("RAG_TIMEOUT", "retrieval backend timed out", retryable=True)
            self._record_error_trace(request, trace_id, rewritten_query, error, started, stage_latencies)
            raise error from exc
        except BackendUnavailable as exc:
            error = RagServiceError("RAG_BACKEND_UNAVAILABLE", "search backend unavailable", retryable=True)
            self._record_error_trace(request, trace_id, rewritten_query, error, started, stage_latencies)
            raise error from exc
        except Exception as exc:
            error = RagServiceError("RAG_PROTOCOL_ERROR", "unexpected retrieval failure", retryable=False)
            self._record_error_trace(request, trace_id, rewritten_query, error, started, stage_latencies)
            raise error from exc

    def _validate_and_build_filters(self, request: RetrievalRequest) -> RetrievalFilters:
        if not isinstance(request.request_id, str) or not request.request_id.strip() or len(request.request_id) > 256:
            raise RagServiceError("RAG_INVALID_REQUEST", "request_id is invalid", retryable=False)
        if _CONTROL_CHARACTERS.search(request.request_id):
            raise RagServiceError("RAG_INVALID_REQUEST", "request_id contains control characters", retryable=False)
        if not isinstance(request.query, str) or not request.query.strip():
            raise RagServiceError("RAG_INVALID_REQUEST", "query must be non-empty", retryable=False)
        if len(request.query) > self.runtime.max_query_characters:
            raise RagServiceError("RAG_INVALID_REQUEST", "query exceeds configured limit", retryable=False)
        if _CONTROL_CHARACTERS.search(request.query):
            raise RagServiceError("RAG_INVALID_REQUEST", "query contains unsupported control characters", retryable=False)
        if len(request.access_scopes) > 64:
            raise RagServiceError("RAG_PERMISSION_DENIED", "effective access scope set is invalid", retryable=False)
        for scope in request.access_scopes:
            if not isinstance(scope, str) or not scope.strip() or len(scope) > 128:
                raise RagServiceError("RAG_PERMISSION_DENIED", "effective access scope set is invalid", retryable=False)
        if len(set(request.access_scopes)) != len(request.access_scopes):
            raise RagServiceError("RAG_PERMISSION_DENIED", "effective access scopes must be unique", retryable=False)
        if len(request.modules) > 64:
            raise RagServiceError("RAG_INVALID_REQUEST", "too many module filters", retryable=False)
        for module in request.modules:
            if not isinstance(module, str) or not module.strip() or len(module) > 128:
                raise RagServiceError("RAG_INVALID_REQUEST", "module filter is invalid", retryable=False)
        if request.required_version is not None and (
            not isinstance(request.required_version, str)
            or not request.required_version.strip()
            or len(request.required_version) > 128
        ):
            raise RagServiceError("RAG_INVALID_REQUEST", "required_version is invalid", retryable=False)
        if not isinstance(request.metadata_filters, Mapping):
            raise RagServiceError("RAG_INVALID_REQUEST", "metadata_filters must be an object", retryable=False)
        if len(request.metadata_filters) > self.runtime.max_filter_count:
            raise RagServiceError("RAG_INVALID_REQUEST", "too many metadata filters", retryable=False)
        filters = dict(request.metadata_filters)
        unknown = set(filters).difference(self.runtime.allowed_metadata_filters)
        if unknown:
            raise RagServiceError(
                "RAG_INVALID_REQUEST",
                "metadata filter is not approved",
                retryable=False,
                details={"invalid_filter_count": len(unknown)},
            )
        for key, value in filters.items():
            if isinstance(value, (Mapping, list, tuple, set)) or value is None:
                raise RagServiceError("RAG_INVALID_REQUEST", f"metadata filter {key} must be scalar", retryable=False)
            if isinstance(value, float) and not math.isfinite(value):
                raise RagServiceError("RAG_INVALID_REQUEST", f"metadata filter {key} is not finite", retryable=False)
        if "version" in filters and request.required_version is not None and filters["version"] != request.required_version:
            raise RagServiceError("RAG_INVALID_REQUEST", "version filters conflict", retryable=False)
        if "module" in filters and request.modules and filters["module"] not in request.modules:
            raise RagServiceError("RAG_INVALID_REQUEST", "module filters conflict", retryable=False)
        requested_status = filters.get("status")
        if requested_status is not None and requested_status not in {"active", "deprecated", "superseded"}:
            raise RagServiceError("RAG_INVALID_REQUEST", "status filter is invalid", retryable=False)
        if requested_status in {"deprecated", "superseded"} and request.required_version is None:
            raise RagServiceError(
                "RAG_INVALID_REQUEST",
                "non-active status requires an exact required_version",
                retryable=False,
            )
        if requested_status is not None:
            allowed_statuses = (str(requested_status),)
        elif request.required_version is not None:
            allowed_statuses = ("active", "deprecated", "superseded")
        else:
            allowed_statuses = ("active",)
        return RetrievalFilters(
            access_scopes=request.access_scopes,
            modules=request.modules,
            required_version=request.required_version,
            allowed_statuses=allowed_statuses,
            metadata_filters=filters,
        )

    def _validate_index_compatibility(self, descriptor: Any) -> None:
        required = (
            "embedding_model",
            "embedding_revision",
            "embedding_dimension",
            "generation",
            "parser_version",
            "chunker_version",
            "document_count",
        )
        if any(not hasattr(descriptor, field) for field in required):
            raise BackendProtocolError("search backend descriptor is malformed")
        if descriptor.embedding_dimension == 0 and descriptor.document_count == 0:
            return
        if self.runtime.recall_mode == "lexical":
            return
        if (
            descriptor.embedding_model != self.embedding_backend.model_name
            or descriptor.embedding_revision != self.embedding_backend.revision
            or descriptor.embedding_dimension != self.embedding_backend.dimension
        ):
            raise IndexIncompatible("query embedding provider does not match the active index")
        manifest_checks = {
            "query_instruction": str(getattr(self.embedding_backend, "query_instruction", "")),
            "document_instruction": str(getattr(self.embedding_backend, "document_instruction", "")),
            "vectors_normalized": bool(getattr(self.embedding_backend, "vectors_normalized", False)),
            "tokenizer_revision": str(getattr(self.embedding_backend, "tokenizer_revision", "unspecified")),
            "max_input_tokens": int(getattr(self.embedding_backend, "max_input_tokens", 0)),
        }
        if any(getattr(descriptor, key, None) != value for key, value in manifest_checks.items()):
            raise IndexIncompatible("query embedding manifest does not match the active index")

    def _embed_query(self, query: str, deadline: float) -> tuple[float, ...]:
        try:
            vector = _call_with_deadline(
                lambda: self.embedding_backend.embed_query(query),
                deadline,
                "embedding",
            )
        except (BackendTimeout, TimeoutError) as exc:
            raise RagServiceError(
                "RAG_TIMEOUT", "query embedding timed out", retryable=True, details={"stage": "embedding"}
            ) from exc
        except EmbeddingUnavailable as exc:
            raise RagServiceError(
                "RAG_EMBEDDING_UNAVAILABLE",
                "query embedding unavailable",
                retryable=True,
                details={"stage": "embedding"},
            ) from exc
        except IndexIncompatible:
            raise
        except BackendProtocolError:
            raise
        except Exception as exc:
            raise RagServiceError(
                "RAG_EMBEDDING_UNAVAILABLE",
                "query embedding failed",
                retryable=True,
                details={"stage": "embedding"},
            ) from exc
        if isinstance(vector, (str, bytes)):
            raise RagServiceError("RAG_PROTOCOL_ERROR", "query embedding is malformed", retryable=False)
        try:
            values = tuple(float(value) for value in vector)
        except (TypeError, ValueError) as exc:
            raise RagServiceError("RAG_PROTOCOL_ERROR", "query embedding is malformed", retryable=False) from exc
        if len(values) != self.embedding_backend.dimension or any(not math.isfinite(value) for value in values):
            raise RagServiceError("RAG_PROTOCOL_ERROR", "query embedding is malformed", retryable=False)
        return values

    def _recall(
        self,
        query: str,
        vector: Sequence[float],
        filters: RetrievalFilters,
        deadline: float,
    ) -> tuple[tuple[SearchHit, ...], tuple[SearchHit, ...]]:
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-recall")
        futures = {}
        if self.runtime.recall_mode in {"hybrid", "lexical"}:
            futures["lexical"] = executor.submit(
                self.search_backend.lexical_search, query, filters, self.config.lexical_top_k
            )
        if self.runtime.recall_mode in {"hybrid", "dense"}:
            futures["dense"] = executor.submit(
                self.search_backend.dense_search, vector, filters, self.config.dense_top_k
            )
        try:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise BackendTimeout("retrieval deadline exhausted before recall")
            done, pending = wait(
                tuple(futures.values()),
                timeout=remaining,
                return_when=ALL_COMPLETED,
            )
            if pending:
                for future in pending:
                    future.cancel()
                raise BackendTimeout("retrieval recall exceeded the total deadline")
            lexical = tuple(futures["lexical"].result()) if "lexical" in futures else ()
            dense = tuple(futures["dense"].result()) if "dense" in futures else ()
        except (BackendTimeout, TimeoutError) as exc:
            raise RagServiceError(
                "RAG_TIMEOUT", "search recall timed out", retryable=True, details={"stage": "recall"}
            ) from exc
        except BackendUnavailable as exc:
            raise RagServiceError(
                "RAG_BACKEND_UNAVAILABLE", "search backend unavailable", retryable=True, details={"stage": "recall"}
            ) from exc
        except BackendProtocolError:
            raise
        except Exception as exc:
            raise RagServiceError(
                "RAG_BACKEND_UNAVAILABLE", "search recall failed", retryable=True, details={"stage": "recall"}
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if len(lexical) > self.config.lexical_top_k or len(dense) > self.config.dense_top_k:
            raise BackendProtocolError("search backend exceeded the requested candidate limit")
        for hit in (*lexical, *dense):
            _validate_search_hit(hit)
            if not chunk_matches_filters(hit.chunk, filters):
                raise BackendProtocolError("search backend returned a hit outside mandatory filters")
        return lexical, dense

    def _rerank(
        self,
        query: str,
        candidates: Sequence[FusionCandidate],
        deadline: float,
    ) -> tuple[list[tuple[FusionCandidate, float]], bool]:
        if not candidates:
            return [], False
        if not self.runtime.use_reranker:
            return self._degraded_ranking(candidates), False
        if self.reranker_backend is None:
            if self.runtime.reranker_failure_mode == "fail":
                raise RagServiceError("RAG_RERANKER_UNAVAILABLE", "reranker is not configured", retryable=False)
            return self._degraded_ranking(candidates), True
        try:
            results = _call_with_deadline(
                lambda: tuple(
                    self.reranker_backend.rerank(
                        query,
                        [candidate.chunk for candidate in candidates],
                    )
                ),
                deadline,
                "reranker",
            )
        except (BackendTimeout, TimeoutError) as exc:
            if self.runtime.reranker_failure_mode == "degrade":
                return self._degraded_ranking(candidates), True
            raise RagServiceError(
                "RAG_TIMEOUT",
                "reranker timed out",
                retryable=True,
                details={"stage": "reranker"},
            ) from exc
        except RerankerUnavailable as exc:
            if self.runtime.reranker_failure_mode == "degrade":
                return self._degraded_ranking(candidates), True
            raise RagServiceError(
                "RAG_RERANKER_UNAVAILABLE",
                "reranker unavailable",
                retryable=True,
                details={"stage": "reranker"},
            ) from exc
        except Exception as exc:
            raise RagServiceError(
                "RAG_PROTOCOL_ERROR",
                "reranker returned an invalid response",
                retryable=False,
                details={"stage": "reranker"},
            ) from exc
        by_id: dict[str, float] = {}
        expected = {candidate.chunk.chunk_id for candidate in candidates}
        for result in results:
            if not hasattr(result, "chunk_id") or not hasattr(result, "score"):
                raise RagServiceError("RAG_PROTOCOL_ERROR", "reranker response is malformed", retryable=False)
            if result.chunk_id in by_id or result.chunk_id not in expected:
                raise RagServiceError("RAG_PROTOCOL_ERROR", "reranker chunk IDs are invalid", retryable=False)
            score = float(result.score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise RagServiceError("RAG_PROTOCOL_ERROR", "reranker score is not normalized", retryable=False)
            by_id[result.chunk_id] = score
        if set(by_id) != expected:
            raise RagServiceError("RAG_PROTOCOL_ERROR", "reranker omitted candidates", retryable=False)
        ranked = [(candidate, by_id[candidate.chunk.chunk_id]) for candidate in candidates]
        ranked.sort(key=lambda item: item[0].chunk.chunk_id)
        ranked.sort(key=lambda item: item[0].chunk.updated_at, reverse=True)
        ranked.sort(key=lambda item: item[0].pre_rerank_rank)
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked, False

    def _degraded_ranking(self, candidates: Sequence[FusionCandidate]) -> list[tuple[FusionCandidate, float]]:
        ranked: list[tuple[FusionCandidate, float]] = []
        for candidate in candidates:
            lexical_confidence = (
                1.0 - math.exp(-max(0.0, candidate.lexical_score or 0.0))
                if candidate.lexical_rank is not None
                else 0.0
            )
            dense_confidence = max(0.0, min(1.0, candidate.dense_score or 0.0))
            ranked.append((candidate, max(lexical_confidence, dense_confidence)))
        ranked.sort(key=lambda item: item[0].chunk.chunk_id)
        ranked.sort(key=lambda item: item[0].chunk.updated_at, reverse=True)
        ranked.sort(key=lambda item: item[0].pre_rerank_rank)
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _expand_and_format(
        self,
        ranked: Sequence[tuple[FusionCandidate, float]],
        filters: RetrievalFilters,
        *,
        top_k: int,
        descriptor_generation: int,
        deadline: float,
    ) -> tuple[RetrievedChunk, ...]:
        selected: list[tuple[FusionCandidate, float, str, str]] = []
        seen_parents: set[str] = set()
        remaining_budget = self.runtime.context_budget_tokens
        for candidate, raw_score in ranked:
            if len(selected) >= top_k or remaining_budget <= 0:
                break
            chunk = candidate.chunk
            if not chunk_matches_filters(chunk, filters) or chunk.parent_id in seen_parents:
                continue
            parent = _call_with_deadline(
                lambda: self.search_backend.get_parent(chunk.parent_id, filters),
                deadline,
                "parent expansion",
            )
            if parent is None:
                raise BackendProtocolError("search backend omitted required parent context")
            self._validate_parent(parent, chunk, filters)
            text = chunk.text
            section = " > ".join(chunk.section_path)
            if self.runtime.expand_parent_context and estimate_tokens(parent.text) <= remaining_budget:
                text = parent.text
                section = " > ".join(parent.section_path)
            text = _truncate_to_budget(text, remaining_budget)
            consumed = estimate_tokens(text)
            if not text or consumed > remaining_budget:
                continue
            remaining_budget -= consumed
            seen_parents.add(chunk.parent_id)
            selected.append((candidate, raw_score, text, section))
        if not selected:
            return ()
        chunks: list[RetrievedChunk] = []
        for rank, (candidate, raw_score, text, section) in enumerate(selected, start=1):
            chunk = candidate.chunk
            normalized_score = min(1.0, max(0.0, raw_score))
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    module=chunk.module,
                    version=chunk.version,
                    section=section,
                    text=text,
                    score=round(normalized_score, 6),
                    rank=rank,
                    citation=(
                        f"[{_citation_component(chunk.title)} | "
                        f"{_citation_component(chunk.version)} | "
                        f"{_citation_component(section)}]"
                    ),
                    metadata={
                        **{
                            key: value
                            for key, value in chunk.metadata.items()
                            if key in {"category", "language"}
                        },
                        "status": chunk.status,
                        "source_type": chunk.source_type,
                        "updated_at": chunk.updated_at,
                        "parent_id": chunk.parent_id,
                        "index_generation": descriptor_generation,
                        "rrf_score": round(candidate.rrf_score, 8),
                        "lexical_rank": candidate.lexical_rank,
                        "dense_rank": candidate.dense_rank,
                        "pre_rerank_rank": candidate.pre_rerank_rank,
                        "rerank_score": round(raw_score, 6),
                        "parser_version": chunk.parser_version,
                        "chunker_version": chunk.chunker_version,
                    },
                )
            )
        return tuple(chunks)

    @staticmethod
    def _validate_parent(parent: ParentRecord, chunk: IndexedChunk, filters: RetrievalFilters) -> None:
        if not parent_matches_filters(parent, filters):
            raise BackendProtocolError("parent context violates mandatory filters")
        if (
            parent.parent_id != chunk.parent_id
            or parent.document_id != chunk.document_id
            or parent.version != chunk.version
            or parent.access_scopes != chunk.access_scopes
            or parent.title != chunk.title
            or parent.module != chunk.module
            or parent.status != chunk.status
            or parent.owner != chunk.owner
            or parent.source_type != chunk.source_type
            or parent.updated_at != chunk.updated_at
        ):
            raise BackendProtocolError("parent and child metadata are inconsistent")

    def _record_trace(
        self,
        request: RetrievalRequest,
        response: RetrievalResponse,
        filters: RetrievalFilters,
        descriptor: Any,
        lexical_hits: Sequence[SearchHit],
        dense_hits: Sequence[SearchHit],
        fused: Sequence[FusionCandidate],
        relevant: Sequence[tuple[FusionCandidate, float]],
        stage_latencies: Mapping[str, float],
        degraded: bool,
    ) -> None:
        trace = {
            "trace_id": response.trace_id,
            "request_id": request.request_id,
            "original_query": self._trace_query(request.query),
            "rewritten_query": self._trace_query(response.rewritten_query),
            "effective_filters": {
                "modules": filters.modules,
                "required_version": filters.required_version,
                "allowed_statuses": filters.allowed_statuses,
                "access_scope_count": len(filters.access_scopes),
                "metadata_filter_keys": tuple(sorted(filters.metadata_filters)),
            },
            "candidate_counts": {
                "lexical": len(lexical_hits),
                "dense": len(dense_hits),
                "fused": len(fused),
                "relevant": len(relevant),
            },
            "rank_changes": tuple(
                {
                    "chunk_id": candidate.chunk.chunk_id,
                    "rrf_rank": candidate.pre_rerank_rank,
                    "rerank_rank": rank,
                }
                for rank, (candidate, _) in enumerate(relevant, start=1)
            ),
            "versions": {
                "index_generation": descriptor.generation,
                "embedding_model": descriptor.embedding_model,
                "embedding_revision": descriptor.embedding_revision,
                "reranker_model": getattr(self.reranker_backend, "model_name", None),
                "reranker_revision": getattr(self.reranker_backend, "revision", None),
                "parser_version": descriptor.parser_version,
                "chunker_version": descriptor.chunker_version,
            },
            "stage_latencies_ms": dict(stage_latencies),
            "latency_ms": response.latency_ms,
            "citations": tuple(chunk.citation for chunk in response.chunks),
            "warnings": response.warnings,
            "degraded": degraded,
        }
        try:
            self.trace_sink.record(trace)
        except Exception:
            # Retrieval correctness must not depend on an optional telemetry sink.
            return

    def _record_error_trace(
        self,
        request: RetrievalRequest,
        trace_id: str,
        rewritten_query: str,
        error: RagServiceError,
        started: float,
        stage_latencies: Mapping[str, float],
    ) -> None:
        try:
            self.trace_sink.record(
                {
                    "trace_id": trace_id,
                    "request_id": getattr(request, "request_id", "invalid"),
                    "original_query": self._trace_query(getattr(request, "query", "")),
                    "rewritten_query": self._trace_query(rewritten_query),
                    "stage_latencies_ms": dict(stage_latencies),
                    "latency_ms": _elapsed_ms(started),
                    "error": {
                        "code": error.code,
                        "retryable": error.retryable,
                        "details": _sanitize_details(error.details),
                    },
                }
            )
        except Exception:
            return

    def _redact(self, text: str) -> str:
        redacted = text
        for pattern in self.runtime.trace_redaction_patterns:
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)
        return redacted

    def _retrieval_mode(self) -> str:
        parts = [self.runtime.recall_mode]
        if self.runtime.recall_mode == "hybrid":
            parts.append("rrf")
        parts.append("rerank" if self.runtime.use_reranker else "no_rerank")
        parts.append("parent" if self.runtime.expand_parent_context else "child")
        return "_".join(parts)

    def _trace_query(self, text: str) -> Mapping[str, Any] | str:
        if self.runtime.trace_query_mode == "redacted":
            return self._redact(text)
        encoded = text.encode("utf-8", errors="replace")
        return {
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "character_count": len(text),
            "byte_count": len(encoded),
        }


def _validate_search_hit(hit: Any) -> None:
    if not isinstance(hit, SearchHit):
        raise BackendProtocolError("search backend returned a malformed hit")
    if not isinstance(hit.chunk, IndexedChunk):
        raise BackendProtocolError("search backend returned a malformed chunk")
    if not isinstance(hit.score, (int, float)) or isinstance(hit.score, bool) or not math.isfinite(hit.score):
        raise BackendProtocolError("search backend returned a non-finite score")


def _call_with_deadline(
    operation: Callable[[], Any],
    deadline: float,
    stage: str,
) -> Any:
    """Run one blocking provider call against the request's remaining deadline."""

    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise BackendTimeout(f"{stage} exceeded the total retrieval deadline")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"rag-{stage.replace(' ', '-')}")
    future = executor.submit(operation)
    try:
        return future.result(timeout=remaining)
    except (FutureTimeoutError, TimeoutError) as exc:
        future.cancel()
        raise BackendTimeout(f"{stage} exceeded the total retrieval deadline") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _truncate_to_budget(text: str, budget: int) -> str:
    if estimate_tokens(text) <= budget:
        return text
    words = text.split()
    if len(words) > 1:
        return " ".join(words[:budget]).strip()
    # Chinese and other no-whitespace content still receives a deterministic bound.
    return text[: max(1, budget)].strip()


def _trace_id(request_id: str) -> str:
    digest = hashlib.sha256(str(request_id).encode("utf-8", errors="replace")).hexdigest()[:20]
    return f"rag-{digest}"


def _citation_component(value: str) -> str:
    normalized = " ".join(str(value).split())
    return (
        normalized.replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("|", "\\|")
    )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _sanitize_details(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "[TRUNCATED]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:256]
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, child in list(value.items())[:32]:
            normalized_key = str(key)[:64]
            if any(term in normalized_key.casefold() for term in ("query", "text", "title", "token", "secret")):
                output[normalized_key] = "[REDACTED]"
            else:
                output[normalized_key] = _sanitize_details(child, depth=depth + 1)
        return output
    if isinstance(value, (list, tuple)):
        return tuple(_sanitize_details(child, depth=depth + 1) for child in value[:32])
    return "[UNSUPPORTED]"


__all__ = [
    "FusionCandidate",
    "HybridRetrievalEngine",
    "InMemoryTraceSink",
    "NullTraceSink",
    "RagRuntimeConfig",
    "RagTraceSink",
    "normalize_query",
    "reciprocal_rank_fusion",
]
