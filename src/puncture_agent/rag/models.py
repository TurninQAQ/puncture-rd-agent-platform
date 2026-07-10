"""Stable data contracts for project-knowledge retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class KnowledgeDocument:
    """A small mock document; production indexing may split it into child chunks."""

    document_id: str
    title: str
    module: str
    version: str
    section: str
    text: str
    access_scopes: tuple[str, ...] = ("public",)
    updated_at: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("document_id", "title", "module", "version", "section", "text"):
            _require_non_empty(getattr(self, field_name), field_name)
        object.__setattr__(self, "access_scopes", tuple(self.access_scopes))
        if not self.access_scopes:
            raise ValueError("access_scopes must not be empty")


@dataclass(frozen=True)
class RetrievalRequest:
    """Input to both the mock and future hybrid retrieval service."""

    request_id: str
    query: str
    modules: tuple[str, ...] = ()
    required_version: str | None = None
    access_scopes: tuple[str, ...] = ("public",)
    top_k: int = 5
    metadata_filters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.request_id, "request_id")
        _require_non_empty(self.query, "query")
        object.__setattr__(self, "modules", tuple(self.modules))
        object.__setattr__(self, "access_scopes", tuple(self.access_scopes))
        if not self.access_scopes:
            raise ValueError("access_scopes must not be empty")
        if not 1 <= self.top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    title: str
    module: str
    version: str
    section: str
    text: str
    score: float
    rank: int
    citation: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("chunk_id", "document_id", "title", "module", "version", "section", "text", "citation"):
            _require_non_empty(getattr(self, field_name), field_name)
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be normalized to [0, 1]")
        if self.rank <= 0:
            raise ValueError("rank must be positive")


@dataclass(frozen=True)
class RetrievalResponse:
    request_id: str
    rewritten_query: str
    chunks: tuple[RetrievedChunk, ...]
    retrieval_mode: str
    trace_id: str
    latency_ms: float
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.request_id, "request_id")
        _require_non_empty(self.rewritten_query, "rewritten_query")
        _require_non_empty(self.retrieval_mode, "retrieval_mode")
        _require_non_empty(self.trace_id, "trace_id")
        object.__setattr__(self, "chunks", tuple(self.chunks))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.latency_ms < 0:
            raise ValueError("latency_ms must not be negative")
        expected_ranks = list(range(1, len(self.chunks) + 1))
        if [chunk.rank for chunk in self.chunks] != expected_ranks:
            raise ValueError("chunk ranks must be contiguous and start at 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RagHealth:
    status: str
    backend: str
    document_count: int
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"UP", "DEGRADED", "DOWN"}:
            raise ValueError("health status must be UP, DEGRADED, or DOWN")
        if self.document_count < 0:
            raise ValueError("document_count must not be negative")


@dataclass(frozen=True)
class EnterpriseRagConfig:
    """Configuration consumed by the future production hybrid RAG client."""

    endpoint: str
    index_name: str
    embedding_model: str
    reranker_model: str
    timeout_seconds: float = 10.0
    dense_top_k: int = 30
    lexical_top_k: int = 30
    rerank_top_k: int = 10

    def __post_init__(self) -> None:
        for field_name in ("endpoint", "index_name", "embedding_model", "reranker_model"):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if min(self.dense_top_k, self.lexical_top_k, self.rerank_top_k) <= 0:
            raise ValueError("all top-k settings must be positive")
