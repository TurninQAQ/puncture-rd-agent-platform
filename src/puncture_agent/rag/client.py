"""RAG service facade with explicit provider dependency injection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from .backends import (
    BackendProtocolError,
    BackendTimeout,
    BackendUnavailable,
    DeterministicEmbeddingBackend,
    DeterministicReranker,
    EmbeddingBackend,
    InMemoryHybridIndex,
    RerankerBackend,
    SearchBackend,
    WritableSearchBackend,
)
from .errors import RagServiceError
from .ingestion import IngestionReport, RagIngestionService, SourceDocument
from .models import EnterpriseRagConfig, RagHealth, RetrievalRequest, RetrievalResponse
from .retrieval import (
    HybridRetrievalEngine,
    NullTraceSink,
    RagRuntimeConfig,
    RagTraceSink,
)


class RagService(Protocol):
    def health(self) -> RagHealth: ...

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse: ...


@dataclass(frozen=True)
class RagDependencies:
    """Explicit provider set; prevents an accidental production-to-mock fallback."""

    search_backend: SearchBackend
    embedding_backend: EmbeddingBackend
    reranker_backend: RerankerBackend | None
    trace_sink: RagTraceSink | None = None


class EnterpriseRagClient:
    """Provider-neutral hybrid RAG facade.

    ``config`` remains the original public configuration object.  Production-only
    backends and policy controls are optional keyword-only constructor extensions.
    Omitting dependencies fails fast; deterministic local behavior must be requested
    explicitly through :meth:`offline`.
    """

    def __init__(
        self,
        config: EnterpriseRagConfig,
        *,
        dependencies: RagDependencies | None = None,
        runtime: RagRuntimeConfig | None = None,
    ) -> None:
        if dependencies is None:
            raise RagServiceError(
                "RAG_INVALID_REQUEST",
                "RAG providers must be explicitly injected; use EnterpriseRagClient.offline for local development",
                retryable=False,
            )
        if dependencies.embedding_backend.model_name != config.embedding_model:
            raise RagServiceError(
                "RAG_INDEX_INCOMPATIBLE",
                "configured embedding model does not match the injected provider",
                retryable=False,
            )
        if dependencies.reranker_backend is not None and dependencies.reranker_backend.model_name != config.reranker_model:
            raise RagServiceError(
                "RAG_INVALID_REQUEST",
                "configured reranker model does not match the injected provider",
                retryable=False,
            )
        self.config = config
        self.dependencies = dependencies
        self.runtime = runtime or RagRuntimeConfig()
        self._engine = HybridRetrievalEngine(
            config,
            dependencies.search_backend,
            dependencies.embedding_backend,
            dependencies.reranker_backend,
            runtime=self.runtime,
            trace_sink=dependencies.trace_sink or NullTraceSink(),
        )
        self._ingestion = (
            RagIngestionService(dependencies.search_backend, dependencies.embedding_backend)
            if _is_writable_backend(dependencies.search_backend)
            else None
        )

    @classmethod
    def offline(
        cls,
        config: EnterpriseRagConfig,
        *,
        runtime: RagRuntimeConfig | None = None,
        trace_sink: RagTraceSink | None = None,
        embedding_dimension: int = 128,
    ) -> "EnterpriseRagClient":
        """Build an explicitly requested deterministic offline implementation."""

        search = InMemoryHybridIndex(config.index_name)
        embedding = DeterministicEmbeddingBackend(
            model_name=config.embedding_model,
            revision="offline-v1",
            dimension=embedding_dimension,
        )
        reranker = DeterministicReranker(model_name=config.reranker_model, revision="offline-v1")
        return cls(
            config,
            dependencies=RagDependencies(search, embedding, reranker, trace_sink),
            runtime=runtime,
        )

    def health(self) -> RagHealth:
        try:
            health = self.dependencies.search_backend.health()
        except (BackendTimeout, TimeoutError):
            return RagHealth(
                status="DOWN",
                backend="provider-neutral-search",
                document_count=0,
                details={"error_code": "RAG_TIMEOUT"},
            )
        except BackendUnavailable:
            return RagHealth(
                status="DOWN",
                backend="provider-neutral-search",
                document_count=0,
                details={"error_code": "RAG_BACKEND_UNAVAILABLE"},
            )
        except BackendProtocolError:
            return RagHealth(
                status="DOWN",
                backend="provider-neutral-search",
                document_count=0,
                details={"error_code": "RAG_PROTOCOL_ERROR"},
            )
        except Exception:
            return RagHealth(
                status="DOWN",
                backend="provider-neutral-search",
                document_count=0,
                details={"error_code": "RAG_PROTOCOL_ERROR"},
            )
        if (
            health.status not in {"UP", "DEGRADED", "DOWN"}
            or not isinstance(health.backend, str)
            or not health.backend.strip()
            or isinstance(health.document_count, bool)
            or not isinstance(health.document_count, int)
            or health.document_count < 0
            or isinstance(health.chunk_count, bool)
            or not isinstance(health.chunk_count, int)
            or health.chunk_count < 0
            or not isinstance(health.details, Mapping)
        ):
            return RagHealth(
                status="DOWN",
                backend="provider-neutral-search",
                document_count=0,
                details={"error_code": "RAG_PROTOCOL_ERROR"},
            )
        details = {
            **dict(health.details),
            "chunk_count": health.chunk_count,
            "embedding_model": self.dependencies.embedding_backend.model_name,
            "embedding_revision": self.dependencies.embedding_backend.revision,
            "reranker_model": getattr(self.dependencies.reranker_backend, "model_name", None),
            "reranker_revision": getattr(self.dependencies.reranker_backend, "revision", None),
        }
        return RagHealth(
            status=health.status,
            backend=health.backend,
            document_count=health.document_count,
            details=details,
        )

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        return self._engine.retrieve(request)

    def ingest(self, document: SourceDocument) -> IngestionReport:
        return self._require_ingestion().ingest(document)

    def update(self, document: SourceDocument, *, expected_checksum: str) -> IngestionReport:
        return self._require_ingestion().update(document, expected_checksum=expected_checksum)

    def rebuild(self, documents: tuple[SourceDocument, ...] | list[SourceDocument]) -> tuple[IngestionReport, ...]:
        return self._require_ingestion().rebuild(documents)

    def delete(self, document_id: str, version: str | None = None):
        return self._require_ingestion().delete(document_id, version)

    def close(self) -> None:
        """Release owned provider transports during application shutdown."""

        closed: set[int] = set()
        for dependency in (
            self.dependencies.search_backend,
            self.dependencies.embedding_backend,
            self.dependencies.reranker_backend,
        ):
            if dependency is None or id(dependency) in closed:
                continue
            closed.add(id(dependency))
            close = getattr(dependency, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "EnterpriseRagClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _require_ingestion(self) -> RagIngestionService:
        if self._ingestion is None:
            raise RagServiceError(
                "RAG_INVALID_REQUEST",
                "the injected search backend is read-only",
                retryable=False,
            )
        return self._ingestion


def _is_writable_backend(backend: SearchBackend) -> bool:
    return all(
        callable(getattr(backend, name, None))
        for name in ("upsert_document", "delete_document", "replace_all")
    )


__all__ = [
    "EnterpriseRagClient",
    "RagDependencies",
    "RagRuntimeConfig",
    "RagService",
    "RagServiceError",
]
