"""RAG service interface and intentionally unimplemented production client."""

from __future__ import annotations

from typing import Any, Protocol

from .models import EnterpriseRagConfig, RagHealth, RetrievalRequest, RetrievalResponse


class RagServiceError(RuntimeError):
    """Normalized retrieval error for retry and routing decisions."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details or {}


class RagService(Protocol):
    def health(self) -> RagHealth: ...

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse: ...


class EnterpriseRagClient:
    """Production hybrid retrieval placeholder.

    Implement this class according to specs/rag-service.md and
    tasks/task-02-rag-service.md. Do not change its public signatures.
    """

    def __init__(self, config: EnterpriseRagConfig) -> None:
        self.config = config

    def health(self) -> RagHealth:
        raise NotImplementedError("Implement RAG backend health checks in task-02")

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        raise NotImplementedError("Implement hybrid retrieval in task-02")
