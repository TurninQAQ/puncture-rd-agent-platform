"""Stable normalized RAG error type shared by retrieval and ingestion layers."""

from __future__ import annotations

from typing import Any


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


__all__ = ["RagServiceError"]
