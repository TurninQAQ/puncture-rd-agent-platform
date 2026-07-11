"""Stable runtime and repository error types."""

from __future__ import annotations


class RunServiceError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class RunRepositoryError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class RunRepositoryNotFound(RunRepositoryError):
    def __init__(self) -> None:
        super().__init__("NOT_FOUND", "run was not found")


class RunRepositoryIdempotencyConflict(RunRepositoryError):
    def __init__(self) -> None:
        super().__init__(
            "IDEMPOTENCY_CONFLICT",
            "idempotency key was used for a different request",
        )


class RunRepositoryVersionConflict(RunRepositoryError):
    def __init__(self) -> None:
        super().__init__("CONFLICT", "run state changed concurrently")


class RunRepositoryTransitionError(RunRepositoryError):
    def __init__(self, message: str = "run state transition is not allowed") -> None:
        super().__init__("CONFLICT", message)


class ExecutionSuperseded(RuntimeError):
    """The execution version no longer owns the Run and must stop publishing."""


__all__ = [
    "ExecutionSuperseded",
    "RunRepositoryError",
    "RunRepositoryIdempotencyConflict",
    "RunRepositoryNotFound",
    "RunRepositoryTransitionError",
    "RunRepositoryVersionConflict",
    "RunServiceError",
]
