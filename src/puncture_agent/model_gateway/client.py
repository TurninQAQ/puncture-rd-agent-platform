"""Model gateway interface and the intentionally unimplemented vLLM adapter."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from .models import GatewayHealth, ModelRequest, ModelResponse, ModelStreamEvent, VllmGatewayConfig


class ModelGatewayError(RuntimeError):
    """Normalized model provider failure consumed by Agent Runtime retry logic."""

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


class ModelGateway(Protocol):
    """Contract implemented by both MockQwenGateway and VllmModelGateway."""

    def health(self) -> GatewayHealth: ...

    def generate(self, request: ModelRequest) -> ModelResponse: ...

    def stream(self, request: ModelRequest) -> Iterator[ModelStreamEvent]: ...


class VllmModelGateway:
    """Production adapter placeholder.

    Implement this class according to specs/qwen-vllm-service.md and
    tasks/task-01-qwen-vllm.md. Do not change the public method signatures.
    """

    def __init__(self, config: VllmGatewayConfig) -> None:
        self.config = config

    def health(self) -> GatewayHealth:
        raise NotImplementedError("Implement the vLLM health probe in task-01")

    def generate(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError("Implement vLLM non-streaming generation in task-01")

    def stream(self, request: ModelRequest) -> Iterator[ModelStreamEvent]:
        raise NotImplementedError("Implement vLLM SSE streaming in task-01")
