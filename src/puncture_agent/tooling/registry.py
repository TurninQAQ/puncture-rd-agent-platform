"""Small, framework-neutral registry for MCP adapters and local tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from contracts.common import ToolResponseEnvelope
from contracts.enums import ErrorCode, ToolExecutionStatus
from contracts.errors import ErrorDetail

ToolHandler = Callable[[Any], ToolResponseEnvelope[Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    version: str
    request_type: type[Any]
    result_type: type[Any]
    description: str
    read_only: bool = True
    destructive: bool = False
    idempotent: bool = True
    open_world: bool = False
    default_timeout_ms: int = 30_000

    def __post_init__(self) -> None:
        if not self.name or not self.version or not self.description:
            raise ValueError("tool name, version, and description are required")
        if self.default_timeout_ms <= 0:
            raise ValueError("default_timeout_ms must be positive")
        if self.read_only and self.destructive:
            raise ValueError("a read-only tool cannot be destructive")


class ToolRegistry:
    """Register handlers without coupling the contracts to MCP or LangGraph."""

    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"tool already registered: {definition.name}")
        self._definitions[definition.name] = definition
        self._handlers[definition.name] = handler

    def list_definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._definitions[name] for name in sorted(self._definitions))

    def get_definition(self, name: str) -> ToolDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def execute(self, name: str, request: Any) -> ToolResponseEnvelope[Any]:
        definition = self.get_definition(name)
        if not isinstance(request, definition.request_type):
            raise TypeError(
                f"{name} requires {definition.request_type.__name__}, "
                f"got {type(request).__name__}"
            )
        try:
            response = self._handlers[name](request)
        except Exception as exc:  # boundary: adapters must never crash the graph
            now = _utc_now()
            return ToolResponseEnvelope(
                request_id=request.context.request_id,
                trace_id=request.context.trace_id,
                tool_name=name,
                tool_version=definition.version,
                status=ToolExecutionStatus.FAILED,
                result=None,
                artifacts=(),
                metrics=(),
                warnings=(),
                error=ErrorDetail(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"unhandled tool exception: {type(exc).__name__}",
                    retryable=False,
                ),
                started_at=now,
                finished_at=now,
            )
        if not isinstance(response, ToolResponseEnvelope):
            raise TypeError(f"{name} handler did not return ToolResponseEnvelope")
        if response.tool_name != name:
            raise ValueError(f"handler response tool_name mismatch for {name}")
        if response.result is not None and not isinstance(response.result, definition.result_type):
            raise TypeError(
                f"{name} returned {type(response.result).__name__}; "
                f"expected {definition.result_type.__name__}"
            )
        return response
