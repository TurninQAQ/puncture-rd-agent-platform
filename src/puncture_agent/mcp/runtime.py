"""MCP-facing runtime for the strongly typed internal tool registry."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import time
from typing import Any, Mapping

from contracts.common import ToolResponseEnvelope
from contracts.enums import ErrorCode, ToolExecutionStatus
from contracts.errors import ErrorDetail
from puncture_agent.tooling.registry import ToolDefinition, ToolRegistry

from .codec import (
    ArtifactResolver,
    ContractDecodeError,
    decode_tool_context,
    decode_tool_request,
    to_mcp_safe_primitive,
)
from .schema import envelope_schema, request_schema


SERVER_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "case-data": (
        "inspect_case_metadata",
        "convert_mcs_to_nifti",
        "validate_label_schema",
    ),
    "segmentation": (
        "run_segmentation",
        "validate_segmentation_result",
        "extract_skin_surface",
    ),
    "planning-safety": (
        "generate_candidate_paths",
        "evaluate_path_safety",
        "evaluate_intraoperative_risk",
        "verify_skin_penetration",
    ),
}

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class McpPrincipal:
    """Transport-authenticated identity passed to the tool boundary."""

    subject: str
    allowed_case_ids: tuple[str, ...]
    allowed_tools: tuple[str, ...] = ("*",)

    def __post_init__(self) -> None:
        if not self.subject.strip():
            raise ValueError("principal subject is required")
        if not self.allowed_case_ids:
            raise ValueError("at least one allowed case ID is required")
        if not self.allowed_tools:
            raise ValueError("at least one allowed tool is required")

    def permits(self, *, tool_name: str, case_id: str, caller: str) -> bool:
        caller_matches = caller == self.subject
        case_allowed = "*" in self.allowed_case_ids or case_id in self.allowed_case_ids
        tool_allowed = "*" in self.allowed_tools or tool_name in self.allowed_tools
        return caller_matches and case_allowed and tool_allowed


@dataclass(frozen=True, slots=True)
class McpCallResult:
    content: tuple[dict[str, Any], ...]
    structured_content: dict[str, Any]
    is_error: bool
    meta: dict[str, Any]

    def to_protocol_result(self) -> dict[str, Any]:
        return {
            "content": [dict(item) for item in self.content],
            "structuredContent": dict(self.structured_content),
            "isError": self.is_error,
            "_meta": dict(self.meta),
        }


class McpToolRuntime:
    """Expose one logical MCP server over an internal :class:`ToolRegistry`."""

    def __init__(
        self,
        registry: ToolRegistry,
        artifact_resolver: ArtifactResolver,
        *,
        server_name: str,
    ) -> None:
        if server_name not in SERVER_TOOL_NAMES:
            raise ValueError(f"unknown MCP server: {server_name}")
        self._registry = registry
        self._artifact_resolver = artifact_resolver
        self.server_name = server_name
        self._tool_names = SERVER_TOOL_NAMES[server_name]
        registered = {definition.name for definition in registry.list_definitions()}
        missing = sorted(set(self._tool_names) - registered)
        if missing:
            raise ValueError("registry is missing MCP tools: " + ", ".join(missing))

    @property
    def tool_names(self) -> tuple[str, ...]:
        return self._tool_names

    def list_tools(self) -> tuple[dict[str, Any], ...]:
        tools = []
        for name in self._tool_names:
            definition = self._registry.get_definition(name)
            tools.append(
                {
                    "name": name,
                    "title": name.replace("_", " ").title(),
                    "description": definition.description,
                    "inputSchema": request_schema(definition.request_type),
                    "outputSchema": envelope_schema(definition.result_type),
                    "annotations": {
                        "readOnlyHint": definition.read_only,
                        "destructiveHint": definition.destructive,
                        "idempotentHint": definition.idempotent,
                        "openWorldHint": definition.open_world,
                    },
                    "execution": {"taskSupport": "forbidden"},
                    "_meta": {
                        "com.turninqaq/toolVersion": definition.version,
                        "com.turninqaq/server": self.server_name,
                        "com.turninqaq/defaultTimeoutMs": definition.default_timeout_ms,
                    },
                }
            )
        return tuple(tools)

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        principal: McpPrincipal,
    ) -> McpCallResult:
        if name not in self._tool_names:
            raise KeyError(f"unknown tool for {self.server_name}: {name}")
        definition = self._registry.get_definition(name)
        if not isinstance(arguments, Mapping):
            response = self._failure_from_arguments(
                definition,
                {},
                ErrorCode.INVALID_ARGUMENT,
                "tool arguments must be an object",
                field_path="$",
            )
            return self._to_call_result(response)
        try:
            context = decode_tool_context(arguments)
        except ContractDecodeError as exc:
            response = self._failure_from_arguments(
                definition,
                arguments,
                ErrorCode.INVALID_ARGUMENT,
                exc.message,
                field_path=exc.path,
            )
            return self._to_call_result(response)
        if not principal.permits(
            tool_name=name,
            case_id=context.case_id,
            caller=context.caller,
        ):
            response = self._failure(
                definition,
                request_id=context.request_id,
                trace_id=context.trace_id,
                code=ErrorCode.PERMISSION_DENIED,
                message="authenticated principal is not allowed to invoke this tool for the case",
                retryable=False,
            )
            return self._to_call_result(response)

        try:
            request = decode_tool_request(
                definition.request_type,
                arguments,
                artifact_resolver=self._artifact_resolver,
            )
        except ContractDecodeError as exc:
            response = self._failure_from_arguments(
                definition,
                arguments,
                ErrorCode.INVALID_ARGUMENT,
                exc.message,
                field_path=exc.path,
            )
            return self._to_call_result(response)

        timeout_seconds = self._timeout_seconds(definition, context.deadline_epoch_ms)
        if timeout_seconds <= 0:
            response = self._failure(
                definition,
                request_id=context.request_id,
                trace_id=context.trace_id,
                code=ErrorCode.TIMEOUT,
                message="tool deadline expired before execution",
                retryable=True,
            )
            return self._to_call_result(response)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"mcp-{name}")
        future = executor.submit(self._registry.execute, name, request)
        try:
            response = future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            # A Python thread cannot be killed safely.  Do not return while a
            # write-like handler can still mutate state in the background.
            # Production native/service adapters must honor the propagated
            # deadline cooperatively so this wait remains bounded.
            executor.shutdown(wait=True, cancel_futures=True)
            response = self._failure(
                definition,
                request_id=context.request_id,
                trace_id=context.trace_id,
                code=ErrorCode.TIMEOUT,
                message="tool execution exceeded its bounded deadline",
                retryable=True,
            )
            return self._to_call_result(response)
        except Exception as exc:  # defensive boundary around custom registries
            executor.shutdown(wait=True, cancel_futures=True)
            response = self._failure(
                definition,
                request_id=context.request_id,
                trace_id=context.trace_id,
                code=ErrorCode.INTERNAL_ERROR,
                message=f"tool runtime rejected an unexpected {type(exc).__name__}",
                retryable=False,
            )
            return self._to_call_result(response)
        executor.shutdown(wait=True, cancel_futures=True)

        if response.request_id != context.request_id or response.trace_id != context.trace_id:
            response = self._failure(
                definition,
                request_id=context.request_id,
                trace_id=context.trace_id,
                code=ErrorCode.CONTRACT_VIOLATION,
                message="tool response identity does not match the request",
                retryable=False,
            )
        elif response.tool_version != definition.version:
            response = self._failure(
                definition,
                request_id=context.request_id,
                trace_id=context.trace_id,
                code=ErrorCode.CONTRACT_VIOLATION,
                message="tool response version does not match the registry definition",
                retryable=False,
            )
        return self._to_call_result(response)

    @staticmethod
    def _timeout_seconds(definition: ToolDefinition, deadline_epoch_ms: int | None) -> float:
        timeout_ms = float(definition.default_timeout_ms)
        if deadline_epoch_ms is not None:
            remaining_ms = deadline_epoch_ms - time.time() * 1000.0
            timeout_ms = min(timeout_ms, remaining_ms)
        return max(0.0, timeout_ms / 1000.0)

    def _failure_from_arguments(
        self,
        definition: ToolDefinition,
        arguments: Mapping[str, Any],
        code: ErrorCode,
        message: str,
        *,
        field_path: str | None = None,
    ) -> ToolResponseEnvelope[Any]:
        context = arguments.get("context")
        context_mapping = context if isinstance(context, Mapping) else {}
        fingerprint = sha256(
            json.dumps(arguments, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        request_id = context_mapping.get("request_id")
        trace_id = context_mapping.get("trace_id")
        if not isinstance(request_id, str) or not request_id.strip():
            request_id = f"mcp-invalid-{fingerprint}"
        if not isinstance(trace_id, str) or not trace_id.strip():
            trace_id = f"mcp-invalid-trace-{fingerprint}"
        return self._failure(
            definition,
            request_id=request_id,
            trace_id=trace_id,
            code=code,
            message=message,
            retryable=False,
            field_path=field_path,
        )

    @staticmethod
    def _failure(
        definition: ToolDefinition,
        *,
        request_id: str,
        trace_id: str,
        code: ErrorCode,
        message: str,
        retryable: bool,
        field_path: str | None = None,
    ) -> ToolResponseEnvelope[Any]:
        now = _utc_now()
        return ToolResponseEnvelope(
            request_id=request_id,
            trace_id=trace_id,
            tool_name=definition.name,
            tool_version=definition.version,
            status=ToolExecutionStatus.FAILED,
            result=None,
            artifacts=(),
            metrics=(),
            warnings=(),
            error=ErrorDetail(
                code=code,
                message=message,
                retryable=retryable,
                field_path=field_path,
            ),
            started_at=now,
            finished_at=now,
        )

    @staticmethod
    def _to_call_result(response: ToolResponseEnvelope[Any]) -> McpCallResult:
        structured = to_mcp_safe_primitive(response)
        encoded = json.dumps(
            structured,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return McpCallResult(
            content=({"type": "text", "text": encoded},),
            structured_content=structured,
            is_error=response.status is ToolExecutionStatus.FAILED,
            meta={
                "com.turninqaq/requestId": response.request_id,
                "com.turninqaq/traceId": response.trace_id,
                "com.turninqaq/toolVersion": response.tool_version,
            },
        )
